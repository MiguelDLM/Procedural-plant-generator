"""
Trait space: a species as a point in a normalised morphological vector space.

Each continuous trait (allometry, crown envelope, branching angles, tropisms,
leaf outline descriptors, venation, colours, bark) is mapped to [0, 1] using a
botanically plausible range (log-scaled for traits spanning orders of magnitude,
such as leaf length or DBH). Categorical traits (architectural model,
phyllotaxis, margin type, compound organisation...) are carried alongside.

This makes new variants a matter of arithmetic:
- blend(a, b, t)      morphological interpolation between two species,
- mutate(p, amount)   intraspecific variability around a species,
- distance(a, b)      morphological distance; nearest_species() for lookup.
"""

from dataclasses import dataclass
import copy
import math
import numpy as np


@dataclass(frozen=True)
class TraitSpec:
    path: str          # Dotted attribute path on BotanicalSpeciesPreset
    lo: float
    hi: float
    log: bool = False
    integer: bool = False
    cv: float = 0.06   # Typical intraspecific coefficient of variation (fraction of range)


TRAITS: list[TraitSpec] = [
    # Allometry
    TraitSpec("allometry.dbh_default_m", 0.03, 4.5, log=True, cv=0.10),
    TraitSpec("allometry.height_a", 0.5, 6.0, cv=0.04),
    TraitSpec("allometry.height_b", 0.40, 0.75, cv=0.03),
    TraitSpec("allometry.height_max_m", 4.0, 95.0, log=True, cv=0.05),
    TraitSpec("allometry.crown_radius_c", 0.05, 1.5, cv=0.06),
    TraitSpec("allometry.crown_radius_d", 0.35, 0.75, cv=0.03),
    TraitSpec("allometry.crown_depth_ratio", 0.15, 0.95, cv=0.06),
    TraitSpec("allometry.pipe_exponent_delta", 1.8, 3.0, cv=0.02),
    TraitSpec("allometry.buttress_amplitude", 0.0, 2.0, cv=0.10),
    TraitSpec("allometry.buttress_decay", 3.0, 25.0, cv=0.05),
    # Architecture
    TraitSpec("architecture.max_order", 0, 4, integer=True, cv=0.0),
    TraitSpec("architecture.branch_angle_mean_deg", 10.0, 95.0, cv=0.05),
    TraitSpec("architecture.twig_angle_mean_deg", 10.0, 95.0, cv=0.05),
    TraitSpec("architecture.divergence_angle_deg", 60.0, 180.0, cv=0.0),
    TraitSpec("architecture.whorl_size", 1, 7, integer=True, cv=0.0),
    TraitSpec("architecture.apical_dominance", 0.0, 1.0, cv=0.05),
    TraitSpec("architecture.gravitropism", -0.6, 0.8, cv=0.05),
    TraitSpec("architecture.phototropism", 0.0, 1.0, cv=0.05),
    TraitSpec("architecture.plagiotropy", 0.0, 1.0, cv=0.05),
    TraitSpec("architecture.internode_length_base_m", 0.1, 1.2, cv=0.06),
    TraitSpec("architecture.internode_decay_per_order", 0.3, 0.95, cv=0.04),
    TraitSpec("architecture.branch_frequency_per_meter", 0.5, 8.0, log=True, cv=0.08),
    TraitSpec("architecture.crookedness", 0.0, 0.6, cv=0.10),
    TraitSpec("architecture.crown_widest_position", 0.02, 0.95, cv=0.05),
    TraitSpec("architecture.crown_fullness", 0.3, 3.0, cv=0.05),
    TraitSpec("architecture.leaf_area_index", 0.5, 12.0, cv=0.08),
    # Leaf outline
    TraitSpec("leaf_morphology.blade_length_cm", 0.3, 150.0, log=True, cv=0.10),
    TraitSpec("leaf_morphology.aspect_ratio", 0.6, 120.0, log=True, cv=0.06),
    TraitSpec("leaf_morphology.widest_position", 0.05, 0.95, cv=0.04),
    TraitSpec("leaf_morphology.base_angle_deg", 10.0, 179.0, cv=0.05),
    TraitSpec("leaf_morphology.apex_angle_deg", 5.0, 179.0, cv=0.05),
    TraitSpec("leaf_morphology.base_curvature", -1.0, 1.0, cv=0.05),
    TraitSpec("leaf_morphology.apex_curvature", -1.0, 1.0, cv=0.05),
    TraitSpec("leaf_morphology.cordate_depth", 0.0, 1.0, cv=0.05),
    TraitSpec("leaf_morphology.base_asymmetry", 0.0, 1.0, cv=0.05),
    TraitSpec("leaf_morphology.apex_notch", 0.0, 0.5, cv=0.04),
    TraitSpec("leaf_morphology.falcate_bend", 0.0, 0.3, cv=0.04),
    TraitSpec("leaf_morphology.lobe_count", 1, 11, integer=True, cv=0.0),
    TraitSpec("leaf_morphology.lobe_depth", 0.0, 0.95, cv=0.06),
    TraitSpec("leaf_morphology.lobe_angle_deg", 15.0, 85.0, cv=0.05),
    TraitSpec("leaf_morphology.lobe_spread_deg", 10.0, 330.0, cv=0.04),
    TraitSpec("leaf_morphology.lobe_width", 0.2, 1.2, cv=0.05),
    TraitSpec("leaf_morphology.lobe_roundness", 0.0, 1.0, cv=0.05),
    TraitSpec("leaf_morphology.lobe_apex_angle_deg", 10.0, 179.0, cv=0.05),
    TraitSpec("leaf_morphology.teeth_count", 0, 80, integer=True, cv=0.08),
    TraitSpec("leaf_morphology.tooth_height_ratio", 0.0, 0.2, cv=0.08),
    TraitSpec("leaf_morphology.tooth_skew", 0.3, 0.92, cv=0.04),
    TraitSpec("leaf_morphology.leaflet_count", 1, 150, integer=True, cv=0.05),
    TraitSpec("leaf_morphology.leaflet_angle_deg", 5.0, 90.0, cv=0.05),
    TraitSpec("leaf_morphology.rachis_length_ratio", 0.2, 30.0, log=True, cv=0.05),
    TraitSpec("leaf_morphology.leaflet_size_gradient", 0.0, 0.8, cv=0.05),
    TraitSpec("leaf_morphology.petiole_length_ratio", 0.0, 1.5, cv=0.08),
    TraitSpec("leaf_morphology.petiole_angle_deg", 10.0, 110.0, cv=0.06),
    TraitSpec("leaf_morphology.mean_leaf_angle_deg", 5.0, 85.0, cv=0.06),
    TraitSpec("leaf_morphology.transverse_curl", 0.0, 1.0, cv=0.08),
    TraitSpec("leaf_morphology.longitudinal_droop", 0.0, 1.0, cv=0.08),
    TraitSpec("leaf_morphology.undulation_amplitude", 0.0, 0.5, cv=0.08),
    TraitSpec("leaf_morphology.gloss", 0.0, 1.0, cv=0.05),
    # Venation
    TraitSpec("venation.vla_mm_per_mm2", 1.0, 20.0, log=True, cv=0.06),
    TraitSpec("venation.secondary_vein_pairs", 0, 30, integer=True, cv=0.06),
    TraitSpec("venation.divergence_angle_deg", 10.0, 88.0, cv=0.04),
    TraitSpec("venation.secondary_curvature", 0.0, 1.0, cv=0.05),
    TraitSpec("venation.reticulation_density", 0.0, 1.0, cv=0.05),
    TraitSpec("venation.vein_contrast", 0.0, 1.5, cv=0.06),
    # Bark
    TraitSpec("bark.feature_scale_m", 0.005, 0.5, log=True, cv=0.08),
    TraitSpec("bark.relief", 0.0, 1.0, cv=0.06),
    TraitSpec("bark.onset_radius_cm", 0.1, 30.0, log=True, cv=0.1),
    TraitSpec("bark.weathering", 0.0, 1.0, cv=0.08),
    TraitSpec("bark.blockiness", 0.0, 1.0, cv=0.08),
    TraitSpec("bark.segments", 0.0, 1.0, cv=0.08),
    TraitSpec("bark.plate_tilt", 0.0, 1.0, cv=0.08),
    TraitSpec("bark.warp", 0.0, 1.0, cv=0.08),
    TraitSpec("bark.moss", 0.0, 1.0, cv=0.1),
    TraitSpec("bark.lichen", 0.0, 1.0, cv=0.1),
    # Roots
    TraitSpec("roots.lateral_count", 1, 16, integer=True, cv=0.0),
    TraitSpec("roots.spread_crown_ratio", 0.3, 4.0, cv=0.08),
    TraitSpec("roots.max_depth_m", 0.3, 60.0, log=True, cv=0.1),
    TraitSpec("roots.beta", 0.90, 0.99, cv=0.02),
    TraitSpec("roots.taproot_share", 0.0, 0.8, cv=0.06),
    TraitSpec("roots.surface_exposure", 0.0, 1.0, cv=0.06),
    TraitSpec("roots.plank", 0.0, 5.0, cv=0.06),
    TraitSpec("roots.buttress_height_dbh", 0.0, 4.0, cv=0.06),
    TraitSpec("roots.deep_roots", 0.0, 1.0, cv=0.06),
    TraitSpec("roots.heart_roots", 3, 16, integer=True, cv=0.0),
    TraitSpec("roots.fine_roots", 0.0, 3.0, cv=0.06),
    TraitSpec("roots.fine_orders", 0, 3, integer=True, cv=0.0),
    TraitSpec("roots.stilt_height_dbh", 0.5, 10.0, cv=0.06),
    TraitSpec("roots.tuber_count", 1, 20, integer=True, cv=0.0),
    TraitSpec("roots.drop_roots", 0, 40, integer=True, cv=0.0),
    TraitSpec("roots.pneumatophores", 0, 80, integer=True, cv=0.0),
    # Mechanics
    TraitSpec("biomechanics.wood_density_g_cm3", 0.15, 1.25, cv=0.05),
    TraitSpec("biomechanics.youngs_modulus_gpa", 3.0, 22.0, cv=0.05),
]

COLOR_TRAITS = [
    "leaf_morphology.adaxial_color",
    "leaf_morphology.abaxial_color",
    "leaf_morphology.vein_color",
    "leaf_morphology.autumn_color",
    "bark.base_color",
    "bark.secondary_color",
    "bark.young_color",
    "bark.inner_color",
]

CATEGORICAL_TRAITS = [
    "architecture.model",
    "architecture.phyllotaxis",
    "leaf_morphology.archetype",
    "leaf_morphology.margin_type",
    "leaf_morphology.lobe_type",
    "leaf_morphology.compound_type",
    "leaf_morphology.terminal_leaflet",
    "venation.pattern",
    "bark.pattern",
    "roots.system",
    "deciduous",
]


def get_path(obj, path: str):
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


def set_path(obj, path: str, value):
    parts = path.split(".")
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], value)


def _norm(spec: TraitSpec, value: float) -> float:
    v = float(np.clip(value, spec.lo, spec.hi))
    if spec.log:
        lo, hi = math.log(max(spec.lo, 1e-6)), math.log(spec.hi)
        return (math.log(max(v, 1e-6)) - lo) / (hi - lo)
    return (v - spec.lo) / (spec.hi - spec.lo)


def _denorm(spec: TraitSpec, x: float):
    x = float(np.clip(x, 0.0, 1.0))
    if spec.log:
        lo, hi = math.log(max(spec.lo, 1e-6)), math.log(spec.hi)
        v = math.exp(lo + x * (hi - lo))
    else:
        v = spec.lo + x * (spec.hi - spec.lo)
    return int(round(v)) if spec.integer else v


def to_vector(preset) -> np.ndarray:
    """Continuous trait vector in [0, 1] (scalar traits followed by colour channels)."""
    vals = [_norm(s, get_path(preset, s.path)) for s in TRAITS]
    for c in COLOR_TRAITS:
        vals.extend(float(x) for x in get_path(preset, c))
    return np.array(vals, dtype=float)


def from_vector(vec: np.ndarray, template):
    """Builds a new preset from a trait vector; categorical traits come from `template`."""
    out = copy.deepcopy(template)
    for i, s in enumerate(TRAITS):
        set_path(out, s.path, _denorm(s, vec[i]))
    k = len(TRAITS)
    for c in COLOR_TRAITS:
        set_path(out, c, tuple(float(np.clip(x, 0.0, 1.0)) for x in vec[k:k + 3]))
        k += 3
    return out


def blend(a, b, t: float):
    """Morphological interpolation; categorical traits switch at t = 0.5."""
    t = float(np.clip(t, 0.0, 1.0))
    vec = (1.0 - t) * to_vector(a) + t * to_vector(b)
    out = from_vector(vec, a if t < 0.5 else b)
    out.scientific_name = a.scientific_name if t < 0.5 else b.scientific_name
    if 0.0 < t < 1.0:
        out.common_name = f"{a.common_name} x {b.common_name} ({t:.2f})"
    return out


def mutate(preset, amount: float = 1.0, seed: int = 0):
    """Intraspecific variant: Gaussian perturbation scaled by each trait's CV."""
    rng = np.random.default_rng(seed)
    vec = to_vector(preset)
    sig = np.array([s.cv for s in TRAITS] + [0.025] * (3 * len(COLOR_TRAITS)))
    vec = np.clip(vec + rng.normal(0.0, 1.0, len(vec)) * sig * float(amount), 0.0, 1.0)
    return from_vector(vec, preset)


def distance(a, b) -> float:
    return float(np.linalg.norm(to_vector(a) - to_vector(b)) / math.sqrt(len(to_vector(a))))


def nearest_species(preset, catalog: dict, exclude: str = None) -> list[tuple[str, float]]:
    ranked = [(k, distance(preset, v)) for k, v in catalog.items() if k != exclude]
    return sorted(ranked, key=lambda kv: kv[1])


# -----------------------------------------------------------------------------
# Generic trait arithmetic for flat profiles (succulents): numeric fields are normalised by
# their UI ranges, colours interpolate per channel, categorical fields switch at t = 0.5.
# -----------------------------------------------------------------------------
def blend_profile(a, b, t: float, ranges: dict):
    from dataclasses import fields as dc_fields, replace as dc_replace
    t = float(np.clip(t, 0.0, 1.0))
    out = {}
    for f in dc_fields(a):
        va, vb = getattr(a, f.name), getattr(b, f.name)
        if isinstance(va, tuple):
            out[f.name] = tuple(float((1 - t) * x + t * y) for x, y in zip(va, vb))
        elif isinstance(va, bool) or not isinstance(va, (int, float)):
            out[f.name] = va if t < 0.5 else vb
        else:
            v = (1 - t) * va + t * vb
            out[f.name] = int(round(v)) if isinstance(va, int) else float(v)
    return dc_replace(a, **out)


def mutate_profile(p, amount: float, seed: int, ranges: dict, cv: float = 0.06):
    from dataclasses import fields as dc_fields, replace as dc_replace
    rng = np.random.default_rng(seed)
    out = {}
    for f in dc_fields(p):
        v = getattr(p, f.name)
        if f.name in ranges and isinstance(v, (int, float)) and not isinstance(v, bool):
            lo, hi = ranges[f.name]
            # Counts (ribs, spines, leaves) vary relative to their value; continuous traits by range
            sigma = cv * amount * (abs(v) if isinstance(v, int) else (hi - lo))
            nv = float(np.clip(v + rng.normal(0.0, sigma), lo, hi))
            out[f.name] = int(round(nv)) if isinstance(v, int) else nv
        elif isinstance(v, tuple):
            out[f.name] = tuple(float(np.clip(x + rng.normal(0.0, 0.02 * amount), 0.0, 1.0)) for x in v)
    return dc_replace(p, **out)
