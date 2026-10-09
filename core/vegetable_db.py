"""
Vegetable presets: root crops, potato and brassica heads.

Sizes are typical market sizes of common cultivars (cm); leaf traits use the tree leaf descriptors.
Storage roots after Esau (1940, carrot anatomy) and crop descriptions; brassica heads after Kieffer,
Fuller & Jellings (1998, Planta 206: 34-43) and Azpeitia et al. (2021, Science 373: 192-197).
"""

from dataclasses import dataclass, field

from .vegetable import VegetableProfile, StorageOrgan, HeadType
from .leaf_morphology import LeafMorphologyProfile, LeafArchetype, MarginType, ARCHETYPE_TEMPLATES
from .leaf_venation import VenationProfile, VenationPattern


@dataclass
class VegetablePreset:
    scientific_name: str
    common_name: str
    family: str
    notes: str
    profile: VegetableProfile
    leaf: LeafMorphologyProfile = field(default_factory=LeafMorphologyProfile)
    venation: VenationProfile = field(default_factory=VenationProfile)


O, HT, A, MT, V = StorageOrgan, HeadType, LeafArchetype, MarginType, VenationPattern


def _mix(a, b, t):
    return tuple(round(x * (1 - t) + y * t, 3) for x, y in zip(a, b))


def _typed(cls, kw: dict) -> dict:
    """Integers given for float fields become floats (exact JSON round trips)."""
    from dataclasses import fields
    floats = {f.name for f in fields(cls) if isinstance(getattr(cls(), f.name), float)}
    return {k: float(v) if k in floats and isinstance(v, int) and not isinstance(v, bool) else v
            for k, v in kw.items()}


def veg(sci, common, family, notes, leaf=None, vein=None, **kw) -> VegetablePreset:
    kw = _typed(VegetableProfile, kw)
    leaf = _typed(LeafMorphologyProfile, dict(leaf or {}))
    vein = _typed(VenationProfile, dict(vein or {}))
    archetype = leaf.pop("archetype", A.OVATE)
    lkw = dict(ARCHETYPE_TEMPLATES.get(archetype, {}))
    lkw.update(leaf)
    adaxial = lkw.get("adaxial_color", (0.20, 0.36, 0.10))
    lkw.setdefault("abaxial_color", _mix(adaxial, (0.62, 0.68, 0.50), 0.4))
    lkw.setdefault("vein_color", _mix(adaxial, (0.78, 0.82, 0.42), 0.5))
    return VegetablePreset(sci, common, family, notes, VegetableProfile(**kw),
                           LeafMorphologyProfile(archetype=archetype, **lkw), VenationProfile(**(vein or {})))


BRASSICA_LEAF = dict(archetype=A.OBOVATE, margin_type=MT.SINUATE, blade_length_cm=30.0, aspect_ratio=1.7,
                     petiole_length_ratio=0.35, undulation_amplitude=0.25, transverse_curl=0.35,
                     longitudinal_droop=0.3, adaxial_color=(0.30, 0.42, 0.36), gloss=0.15,
                     vein_color=(0.70, 0.76, 0.66))
BRASSICA_VEIN = dict(pattern=V.CRASPEDODROMOUS, secondary_vein_pairs=8, vein_contrast=0.9, primary_radius_mm=2.5)

VEGETABLE_CATALOG: dict[str, VegetablePreset] = {
    "daucus_carota": veg(
        "Daucus carota subsp. sativus", "Carrot (zanahoria)", "Apiaceae",
        "Conical orange taproot with green shoulders, fine growth rings and rootlets in vertical ranks; finely "
        "divided (tripinnate) leaves.",
        leaf=dict(archetype=A.PINNATE_COMPOUND, margin_type=MT.ENTIRE, blade_length_cm=3.0, aspect_ratio=3.5,
                  leaflet_count=9, leaflet_angle_deg=55, rachis_length_ratio=7.0, petiole_length_ratio=1.5,
                  longitudinal_droop=0.4, adaxial_color=(0.18, 0.38, 0.10)),
        vein=dict(pattern=V.CRASPEDODROMOUS, secondary_vein_pairs=3),
        organ=O.TAPROOT, leaf_count=10, leaf_size=1.0, elevation_outer_deg=45, elevation_inner_deg=78,
        root_length_cm=18.0, root_diameter_cm=3.2, widest_position=0.06, shoulder=0.45, taper=1.0, exposure=0.04,
        tail_cm=4.0, rings=0.45, rootlets=0.35, rootlet_ranks=4, root_color=(0.95, 0.45, 0.08),
        shoulder_color=(0.40, 0.48, 0.15), shoulder_tint=0.6, stem_color=(0.35, 0.50, 0.20)),
    "raphanus_sativus": veg(
        "Raphanus sativus", "Radish (rábano)", "Brassicaceae",
        "Globose red root, white at the tip, mostly above the soil; lyrate (pinnately lobed) rough leaves.",
        leaf=dict(archetype=A.PINNATE_LOBED, margin_type=MT.DENTATE, blade_length_cm=12.0, aspect_ratio=2.8,
                  widest_position=0.75, lobe_count=4, lobe_depth=0.6, teeth_count=14, tooth_height_ratio=0.03,
                  petiole_length_ratio=0.3, adaxial_color=(0.20, 0.38, 0.14)),
        vein=dict(pattern=V.CRASPEDODROMOUS, secondary_vein_pairs=5),
        organ=O.TAPROOT, leaf_count=8, elevation_outer_deg=40, elevation_inner_deg=75, root_length_cm=3.5,
        root_diameter_cm=3.0, widest_position=0.45, shoulder=0.5, taper=0.5, exposure=0.3, tail_cm=6.0,
        rings=0.05, rootlets=0.15, rootlet_ranks=2, root_color=(0.82, 0.08, 0.22), shoulder_tint=0.0,
        tip_color=(0.96, 0.94, 0.92), tip_tint=0.6, stem_color=(0.55, 0.20, 0.25)),
    "beta_vulgaris": veg(
        "Beta vulgaris subsp. vulgaris", "Beetroot (betabel)", "Amaranthaceae",
        "Round dark-red storage root with ring scars, the top above the soil; glossy leaves with red petioles and "
        "veins.",
        leaf=dict(archetype=A.OVATE, margin_type=MT.SINUATE, blade_length_cm=15.0, aspect_ratio=1.7,
                  petiole_length_ratio=0.9, undulation_amplitude=0.15, adaxial_color=(0.16, 0.30, 0.10),
                  vein_color=(0.62, 0.08, 0.15), gloss=0.6),
        vein=dict(pattern=V.BROCHIDODROMOUS, secondary_vein_pairs=8, vein_contrast=0.6),
        organ=O.TAPROOT, leaf_count=12, elevation_outer_deg=45, elevation_inner_deg=80, root_length_cm=7.0,
        root_diameter_cm=7.0, widest_position=0.5, shoulder=0.55, taper=0.45, exposure=0.3, tail_cm=10.0,
        rings=0.25, rootlets=0.2, rootlet_ranks=2, root_color=(0.35, 0.05, 0.12),
        shoulder_color=(0.40, 0.12, 0.15), shoulder_tint=0.3, stem_color=(0.60, 0.10, 0.18)),
    "brassica_rapa_rapa": veg(
        "Brassica rapa subsp. rapa", "Turnip (nabo)", "Brassicaceae",
        "Flattened globe, white below the soil and purple where the light reaches it.",
        leaf=dict(archetype=A.PINNATE_LOBED, margin_type=MT.DENTATE, blade_length_cm=20.0, aspect_ratio=2.6,
                  widest_position=0.7, lobe_count=4, lobe_depth=0.5, teeth_count=16, tooth_height_ratio=0.03,
                  petiole_length_ratio=0.3, adaxial_color=(0.22, 0.40, 0.16)),
        vein=dict(pattern=V.CRASPEDODROMOUS, secondary_vein_pairs=6),
        organ=O.TAPROOT, leaf_count=10, root_length_cm=7.0, root_diameter_cm=8.5, widest_position=0.4,
        shoulder=0.4, taper=0.45, exposure=0.35, tail_cm=8.0, rings=0.1, rootlets=0.2, rootlet_ranks=2,
        root_color=(0.94, 0.92, 0.86), shoulder_color=(0.55, 0.15, 0.45), shoulder_tint=0.9),
    "solanum_tuberosum": veg(
        "Solanum tuberosum", "Potato (papa)", "Solanaceae",
        "Bushy herb of several leafy stems with odd-pinnate leaves; stolons from the underground nodes end in "
        "tubers whose eyes spiral toward the rose end.",
        leaf=dict(archetype=A.PINNATE_COMPOUND, margin_type=MT.ENTIRE, blade_length_cm=6.0, aspect_ratio=1.6,
                  leaflet_count=3, leaflet_angle_deg=60, rachis_length_ratio=3.5, petiole_length_ratio=0.3,
                  undulation_amplitude=0.1, adaxial_color=(0.18, 0.36, 0.12)),
        vein=dict(pattern=V.BROCHIDODROMOUS, secondary_vein_pairs=6),
        organ=O.TUBERS, stem_count=5, stem_height_cm=45.0, stem_radius_mm=6.0, leaf_count=8, leaf_size=1.6,
        elevation_outer_deg=10, elevation_inner_deg=45, tuber_count=8, tuber_length_cm=8.0, tuber_diameter_cm=6.0,
        tuber_depth_cm=12.0, stolon_length_cm=15.0, eyes=9, eye_depth=0.1, tuber_color=(0.78, 0.62, 0.42),
        tuber_dots=0.3, stem_color=(0.30, 0.42, 0.18)),
    "brassica_oleracea_botrytis": veg(
        "Brassica oleracea var. botrytis", "Cauliflower (coliflor)", "Brassicaceae",
        "White curd of packed meristem domes in golden-angle spirals, wrapped by large glaucous leaves.",
        leaf=BRASSICA_LEAF, vein=BRASSICA_VEIN,
        organ=O.HEAD, stem_height_cm=12.0, stem_radius_mm=25.0, leaf_count=16, elevation_outer_deg=12,
        elevation_inner_deg=55, head_wrap=0.8, head_type=HT.CURD, head_diameter_cm=16.0, head_height_ratio=0.45,
        head_levels=3, florets=21, floret_scale=0.4, head_color=(0.95, 0.92, 0.80),
        branch_color=(0.90, 0.90, 0.75), stem_color=(0.60, 0.70, 0.55)),
    "brassica_oleracea_italica": veg(
        "Brassica oleracea var. italica", "Broccoli (brócoli)", "Brassicaceae",
        "Dome of green flower-bud clusters on thick pale branches, on a tall stem.",
        leaf=BRASSICA_LEAF, vein=BRASSICA_VEIN,
        organ=O.HEAD, stem_height_cm=25.0, stem_radius_mm=22.0, leaf_count=12, elevation_outer_deg=15,
        elevation_inner_deg=50, head_wrap=0.2, head_type=HT.BUDS, head_diameter_cm=14.0, head_height_ratio=0.5,
        head_levels=3, florets=13, floret_scale=0.4, bud_size_mm=2.0, head_color=(0.15, 0.30, 0.12),
        branch_color=(0.55, 0.68, 0.40), stem_color=(0.55, 0.68, 0.45)),
    "brassica_oleracea_romanesco": veg(
        "Brassica oleracea var. botrytis 'Romanesco'", "Romanesco (coliflor romanesco)", "Brassicaceae",
        "Lime-green conical head of cones on cones arranged in Fibonacci spirals: a natural fractal.",
        leaf=BRASSICA_LEAF, vein=BRASSICA_VEIN,
        organ=O.HEAD, stem_height_cm=12.0, stem_radius_mm=25.0, leaf_count=14, elevation_outer_deg=15,
        elevation_inner_deg=50, head_wrap=0.5, head_type=HT.CONES, head_diameter_cm=15.0, head_height_ratio=0.85,
        head_levels=3, florets=34, floret_scale=0.33, head_color=(0.55, 0.75, 0.25),
        branch_color=(0.60, 0.75, 0.35), stem_color=(0.60, 0.70, 0.50)),
}

VEGETABLE_RANGES = {
    "stem_count": (1, 12), "stem_height_cm": (0.0, 300.0), "stem_radius_mm": (0.5, 60.0), "leaf_count": (0, 60),
    "leaf_size": (0.1, 3.0), "elevation_outer_deg": (-10.0, 90.0), "elevation_inner_deg": (-10.0, 90.0),
    "head_wrap": (0.0, 1.0), "root_length_cm": (0.5, 100.0), "root_diameter_cm": (0.3, 40.0),
    "widest_position": (0.02, 0.9), "shoulder": (0.05, 1.5), "taper": (0.1, 3.0), "exposure": (0.0, 0.9),
    "tail_cm": (0.0, 40.0), "rings": (0.0, 1.0), "rootlets": (0.0, 1.0), "rootlet_ranks": (1, 8),
    "shoulder_tint": (0.0, 1.0), "tip_tint": (0.0, 1.0), "tuber_count": (0, 40), "tuber_length_cm": (0.5, 30.0),
    "tuber_diameter_cm": (0.5, 20.0), "tuber_depth_cm": (1.0, 50.0), "stolon_length_cm": (0.0, 80.0),
    "eyes": (0, 30), "eye_depth": (0.0, 0.4), "tuber_dots": (0.0, 1.0), "head_diameter_cm": (2.0, 50.0),
    "head_height_ratio": (0.1, 1.5), "head_levels": (1, 4), "florets": (3, 89), "floret_scale": (0.1, 0.7),
    "bud_size_mm": (0.5, 6.0),
}


def vegetable_items() -> list[tuple[str, str, str]]:
    return [(k, f"{v.scientific_name} ({v.common_name})", f"{v.family} · {v.profile.organ.value}. {v.notes}")
            for k, v in sorted(VEGETABLE_CATALOG.items(), key=lambda kv: kv[1].scientific_name)]
