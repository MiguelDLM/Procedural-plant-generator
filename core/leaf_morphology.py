"""
Botanical Leaf Morphology: a descriptor-driven, implicit-field leaf shape model.

Every foliar unit (simple leaf, lobed leaf, compound leaf, needle fascicle or
conifer spray) is decomposed into *blades*: an axis (the midvein of a lamina,
lobe, leaflet or needle) with a half-width profile w(t) along it. The full leaf
outline is the (smooth) union of its blades, evaluated as an implicit field
f(x, y) < 0 inside. This turns the qualitative vocabulary of the Manual of Leaf
Architecture (Ellis et al., 2009) into a small set of continuous numbers:

    aspect ratio (L/W), position of widest point, base angle, apex angle,
    base/apex curvature (concave ... convex), cordate depth, base asymmetry,
    apical notch, falcate bend, lobation (pinnate/palmate, count, sinus depth),
    margin (tooth type, count, height, skew) and compound organisation.

The half-width profile of a blade is two cubic Bezier segments (base -> widest
point -> apex) whose end tangents are the botanical base and apex angles, so a
cuneate, rounded, attenuate, acute or acuminate leaf is a direct function of
measurable angles rather than an ad-hoc polar harmonic series.

Coordinates are expressed in "blade length units" (BLU): 1.0 is the length of
one lamina (or leaflet / needle), +Y points from the petiole insertion to the
apex, +X is the right-hand side when viewed from the adaxial (upper) surface.
"""

from dataclasses import dataclass, field, fields, replace
from enum import Enum
import math
import numpy as np


class LeafArchetype(str, Enum):
    """Shape templates (starting points; every field remains editable)."""
    OVATE = "Ovate"
    ELLIPTIC = "Elliptic"
    OBOVATE = "Obovate"
    LANCEOLATE = "Lanceolate"
    LINEAR = "Linear"
    CORDATE = "Cordate"
    DELTOID = "Deltoid"
    ORBICULAR = "Orbicular"
    RHOMBIC = "Rhombic"
    FALCATE = "Falcate"
    PALMATE_LOBED = "Palmate_Lobed"
    PINNATE_LOBED = "Pinnate_Lobed"
    TULIP = "Tulip"
    FLABELLATE = "Flabellate"
    PINNATE_COMPOUND = "Pinnate_Compound"
    PALMATE_COMPOUND = "Palmate_Compound"
    NEEDLE_FASCICLE = "Needle_Fascicle"
    NEEDLE_SPRAY = "Needle_Spray"
    SCALE_SPRAY = "Scale_Spray"
    PALM_FROND = "Palm_Frond"


class MarginType(str, Enum):
    ENTIRE = "Entire"                  # Smooth (Magnolia, Ficus)
    SERRATE = "Serrate"                # Apically pointing teeth (Prunus)
    DOUBLY_SERRATE = "Doubly_Serrate"  # Teeth on teeth (Betula, Ulmus)
    DENTATE = "Dentate"                # Outward, symmetric teeth (Castanea)
    CRENATE = "Crenate"                # Rounded teeth (Populus tremula)
    SINUATE = "Sinuate"                # Shallow waves (Fagus)
    SPINOSE = "Spinose"                # Spine-tipped teeth (Ilex, Quercus agrifolia)


class LobeType(str, Enum):
    NONE = "None"
    PINNATE = "Pinnate"    # Lobes along the midvein (Quercus)
    PALMATE = "Palmate"    # Lobes radiating from the petiole (Acer)


class CompoundType(str, Enum):
    SIMPLE = "Simple"
    PINNATE = "Pinnate"    # Leaflets along a rachis (Fraxinus, Juglans)
    PALMATE = "Palmate"    # Leaflets from one point (Aesculus)
    FASCICLE = "Fascicle"  # Needles bundled at one point (Pinus)
    SPRAY = "Spray"        # Needles / scales along a shoot (Picea, Taxodium)


@dataclass
class LeafMorphologyProfile:
    """Measurable leaf descriptors. Lengths are relative to blade length unless noted."""
    archetype: LeafArchetype = LeafArchetype.OVATE
    margin_type: MarginType = MarginType.SERRATE

    # Lamina (or leaflet / needle) size and outline
    blade_length_cm: float = 8.5
    aspect_ratio: float = 1.65         # Length / width
    widest_position: float = 0.40      # 0 = base, 1 = apex (ovate < 0.5 < obovate)
    base_angle_deg: float = 90.0       # Angle enclosed by the base (acute < 90 < obtuse)
    apex_angle_deg: float = 70.0       # Angle enclosed by the apex
    base_curvature: float = 0.3        # -1 concave (attenuate) .. +1 convex (rounded)
    apex_curvature: float = 0.0        # -1 concave (acuminate) .. +1 convex (obtuse)
    cordate_depth: float = 0.0         # Basal auricle size (Tilia, Cercis)
    base_asymmetry: float = 0.0        # Oblique base (Ulmus, Tilia)
    apex_notch: float = 0.0            # Emarginate apex depth (Liriodendron, Ginkgo)
    falcate_bend: float = 0.0          # Sickle curvature of the midvein (Eucalyptus)

    # Lobation
    lobe_type: LobeType = LobeType.NONE
    lobe_count: int = 5                # Palmate: total lobes. Pinnate: lobes per side
    lobe_depth: float = 0.5            # Sinus depth: 0 = unlobed, 1 = to the midvein
    lobe_angle_deg: float = 50.0       # Pinnate lobe divergence from the midvein
    lobe_spread_deg: float = 200.0     # Palmate: total angle between outermost lobes
    lobe_width: float = 0.55           # Lobe breadth relative to its angular sector
    lobe_roundness: float = 0.5        # Sinus rounding (0 = V sinus, 1 = U sinus)
    lobe_apex_angle_deg: float = 80.0  # Apex angle of each lobe

    # Margin teeth
    teeth_count: int = 24              # Teeth per side along one lamina
    tooth_height_ratio: float = 0.035  # Tooth depth relative to lamina width
    tooth_skew: float = 0.75           # 0.5 symmetric (dentate) .. 0.9 strongly forward

    # Compound organisation / needle units
    compound_type: CompoundType = CompoundType.SIMPLE
    leaflet_count: int = 7             # Pinnate: pairs. Palmate/fascicle: total. Spray: per side
    leaflet_angle_deg: float = 60.0    # Leaflet / needle angle from rachis
    rachis_length_ratio: float = 2.5   # Rachis length / leaflet length
    terminal_leaflet: bool = True      # Odd-pinnate
    leaflet_size_gradient: float = 0.3 # Basal leaflets smaller by this fraction

    # Petiole
    petiole_length_ratio: float = 0.35
    petiole_radius_mm: float = 1.2
    petiole_angle_deg: float = 50.0    # Insertion angle between petiole and shoot
    mean_leaf_angle_deg: float = 40.0  # Mean lamina inclination from horizontal (planophile ~25, spherical ~57, erectophile ~70)

    # Physical / 3D
    thickness_mm: float = 0.22
    transverse_curl: float = 0.25
    longitudinal_droop: float = 0.35
    undulation_amplitude: float = 0.05

    # Colour (sRGB 0..1)
    adaxial_color: tuple = (0.20, 0.36, 0.10)
    abaxial_color: tuple = (0.42, 0.52, 0.30)
    vein_color: tuple = (0.55, 0.62, 0.30)
    autumn_color: tuple = (0.80, 0.55, 0.12)
    gloss: float = 0.35


def _template(**kw) -> dict:
    return kw


# Templates per archetype (values typical of the named exemplar taxa)
ARCHETYPE_TEMPLATES: dict[LeafArchetype, dict] = {
    LeafArchetype.OVATE: _template(aspect_ratio=1.6, widest_position=0.36, base_angle_deg=140, apex_angle_deg=60,
                                   base_curvature=0.5, apex_curvature=-0.1, lobe_type=LobeType.NONE,
                                   compound_type=CompoundType.SIMPLE),
    LeafArchetype.ELLIPTIC: _template(aspect_ratio=1.9, widest_position=0.5, base_angle_deg=80, apex_angle_deg=70,
                                      base_curvature=0.4, apex_curvature=0.3, lobe_type=LobeType.NONE,
                                      compound_type=CompoundType.SIMPLE),
    LeafArchetype.OBOVATE: _template(aspect_ratio=1.8, widest_position=0.66, base_angle_deg=55, apex_angle_deg=110,
                                     base_curvature=-0.1, apex_curvature=0.6, lobe_type=LobeType.NONE,
                                     compound_type=CompoundType.SIMPLE),
    LeafArchetype.LANCEOLATE: _template(aspect_ratio=4.5, widest_position=0.35, base_angle_deg=50, apex_angle_deg=25,
                                        base_curvature=0.2, apex_curvature=-0.3, lobe_type=LobeType.NONE,
                                        compound_type=CompoundType.SIMPLE),
    LeafArchetype.LINEAR: _template(aspect_ratio=9.0, widest_position=0.45, base_angle_deg=30, apex_angle_deg=20,
                                    base_curvature=0.0, apex_curvature=0.0, lobe_type=LobeType.NONE,
                                    compound_type=CompoundType.SIMPLE),
    LeafArchetype.CORDATE: _template(aspect_ratio=1.15, widest_position=0.3, base_angle_deg=170, apex_angle_deg=55,
                                     base_curvature=0.8, apex_curvature=-0.5, cordate_depth=0.35,
                                     lobe_type=LobeType.NONE, compound_type=CompoundType.SIMPLE),
    LeafArchetype.DELTOID: _template(aspect_ratio=1.1, widest_position=0.12, base_angle_deg=175, apex_angle_deg=65,
                                     base_curvature=0.7, apex_curvature=-0.2, lobe_type=LobeType.NONE,
                                     compound_type=CompoundType.SIMPLE),
    LeafArchetype.ORBICULAR: _template(aspect_ratio=1.02, widest_position=0.45, base_angle_deg=150,
                                       apex_angle_deg=130, base_curvature=0.9, apex_curvature=0.9,
                                       lobe_type=LobeType.NONE, compound_type=CompoundType.SIMPLE),
    LeafArchetype.RHOMBIC: _template(aspect_ratio=1.45, widest_position=0.42, base_angle_deg=100, apex_angle_deg=60,
                                     base_curvature=-0.4, apex_curvature=-0.4, lobe_type=LobeType.NONE,
                                     compound_type=CompoundType.SIMPLE),
    LeafArchetype.FALCATE: _template(aspect_ratio=6.0, widest_position=0.35, base_angle_deg=45, apex_angle_deg=20,
                                     base_curvature=0.1, apex_curvature=-0.4, falcate_bend=0.12,
                                     lobe_type=LobeType.NONE, compound_type=CompoundType.SIMPLE),
    LeafArchetype.PALMATE_LOBED: _template(aspect_ratio=1.0, widest_position=0.45, base_angle_deg=70,
                                           apex_angle_deg=45, base_curvature=0.2, apex_curvature=-0.3,
                                           lobe_type=LobeType.PALMATE, lobe_count=5, lobe_depth=0.5,
                                           lobe_spread_deg=210, compound_type=CompoundType.SIMPLE),
    LeafArchetype.PINNATE_LOBED: _template(aspect_ratio=1.9, widest_position=0.65, base_angle_deg=60,
                                           apex_angle_deg=110, base_curvature=0.2, apex_curvature=0.7,
                                           lobe_type=LobeType.PINNATE, lobe_count=4, lobe_depth=0.55,
                                           lobe_angle_deg=55, compound_type=CompoundType.SIMPLE),
    LeafArchetype.TULIP: _template(aspect_ratio=0.95, widest_position=0.55, base_angle_deg=165, apex_angle_deg=178,
                                   base_curvature=0.7, apex_curvature=0.9, apex_notch=0.16,
                                   lobe_type=LobeType.PINNATE, lobe_count=1, lobe_depth=0.45, lobe_angle_deg=70,
                                   lobe_apex_angle_deg=90, lobe_width=0.9, compound_type=CompoundType.SIMPLE),
    LeafArchetype.FLABELLATE: _template(aspect_ratio=1.25, widest_position=0.86, base_angle_deg=75,
                                        apex_angle_deg=178, base_curvature=-0.1, apex_curvature=1.0,
                                        apex_notch=0.14, lobe_type=LobeType.NONE,
                                        compound_type=CompoundType.SIMPLE),
    LeafArchetype.PINNATE_COMPOUND: _template(aspect_ratio=2.6, widest_position=0.45, base_angle_deg=70,
                                              apex_angle_deg=45, base_curvature=0.3, apex_curvature=-0.3,
                                              lobe_type=LobeType.NONE, compound_type=CompoundType.PINNATE,
                                              leaflet_count=4, leaflet_angle_deg=65, rachis_length_ratio=2.6),
    LeafArchetype.PALMATE_COMPOUND: _template(aspect_ratio=2.6, widest_position=0.68, base_angle_deg=40,
                                              apex_angle_deg=70, base_curvature=-0.3, apex_curvature=0.0,
                                              lobe_type=LobeType.NONE, compound_type=CompoundType.PALMATE,
                                              leaflet_count=7, lobe_spread_deg=230),
    LeafArchetype.NEEDLE_FASCICLE: _template(aspect_ratio=70.0, widest_position=0.5, base_angle_deg=20,
                                             apex_angle_deg=20, base_curvature=0.0, apex_curvature=0.0,
                                             lobe_type=LobeType.NONE, compound_type=CompoundType.FASCICLE,
                                             leaflet_count=2, lobe_spread_deg=24, margin_type=MarginType.ENTIRE),
    LeafArchetype.NEEDLE_SPRAY: _template(aspect_ratio=18.0, widest_position=0.5, base_angle_deg=30,
                                          apex_angle_deg=30, base_curvature=0.0, apex_curvature=0.2,
                                          lobe_type=LobeType.NONE, compound_type=CompoundType.SPRAY,
                                          leaflet_count=26, leaflet_angle_deg=55, rachis_length_ratio=6.0,
                                          margin_type=MarginType.ENTIRE),
    LeafArchetype.SCALE_SPRAY: _template(aspect_ratio=4.0, widest_position=0.3, base_angle_deg=60, apex_angle_deg=35,
                                         base_curvature=0.3, apex_curvature=0.0, lobe_type=LobeType.NONE,
                                         compound_type=CompoundType.SPRAY, leaflet_count=40,
                                         leaflet_angle_deg=25, rachis_length_ratio=12.0,
                                         margin_type=MarginType.ENTIRE),
    LeafArchetype.PALM_FROND: _template(aspect_ratio=18.0, widest_position=0.4, base_angle_deg=30, apex_angle_deg=15,
                                        base_curvature=0.0, apex_curvature=-0.2, lobe_type=LobeType.NONE,
                                        compound_type=CompoundType.PINNATE, leaflet_count=60,
                                        leaflet_angle_deg=40, rachis_length_ratio=9.0, terminal_leaflet=False,
                                        leaflet_size_gradient=0.6, margin_type=MarginType.ENTIRE),
}


def apply_archetype_template(profile: LeafMorphologyProfile, archetype: LeafArchetype) -> LeafMorphologyProfile:
    """Returns a copy of `profile` re-shaped to the given archetype template."""
    return replace(profile, archetype=archetype, **ARCHETYPE_TEMPLATES.get(archetype, {}))


# -----------------------------------------------------------------------------
# Blade primitive
# -----------------------------------------------------------------------------
T_GRID = np.linspace(0.0, 1.0, 129)


def _bezier(p0, p1, p2, p3, n):
    t = np.linspace(0.0, 1.0, n)[:, None]
    return ((1 - t) ** 3) * p0 + 3 * ((1 - t) ** 2) * t * p1 + 3 * (1 - t) * (t ** 2) * p2 + (t ** 3) * p3


def half_width_profile(
    aspect_ratio: float,
    widest_position: float,
    base_angle_deg: float,
    apex_angle_deg: float,
    base_curvature: float,
    apex_curvature: float,
) -> np.ndarray:
    """
    Half-width w(t) of a unit-length lamina sampled on T_GRID.

    The margin is two cubic Bezier segments: base (0,0) -> widest point (W,p),
    leaving the base at half the base angle from the midvein and arriving
    vertically; then widest point -> apex (0,1), arriving at half the apex angle.
    Handle lengths encode curvature (concave = short outer handle, convex = long).
    """
    W = 0.5 / max(0.3, aspect_ratio)
    p = float(np.clip(widest_position, 0.05, 0.95))
    hb = math.radians(np.clip(base_angle_deg, 5.0, 179.0) * 0.5)
    ha = math.radians(np.clip(apex_angle_deg, 5.0, 179.0) * 0.5)
    cb = float(np.clip(base_curvature, -1.0, 1.0))
    ca = float(np.clip(apex_curvature, -1.0, 1.0))

    # Base segment
    b0 = np.array([0.0, 0.0])
    b3 = np.array([W, p])
    d_b = math.hypot(W, p)
    h0 = d_b * (0.20 + 0.40 * (cb + 1.0) * 0.5)
    h1 = d_b * (0.15 + 0.35 * (cb + 1.0) * 0.5)
    b1 = b0 + h0 * np.array([math.sin(hb), math.cos(hb)])
    b2 = b3 - np.array([0.0, min(h1, p * 0.95)])
    b1[1] = min(b1[1], b2[1])
    base = _bezier(b0, b1, b2, b3, 200)

    # Apex segment
    a0 = b3
    a3 = np.array([0.0, 1.0])
    d_a = math.hypot(W, 1.0 - p)
    h2 = d_a * (0.15 + 0.40 * (ca + 1.0) * 0.5)
    h3 = d_a * (0.45 - 0.30 * (ca + 1.0) * 0.5)
    a1 = a0 + np.array([0.0, min(h2, (1.0 - p) * 0.95)])
    a2 = a3 - h3 * np.array([-math.sin(ha), math.cos(ha)])
    a2[0] = max(0.0, a2[0])
    a2[1] = max(a2[1], a1[1])
    apex = _bezier(a0, a1, a2, a3, 200)

    pts = np.vstack([base, apex[1:]])
    ys = np.maximum.accumulate(pts[:, 1])
    ys, idx = np.unique(ys, return_index=True)
    xs = np.maximum(pts[idx, 0], 0.0)
    w = np.interp(T_GRID, ys, xs)
    w[0] = 0.0
    w[-1] = 0.0
    return w


@dataclass
class ToothSpec:
    margin: MarginType
    count: int
    height: float  # absolute BLU
    skew: float


@dataclass
class Blade:
    """One lamina-like element: leaf blade, lobe, leaflet, needle or rachis."""
    origin: np.ndarray            # (2,) BLU
    angle: float                  # Radians from +Y, positive rotates toward +X
    length: float                 # BLU
    half_width: np.ndarray        # Half-width table on T_GRID (BLU, unit-length shape * length)
    teeth: ToothSpec | None = None
    vein_order: int = 1           # Order of this blade's own midvein
    secondary_pairs: int = 0      # -1 = derived from the venation profile, 0 = none
    notch: float = 0.0            # Emarginate apex depth (fraction of length)
    asymmetry: float = 0.0
    bend: float = 0.0             # Falcate bow (fraction of length)
    blend: float = 0.0            # Smooth-union radius with previous blades (BLU)
    is_axis: bool = False         # Petiole / rachis (drawn as stalk tissue)
    role: str = "lamina"

    @property
    def axis(self) -> np.ndarray:
        return np.array([math.sin(self.angle), math.cos(self.angle)])

    @property
    def normal(self) -> np.ndarray:
        return np.array([math.cos(self.angle), -math.sin(self.angle)])

    @property
    def max_half_width(self) -> float:
        return float(np.max(self.half_width))

    def local_to_global(self, t_abs: np.ndarray, s: np.ndarray) -> np.ndarray:
        """Converts blade-local (t along axis in BLU, s lateral) to global XY."""
        tn = np.clip(t_abs / max(1e-6, self.length), 0.0, 1.0)
        s_bent = s + self.bend * self.length * tn ** 2
        return self.origin[None, :] + t_abs[:, None] * self.axis[None, :] + s_bent[:, None] * self.normal[None, :]

    def width_at(self, tn: np.ndarray) -> np.ndarray:
        return np.interp(tn, T_GRID, self.half_width, left=0.0, right=0.0)

    def tooth_wave(self, tn: np.ndarray) -> np.ndarray:
        """Margin offset (<= 0 cuts sinuses into the envelope, > 0 spines)."""
        spec = self.teeth
        if spec is None or spec.margin == MarginType.ENTIRE or spec.count <= 0 or spec.height <= 0.0:
            return np.zeros_like(tn)
        phase = tn * spec.count
        frac = phase - np.floor(phase)
        skew = float(np.clip(spec.skew, 0.1, 0.9))
        saw = np.where(frac < skew, frac / skew, (1.0 - frac) / (1.0 - skew))
        m = spec.margin
        if m == MarginType.SERRATE:
            wave = saw ** 1.3 - 1.0
        elif m == MarginType.DOUBLY_SERRATE:
            f2 = (phase * 3.0) % 1.0
            saw2 = np.where(f2 < skew, f2 / skew, (1.0 - f2) / (1.0 - skew))
            wave = (saw ** 1.3 - 1.0) + 0.35 * (saw2 - 1.0) * saw
        elif m == MarginType.DENTATE:
            tri = 1.0 - np.abs(2.0 * frac - 1.0)
            wave = tri - 1.0
        elif m == MarginType.CRENATE:
            wave = np.sqrt(np.clip(np.sin(math.pi * frac), 0.0, 1.0)) - 1.0
        elif m == MarginType.SINUATE:
            wave = 0.5 * (np.cos(2.0 * math.pi * frac) - 1.0)
        elif m == MarginType.SPINOSE:
            spike = (1.0 - np.abs(2.0 * frac - 1.0)) ** 4
            wave = 1.6 * spike - 0.6
        else:
            wave = np.zeros_like(tn)
        envelope = np.clip((tn - 0.06) / 0.18, 0.0, 1.0) * np.clip((0.97 - tn) / 0.12, 0.0, 1.0)
        return wave * spec.height * envelope

    def field(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Approximate signed distance (BLU): negative inside the blade."""
        dx = X - self.origin[0]
        dy = Y - self.origin[1]
        ax, ay = self.axis
        nx, ny = self.normal
        t_abs = dx * ax + dy * ay
        s = dx * nx + dy * ny
        L = max(1e-6, self.length)
        tn = t_abs / L
        tnc = np.clip(tn, 0.0, 1.0)
        if self.bend != 0.0:
            s = s - self.bend * L * tnc ** 2
        if self.asymmetry != 0.0:
            # Oblique base: one side starts higher on the midvein
            shift = self.asymmetry * 0.18
            tn_side = np.where(s < 0.0, (tnc - shift) / (1.0 - shift), tnc)
            w = np.where(tn_side < 0.0, 0.0, self.width_at(np.clip(tn_side, 0.0, 1.0)))
        else:
            w = self.width_at(tnc)
        w = w + self.tooth_wave(tnc)
        d = np.abs(s) - w
        d = np.maximum(d, np.maximum(-t_abs, t_abs - L))
        if self.notch > 0.0:
            nd = self.notch
            depth_in = (tn - (1.0 - nd)) / nd
            notch_w = self.max_half_width * 0.35
            d = np.maximum(d, np.where(depth_in > 0.0, depth_in * notch_w - np.abs(s), -1e9))
        return d

    def bbox(self) -> tuple[float, float, float, float]:
        t = np.array([0.0, self.length * 0.5, self.length])
        wmax = self.max_half_width * 1.6 + (self.teeth.height if self.teeth else 0.0) + 1e-3
        pts = []
        for s in (-wmax, wmax):
            pts.append(self.local_to_global(t, np.full(3, s)))
        pts = np.vstack(pts)
        return float(pts[:, 0].min()), float(pts[:, 1].min()), float(pts[:, 0].max()), float(pts[:, 1].max())


def smooth_min(a: np.ndarray, b: np.ndarray, k: float) -> np.ndarray:
    if k <= 1e-9:
        return np.minimum(a, b)
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return b * (1.0 - h) + a * h - k * h * (1.0 - h)


# -----------------------------------------------------------------------------
# Leaf shape model (blade decomposition)
# -----------------------------------------------------------------------------
@dataclass
class LeafShapeModel:
    blades: list[Blade]
    petiole_length: float          # BLU
    petiole_half_width: float      # BLU
    bounds: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)
    laminae: list[int] = field(default_factory=list)  # Blade indices that carry venation

    def field(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        out = np.full(X.shape, 1e3, dtype=float)
        for blade in self.blades:
            out = smooth_min(out, blade.field(X, Y), blade.blend)
        return out


class LeafShapeBuilder:
    """Compiles a LeafMorphologyProfile into blades."""

    def __init__(self, profile: LeafMorphologyProfile):
        self.p = profile

    def _lamina_profile(self, aspect=None, widest=None, apex_angle=None, base_angle=None,
                        base_curv=None, apex_curv=None) -> np.ndarray:
        p = self.p
        return half_width_profile(
            aspect if aspect is not None else p.aspect_ratio,
            widest if widest is not None else p.widest_position,
            base_angle if base_angle is not None else p.base_angle_deg,
            apex_angle if apex_angle is not None else p.apex_angle_deg,
            base_curv if base_curv is not None else p.base_curvature,
            apex_curv if apex_curv is not None else p.apex_curvature,
        )

    def _teeth(self, width_blu: float, count: int | None = None) -> ToothSpec | None:
        p = self.p
        if p.margin_type == MarginType.ENTIRE:
            return None
        return ToothSpec(p.margin_type, int(count if count is not None else p.teeth_count),
                         p.tooth_height_ratio * width_blu * 2.0, p.tooth_skew)

    def _simple_lamina(self, origin, angle, length, secondary_pairs, vein_order=1, role="lamina") -> list[Blade]:
        """Unlobed lamina (+ cordate auricles) of a given length."""
        p = self.p
        hw = self._lamina_profile() * length
        width = float(hw.max()) * 2.0
        main = Blade(np.asarray(origin, float), angle, length, hw, self._teeth(width),
                     vein_order, secondary_pairs, notch=p.apex_notch, asymmetry=p.base_asymmetry,
                     bend=p.falcate_bend, role=role)
        blades = [main]
        if p.cordate_depth > 0.01:
            aur_len = length * (0.12 + 0.30 * p.cordate_depth)
            aur_hw = half_width_profile(1.1, 0.5, 160, 160, 1.0, 1.0) * aur_len
            for side in (-1.0, 1.0):
                a = angle + side * math.radians(112.0)
                start = np.asarray(origin, float) + main.axis * length * 0.10 + main.normal * side * length * 0.04
                blades.append(Blade(start, a, aur_len, aur_hw, None, vein_order + 1, 0,
                                    blend=length * 0.16, role="auricle"))
        return blades

    def build(self) -> LeafShapeModel:
        p = self.p
        blades: list[Blade] = []
        laminae: list[int] = []
        sec = 0

        petiole_len = max(0.0, p.petiole_length_ratio)
        petiole_hw = max(0.004, (p.petiole_radius_mm * 0.1) / max(0.5, p.blade_length_cm))

        def add(bl_list, venated=True):
            for b in bl_list:
                if venated and b.role != "auricle":
                    laminae.append(len(blades))
                blades.append(b)

        ct = p.compound_type
        if ct == CompoundType.SIMPLE:
            if p.lobe_type == LobeType.PALMATE:
                add(self._palmate_lobes())
            elif p.lobe_type == LobeType.PINNATE:
                add(self._pinnate_lobes())
            else:
                add(self._simple_lamina((0.0, 0.0), 0.0, 1.0, -1))
        elif ct == CompoundType.PINNATE:
            add(self._pinnate_compound(petiole_hw))
        elif ct == CompoundType.PALMATE:
            add(self._palmate_compound())
        elif ct == CompoundType.FASCICLE:
            add(self._fascicle())
        elif ct == CompoundType.SPRAY:
            add(self._spray(petiole_hw))

        # Petiole as stalk tissue from (0, -petiole_len) to the insertion point
        if petiole_len > 1e-3:
            stalk = Blade(np.array([0.0, -petiole_len]), 0.0, petiole_len,
                          np.full(T_GRID.shape, petiole_hw) * np.clip(T_GRID * 40.0, 0.4, 1.0),
                          None, 1, 0, is_axis=True, role="petiole")
            stalk.half_width[-1] = petiole_hw
            blades.append(stalk)

        model = LeafShapeModel(blades=blades, petiole_length=petiole_len, petiole_half_width=petiole_hw,
                               laminae=laminae)
        boxes = np.array([b.bbox() for b in blades])
        pad = 0.02
        model.bounds = (float(boxes[:, 0].min() - pad), float(min(boxes[:, 1].min(), -petiole_len) - pad),
                        float(boxes[:, 2].max() + pad), float(boxes[:, 3].max() + pad))
        return model

    # --- lobed simple leaves -------------------------------------------------
    def _palmate_lobes(self) -> list[Blade]:
        p = self.p
        k = max(1, int(p.lobe_count))
        spread = math.radians(np.clip(p.lobe_spread_deg, 10.0, 330.0))
        angles = np.linspace(-spread / 2, spread / 2, k) if k > 1 else np.array([0.0])
        sector = spread / max(1, k - 1) if k > 1 else math.radians(60)
        blades: list[Blade] = []
        depth = float(np.clip(p.lobe_depth, 0.0, 0.97))
        for a in angles:
            rel = abs(a) / max(1e-6, spread / 2) if k > 1 else 0.0
            L = 1.0 - 0.45 * rel ** 1.6
            # Lobe breadth from its angular sector; deeper sinuses -> free-standing narrower lobes
            hw_target = max(0.04, p.lobe_width * math.sin(min(sector, math.pi * 0.9) / 2) * L * (1.2 - 0.4 * depth))
            aspect = L / (2.0 * hw_target)
            hw = self._lamina_profile(aspect=aspect, apex_angle=p.lobe_apex_angle_deg,
                                      base_angle=60.0, base_curv=0.0) * L
            blade = Blade(np.zeros(2), float(a), L, hw, self._teeth(hw_target * 2.0, max(3, int(p.teeth_count * L))),
                          1, -1, notch=p.apex_notch if abs(a) < 1e-6 else 0.0,
                          blend=0.06 * p.lobe_roundness, role="lobe")
            blades.append(blade)
        # Palm: the undivided central lamina, its radius set by sinus depth
        palm_r = max(0.05, (1.0 - depth) * 0.78)
        palm_hw = half_width_profile(1.0, 0.5, 170, 170, 1.0, 1.0) * palm_r * 1.25
        palm = Blade(np.array([0.0, -palm_r * 0.15]), 0.0, palm_r * 1.25, palm_hw, None, 1, 0,
                     blend=0.03 + 0.10 * p.lobe_roundness, role="palm")
        return [palm] + blades

    def _pinnate_lobes(self) -> list[Blade]:
        p = self.p
        env = self._lamina_profile()  # Overall envelope (unit length)
        depth = float(np.clip(p.lobe_depth, 0.0, 0.95))
        # Central body: full envelope at the terminal lobe, narrowed elsewhere
        taper = np.clip((T_GRID - 0.78) / 0.15, 0.0, 1.0)
        body_hw = env * (1.0 - depth * (1.0 - taper)) + 0.004
        body_hw[0] = 0.0
        body_hw[-1] = 0.0
        body = Blade(np.zeros(2), 0.0, 1.0, body_hw, self._teeth(float(env.max()) * 2, max(2, p.teeth_count // 4)),
                     1, 0, notch=p.apex_notch, role="body")
        blades = [body]
        n = max(1, int(p.lobe_count))
        alpha = math.radians(np.clip(p.lobe_angle_deg, 15.0, 85.0))
        t0, t1 = 0.14, 0.80
        spacing = (t1 - t0) / n
        for i in range(n):
            ti = t0 + spacing * (i + 0.5)
            for side in (-1.0, 1.0):
                # Lobe length: ray from the midvein until it meets the envelope
                ls = np.linspace(0.0, 0.8, 200)
                xs = ls * math.sin(alpha)
                ys = ti + ls * math.cos(alpha)
                inside = xs <= np.interp(np.clip(ys, 0, 1), T_GRID, env) * 1.02
                L = float(ls[np.argmax(~inside)] if np.any(~inside) else ls[-1])
                L = max(0.05, L)
                hw_target = max(0.02, spacing * math.sin(alpha) * 0.5 * (0.6 + 0.7 * p.lobe_width))
                hw = half_width_profile(L / (2 * hw_target), 0.55, 80, p.lobe_apex_angle_deg, 0.3, 0.4) * L
                blades.append(Blade(np.array([0.0, ti]), side * alpha, L, hw,
                                    self._teeth(hw_target * 2.0, max(1, p.teeth_count // max(1, n))), 2, -1,
                                    blend=spacing * (0.10 + 0.35 * p.lobe_roundness), role="lobe"))
        return blades

    # --- compound leaves and needle units ------------------------------------
    def _leaflet(self, origin, angle, scale) -> Blade:
        hw = self._lamina_profile() * scale
        return Blade(np.asarray(origin, float), angle, scale, hw, self._teeth(float(hw.max()) * 2.0), 1, -1,
                     notch=self.p.apex_notch, asymmetry=self.p.base_asymmetry * 0.5, bend=self.p.falcate_bend,
                     role="leaflet")

    def _pinnate_compound(self, rachis_hw: float) -> list[Blade]:
        p = self.p
        R = max(0.2, p.rachis_length_ratio)
        n = max(1, int(p.leaflet_count))
        alpha = math.radians(np.clip(p.leaflet_angle_deg, 10.0, 90.0))
        blades = []
        rachis = Blade(np.zeros(2), 0.0, R, np.full(T_GRID.shape, rachis_hw) * (1.0 - 0.6 * T_GRID),
                       None, 1, 0, is_axis=True, role="rachis")
        blades.append(rachis)
        for i in range(n):
            ti = R * (0.10 + 0.86 * (i / max(1, n - 1) if n > 1 else 0.5))
            scale = 1.0 - p.leaflet_size_gradient * (1.0 - i / max(1, n - 1)) if n > 1 else 1.0
            if p.terminal_leaflet is False and n > 6:
                # Fronds: leaflets shorten toward the tip as well
                scale *= 1.0 - 0.55 * (i / (n - 1)) ** 2
            for side in (-1.0, 1.0):
                blades.append(self._leaflet((0.0, ti), side * alpha, max(0.1, scale)))
        if p.terminal_leaflet:
            blades.append(self._leaflet((0.0, R), 0.0, 1.0))
        return blades

    def _palmate_compound(self) -> list[Blade]:
        p = self.p
        k = max(1, int(p.leaflet_count))
        spread = math.radians(np.clip(p.lobe_spread_deg, 10.0, 330.0))
        angles = np.linspace(-spread / 2, spread / 2, k) if k > 1 else np.array([0.0])
        blades = []
        for a in angles:
            rel = abs(a) / max(1e-6, spread / 2) if k > 1 else 0.0
            blades.append(self._leaflet((0.0, 0.0) + 0.03 * np.array([math.sin(a), math.cos(a)]), float(a),
                                        1.0 - 0.45 * rel ** 1.5))
        return blades

    def _needle(self, origin, angle, scale) -> Blade:
        hw = self._lamina_profile() * scale
        return Blade(np.asarray(origin, float), angle, scale, hw, None, 1, 0, role="needle")

    def _fascicle(self) -> list[Blade]:
        p = self.p
        k = max(1, int(p.leaflet_count))
        spread = math.radians(np.clip(p.lobe_spread_deg, 0.0, 120.0))
        angles = np.linspace(-spread / 2, spread / 2, k) if k > 1 else np.array([0.0])
        blades = [self._needle((0.0, 0.0), float(a), 1.0) for a in angles]
        sheath = Blade(np.array([0.0, -0.02]), 0.0, 0.06, np.full(T_GRID.shape, 0.012), None, 1, 0,
                       is_axis=True, role="sheath")
        return blades + [sheath]

    def _spray(self, rachis_hw: float) -> list[Blade]:
        """Shoot with needles/scales on both flanks plus alternate lateral sub-shoots (flattened branchlet)."""
        p = self.p
        R = max(0.5, p.rachis_length_ratio)
        rng = np.random.default_rng(7)
        blades = self._needle_shoot(np.zeros(2), 0.0, R, max(2, int(p.leaflet_count)), rachis_hw, rng)
        n_sub = int(np.clip(round(R / 2.5), 0, 6))
        for k in range(n_sub):
            t = R * (0.15 + 0.6 * k / max(1, n_sub))
            side = 1.0 if k % 2 == 0 else -1.0
            sub_len = R * 0.55 * (1.0 - 0.5 * k / max(1, n_sub))
            blades += self._needle_shoot(np.array([0.0, t]), side * math.radians(42.0), sub_len,
                                         max(2, int(p.leaflet_count * sub_len / R)), rachis_hw * 0.7, rng)
        return blades

    def _needle_shoot(self, origin, angle, R, n, rachis_hw, rng) -> list[Blade]:
        p = self.p
        alpha = math.radians(np.clip(p.leaflet_angle_deg, 5.0, 90.0))
        scales = p.aspect_ratio < 6.0
        if scales:
            # Scale-leaved shoots (Cupressaceae): the axis is clothed in green appressed scales
            hw = max(rachis_hw * 0.8, self._lamina_profile().max() * 0.9)
            axis = Blade(np.asarray(origin, float), angle, R, np.full(T_GRID.shape, hw) * (1.0 - 0.4 * T_GRID),
                         None, 1, 0, role="scale_axis")
        else:
            axis = Blade(np.asarray(origin, float), angle, R,
                         np.full(T_GRID.shape, rachis_hw * 0.8) * (1.0 - 0.5 * T_GRID), None, 1, 0,
                         is_axis=True, role="rachis")
        blades = [axis]
        for i in range(n):
            u = (i + 0.5) / n
            scale = 1.0 - 0.45 * u ** 2   # Shorter toward the growing tip
            for side in (-1.0, 1.0):
                for row, row_angle in enumerate((alpha, alpha * 0.55)):
                    pos = axis.origin + axis.axis * (u + 0.25 * row / n) * R
                    blades.append(self._needle(pos, angle + side * (row_angle + rng.normal(0.0, 0.06)),
                                               scale * (1.0 - 0.18 * row)))
        return blades

def _transform_blade(b: Blade, pivot: np.ndarray, theta: float, scale: float, offset: np.ndarray) -> Blade:
    """Rotates a blade clockwise by theta about `pivot`, scales it about the pivot, then translates it."""
    c, s = math.cos(theta), math.sin(theta)
    rel = (b.origin - pivot) * scale
    new_origin = np.array([rel[0] * c + rel[1] * s, -rel[0] * s + rel[1] * c]) + offset
    teeth = None
    if b.teeth is not None:
        teeth = ToothSpec(b.teeth.margin, b.teeth.count, b.teeth.height * scale, b.teeth.skew)
    return replace(b, origin=new_origin, angle=b.angle + theta, length=b.length * scale,
                   half_width=b.half_width * scale, teeth=teeth, blend=b.blend * scale)


def build_shoot_model(profile: "LeafMorphologyProfile", n_leaves: int, opposite: bool = False,
                      seed: int = 5) -> "LeafShapeModel":
    """
    A leafy shoot (one metamer sequence) for cluster foliage cards: a stem with
    `n_leaves` leaves inserted alternately (or in opposite pairs) at the petiole
    angle, younger leaves toward the tip being smaller.
    """
    rng = np.random.default_rng(seed)
    leaf = LeafShapeBuilder(profile).build()
    pivot = np.array([0.0, -leaf.petiole_length])
    insert = math.radians(np.clip(profile.petiole_angle_deg, 10.0, 85.0))
    unit = leaf.bounds[3] + leaf.petiole_length
    internode = max(0.12, unit * 0.32)
    blades: list[Blade] = []
    laminae: list[int] = []
    nodes = max(1, int(math.ceil(n_leaves / (2 if opposite else 1))))
    y = internode * 0.4
    count = 0
    for i in range(nodes):
        age = i / max(1, nodes - 1)
        scale = 1.0 - 0.35 * age ** 1.5
        sides = (-1.0, 1.0) if opposite else ((-1.0 if i % 2 else 1.0),)
        for side in sides:
            if count >= n_leaves:
                break
            theta = side * insert * rng.uniform(0.8, 1.1) * (1.0 - 0.35 * age)
            for k, b in enumerate(leaf.blades):
                if k in leaf.laminae:
                    laminae.append(len(blades))
                blades.append(_transform_blade(b, pivot, theta, scale, np.array([0.0, y])))
            count += 1
        y += internode * (1.0 - 0.3 * age)
    stem_len = y + internode * 0.2
    stem_hw = leaf.petiole_half_width * 1.5
    stem = Blade(np.zeros(2), 0.0, stem_len, np.full(T_GRID.shape, stem_hw) * (1.0 - 0.5 * T_GRID), None, 1, 0,
                 is_axis=True, role="stem")
    blades.append(stem)
    model = LeafShapeModel(blades=blades, petiole_length=0.0, petiole_half_width=stem_hw, laminae=laminae)
    boxes = np.array([b.bbox() for b in blades])
    pad = 0.02
    model.bounds = (float(boxes[:, 0].min() - pad), float(min(boxes[:, 1].min(), 0.0) - pad),
                    float(boxes[:, 2].max() + pad), float(boxes[:, 3].max() + pad))
    return model


# -----------------------------------------------------------------------------
# Leaf engine: shape model, outline polygon and 3D leaf card
# -----------------------------------------------------------------------------
class LeafMorphologyEngine:
    """Builds the leaf shape model and a low-poly curved card for alpha-mapped foliage."""

    def __init__(self, profile: LeafMorphologyProfile = None, shoot_leaves: int = 0, opposite: bool = False):
        self.profile = profile or LeafMorphologyProfile()
        self.shoot_leaves = int(shoot_leaves)
        self.opposite = opposite
        self._model: LeafShapeModel | None = None

    @property
    def shape_model(self) -> LeafShapeModel:
        """The foliar unit drawn on one card: a single leaf, or a leafy shoot when shoot_leaves > 0."""
        if self._model is None:
            if self.shoot_leaves > 0:
                self._model = build_shoot_model(self.profile, self.shoot_leaves, self.opposite)
            else:
                self._model = LeafShapeBuilder(self.profile).build()
        return self._model

    def rasterize_mask(self, resolution: int = 256) -> tuple[np.ndarray, tuple]:
        """Boolean inside-mask of the leaf over its bounds (row 0 = bottom)."""
        x0, y0, x1, y1 = self.shape_model.bounds
        aspect = (x1 - x0) / max(1e-6, y1 - y0)
        h = resolution if aspect <= 1 else max(16, int(resolution / aspect))
        w = max(16, int(h * aspect)) if aspect <= 1 else resolution
        xs = x0 + (np.arange(w) + 0.5) / w * (x1 - x0)
        ys = y0 + (np.arange(h) + 0.5) / h * (y1 - y0)
        X, Y = np.meshgrid(xs, ys)
        return self.shape_model.field(X, Y) < 0.0, (x0, y0, x1, y1)

    def generate_boundary_2d(self, num_points: int = 128) -> np.ndarray:
        """Polar outline sampled around the lamina centroid (normalised so the lamina spans y in [0, 1])."""
        mask, (x0, y0, x1, y1) = self.rasterize_mask(256)
        h, w = mask.shape
        ys, xs = np.nonzero(mask)
        px = x0 + (xs + 0.5) / w * (x1 - x0)
        py = y0 + (ys + 0.5) / h * (y1 - y0)
        keep = py >= 0.0  # Exclude the petiole
        px, py = px[keep], py[keep]
        cx, cy = px.mean(), py.mean()
        ang = np.arctan2(py - cy, px - cx)
        rad = np.hypot(px - cx, py - cy)
        bins = np.linspace(-math.pi, math.pi, num_points + 1)
        idx = np.clip(np.digitize(ang, bins) - 1, 0, num_points - 1)
        r = np.zeros(num_points)
        np.maximum.at(r, idx, rad)
        centers = 0.5 * (bins[:-1] + bins[1:])
        pts = np.stack([cx + r * np.cos(centers), cy + r * np.sin(centers)], axis=1)
        ymin, ymax = pts[:, 1].min(), pts[:, 1].max()
        pts[:, 1] = (pts[:, 1] - ymin) / max(1e-6, ymax - ymin)
        pts[:, 0] = pts[:, 0] / max(1e-6, ymax - ymin)
        return pts

    def generate_3d_leaf_mesh(self, grid_x: int = 3, grid_y: int = 6) -> dict:
        """
        Curved quad card spanning the leaf bounds; the outline and venation come
        from the alpha/colour texture. The petiole base sits at the origin and
        the leaf extends along +Y with its adaxial side facing +Z.

        Returns 'vertices' (N,3) float32, 'faces' (F,4) int32, 'uvs' (N,2) float32,
        plus blade dimensions in meters.
        """
        p = self.profile
        model = self.shape_model
        scale = p.blade_length_cm * 0.01
        x0, y0, x1, y1 = model.bounds
        grid_x = max(2, int(grid_x))
        grid_y = max(2, int(grid_y))
        u = np.linspace(0.0, 1.0, grid_x)
        v = np.linspace(0.0, 1.0, grid_y)
        U, V = np.meshgrid(u, v)
        X = x0 + U * (x1 - x0)
        Y = y0 + V * (y1 - y0)
        base_y = -model.petiole_length
        length_total = max(1e-6, y1 - base_y)
        half_extent = max(1e-6, max(abs(x0), abs(x1)))

        y_rel = np.clip((Y - base_y) / length_total, 0.0, 1.0)
        x_rel = X / half_extent
        z = -p.longitudinal_droop * y_rel ** 2.2 * length_total * 0.45
        z = z + p.transverse_curl * np.abs(x_rel) ** 1.8 * half_extent * 0.5
        z = z + p.undulation_amplitude * np.sin(y_rel * 16.0) * np.abs(x_rel) ** 2 * length_total * 0.06
        # Petiole base at the origin
        verts = np.stack([X, Y - base_y, z - z.min() * 0.0], axis=-1).reshape(-1, 3) * scale
        verts[:, 2] -= verts[np.argmin(np.abs(verts[:, 0]) + np.abs(verts[:, 1])), 2]
        uvs = np.stack([U, V], axis=-1).reshape(-1, 2)

        j, i = np.meshgrid(np.arange(grid_y - 1), np.arange(grid_x - 1), indexing="ij")
        i0 = (j * grid_x + i).ravel()
        faces = np.stack([i0, i0 + 1, i0 + grid_x + 1, i0 + grid_x], axis=1)

        lam = [model.blades[k] for k in model.laminae] or model.blades
        return {
            "vertices": verts.astype(np.float32),
            "faces": faces.astype(np.int32),
            "uvs": uvs.astype(np.float32),
            "blade_length_m": scale,
            "blade_width_m": scale * 2.0 * max(b.max_half_width for b in lam),
            "bounds_blu": model.bounds,
        }
