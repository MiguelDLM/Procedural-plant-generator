"""
Orchids (Orchidaceae): monopodial (Phalaenopsis), sympodial with pseudobulbs (Cattleya, Dendrobium, Oncidium,
Cymbidium, Prosthechea) or without them (Paphiopedilum), and climbing (Vanilla).

Flower. Two trimerous perianth whorls: three sepals (one dorsal, two lateral, sometimes fused into a synsepal)
and three petals, two lateral and the median one modified into the labellum (lip); stamens and style are fused
into the column (gynostemium) that ends in the anther cap over the pollinia, with the stigma below it. The lip
is the adaxial median tepal: its distinct identity comes from a competition between two MADS-box complexes (the
"perianth code" of Hsu et al. 2015, Nature Plants 1: 15046; "orchid code" of Mondragon-Palomino & Theissen 2009),
so the model gives it its own outline, rolling and ornaments instead of scaling a petal. Geometry:

- Tepals: blades with the leaf half-width model (base / apex angles, widest point) plus claw, cupping,
  reflexion, twist and wavy margins, radiating from the top of the ovary at the floral-diagram azimuths.
- Lip outline: union (smooth maximum) of a claw, two lateral lobes and a mid lobe (with an isthmus), the way
  lobed lips are described (Phalaenopsis, Oncidium, Cymbidium). The lamina bends without stretching: each
  cross-section is integrated from its curvature, so the lateral lobes fold up around the column along hinge
  lines (Phalaenopsis), the whole lip rolls into a tube around it (Cattleya, Dendrobium, Vanilla: lip fused
  to the column) or into a pouch (Paphiopedilum). Callus ridges or pads and the two apical cirrhi of
  Phalaenopsis are added on the lip; the margin ripples follow McCord & Wuensche (2008, GRAPP: sum of sine
  waves growing toward the edge).
- Resupination: before anthesis the lip is uppermost (adaxial, toward the axis); the pedicel/ovary twists
  until it is lowermost. The twist is gravitropic (auxin-driven; the lip mass acts as a lever: Rowe et al.
  2025, J. Exp. Bot. 76: 433), so it is computed as the angle that brings the lip down, times `resupination`
  (0 = non-resupinate, lip uppermost: Prosthechea cochleata). The six-ribbed ovary shows the twist.
- Measurements of the presets: USPTO plant patents (Phalaenopsis 'Northstar' PP25332), Flora of China
  (Dendrobium nobile), POWO (Paphiopedilum insigne), Flora of the Bahamas (Oncidium sphacelatum), Flora of
  North America (Prosthechea cochleata) and the vanilla treatments cited in core/orchid_db.py.

Inflorescence: racemes or panicles from the leaf axils (3rd-4th leaf below the apex in Phalaenopsis), the apex
or the base of a pseudobulb or the nodes of old canes. Its axis is an elastic cantilever (Euler-Bernoulli,
d(theta)/ds = M(s)/EI) loaded by its flowers: erect, arching or pendent depending on `spike_flex`, after a
young, negatively gravitropic peduncle. Flowers open acropetally (buds toward the tip) and turn to the light.

Vegetative body: a monopodial stem with two-ranked fleshy leaves and aerial roots; sympodial growths along a
zigzag rhizome, each a pseudobulb (radius profile from its widest point and fullness, nodes, ridges, wrinkling
with age, papery sheaths) with leaves at its apex, at its base (fan) or along it; old growths are leafless
backbulbs. Aerial roots are thick (velamen) with a green growing tip, agravitropic and wandering; in Vanilla one
root per node, opposite the leaf, clings to the support.

Plant space: substrate surface at z = 0.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from .leaf_morphology import half_width_profile, T_GRID
from .mesh_engine import MeshData
from .vine import frames, arc_length, tube
from .vegetable import revolve

UP = np.array([0.0, 0.0, 1.0])
GOLDEN = math.radians(137.50776)
ATTRS = ("orc_part", "orc_u", "orc_v", "orc_s", "orc_t", "orc_rand")
# orc_part codes (material): organ identity
(LEAF, STEM, ROOT, SEPAL, PETAL, LIP, COLUMN, ANTHER, CALLUS, BRACT, SPIKE, BUD, SUPPORT, CAPSULE,
 STIGMA) = range(15)
LIGHT = np.array([0.0, -1.0, 0.0])          # Flowers turn toward the front of the scene (-Y)
BEND_GAIN = 14.0                            # Scales spike_flex to the bending compliance of the axis


class OrchidHabit(str, Enum):
    MONOPODIAL = "Monopodial"    # One stem growing from its apex (Phalaenopsis, Vanda)
    SYMPODIAL = "Sympodial"      # Successive growths along a rhizome (Cattleya, Dendrobium, Oncidium, Paphiopedilum)
    CLIMBING = "Climbing"        # Long monopodial vine on a support, rooting at the nodes (Vanilla)


class LeafPlacement(str, Enum):
    APEX = "Apex"                # 1-3 leaves on top of the pseudobulb (Cattleya, Oncidium)
    BASE = "Base"                # Two-ranked fan sheathing the base of the growth (Cymbidium, Paphiopedilum)
    ALONG = "Along"              # Two-ranked along the stem or cane (Phalaenopsis, Dendrobium, Vanilla)


class InflOrigin(str, Enum):
    AXIL = "Leaf axil"           # Phalaenopsis (3rd-4th leaf below the apex), Vanilla
    APEX = "Apex"                # Cattleya (from a sheath), Paphiopedilum and Prosthechea
    BASE = "Base"                # Beside the newest pseudobulb (Oncidium, Cymbidium)
    NODES = "Upper nodes"        # Nodes of old leafless canes (Dendrobium nobile)


@dataclass
class OrchidProfile:
    """Habit, leaves, roots, inflorescence and flower of an orchid. Lengths in cm unless noted."""
    # Habit and stems
    habit: OrchidHabit = OrchidHabit.MONOPODIAL
    growths: int = 4                     # Sympodial growths (pseudobulbs / fans) along the rhizome
    rhizome_cm: float = 3.0              # Rhizome length between successive growths
    growth_angle_deg: float = 25.0       # The rhizome turns alternately by this angle at each growth (zigzag)
    leafless_growths: int = 0            # Oldest growths that have shed their leaves (backbulbs)
    stem_length_cm: float = 4.0          # Monopodial stem; climbing: length of the vine
    stem_radius_mm: float = 6.0          # Stem / rhizome radius
    internode_cm: float = 1.0            # Between leaves along a stem or cane
    pseudobulb_cm: float = 0.0           # Pseudobulb length (0 = none)
    pseudobulb_diameter_cm: float = 2.0
    bulb_widest: float = 0.4             # Position of the widest point (0.3 ovoid .. 0.8 club-shaped)
    bulb_fullness: float = 0.8           # 0.2 cylindrical cane .. 1 spindle / egg
    bulb_flatten: float = 0.0            # Lateral compression (Oncidium, Prosthechea)
    bulb_ridges: int = 0                 # Longitudinal ridges and furrows (sulcate pseudobulbs)
    bulb_lean_deg: float = 10.0          # Lean of the growths away from the rhizome axis
    sheath_cover: float = 0.0            # Share of the pseudobulb covered by papery sheaths, from the base
    support_height_m: float = 1.5        # Climbing: height of the post the vine climbs
    support_radius_cm: float = 5.0
    stem_color: tuple = (0.38, 0.48, 0.20)
    sheath_color: tuple = (0.78, 0.72, 0.58)

    # Leaves
    leaf_placement: LeafPlacement = LeafPlacement.ALONG
    leaves: int = 6                      # Living leaves (per growth when sympodial)
    leaf_length_cm: float = 20.0
    leaf_width_cm: float = 8.0
    leaf_thickness_mm: float = 2.0       # Fleshy leaves: Phalaenopsis ~2, Vanilla ~2, Cymbidium ~0.8
    leaf_widest: float = 0.6             # Position of the widest point
    leaf_apex_deg: float = 120.0         # Angle at the apex (obtuse ~120, acute ~50)
    leaf_angle_deg: float = 60.0         # Insertion angle from the vertical
    leaf_droop: float = 0.3              # Arching of the blade under its weight
    leaf_fold: float = 0.3               # Conduplicate V-fold along the midrib (keel)
    leaf_twist_deg: float = 10.0
    leaf_size_gradient: float = 0.3      # Youngest leaves smaller by this fraction
    leaf_color: tuple = (0.12, 0.30, 0.10)
    leaf_mottle: float = 0.0             # Tessellated / mottled leaves (Paphiopedilum)
    mottle_color: tuple = (0.30, 0.45, 0.22)

    # Roots
    roots: int = 8                       # Aerial roots (per growth when sympodial)
    root_diameter_mm: float = 5.0        # Thick roots: velamen around the cortex
    root_length_cm: float = 25.0
    root_wander: float = 0.6             # Meandering of the agravitropic aerial roots
    aerial_share: float = 0.5            # Share of the roots growing in the air (others go into the substrate)
    root_color: tuple = (0.78, 0.80, 0.72)      # Velamen (silvery when dry)
    root_tip_color: tuple = (0.40, 0.62, 0.25)  # Green growing tip

    # Inflorescence
    infl_origin: InflOrigin = InflOrigin.AXIL
    inflorescences: int = 1
    flowers: int = 8                     # Flowers per inflorescence (per branch when branched)
    peduncle_cm: float = 25.0            # Stalk below the first flower
    rachis_cm: float = 25.0              # Flower-bearing part
    spike_radius_mm: float = 2.5
    spike_angle_deg: float = 30.0        # Emergence angle from the vertical
    spike_flex: float = 0.5              # 0 stiff, erect .. 1 pendent under the weight of the flowers
    branches: int = 0                    # Lateral branches (panicle: Oncidium)
    branch_ratio: float = 0.4            # Branch length relative to the rachis
    divergence_deg: float = 180.0        # Between successive flowers (180 two-ranked, 137.5 spiral)
    pedicel_cm: float = 4.0              # Pedicel and ovary
    flower_facing: float = 0.7           # 0 facing away from the axis .. 1 all turned to the light
    maturation: float = 0.3              # Acropetal opening: share of buds at the tip
    bract_mm: float = 4.0                # Floral and peduncle bracts
    spike_color: tuple = (0.25, 0.35, 0.18)
    fruit_set: float = 0.0               # Share of flowers replaced by capsules (pods)
    capsule_cm: float = 5.0
    capsule_diameter_cm: float = 1.2
    capsule_color: tuple = (0.35, 0.45, 0.15)

    # Flower: perianth
    resupination: float = 1.0            # 1 lip lowermost (gravitropic twist of the ovary) .. 0 lip uppermost
    flower_tilt_deg: float = 10.0        # Face tilted upward
    sepal_length_cm: float = 4.5         # Dorsal sepal
    sepal_width_cm: float = 3.0
    sepal_widest: float = 0.5
    sepal_apex_deg: float = 110.0
    lateral_sepal_deg: float = 35.0      # Lateral sepals below the horizontal
    lateral_sepal_scale: float = 1.0     # Lateral / dorsal sepal size
    synsepal: float = 0.0                # Lateral sepals joined behind the lip (Paphiopedilum: 1)
    sepal_cup: float = 0.15
    sepal_reflex_deg: float = 10.0       # Bending back along the sepal
    sepal_twist_deg: float = 0.0
    sepal_wave: float = 0.0
    petal_length_cm: float = 5.0
    petal_width_cm: float = 4.0
    petal_widest: float = 0.6
    petal_apex_deg: float = 140.0
    petal_claw: float = 0.2              # Narrow stalk-like base (fraction of the length)
    petal_angle_deg: float = 10.0        # Above the horizontal (negative: drooping, Paphiopedilum)
    petal_cup: float = 0.1
    petal_reflex_deg: float = 5.0
    petal_twist_deg: float = 0.0
    petal_wave: float = 0.0              # Wavy / frilled margins (Cattleya)
    perianth_forward_deg: float = 8.0    # Tepals raised forward from the floral plane (cupped flowers)

    # Flower: labellum (lip)
    lip_length_cm: float = 2.0
    lip_width_cm: float = 2.5            # Across the spread lateral lobes (or the whole lip)
    lip_angle_deg: float = 35.0          # Base direction below the floral axis (0 forward .. 90 straight down)
    lip_deflex_deg: float = 30.0         # Further bending down along the lip (recurved mid lobe)
    lip_claw: float = 0.25               # Half-width of the narrow base, relative to the lip
    side_lobes: float = 0.0              # Lateral lobes, relative half-width (0 = entire lip)
    side_lobe_pos: float = 0.35          # Their centre along the lip
    side_lobe_length: float = 0.45       # Their extent along the lip
    side_lobe_erect_deg: float = 0.0     # Fold of the lateral lobes up around the column
    midlobe_width: float = 0.6           # Mid lobe width relative to the lip width
    isthmus: float = 0.0                 # Narrowing between lateral lobes and mid lobe (pandurate lips)
    lip_widest: float = 0.5
    lip_apex_deg: float = 120.0
    lip_roll: float = 0.0                # 0 flat .. 1 rolled into a tube around the column
    lip_roll_extent: float = 0.5         # Share of the lip that is rolled, from the base
    lip_sac: float = 0.0                 # Inflated pouch (Paphiopedilum slipper)
    lip_wave: float = 0.0                # Undulate / crisped margin
    lip_waves: float = 4.0               # Waves along the margin
    callus_mm: float = 0.0               # Height of the callus (pad or ridges) on the lip disc
    callus_ridges: int = 0               # 0 one fleshy pad; >0 parallel keels (lamellae)
    callus_pos: float = 0.35             # Centre of the callus along the lip
    cirrhi_mm: float = 0.0               # Two filiform appendages at the lip apex (Phalaenopsis)

    # Flower: column
    column_length_cm: float = 1.0
    column_width_mm: float = 6.0
    column_arch: float = 0.3             # Arching of the column toward the lip
    staminode_mm: float = 0.0            # Shield-like staminode on the column (Paphiopedilum)
    mentum_mm: float = 0.0               # Chin formed by the column foot and the lateral sepals (Dendrobium)

    # Flower colours and pigmentation patterns (full colour, spots, venation: PeMYB2 / 11 / 12, Hsu et al. 2015)
    sepal_color: tuple = (0.97, 0.96, 0.97)
    petal_color: tuple = (0.97, 0.96, 0.97)
    lip_color: tuple = (0.95, 0.85, 0.40)
    lip_throat_color: tuple = (0.95, 0.80, 0.25)
    callus_color: tuple = (0.95, 0.75, 0.15)
    column_color: tuple = (0.96, 0.95, 0.92)
    anther_color: tuple = (0.95, 0.85, 0.55)
    pattern_color: tuple = (0.65, 0.10, 0.35)
    tip_color: tuple = (0.75, 0.30, 0.60)
    tip_amount: float = 0.0              # Coloured tips of sepals, petals and lip (Dendrobium nobile)
    spots: float = 0.0                   # Spots on sepals and petals
    spot_size_mm: float = 2.0
    spot_stretch: float = 1.0            # >1 transverse bars (Oncidium sepals)
    veins: float = 0.0                   # Coloured venation on sepals and petals
    lip_spots: float = 0.0
    lip_veins: float = 0.0
    sheen: float = 0.3                   # Crystalline / waxy sheen of the tepals


@dataclass
class OrchidResult:
    leaves: MeshData
    stems: MeshData
    roots: MeshData
    flowers: MeshData
    spikes: MeshData
    support: MeshData
    stats: dict = field(default_factory=dict)


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------
def _unit(v):
    v = np.asarray(v, float)
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def _rotate(v, axis, ang):
    axis = _unit(axis)
    c, s = math.cos(ang), math.sin(ang)
    return v * c + np.cross(axis, v) * s + axis * (axis @ v) * (1 - c)


def _perp(d):
    p = np.cross(d, UP)
    if np.linalg.norm(p) < 1e-6:
        p = np.cross(d, [1.0, 0.0, 0.0])
    return _unit(p)


def _proj(v, n):
    """Component of v perpendicular to the unit vector n (unit, or zero when parallel)."""
    w = v - (v @ n) * n
    return _unit(w) if np.linalg.norm(w) > 1e-6 else np.zeros(3)


def _smooth(x, a, b):
    t = np.clip((np.asarray(x, float) - a) / np.maximum(np.asarray(b, float) - a, 1e-9), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def _signed_angle(a, b, axis):
    return math.atan2(float(np.cross(a, b) @ axis), float(a @ b))


def _attrs(n, part, u=0.0, v=0.0, s=0.0, t=0.0, rnd=0.0):
    f = lambda a: np.broadcast_to(np.asarray(a, float), (n,)).astype(np.float32).copy()
    return {"orc_part": f(part), "orc_u": f(u), "orc_v": f(v), "orc_s": f(s), "orc_t": f(t), "orc_rand": f(rnd)}


def _merge(parts) -> MeshData:
    parts = [p for p in parts if p is not None and len(p.vertices)]
    for p in parts:
        n = len(p.vertices)
        for k in ATTRS:
            p.point_attributes[k] = np.asarray(p.point_attributes.get(k, np.zeros(n)), np.float32)
        for k in list(p.point_attributes):
            if k not in ATTRS:
                del p.point_attributes[k]
    return MeshData.concatenate(parts)


def _grid(V, part, u, v, s=0.0, t=0.0, rnd=0.0, closed=False, caps=False) -> MeshData:
    """Quad grid from rows of points V (n, m, 3); `closed` joins the last column to the first (tubes), `caps`
    closes both ends of a closed grid with triangle fans. Attributes broadcast to (n, m)."""
    V = np.asarray(V, float)
    n, m = V.shape[:2]
    full = lambda a: np.broadcast_to(np.asarray(a, float), (n, m)).reshape(-1)
    verts = [V.reshape(-1, 3)]
    A = {k: [full(a)] for k, a in zip(ATTRS, (part, u, v, s, t, rnd))}
    mm = m if closed else m - 1
    j, i = np.meshgrid(np.arange(n - 1), np.arange(mm), indexing="ij")
    a = (j * m + i).ravel()
    b = (j * m + (i + 1) % m).ravel()
    loops = [np.stack([a, b, b + m, a + m], 1).reshape(-1)]
    totals = [np.full(len(a), 4)]
    uvs = [np.stack([np.repeat(np.arange(m)[None, :] / max(m - 1, 1), n, 0).reshape(-1),
                     np.repeat(np.linspace(0, 1, n)[:, None], m, 1).reshape(-1)], 1)]
    if closed and caps:
        base = n * m
        for row, k in ((0, 0), (n - 1, 1)):
            c = V[row].mean(0)
            verts.append(c[None, :])
            for key in A:
                A[key].append(A[key][0][row * m:row * m + 1])
            uvs.append(np.array([[0.5, float(k)]]))
            ring = row * m + np.arange(m)
            ctr = np.full(m, base + k)
            tri = np.stack([ring, np.roll(ring, -1), ctr], 1) if k else np.stack([np.roll(ring, -1), ring, ctr], 1)
            loops.append(tri.reshape(-1))
            totals.append(np.full(m, 3))
    Vall = np.concatenate(verts)
    lv = np.concatenate(loops).astype(np.int32)
    lt = np.concatenate(totals).astype(np.int32)
    ls = np.concatenate([[0], np.cumsum(lt)[:-1]]).astype(np.int32)
    uv = np.concatenate(uvs)[lv].astype(np.float32)
    return MeshData(Vall.astype(np.float32), lv, ls, lt, uv,
                    {k: np.concatenate(a).astype(np.float32) for k, a in A.items()})


def _blob(center, axis, length, width, part, flat=1.0, n=7, nth=10, rnd=0.0, side=None, point=0.7) -> MeshData:
    """Closed ovoid (buds, anther cap, callus, stigma) along `axis`, centred at `center`; `flat` scales the
    width along `side`."""
    t = np.linspace(0.0, 1.0, n + 2)[1:-1]
    rho = 0.5 * width * np.sin(np.pi * t) ** point
    m = revolve(rho, length * (t - 0.5), nth, {"orc_part": np.full(len(t), float(part)), "orc_u": t,
                                               "orc_rand": np.full(len(t), rnd)})
    z = _unit(axis)
    e1 = _perp(z) if side is None else _proj(np.asarray(side, float), z)
    if not np.any(e1):
        e1 = _perp(z)
    e2 = np.cross(z, e1)
    Vl = m.vertices.astype(float)
    V = center + Vl[:, 0:1] * e1 * flat + Vl[:, 1:2] * e2 + Vl[:, 2:3] * z
    m.vertices = V.astype(np.float32)
    return m


def _tube(P, R, part, sides=6, u=None, rnd=0.0, cap=True) -> MeshData:
    P = np.asarray(P, float)
    n = len(P)
    uu = np.linspace(0, 1, n) if u is None else u
    return tube(P, np.broadcast_to(np.asarray(R, float), (n,)), sides,
                {"orc_part": np.full(n, float(part)), "orc_u": uu, "orc_s": arc_length(P),
                 "orc_rand": np.full(n, rnd)}, cap_end=cap)


def _ring(th, N, B):
    """Unit ring directions (n, m, 3) around a curve with frames N, B."""
    return np.cos(th)[None, :, None] * N[:, None, :] + np.sin(th)[None, :, None] * B[:, None, :]


def _bezier(p0, p1, p2, p3, n):
    t = np.linspace(0, 1, n)[:, None]
    return (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3


def _shape(aspect, widest, apex_deg, base_deg=70.0, apex_curv=0.5):
    """Normalised half-width on T_GRID (max 1) of a blade (leaf half-width model)."""
    hw = np.maximum(np.asarray(half_width_profile(max(0.3, aspect), widest, base_deg, apex_deg, 0.0, apex_curv),
                               float), 0.0)
    return hw / max(float(hw.max()), 1e-9)


# -----------------------------------------------------------------------------
# Engine
# -----------------------------------------------------------------------------
class OrchidEngine:
    def __init__(self, profile: OrchidProfile = None):
        self.p = profile or OrchidProfile()
        p = self.p
        self._sepal = _shape(p.sepal_length_cm / max(p.sepal_width_cm, 1e-3), p.sepal_widest, p.sepal_apex_deg)
        self._petal = _shape(p.petal_length_cm / max(p.petal_width_cm, 1e-3), p.petal_widest, p.petal_apex_deg)
        mid_len = p.lip_length_cm * (1.0 - (p.side_lobe_pos if p.side_lobes > 0 else 0.0))
        mid_w = p.lip_width_cm * (p.midlobe_width if p.side_lobes > 0 else 1.0)
        self._mid = _shape(mid_len / max(mid_w, 1e-3), p.lip_widest, p.lip_apex_deg, base_deg=120.0)
        self._leaf = _shape(p.leaf_length_cm / max(p.leaf_width_cm, 1e-3), p.leaf_widest, p.leaf_apex_deg,
                            base_deg=40.0)

    # ================================================================= flower
    def tepal(self, origin, out, fwd, L, W, shape, claw, cup, forward_deg, reflex_deg, twist_deg, wave, part,
              rng, detail, mirror=1.0) -> MeshData:
        """Sepal or petal: blade leaving `origin` along `out` (in the floral plane), raised toward the front
        `fwd` by forward_deg, bending back by reflex_deg; parabolic cupping toward the front, twist and wavy
        margins. `mirror` (+1/-1) makes left and right organs symmetric."""
        nu = max(6, int(14 * detail))
        nv = 2 * max(2, int(4 * detail)) + 1
        u = np.linspace(0, 1, nu)
        v = np.linspace(-1, 1, nv)
        lat = _unit(np.cross(fwd, out))
        tf = np.linspace(0, 1, 40)
        ang = math.radians(forward_deg) - math.radians(reflex_deg) * tf ** 1.5
        dirs = np.cos(ang)[:, None] * out + np.sin(ang)[:, None] * fwd
        mid = origin + np.concatenate([[np.zeros(3)], np.cumsum((dirs[:-1] + dirs[1:]) * 0.5 * L / 39, axis=0)])
        nrm = _unit(np.cross(dirs, lat))
        M = np.stack([np.interp(u, tf, mid[:, k]) for k in range(3)], -1)
        N = _unit(np.stack([np.interp(u, tf, nrm[:, k]) for k in range(3)], -1))
        hw = 0.5 * W * np.interp(u, T_GRID, shape)
        if claw > 0:
            hw = hw * (0.22 + 0.78 * _smooth(u, 0.0, claw * 1.6))
        hw = np.maximum(hw, 0.0015 * (1 - u))                     # Attached to the receptacle
        X = hw[:, None] * v[None, :]
        Z = cup * hw[:, None] * v[None, :] ** 2
        if wave > 0:
            ph = rng.uniform(0, 2 * math.pi)
            Z = Z + wave * W * 0.12 * np.sin(2 * math.pi * 3.0 * u[:, None] + ph + 2.0 * v[None, :]) * \
                np.abs(v[None, :]) ** 3 * _smooth(u, 0.1, 0.5)[:, None]
        if twist_deg:
            a = math.radians(twist_deg) * mirror * u[:, None]
            X, Z = X * np.cos(a) - Z * np.sin(a), X * np.sin(a) + Z * np.cos(a)
        P = M[:, None, :] + X[..., None] * lat + Z[..., None] * N[:, None, :]
        return _grid(P, part, u[:, None], v[None, :], u[:, None] * L, X, rng.random())

    def _lip_outline(self, t):
        """Half-width of the lip (fraction of half the lip width) along it: smooth union of the claw, the
        lateral lobes and the mid lobe (with an isthmus)."""
        p = self.p
        side = p.side_lobes
        t0 = p.side_lobe_pos if side > 0 else 0.0
        tau = np.clip((t - t0) / max(1.0 - t0, 1e-3), 0, 1)
        mid = np.interp(tau, T_GRID, self._mid) * (p.midlobe_width if side > 0 else 1.0)
        if side > 0:
            mid = mid * (t >= t0) * (1.0 - p.isthmus * np.exp(-((tau - 0.05) / 0.12) ** 2))
            ext = max(p.side_lobe_length * 0.5, 0.05)
            sl = side * np.cos(np.clip((t - p.side_lobe_pos) / ext, -1, 1) * math.pi / 2) ** 1.3
        else:
            sl = np.zeros_like(t)
        claw = p.lip_claw * (1.0 - _smooth(t, t0 + 0.05, t0 + 0.3)) if side > 0 else \
            p.lip_claw * (1.0 - _smooth(t, 0.05, 0.3))
        q = 6.0
        return (np.maximum(claw, 1e-4) ** q + np.maximum(sl, 0) ** q + np.maximum(mid, 0) ** q) ** (1 / q), claw, mid

    def labellum(self, origin, down, fwd, rng, detail, scale=1.0):
        """Lip: lobed outline; the lamina bends without stretching (cross-sections integrated from their
        bending angle) to fold its lateral lobes up, roll into a tube or inflate into a pouch; callus,
        margin ripples and apical cirrhi. Returns (mesh, apex point)."""
        p = self.p
        L = p.lip_length_cm * 0.01 * scale
        Wl = p.lip_width_cm * 0.01 * scale
        nu = max(10, int(26 * detail))
        nh = max(4, int(8 * detail))                              # Samples per half cross-section
        t = np.linspace(0, 1, nu)
        lat = _unit(np.cross(fwd, down))
        hwf, clawf, midf = self._lip_outline(t)
        hw = 0.5 * Wl * hwf
        hw = np.maximum(hw, 0.0008)
        # Midline in the (fwd, down) plane: projects forward under the column, then deflexes downward
        tf = np.linspace(0, 1, 48)
        a = math.radians(p.lip_angle_deg) + math.radians(p.lip_deflex_deg) * _smooth(tf, 0.2, 1.0) ** 1.2
        if p.lip_sac > 0:                                         # Pouch: toe curves forward and up
            a = a - p.lip_sac * math.radians(70) * _smooth(tf, 0.55, 1.0)
        dirs = np.cos(a)[:, None] * fwd + np.sin(a)[:, None] * down
        mid = origin + np.concatenate([[np.zeros(3)], np.cumsum((dirs[:-1] + dirs[1:]) * 0.5 * L / 47, axis=0)])
        nin = _unit(np.cross(dirs, lat))                         # Inner (upper) face: toward the column
        M = np.stack([np.interp(t, tf, mid[:, k]) for k in range(3)], -1)
        Nin = _unit(np.stack([np.interp(t, tf, nin[:, k]) for k in range(3)], -1))
        # Bending of each cross-section: roll (uniform) + lateral lobes folding up at a hinge
        e = p.lip_roll_extent
        roll = math.pi * p.lip_roll * (1.0 - _smooth(t, e - 0.12, e + 0.12)) if e < 1 else \
            np.full_like(t, math.pi * p.lip_roll)
        if p.lip_sac > 0:
            # Pouch: margins incurved over a mouth near the base, overlapping toward the closed toe
            roll = roll + math.pi * p.lip_sac * (0.7 + 0.45 * _smooth(t, 0.15, 0.65))
        # `lip_width_cm` is the apparent width: a rolled lamina must be wider (arc of total angle 2*roll)
        rl = np.clip(roll, 1e-3, None)
        hw = hw * np.where(rl < math.pi / 2, rl / np.sin(rl), rl)
        if p.side_lobes > 0:
            ext = max(p.side_lobe_length * 0.5, 0.05)
            bump = np.cos(np.clip((t - p.side_lobe_pos) / (ext * 1.3), -1, 1) * math.pi / 2) ** 2
            side = math.radians(p.side_lobe_erect_deg) * bump
            v0 = np.clip(np.maximum(clawf, midf) / np.maximum(hwf, 1e-6), 0.15, 0.85)
        else:
            side = np.zeros_like(t)
            v0 = np.full_like(t, 0.5)
        vh = np.linspace(0, 1, nh + 1)                            # 0 centre .. 1 margin
        vm = 0.5 * (vh[:-1] + vh[1:])
        phi = roll[:, None] * vm[None, :] + side[:, None] * _smooth(vm[None, :], v0[:, None], v0[:, None] + 0.3)
        ds = hw[:, None] / nh
        Xh = np.concatenate([np.zeros((nu, 1)), np.cumsum(ds * np.cos(phi), 1)], 1)
        Yh = np.concatenate([np.zeros((nu, 1)), np.cumsum(ds * np.sin(phi), 1)], 1)
        X = np.concatenate([-Xh[:, :0:-1], Xh], 1)                 # Mirror: -margin .. centre .. +margin
        Y = np.concatenate([Yh[:, :0:-1], Yh], 1)
        v = np.concatenate([-vh[:0:-1], vh])
        if p.lip_sac > 0:                                          # Inflated floor of the pouch
            Y = Y - p.lip_sac * 0.9 * hw[:, None] * (1 - v[None, :] ** 2) * np.sin(math.pi * t[:, None]) ** 0.7
        part = np.full(X.shape, float(LIP))
        if p.callus_mm > 0:
            ch = p.callus_mm * 0.001 * scale
            along = np.exp(-((t - p.callus_pos) / 0.12) ** 2)
            if p.callus_ridges > 0:
                k = np.arange(p.callus_ridges)
                vk = (k - (p.callus_ridges - 1) / 2) * (0.5 / max(p.callus_ridges, 1))
                across = np.exp(-((v[None, :, None] - vk[None, None, :]) / 0.07) ** 2).max(-1)[0]
                along = np.exp(-((t - p.callus_pos) / 0.25) ** 4)
            else:
                across = np.clip(1 - (v / 0.38) ** 2, 0, 1) ** 0.6
            cal = ch * along[:, None] * across[None, :]
            Y = Y + cal
            part = np.where(cal > 0.3 * ch, float(CALLUS), part)
        if p.lip_wave > 0:
            ph = rng.uniform(0, 2 * math.pi)
            Y = Y + p.lip_wave * Wl * 0.09 * np.sin(2 * math.pi * p.lip_waves * t[:, None] + ph + 3 * v[None, :]) * \
                np.abs(v[None, :]) ** 3 * _smooth(t, 0.25, 0.7)[:, None]
        P = M[:, None, :] + X[..., None] * lat + Y[..., None] * Nin[:, None, :]
        uu = np.repeat(t[:, None], len(v), 1)
        parts = [_grid(P, part, uu, v[None, :], uu * L, X, rng.random())]
        apex = P[-1, len(v) // 2]
        if p.cirrhi_mm > 0:                                        # Two filiform appendages at the apex
            j = len(v) // 2
            for sgn in (-1, 1):
                k = int(np.argmin(np.abs(t - 0.93)))
                q = P[k, j + sgn * max(1, nh // 3)]
                d = _unit(dirs[-1] + lat * sgn * 0.9)
                pts = [q]
                lc = p.cirrhi_mm * 0.001 * scale
                for i in range(8):
                    d = _unit(_rotate(d, np.cross(d, nin[-1]), -0.22))   # Curling toward the inner face
                    pts.append(pts[-1] + d * lc / 8)
                parts.append(_tube(pts, np.linspace(0.0005, 0.0002, 9) * scale, CALLUS, 4))
        return _merge(parts), apex

    def column(self, base, fwd, down, rng, detail, scale=1.0) -> MeshData:
        """Column (gynostemium): clavate, arching toward the lip; anther cap at the apex, stigma cavity below
        it; optional shield-like staminode (Paphiopedilum) and mentum (Dendrobium)."""
        p = self.p
        Lc = p.column_length_cm * 0.01 * scale
        rc = p.column_width_mm * 0.0005 * scale
        t = np.linspace(0, 1, 8)
        pts = base[None, :] + fwd[None, :] * Lc * t[:, None] + \
            down[None, :] * (p.column_arch * Lc * 0.3 * t ** 2)[:, None]
        R = rc * (0.7 + 0.35 * t) * np.sqrt(np.clip(1 - ((t - 0.9) / 0.11).clip(0) ** 2, 0.15, 1))
        parts = [_tube(pts, R, COLUMN, max(6, int(10 * detail)))]
        tip = pts[-1]
        tan = _unit(pts[-1] - pts[-2])
        parts.append(_blob(tip - down * rc * 0.25 + tan * rc * 0.1, tan, rc * 1.2, rc * 1.5, ANTHER, flat=0.8,
                           side=np.cross(tan, down)))
        parts.append(_blob(tip + down * rc * 0.75 - tan * rc * 0.4, down, rc * 0.25, rc * 1.0, STIGMA, flat=0.7,
                           side=np.cross(tan, down)))
        if p.staminode_mm > 0:
            st = p.staminode_mm * 0.001 * scale
            parts.append(_blob(tip + tan * rc * 0.6 - down * rc * 0.2, fwd, st * 0.18, st, ANTHER, flat=0.85,
                               side=np.cross(fwd, down), point=0.5))
        if p.mentum_mm > 0:
            mt = p.mentum_mm * 0.001 * scale
            d = _unit(-fwd * 0.75 + down * 0.65)
            q = base + d * mt * 0.5
            parts.append(_blob(q, d, mt * 1.1, rc * 1.6, SEPAL, flat=0.8, point=0.5))
        return _merge(parts)

    def flower(self, center, fwd, adaxial, rng, detail, openness=1.0, scale=1.0):
        """One flower at `center` facing `fwd`; `adaxial` points to the axis that bears it (the lip is on that
        side before resupination). Returns (mesh, twist angle applied)."""
        p = self.p
        fwd = _unit(fwd)
        adax = _proj(adaxial, fwd)
        if not np.any(adax):
            adax = _proj(UP, fwd) if np.any(_proj(UP, fwd)) else _perp(fwd)
        dorsal_pre = -adax
        target = _proj(UP, fwd)
        twist = 0.0
        if np.any(target):
            twist = p.resupination * _signed_angle(dorsal_pre, target, fwd)
        dorsal = _rotate(dorsal_pre, fwd, twist)
        down = -dorsal
        right = _unit(np.cross(dorsal, fwd))
        o = float(np.clip(openness, 0, 1))
        if o < 0.55:                                                # Bud
            bl = p.sepal_length_cm * 0.01 * scale * 0.45 * (0.45 + 0.55 * o / 0.55)
            return _blob(center + fwd * bl * 0.45, fwd, bl, bl * 0.8, BUD, rnd=rng.random(), point=0.8), twist
        s = scale * (0.85 + 0.15 * o)
        fwd_extra = (1.0 - o) * 110.0                               # Half-open: tepals still cupped forward
        parts = []
        rc = p.column_width_mm * 0.0005 * s
        r0 = rc * 0.9

        def out_dir(psi):
            return _unit(math.cos(math.radians(psi)) * right + math.sin(math.radians(psi)) * dorsal)
        f0 = p.perianth_forward_deg + fwd_extra
        # Sepals (outer whorl, slightly behind)
        Ls, Ws = p.sepal_length_cm * 0.01 * s, p.sepal_width_cm * 0.01 * s
        sep = dict(cup=p.sepal_cup, forward_deg=f0, reflex_deg=p.sepal_reflex_deg * o, twist_deg=p.sepal_twist_deg,
                   wave=p.sepal_wave, part=SEPAL, rng=rng, detail=detail)
        parts.append(self.tepal(center - fwd * 0.001 * s + out_dir(90) * r0, out_dir(90), fwd, Ls, Ws, self._sepal,
                                0.0, mirror=1.0, **sep))
        ls = p.lateral_sepal_scale
        if p.synsepal >= 0.85:
            d = out_dir(270)
            parts.append(self.tepal(center - fwd * 0.0015 * s + d * r0, d, fwd, Ls * ls, Ws * ls * 1.5, self._sepal,
                                    0.0, mirror=1.0, **sep))
        else:
            for sgn, base_psi in ((1, -p.lateral_sepal_deg), (-1, 180 + p.lateral_sepal_deg)):
                psi = base_psi + p.synsepal * ((-90.0) - base_psi if sgn > 0 else 270.0 - base_psi)
                d = out_dir(psi)
                parts.append(self.tepal(center - fwd * 0.0012 * s + d * r0, d, fwd, Ls * ls, Ws * ls, self._sepal,
                                        0.0, mirror=sgn, **sep))
        # Petals (inner whorl)
        Lp, Wp = p.petal_length_cm * 0.01 * s, p.petal_width_cm * 0.01 * s
        for sgn, psi in ((1, p.petal_angle_deg), (-1, 180 - p.petal_angle_deg)):
            d = out_dir(psi)
            parts.append(self.tepal(center + fwd * 0.0005 * s + d * r0, d, fwd, Lp, Wp, self._petal, p.petal_claw,
                                    p.petal_cup, f0 + 4, p.petal_reflex_deg * o, p.petal_twist_deg, p.petal_wave,
                                    PETAL, rng, detail, mirror=sgn))
        # Column along the floral axis, lip below it
        col_base = center + fwd * 0.001 * s
        parts.append(self.column(col_base, fwd, down, rng, detail, s))
        lip, _ = self.labellum(col_base + down * rc * 0.9, down, fwd, rng, detail, s)
        parts.append(lip)
        return _merge(parts), twist

    # ================================================================= inflorescence
    def _axis(self, origin, heading, length, n_flowers, rng, flex=None, angle=None, taper=0.6):
        """Inflorescence axis. The young peduncle leaves the axil at `spike_angle_deg` and turns toward a
        steeper gravitropic set-point angle; the axis then bends under the moment of its flowers as an elastic
        cantilever, d(theta)/ds = M(s) / EI(s), with EI ~ r(s)^4 (tapering axis) and a stiffer, lignified
        peduncle, so the bending concentrates in the young rachis (arching sprays)."""
        p = self.p
        flex = p.spike_flex if flex is None else flex
        th0 = math.radians(p.spike_angle_deg if angle is None else angle)
        n = 48
        s = np.linspace(0, length, n + 1)
        ds = length / n
        grav = th0 * (0.3 + 0.7 * np.exp(-s / 0.06))         # Toward the gravitropic set-point angle
        ped = min(p.peduncle_cm * 0.01, length * 0.95)
        sk = ped + (length - ped) * (np.arange(n_flowers) + 0.5) / max(n_flowers, 1) if length > ped else \
            np.full(n_flowers, length)
        wk = np.ones(n_flowers)
        r = 1.0 - (1.0 - taper) * s / length
        EI = r ** 4 * (1.0 + 7.0 * (1.0 - _smooth(s, ped * 0.6, ped * 1.05)))
        theta = grav.copy()
        for _ in range(8):
            x = np.concatenate([[0.0], np.cumsum(np.sin(theta[:-1]) * ds)])
            xk = np.interp(sk, s, x)
            M = ((xk[None, :] - x[:, None]) * wk[None, :] * (sk[None, :] > s[:, None])).sum(1)
            own = 0.25 * n_flowers / length                       # Own weight per metre
            for i in range(n + 1):
                M[i] += own * ((x[i:] - x[i]) * ds).sum()
            kappa = flex * BEND_GAIN * M / (n_flowers * length * length * EI)
            new = grav + np.concatenate([[0.0], np.cumsum(kappa[:-1] * ds)])
            theta = 0.5 * theta + 0.5 * np.clip(new, 0.0, math.radians(175))
        h = _unit(np.array([heading[0], heading[1], 0.0])) if np.linalg.norm(heading[:2]) > 1e-6 else \
            np.array([1.0, 0.0, 0.0])
        wob = rng.normal(0, 0.03, (n + 1, 3))
        dirs = _unit(np.cos(theta)[:, None] * UP + np.sin(theta)[:, None] * h + wob * 0.3)
        P = origin + np.concatenate([[np.zeros(3)], np.cumsum(dirs[:-1] * ds, 0)])
        return P, s, sk

    def inflorescence(self, origin, heading, rng, detail, n_flowers=None, length_scale=1.0, branch=False):
        """Raceme (or panicle with `branches`) with bracts, pedicels, flowers, buds and capsules.
        Returns (spike mesh parts, flower mesh parts, stats)."""
        p = self.p
        nf = max(1, p.flowers if n_flowers is None else n_flowers)
        length = (p.peduncle_cm + p.rachis_cm) * 0.01 * length_scale
        P, s, sk = self._axis(origin, heading, max(length, 0.01), nf, rng,
                              flex=p.spike_flex * (1.2 if branch else 1.0))
        r0 = p.spike_radius_mm * 0.001 * (0.7 if branch else 1.0)
        spikes = [_tube(P, np.linspace(r0, r0 * 0.6, len(P)), SPIKE, max(5, int(8 * detail)))]
        T, _, _ = frames(P)
        flowers = []
        # Peduncle bracts (sheathing, every ~4 cm)
        ped = p.peduncle_cm * 0.01 * length_scale
        if p.bract_mm > 0:
            for z in np.arange(0.04, max(ped - 0.01, 0.0), 0.045):
                i = min(int(np.searchsorted(s, z)), len(P) - 1)
                flowers.append(None)
                spikes.append(_blob(P[i] + T[i] * p.bract_mm * 0.0004, T[i], p.bract_mm * 0.0018, r0 * 2.6,
                                    BRACT, point=0.45))
        # Panicle branches from the lower rachis
        if p.branches > 0 and not branch:
            for b in range(p.branches):
                z = ped + (s[-1] - ped) * (0.08 + 0.55 * b / max(p.branches, 1))
                i = min(int(np.searchsorted(s, z)), len(P) - 1)
                hb = _rotate(_perp(T[i]), T[i], b * GOLDEN)
                sp, fl, _ = self.inflorescence(P[i], hb, rng, detail * 0.8,
                                               n_flowers=max(2, nf // 2), length_scale=p.branch_ratio, branch=True)
                spikes += sp
                flowers += fl
        open_until = 1.0 - p.maturation
        twists = []
        side0 = _perp(T[0])
        for k in range(nf):
            z = sk[k] if k < len(sk) else s[-1]
            i = min(int(np.searchsorted(s, z)), len(P) - 1)
            x = np.array([np.interp(z, s, P[:, q]) for q in range(3)])
            t_dir = T[i]
            side = _proj(_rotate(side0, t_dir, k * math.radians(p.divergence_deg)), t_dir)
            if not np.any(side):
                side = _perp(t_dir)
            pos = k / max(nf - 1, 1)
            o = float(np.clip((open_until - pos) / 0.12 + 1.0, 0.0, 1.0)) if nf > 1 else 1.0
            Lp = p.pedicel_cm * 0.01 * (0.55 + 0.45 * o)
            d_ped = _unit(t_dir * math.cos(math.radians(50)) + side * math.sin(math.radians(50)))
            horiz = _unit(np.array([side[0], side[1], 0.0])) if np.linalg.norm(side[:2]) > 1e-6 else LIGHT
            face = _unit(horiz * (1 - p.flower_facing) + LIGHT * p.flower_facing * 1.5 +
                         UP * math.sin(math.radians(p.flower_tilt_deg)))
            if face @ t_dir > 0.9:
                face = _unit(face + side)
            capsule = o >= 1.0 and rng.random() < p.fruit_set
            if p.bract_mm > 0:
                spikes.append(_blob(x + side * r0 * 0.6, _unit(t_dir + side), p.bract_mm * 0.001, r0 * 2.0, BRACT,
                                    point=0.45))
            if capsule:
                hang = _unit(d_ped * 0.3 - UP)
                end = x + d_ped * Lp * 0.4
                spikes.append(_tube(_bezier(x, x + d_ped * Lp * 0.2, end - d_ped * Lp * 0.1, end, 6),
                                    r0 * 0.6, SPIKE, 5))
                flowers.append(self.capsule(end, hang, rng, detail))
                continue
            end = x + d_ped * Lp * 0.55 + face * Lp * 0.45
            C = _bezier(x, x + d_ped * Lp * 0.5, end - face * Lp * 0.35, end, 10)
            fl, tw = self.flower(end, face, -_unit(C[-1] - C[-3]) + t_dir * 2.0, rng, detail, o)
            twists.append(abs(tw))
            spikes.append(self.ovary(C, r0 * 0.55, tw, detail))
            flowers.append(fl)
        return spikes, flowers, {"twists": twists}

    def ovary(self, C, r, twist, detail) -> MeshData:
        """Pedicel and inferior ovary: six ribs (three carpels) spiralling by the resupination twist."""
        n = len(C)
        T, N, B = frames(C)
        m = max(8, int(12 * detail))
        th = np.linspace(0, 2 * math.pi, m, endpoint=False)
        t = np.linspace(0, 1, n)
        ribs = 1.0 + 0.16 * np.cos(6 * (th[None, :] + twist * _smooth(t, 0.3, 1.0)[:, None])) * \
            _smooth(t, 0.4, 0.8)[:, None]
        R = r * (1.0 + 0.5 * _smooth(t, 0.55, 0.95))[:, None] * ribs
        V = C[:, None, :] + R[..., None] * _ring(th, N, B)
        return _grid(V, SPIKE, t[:, None], th[None, :] / (2 * math.pi), closed=True, caps=True)

    def capsule(self, base, axis, rng, detail) -> MeshData:
        """Capsule (pod): six-ribbed, spindle-shaped; Vanilla 'beans' are long and slender."""
        p = self.p
        L = p.capsule_cm * 0.01
        D = p.capsule_diameter_cm * 0.01
        n = max(10, int(18 * detail))
        t = np.linspace(0, 1, n)
        nth = max(8, int(12 * detail))
        rho = 0.5 * D * np.sin(np.pi * np.clip(t, 0.02, 0.98)) ** 0.45
        th = np.linspace(0, 2 * math.pi, nth, endpoint=False)
        R = rho[:, None] * (1 + 0.08 * np.cos(6 * th[None, :]))
        # Bends under its weight
        pts = [base]
        d = _unit(axis)
        for i in range(1, n):
            d = _unit(d - 0.03 * UP)
            pts.append(pts[-1] + d * L / (n - 1))
        C = np.array(pts)
        T, N, B = frames(C)
        V = C[:, None, :] + R[..., None] * _ring(th, N, B)
        return _grid(V, CAPSULE, t[:, None], th[None, :] / (2 * math.pi), rnd=rng.random(), closed=True, caps=True)

    # ================================================================= vegetative organs
    def leaf(self, base, d0, side0, L, W, rng, detail, base_w=None) -> MeshData:
        """Fleshy leaf: closed lens-shaped cross-section (thickness tapering to the margins and the tip),
        conduplicate V-fold, arching under its weight."""
        p = self.p
        n = max(10, int(min(40, L / 0.008) * detail))
        m = 12 if detail >= 0.6 else 8
        droop = p.leaf_droop
        d = _unit(d0)
        P, D = [np.asarray(base, float)], [d]
        total = math.radians(150) * droop * rng.uniform(0.85, 1.15)
        for j in range(n):
            ang = total * 2 * ((j + 0.5) / n) / n
            axis = np.cross(d, -UP)
            if np.linalg.norm(axis) > 1e-6 and d @ -UP < 0.95:
                d = _rotate(d, axis, ang)
            P.append(P[-1] + d * L / n)
            D.append(d)
        P, D = np.array(P), np.array(D)
        u = np.linspace(0, 1, len(P))
        hw = 0.5 * W * np.interp(u, T_GRID, self._leaf)
        bw = (base_w if base_w is not None else 0.35 * W) * 0.5
        hw = np.maximum(hw, bw * (1 - _smooth(u, 0.0, 0.25)))
        hw[-1] = 0.0
        th = p.leaf_thickness_mm * 0.001 * (1.0 - 0.6 * u) * 0.5 + 1e-4
        th = np.maximum(th, 1e-4)
        tw = math.radians(p.leaf_twist_deg) * rng.uniform(-1, 1)
        side0 = _proj(np.asarray(side0, float), D[0])
        if not np.any(side0):
            side0 = _perp(D[0])
        rows = []
        a = np.linspace(0, 2 * math.pi, m, endpoint=False)
        for j in range(len(P)):
            sd = _proj(side0, D[j])
            if not np.any(sd):
                sd = _perp(D[j])
            sd = _rotate(sd, D[j], tw * u[j])
            nr = _unit(np.cross(sd, D[j]))
            if nr @ UP < 0:
                nr = -nr
            x = hw[j] * np.cos(a)
            y = th[j] * np.sin(a) + p.leaf_fold * 0.35 * np.abs(x) * (1 - 0.5 * u[j])
            rows.append(P[j] + x[:, None] * sd + y[:, None] * nr)
        V = np.array(rows)
        return _grid(V, LEAF, u[:, None], np.cos(a)[None, :], u[:, None] * L, np.cos(a)[None, :] * hw[:, None],
                     rng.random(), closed=True, caps=True)

    def body(self, base, axis, length, radius, rng, detail, age=0.0, flat_dir=None, nodes=0, cane=False) -> MeshData:
        """Pseudobulb or cane: radius profile from its widest point and fullness, lateral compression,
        ridges, constrictions at the nodes and wrinkles that deepen with age (shrivelled backbulbs)."""
        p = self.p
        n = max(12, int(28 * detail))
        m = max(10, int(16 * detail))
        t = np.linspace(0, 1, n)
        a = math.log(0.5) / math.log(float(np.clip(p.bulb_widest, 0.05, 0.95)))
        prof = np.sin(np.pi * np.clip(t ** a, 0, 1)) ** max(p.bulb_fullness, 0.05)
        base_frac = min(1.0, p.stem_radius_mm * 0.001 / max(radius, 1e-5))
        prof = np.maximum(prof, base_frac * (1 - _smooth(t, 0.0, 0.3)))
        if cane:                                                  # Truncate the top of a cane at a node
            prof = np.maximum(prof, 0.55 * (1 - _smooth(t, 0.9, 1.0)))
        prof[-1] = 0.0
        R = radius * prof * (1.0 - 0.12 * age)
        th = np.linspace(0, 2 * math.pi, m, endpoint=False)
        ring = np.ones((n, m))
        if p.bulb_ridges > 0:
            ring = ring + (0.05 + 0.08 * age) * np.cos(p.bulb_ridges * th)[None, :]
        if age > 0:
            ph = rng.uniform(0, 6.28, 3)
            ring = ring + 0.07 * age * np.sin(7 * th[None, :] + ph[0] + 3 * t[:, None]) * np.sin(np.pi * t)[:, None]
        nd = np.full(n, 9.0)
        if nodes > 0:
            zn = (np.arange(1, nodes + 1) / (nodes + 1))
            dist = np.abs(t[:, None] - zn[None, :]).min(1) * length
            nd = dist / max(radius, 1e-5)
            ring = ring * (1.0 - 0.06 * np.exp(-(nd / 0.25) ** 2))[:, None]
        # Path: slight curve
        bend = _perp(axis) * rng.uniform(-1, 1) * 0.08 * length
        C = base[None, :] + axis[None, :] * length * t[:, None] + bend[None, :] * np.sin(np.pi * t)[:, None] ** 2
        T, N, B = frames(C)
        if flat_dir is not None:                                   # Compressed across `flat_dir`
            fd = _proj(np.asarray(flat_dir, float), axis)
            if np.any(fd):
                N = np.array([fd] * n)
                B = _unit(np.cross(T, N))
        e1 = np.cos(th)[None, :, None] * N[:, None, :] * (1 - p.bulb_flatten)
        e2 = np.sin(th)[None, :, None] * B[:, None, :]
        V = C[:, None, :] + (R[:, None] * ring)[..., None] * (e1 + e2)
        return _grid(V, STEM, t[:, None], (th / (2 * math.pi))[None, :], t[:, None] * length, nd[:, None],
                     rng.random(), closed=True, caps=True)

    def root(self, start, d0, length, rng, detail, aerial=True, radius=None) -> list:
        """Aerial or substrate root: random walk (agravitropic in the air, gravitropic in the substrate),
        green growing tip, occasional branch."""
        p = self.p
        r = (p.root_diameter_mm * 0.0005) if radius is None else radius
        steps = max(6, int(length / 0.012 * min(detail, 1.0)))
        d = _unit(d0)
        pts = [np.asarray(start, float)]
        for i in range(steps):
            noise = rng.normal(0, 1, 3) * 0.35 * p.root_wander
            pull = np.array([0.0, 0.0, -0.06]) if aerial else np.array([0.0, 0.0, -0.45])
            d = _unit(d + noise * 0.5 + pull)
            nxt = pts[-1] + d * length / steps
            if aerial and nxt[2] < r:                              # Creeps over the substrate surface
                nxt[2] = r + 0.0005
                d[2] = abs(d[2]) * 0.3
                d = _unit(d)
            pts.append(nxt)
        P = np.array(pts)
        u = np.linspace(0, 1, len(P))
        R = r * (1.0 - 0.45 * _smooth(u, 0.85, 1.0))
        out = [_tube(P, R, ROOT, max(5, int(8 * detail)), u=u)]
        if length > 0.08 and rng.random() < 0.35:
            k = int(len(P) * rng.uniform(0.3, 0.6))
            out += self.root(P[k], _unit(np.cross(d0, UP) * rng.choice([-1, 1]) + rng.normal(0, 0.3, 3)),
                             length * 0.35, rng, detail * 0.8, aerial, r * 0.7)
        return out

    # ================================================================= plants
    def _leaves_along(self, C, s, nodes_s, rng, detail, az0, sizes=1.0, plane=None, outward=None):
        """Two-ranked leaves at the given positions along the curve C (arc lengths s)."""
        p = self.p
        T, _, _ = frames(C)
        out = []
        nn = len(nodes_s)
        for k, z in enumerate(nodes_s):
            i = min(int(np.searchsorted(s, z)), len(C) - 1)
            x = np.array([np.interp(z, s, C[:, q]) for q in range(3)])
            ref = _perp(T[i]) if plane is None else _proj(plane, T[i])
            if not np.any(ref):
                ref = _perp(T[i])
            radial = _rotate(ref, T[i], az0 + k * math.pi + rng.normal(0, 0.12))
            if outward is not None and radial @ outward < 0:
                radial = _rotate(radial, T[i], 0.0)
            youth = (k + 1) / nn
            g = (1.0 - p.leaf_size_gradient * youth ** 2) * (sizes if np.isscalar(sizes) else sizes[k])
            ang = math.radians(p.leaf_angle_deg * (1.0 - 0.25 * youth ** 3))
            d0 = _unit(T[i] * math.cos(ang) + radial * math.sin(ang))
            L = p.leaf_length_cm * 0.01 * g * rng.uniform(0.92, 1.05)
            W = p.leaf_width_cm * 0.01 * (0.7 + 0.3 * g)
            rr = p.stem_radius_mm * 0.001
            out.append(self.leaf(x + radial * rr * 0.3, d0, np.cross(T[i], radial), L, W, rng, detail,
                                 base_w=rr * 1.6))
        return out

    def monopodial(self, rng, detail):
        p = self.p
        leaves, stems, roots, flowers, spikes = [], [], [], [], []
        N = max(1, p.leaves)
        rs = p.stem_radius_mm * 0.001
        H = max(p.stem_length_cm * 0.01, (N + 1) * p.internode_cm * 0.01)
        n = 12
        C = np.array([[0.0, 0.0, -0.01 + (H + 0.01) * q] for q in np.linspace(0, 1, n)])
        C[:, 0] += 0.004 * np.sin(np.linspace(0, 3, n))
        s = arc_length(C)
        stems.append(_tube(C, np.full(n, rs), STEM, max(8, int(12 * detail))))
        z_nodes = s[-1] - (np.arange(N)[::-1] + 0.5) * p.internode_cm * 0.01
        z_nodes = np.clip(z_nodes, 0.01, s[-1])
        az0 = 0.0                                                   # Leaves in two ranks along X
        leaves += self._leaves_along(C, s, z_nodes, rng, detail, az0, plane=np.array([1.0, 0.0, 0.0]))
        # Aerial roots from the lower stem, between the leaves
        for k in range(p.roots):
            z = rng.uniform(0.005, max(0.01, z_nodes[min(len(z_nodes) - 1, 2)]))
            x = np.array([np.interp(z, s, C[:, q]) for q in range(3)])
            a = rng.uniform(0, 2 * math.pi)
            radial = np.array([math.cos(a), math.sin(a), 0.0])
            aerial = rng.random() < p.aerial_share
            d0 = radial + UP * (rng.uniform(-0.2, 0.5) if aerial else -1.2)
            roots += self.root(x + radial * rs * 0.6, d0, p.root_length_cm * 0.01 * rng.uniform(0.6, 1.2), rng,
                               detail, aerial)
        # Spikes from the axils of the 3rd-4th leaf below the apex
        for k in range(p.inflorescences):
            li = max(0, N - 3 - k)
            z = z_nodes[li] - 0.004
            x = np.array([np.interp(z, s, C[:, q]) for q in range(3)])
            side = 1.0 if (li % 2 == 0) else -1.0
            heading = np.array([side, 0.25 * (1 if k % 2 else -1), 0.0])
            sp, fl, st = self.inflorescence(x + _unit(heading) * rs, heading, np.random.default_rng(
                int(rng.integers(1 << 30))), detail)
            spikes += sp
            flowers += fl
        return leaves, stems, roots, flowers, spikes, []

    def sympodial(self, rng, detail):
        p = self.p
        leaves, stems, roots, flowers, spikes = [], [], [], [], []
        G = max(1, p.growths)
        rr = p.stem_radius_mm * 0.001
        step = p.rhizome_cm * 0.01
        h = np.array([1.0, 0.0, 0.0])
        B = np.array([-(G - 1) * step * 0.5, 0.0, rr * 0.8])
        bases, heads = [], []
        for g in range(G):
            bases.append(B.copy())
            heads.append(h.copy())
            if g < G - 1:
                h_next = _rotate(h, UP, math.radians(p.growth_angle_deg) * (1 if g % 2 else -1) * rng.uniform(0.6, 1.2))
                nxt = B + h_next * step
                mid = (B + nxt) / 2 + UP * rr * 0.6
                if step > 0.002:
                    stems.append(_tube(_bezier(B, mid, mid, nxt, 6), rr, STEM, 7))
                B, h = nxt, h_next
        Lb = p.pseudobulb_cm * 0.01
        Rb = p.pseudobulb_diameter_cm * 0.005 if Lb > 0 else rr      # Pseudobulb (or stem) radius
        newest = G - 1
        infl_done = 0
        for g in range(G):
            age = (newest - g) / max(G - 1, 1)
            Bg, hg = bases[g], heads[g]
            lean_dir = _unit(np.cross(hg, UP)) * (1 if g % 2 else -1)
            ax = _unit(UP + lean_dir * math.tan(math.radians(p.bulb_lean_deg)) * rng.uniform(0.5, 1.2) + hg * 0.15)
            leafy = g >= p.leafless_growths
            cane = p.leaf_placement == LeafPlacement.ALONG
            n_nodes = int(Lb / max(p.internode_cm * 0.01, 0.005)) if cane and Lb > 0 else (2 if Lb > 0 else 0)
            if Lb > 0:
                stems.append(self.body(Bg - ax * rr * 0.5, ax, Lb, Rb, rng, detail,
                                       age=age * 0.8 if not leafy else age * 0.3, flat_dir=hg, nodes=n_nodes,
                                       cane=cane))
                top = Bg + ax * Lb * (0.94 if not cane else 0.97)
            else:
                stub = max(0.02, 0.01 + p.leaves * 0.004) if p.leaf_placement == LeafPlacement.BASE else 0.02
                stems.append(_tube(np.array([Bg - UP * 0.005, Bg + ax * stub]), rr * 1.2, STEM, 7))
                top = Bg + ax * stub
            # Leaves
            if leafy:
                if p.leaf_placement == LeafPlacement.APEX:
                    for k in range(p.leaves):
                        sp = math.radians(18) * (k - (p.leaves - 1) / 2) * 2
                        rad = _rotate(lean_dir, ax, math.pi / 2 + sp + rng.normal(0, 0.1))
                        ang = math.radians(p.leaf_angle_deg) * rng.uniform(0.7, 1.1)
                        rad_h = _rotate(hg, UP, math.pi * k + rng.normal(0, 0.2))
                        d0 = _unit(ax * math.cos(ang) + rad_h * math.sin(ang))
                        L = p.leaf_length_cm * 0.01 * rng.uniform(0.85, 1.05) * (1 - 0.15 * k)
                        leaves.append(self.leaf(top - ax * 0.003, d0, np.cross(ax, rad_h), L,
                                                p.leaf_width_cm * 0.01, rng, detail,
                                                base_w=Rb * 0.8))
                elif p.leaf_placement == LeafPlacement.BASE:
                    plane = _unit(np.cross(hg, UP))                  # Fan across the rhizome
                    fan_h = max(Lb * 0.6, 0.01 + p.leaves * 0.004)
                    Cf = np.array([Bg, Bg + ax * fan_h])
                    sf = arc_length(Cf)
                    z = np.linspace(0.002, fan_h * 0.9, p.leaves)
                    leaves += self._leaves_along(Cf, sf, z, rng, detail, rng.uniform(-0.2, 0.2), plane=plane)
                else:
                    Ct = np.array([Bg, Bg + ax * max(Lb, 0.02)])
                    st = arc_length(Ct)
                    zs = np.linspace(st[-1] * 0.45, st[-1] * 0.98, p.leaves)
                    leaves += self._leaves_along(Ct, st, zs, rng, detail, rng.uniform(0, 1), plane=hg,
                                                 sizes=1.0)
            # Roots from the base of each growth
            for k in range(p.roots):
                a = rng.uniform(0, 2 * math.pi)
                radial = np.array([math.cos(a), math.sin(a), 0.0])
                aerial = rng.random() < p.aerial_share
                d0 = radial + UP * (rng.uniform(-0.1, 0.4) if aerial else -1.0)
                roots += self.root(Bg + radial * rr * 0.5, d0, p.root_length_cm * 0.01 * rng.uniform(0.5, 1.1), rng,
                                   detail, aerial)
            # Inflorescences
            sub = np.random.default_rng(int(rng.integers(1 << 30)))
            if p.infl_origin == InflOrigin.APEX and g >= newest - (p.inflorescences - 1) and leafy:
                if Lb > 0:                                          # Sheath (spathe) at the apex
                    spikes.append(_blob(top + ax * 0.012, ax, 0.035, Rb * 1.2, BRACT,
                                        flat=0.5, point=0.5))
                sp, fl, _ = self.inflorescence(top + ax * 0.004, _unit(hg * 0.5 + LIGHT * 0.3), sub, detail)
                spikes += sp
                flowers += fl
            elif p.infl_origin == InflOrigin.BASE and g >= newest - (p.inflorescences - 1):
                o = Bg + hg * rr * 0.6 + UP * 0.002                  # From the base, inside the sheaths
                sp, fl, _ = self.inflorescence(o, _unit(hg + lean_dir * 0.3), sub, detail)
                spikes += sp
                flowers += fl
            elif p.infl_origin == InflOrigin.NODES and Lb > 0 and infl_done < p.inflorescences and \
                    (not leafy or g == newest - 1 or G == 1):
                k_per = max(1, (p.inflorescences - infl_done) // max(1, newest - g + 1) if g < newest else 1)
                for k in range(k_per):
                    f = 0.55 + 0.4 * (k + 0.5) / k_per
                    rad = _rotate(lean_dir, ax, k * 2.4)
                    o = Bg + ax * Lb * f + rad * Rb * 0.8
                    sp, fl, _ = self.inflorescence(o, rad, sub, detail)
                    spikes += sp
                    flowers += fl
                    infl_done += 1
            elif p.infl_origin == InflOrigin.AXIL and g == newest:
                sp, fl, _ = self.inflorescence(Bg + ax * max(Lb, 0.02) * 0.5, hg, sub, detail)
                spikes += sp
                flowers += fl
        return leaves, stems, roots, flowers, spikes, []

    def climbing(self, rng, detail):
        """Vanilla: thick green vine winding up a post, one leaf and one clinging root per node (opposite),
        arching over the top and hanging; axillary racemes on the upper nodes."""
        p = self.p
        leaves, stems, roots, flowers, spikes, support = [], [], [], [], [], []
        Rp = p.support_radius_cm * 0.01
        Hs = p.support_height_m
        rs = p.stem_radius_mm * 0.001
        nP = 16
        pc = np.array([[0.0, 0.0, z] for z in np.linspace(-0.05, Hs, nP)])
        support.append(_tube(pc, Rp * (1 + 0.04 * np.sin(np.linspace(0, 9, nP))), SUPPORT, 14, cap=True))
        total = p.stem_length_cm * 0.01
        ds = 0.01
        pts = []
        a = rng.uniform(0, 2 * math.pi)
        Rh = Rp + rs * 1.05
        pos = np.array([Rh * math.cos(a), Rh * math.sin(a), -0.02])
        climb = math.radians(28)                                    # Gentle helix up the post
        d = UP
        over = False
        for i in range(int(total / ds)):
            pts.append(pos.copy())
            if not over:
                tang = np.array([-math.sin(a), math.cos(a), 0.0])
                d = _unit(UP * math.cos(climb) + tang * math.sin(climb))
                pos = pos + d * ds
                a += (d @ tang) * ds / Rh
                pos[:2] = Rh * np.array([math.cos(a), math.sin(a)])
                if pos[2] >= Hs + rs:
                    over = True
                    out = _unit(np.array([math.cos(a), math.sin(a), 0.0]))
                    d = _unit(UP + out * 0.3)
            else:
                d = _unit(d + out * 0.05 - UP * 0.08)                 # Arches over and hangs
                pos = pos + d * ds
        C = np.array(pts)
        s = arc_length(C)
        stems.append(_tube(C, np.full(len(C), rs), STEM, max(8, int(10 * detail))))
        T, _, _ = frames(C)
        node_s = np.arange(p.internode_cm * 0.01, s[-1] - 0.02, p.internode_cm * 0.01)
        n_nodes = len(node_s)
        upper = [k for k in range(n_nodes) if node_s[k] > s[-1] * 0.55]
        infl_nodes = upper[::max(1, len(upper) // max(1, p.inflorescences))][:p.inflorescences]
        for k, z in enumerate(node_s):
            i = min(int(np.searchsorted(s, z)), len(C) - 1)
            x = C[i]
            axis_pt = np.array([0.0, 0.0, x[2]])
            to_post = _proj(axis_pt - x, T[i])
            on_post = x[2] < Hs and np.linalg.norm(x[:2]) < Rh * 1.5
            away = -to_post if np.any(to_post) else _perp(T[i])
            # Two ranks: leaves alternate left / right of the side facing away from the post
            lr = _unit(np.cross(T[i], away)) * (1 if k % 2 else -1)
            radial = _unit(away * 0.6 + lr * 0.8)
            ang = math.radians(p.leaf_angle_deg)
            d0 = _unit(T[i] * math.cos(ang) + radial * math.sin(ang))
            L = p.leaf_length_cm * 0.01 * rng.uniform(0.85, 1.05)
            leaves.append(self.leaf(x + radial * rs * 0.3, d0, np.cross(T[i], radial), L, p.leaf_width_cm * 0.01,
                                    rng, detail, base_w=rs * 1.2))
            # Root opposite the leaf: clings to the post, or dangles in the air
            if on_post and np.any(to_post):
                r0 = x - radial * rs * 0.5
                tip = np.array([0.0, 0.0, x[2]]) + _unit(np.array([x[0], x[1], 0.0])) * (Rp + 0.001)
                hug = tip - UP * rng.uniform(0.03, 0.07)
                roots.append(_tube(_bezier(r0, r0 + (tip - r0) * 0.6, tip + UP * 0.01, hug, 8),
                                   p.root_diameter_mm * 0.0005, ROOT, 5, u=np.linspace(0, 1, 8)))
            elif k % 2 == 0 and p.roots > 0:
                roots += self.root(x - radial * rs * 0.5, -radial * 0.3 - UP, p.root_length_cm * 0.01, rng, detail,
                                   True)
            if k in infl_nodes:
                sp, fl, _ = self.inflorescence(x + radial * rs, _unit(radial + T[i] * 0.3),
                                               np.random.default_rng(int(rng.integers(1 << 30))), detail)
                spikes += sp
                flowers += fl
        # Ground roots at the base
        for k in range(max(2, p.roots // 3)):
            a2 = rng.uniform(0, 2 * math.pi)
            roots += self.root(C[0], np.array([math.cos(a2), math.sin(a2), -1.0]), p.root_length_cm * 0.01, rng,
                               detail, False)
        return leaves, stems, roots, flowers, spikes, support

    def generate(self, seed: int = 0, detail: float = 1.0, with_roots: bool = True) -> OrchidResult:
        p = self.p
        rng = np.random.default_rng(seed)
        if p.habit == OrchidHabit.MONOPODIAL:
            parts = self.monopodial(rng, detail)
        elif p.habit == OrchidHabit.CLIMBING:
            parts = self.climbing(rng, detail)
        else:
            parts = self.sympodial(rng, detail)
        leaves, stems, roots, flowers, spikes, support = parts
        flowers = [f for f in flowers if f is not None]
        res = OrchidResult(_merge(leaves), _merge(stems), _merge(roots) if with_roots else MeshData.empty(),
                           _merge(flowers), _merge(spikes), _merge(support),
                           {"leaves": len(leaves), "flowers": len(flowers)})
        return res


def single_flower(profile: OrchidProfile, seed: int = 0, detail: float = 1.0) -> MeshData:
    """One open flower facing -Y (front view), lip down: for close-ups and tests."""
    eng = OrchidEngine(profile)
    m, _ = eng.flower(np.zeros(3), LIGHT, UP, np.random.default_rng(seed), detail)
    return m
