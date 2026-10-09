"""
Climbing and trailing plants (vines) grown along user-controlled guide paths.

The guide is a polyline in plant space: a parametric base shape (pole, arch, obelisk spiral, fence, wall,
ground run) or the splines of any curve in the Blender scene, one stem per spline. How the stem uses the
guide depends on the climbing mode (Gianoli 2015; Isnard & Silk 2009):

- Twining (Ipomoea, Phaseolus, Humulus, Wisteria): the stem coils around the guide, which stands for the
  support, as a helix of given radius, pitch and handedness. Most twiners form right-handed helices
  (counter-clockwise seen from above); Humulus and Lonicera are left-handed.
- Tendril (Pisum, Vitis, Cucumis, Passiflora): the stem follows the guide with a slight meander and holds
  on with tendrils. A tendril that has caught the support coils with a perversion, a point where the coil
  handedness reverses, because both ends are fixed (Gerbode et al. 2012, Science 337:1087); a free tendril
  coils at the tip only.
- Clinging (Hedera, Parthenocissus): the stem is pressed along the guide, held by adventitious rootlets on
  the support side; the laminae face away from the support.
- Trailing (Cucurbita, Citrullus): the stem creeps along the guide on the ground, roots at the nodes, and
  the fruits rest on the soil.

The stem is a sequence of phytomers (node, internode, leaf, axillary bud). Internodes and leaves expand
over a zone behind the apex; past the guide end a free searcher tip with an apical hook continues the
shoot (Vecchiato et al. 2023). Nodes may carry lateral shoots (which droop under their weight), tendrils,
flower sites, fruits (any fruit preset, including grape bunches) and nodal roots. Leaves reuse the leaf morphology engine of the trees (textured
cards), so every leaf trait of the tree presets is available.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from .foliage import FoliageInstances
from .leaf_morphology import LeafMorphologyProfile, LeafMorphologyEngine
from .leaf_venation import VenationProfile
from .mesh_engine import MeshData, BotanicalMeshEngine
from .fruit import FruitProfile, body_mesh as fruit_body_mesh, hanging_fruit, pole_depth as fruit_pole_depth

UP = np.array([0.0, 0.0, 1.0])


class ClimbingMode(str, Enum):
    TWINING = "Twining"      # Stem coils around the support (Ipomoea, Phaseolus, Humulus, Wisteria)
    TENDRIL = "Tendril"      # Stem follows the support, tendrils grasp it (Pisum, Vitis, Cucumis, Passiflora)
    CLINGING = "Clinging"    # Pressed against the support by adhesive rootlets (Hedera, Parthenocissus)
    TRAILING = "Trailing"    # Creeping on the ground (Cucurbita, Citrullus, Ipomoea batatas)


class Chirality(str, Enum):
    RIGHT = "Right"          # Right-handed helix: counter-clockwise seen from above (most twiners)
    LEFT = "Left"            # Left-handed helix: clockwise seen from above (Humulus, Lonicera)


class LeafArrangement(str, Enum):
    ALTERNATE = "Alternate"      # Spiral, golden angle
    DISTICHOUS = "Distichous"    # Two ranks, 180 deg (Vitis, Hedera, Cucurbita)
    OPPOSITE = "Opposite"        # Pairs, successive pairs at 90 deg (Humulus, Lonicera)


class TendrilMode(str, Enum):
    NONE = "None"
    NODE = "Node"            # One tendril per node, beside or opposite the leaf (Vitis, Cucurbita, Passiflora)
    LEAF_TIP = "Leaf Tip"    # Terminal leaflets turned into tendrils (Pisum, Lathyrus)


class GuideShape(str, Enum):
    POLE = "Pole"            # Vertical stake
    ARCH = "Arch"            # Garden arch: up one side and over
    SPIRAL = "Spiral"        # Obelisk / tuteur: helix narrowing upward
    FENCE = "Fence"          # Up a post, then along a rail
    WALL = "Wall"            # Zig-zag climb on a vertical plane
    GROUND = "Ground"        # Meandering run on the soil


@dataclass
class VineProfile:
    """Habit of a climbing or trailing plant. Lengths are absolute; the guide sets the overall extent."""
    mode: ClimbingMode = ClimbingMode.TWINING
    chirality: Chirality = Chirality.RIGHT
    coil_radius_cm: float = 1.5        # Helix radius around the guide (support radius + stem)
    coil_pitch_cm: float = 12.0        # Rise per helix turn along the guide
    wander_cm: float = 2.0             # Lateral meander of the stem around the guide
    tip_length_cm: float = 25.0        # Free searcher tip beyond the reached part of the guide
    tip_hook: float = 0.5              # Apical hook / nutation curl of the free tip

    # Stem
    stem_radius_mm: float = 2.5        # At the base
    stem_taper: float = 0.6            # Fraction of the radius lost toward the tip
    internode_cm: float = 10.0         # Mature internode length

    # Leaves on the stem (the leaf itself is described by the 'leaf' section)
    leaf_arrangement: LeafArrangement = LeafArrangement.ALTERNATE
    leaf_size: float = 1.0             # Scale of mature leaves relative to the leaf blade length
    young_leaf_size: float = 0.3       # Size of the youngest leaf relative to a mature one
    expansion_zone_cm: float = 40.0    # Length behind the apex over which internodes and leaves expand
    leaf_facing: float = 0.6           # How strongly laminae turn away from the support, toward the light
    basal_leaf_loss: float = 0.0       # Fraction of the stem (from the base) that has shed its leaves

    # Lateral shoots
    branch_probability: float = 0.1    # Chance that a node bears a lateral shoot
    branch_length_cm: float = 60.0
    branch_angle_deg: float = 45.0     # Insertion angle from the stem
    branch_droop: float = 0.4          # Gravity bending of unsupported lateral shoots

    # Tendrils
    tendril_mode: TendrilMode = TendrilMode.NONE
    tendril_length_cm: float = 15.0
    tendril_branches: int = 1          # Simple (1) or branched (Vitis 2, Pisum 3-5)
    tendril_coils: float = 5.0         # Turns of a free tendril's coiled tip
    tendril_coil_mm: float = 3.0       # Coil radius
    tendril_radius_mm: float = 0.6
    tendril_reach: float = 0.6         # Fraction of tendrils that have caught the support

    # Nodal / adventitious roots
    aerial_roots: float = 0.0          # Density of rootlets at the nodes
    leafless: bool = False             # Leaves reduced to scales (dodder, Cuscuta)
    haustoria: float = 0.0             # Parasites: haustoria pressed into the host at the coils (per node)
    rootlet_length_cm: float = 2.0

    # Stem colour
    stem_color: tuple = (0.24, 0.42, 0.14)
    stem_color_old: tuple = (0.38, 0.30, 0.20)   # Lignified base
    woodiness: float = 0.0             # Fraction of the stem, from the base, that is lignified
    hairiness: float = 0.2             # Pubescence (soft sheen)

    # Fruits (their shape comes from a fruit preset, core.fruit)
    fruit_count: int = 0               # Per stem


@dataclass
class VineResult:
    stem: MeshData
    tendrils: MeshData
    leaves: MeshData
    fruits: MeshData
    roots: MeshData
    foliage: FoliageInstances
    leaf_card: dict
    leaf_engine: LeafMorphologyEngine
    flower_pos: np.ndarray
    flower_dir: np.ndarray
    stem_count: int = 0
    node_count: int = 0
    leaf_count: int = 0
    tendril_count: int = 0
    fruit_count: int = 0
    stats: dict = field(default_factory=dict)


# -----------------------------------------------------------------------------
# Polyline helpers
# -----------------------------------------------------------------------------
def _unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def _smoothstep(e0, e1, x):
    t = np.clip((np.asarray(x, float) - e0) / max(e1 - e0, 1e-12), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def clean_polyline(P) -> np.ndarray:
    P = np.asarray(P, dtype=float).reshape(-1, 3)
    if len(P) < 2:
        return P
    keep = np.concatenate([[True], np.linalg.norm(np.diff(P, axis=0), axis=1) > 1e-6])
    return P[keep]


def arc_length(P) -> np.ndarray:
    if len(P) < 2:
        return np.zeros(len(P))
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])


def resample(P, step: float) -> tuple[np.ndarray, np.ndarray]:
    """Points every `step` of arc length (endpoints kept) and their arc length."""
    s = arc_length(P)
    L = s[-1] if len(s) else 0.0
    n = max(2, int(math.ceil(L / max(step, 1e-6))) + 1)
    t = np.linspace(0.0, L, n)
    Q = np.stack([np.interp(t, s, P[:, k]) for k in range(3)], axis=1)
    return Q, t


def sample_at(P, s, t) -> np.ndarray:
    """Points of polyline P (arc lengths s) at arc lengths t."""
    t = np.clip(np.asarray(t, float), 0.0, s[-1])
    return np.stack([np.interp(t, s, P[:, k]) for k in range(3)], axis=-1)


def frames(P) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rotation-minimising frames (double reflection, Wang et al. 2008): tangents T, normals N, binormals B
    with N x B = T."""
    n = len(P)
    T = np.zeros((n, 3))
    if n >= 2:
        T[1:-1] = P[2:] - P[:-2]
        T[0] = P[1] - P[0]
        T[-1] = P[-1] - P[-2]
    else:
        T[:] = UP
    T = _unit(T)
    N = np.zeros_like(T)
    ref = np.array([1.0, 0.0, 0.0]) if abs(T[0, 2]) > 0.9 else UP
    N[0] = _unit(np.cross(np.cross(T[0], ref), T[0]))
    for i in range(n - 1):
        v1 = P[i + 1] - P[i]
        c1 = v1 @ v1
        if c1 < 1e-18:
            N[i + 1] = N[i]
            continue
        rL = N[i] - (2.0 / c1) * (v1 @ N[i]) * v1
        tL = T[i] - (2.0 / c1) * (v1 @ T[i]) * v1
        v2 = T[i + 1] - tL
        c2 = v2 @ v2
        N[i + 1] = rL - (2.0 / c2) * (v2 @ rL) * v2 if c2 > 1e-18 else rL
        N[i + 1] = _unit(N[i + 1] - (N[i + 1] @ T[i + 1]) * T[i + 1])
    B = np.cross(T, N)
    return T, N, B


def _smooth_noise(s, wavelength, rng, octaves=2) -> np.ndarray:
    """Smooth 1-D noise in [-1, 1] along arc length s (sum of random sines)."""
    out = np.zeros_like(s, dtype=float)
    amp, total = 1.0, 0.0
    for k in range(octaves):
        f = 2 * math.pi / max(wavelength / (2 ** k), 1e-6)
        out += amp * np.sin(f * s * rng.uniform(0.8, 1.25) + rng.uniform(0, 2 * math.pi))
        total += amp
        amp *= 0.5
    return out / total


# -----------------------------------------------------------------------------
# Meshes
# -----------------------------------------------------------------------------
def tube(P, R, sides: int, attrs: dict | None = None, cap_end=True, uv_scale=1.0) -> MeshData:
    """Generalised cylinder along P with radii R (rotation-minimising frames). `attrs`: per-point arrays
    (len(P)) copied to the ring vertices; the closing tip vertex repeats the last value."""
    P = np.asarray(P, float)
    n = len(P)
    if n < 2:
        return MeshData.empty()
    sides = max(3, int(sides))
    T, N, B = frames(P)
    a = np.linspace(0.0, 2 * math.pi, sides + 1)               # Seam vertex duplicated for clean UVs
    ca, sa = np.cos(a), np.sin(a)
    R = np.asarray(R, float)
    V = P[:, None, :] + R[:, None, None] * (ca[None, :, None] * N[:, None, :] + sa[None, :, None] * B[:, None, :])
    V = V.reshape(-1, 3)
    m = sides + 1
    j, i = np.meshgrid(np.arange(n - 1), np.arange(sides), indexing="ij")
    i0 = (j * m + i).ravel()
    quads = np.stack([i0, i0 + 1, i0 + m + 1, i0 + m], axis=1)
    s = arc_length(P)
    circ = 2 * math.pi * max(float(np.mean(R)), 1e-5)
    U = np.repeat((np.arange(m) / sides)[None, :], n, 0).reshape(-1)
    Vv = np.repeat((s / circ * uv_scale)[:, None], m, 1).reshape(-1)
    uv_vert = np.stack([U, Vv], axis=1)
    loops = [quads.reshape(-1)]
    totals = [np.full(len(quads), 4)]
    verts = [V]
    pattr = {k: np.repeat(np.asarray(v, np.float32)[:, None], m, 1).reshape(-1) for k, v in (attrs or {}).items()}
    if cap_end:
        tip = len(V)
        verts.append(P[-1:] + T[-1:] * R[-1] * 0.6)
        ring = (n - 1) * m + np.arange(sides)
        tris = np.stack([ring, ring + 1, np.full(sides, tip)], axis=1)
        loops.append(tris.reshape(-1))
        totals.append(np.full(sides, 3))
        uv_vert = np.concatenate([uv_vert, [[0.5, Vv[-1] + 0.1]]])
        for k in pattr:
            pattr[k] = np.concatenate([pattr[k], pattr[k][-1:]])
    lv = np.concatenate(loops).astype(np.int32)
    lt = np.concatenate(totals).astype(np.int32)
    ls = np.concatenate([[0], np.cumsum(lt)[:-1]]).astype(np.int32)
    return MeshData(np.concatenate(verts).astype(np.float32), lv, ls, lt, uv_vert[lv].astype(np.float32),
                    {k: v.astype(np.float32) for k, v in pattr.items()})


def _merge(parts, names) -> MeshData:
    """Concatenate meshes that may lack some attributes (filled with zeros)."""
    parts = [p for p in parts if len(p.vertices)]
    for p in parts:
        for k in names:
            p.point_attributes.setdefault(k, np.zeros(len(p.vertices), np.float32))
        for k in list(p.point_attributes):
            if k not in names:
                del p.point_attributes[k]
    return MeshData.concatenate(parts)


# -----------------------------------------------------------------------------
# Guides
# -----------------------------------------------------------------------------
def guide_shape(shape: GuideShape | str, height_m: float = 2.0, width_m: float = 1.5, turns: float = 3.0,
                n: int = 200) -> list[np.ndarray]:
    """Base guide paths (plant space, starting at the origin on the ground)."""
    shape = GuideShape(shape)
    H, W = max(height_m, 0.05), max(width_m, 0.05)
    u = np.linspace(0.0, 1.0, n)
    if shape == GuideShape.POLE:
        P = np.stack([0 * u, 0 * u, H * u], 1)
    elif shape == GuideShape.ARCH:
        a = np.pi * u
        P = np.stack([W * 0.5 * (1 - np.cos(a)), 0 * u, H * np.sin(a) ** 0.45], 1)
        # Vertical legs: the arch is a stretched half-superellipse starting on the ground
    elif shape == GuideShape.SPIRAL:
        a = 2 * np.pi * turns * u
        r = W * 0.5 * (1.0 - 0.65 * u)
        P = np.stack([r * np.cos(a) - W * 0.5, r * np.sin(a), H * u], 1)
    elif shape == GuideShape.FENCE:
        k = 0.35                                               # Share of the path on the post
        up = u[u < k] / k
        run = (u[u >= k] - k) / (1 - k)
        post = np.stack([0 * up, 0 * up, 0.75 * H * up], 1)
        rail = np.stack([W * run, 0 * run, 0.75 * H + 0.1 * H * np.sin(2 * np.pi * 2 * run)], 1)
        P = np.concatenate([post, rail])
    elif shape == GuideShape.WALL:
        P = np.stack([0.18 * W * np.sin(2 * np.pi * 1.5 * u) * (0.3 + u), 0 * u, H * u], 1)
    else:
        P = np.stack([W * u, 0.15 * W * np.sin(2 * np.pi * 1.2 * u), 0 * u], 1)
    return [P]


def guide_side(P) -> np.ndarray | None:
    """Unit normal of the plane that contains a (nearly) planar guide, e.g. a path drawn on a wall;
    None for straight or non-planar guides. Points to the -Y side by convention (flip in the UI)."""
    if len(P) < 3:
        return None
    C = P - P.mean(0)
    w, v = np.linalg.eigh(C.T @ C / len(P))
    if w[1] < 1e-6 or w[0] > 0.04 * w[1]:
        return None
    nrm = v[:, 0]
    return -nrm if nrm[1] > 0 else nrm


# -----------------------------------------------------------------------------
# Engine
# -----------------------------------------------------------------------------
@dataclass
class _Shoot:
    P: np.ndarray        # Centreline
    s: np.ndarray        # Arc length
    R: np.ndarray        # Radius
    out: np.ndarray      # Unit "away from the support" vector per point
    main: bool
    ground: float | None # Ground height for trailing shoots
    soil: float = 0.0    # Soil level (height of the guide's first point), any mode

    def grounded(self, z, tol=0.15) -> bool:
        """A trailing shoot lies on the soil here (it may also climb, e.g. a squash on a trellis)."""
        return self.ground is not None and z - self.ground < tol


class VineEngine:
    def __init__(self, profile: VineProfile = None, leaf: LeafMorphologyProfile = None,
                 venation: VenationProfile = None):
        self.p = profile or VineProfile()
        self.leaf = leaf or LeafMorphologyProfile()
        self.venation = venation or VenationProfile()
        self.leaf_engine = LeafMorphologyEngine(self.leaf)

    # ----------------------------------------------------------------- stems
    def _main_shoot(self, G, growth, rng, side) -> _Shoot | None:
        p = self.p
        G = clean_polyline(G)
        if len(G) < 2:
            return None
        sg = arc_length(G)
        Lg = sg[-1] * float(np.clip(growth, 0.0, 1.0))
        tip = p.tip_length_cm * 0.01
        if Lg + tip < 0.02:
            return None
        step = 0.01 if p.mode != ClimbingMode.TWINING else min(0.01, p.coil_pitch_cm * 0.01 / 24)
        mode = p.mode
        if Lg > 0.005:
            Q = sample_at(G, sg, np.linspace(0.0, Lg, max(2, int(math.ceil(Lg / step)) + 1)))
            s = arc_length(Q)
            T, N, B = frames(Q)
            wn = p.wander_cm * 0.01
            if mode == ClimbingMode.TWINING:
                sign = 1.0 if p.chirality == Chirality.RIGHT else -1.0
                phi = sign * 2 * math.pi * s / max(p.coil_pitch_cm * 0.01, 1e-3) + rng.uniform(0, 2 * math.pi)
                r = p.coil_radius_cm * 0.01 * _smoothstep(0.0, 0.25 * p.coil_pitch_cm * 0.01, s)
                r = r * (1.0 + 0.25 * (wn > 0) * _smooth_noise(s, 0.3, rng))
                off = np.cos(phi)[:, None] * N + np.sin(phi)[:, None] * B
                C = Q + r[:, None] * off
                out = _unit(off)
            elif mode == ClimbingMode.TRAILING:
                h = np.cross(T, UP)
                h = np.where(np.linalg.norm(h, axis=1, keepdims=True) < 1e-6, N, h)
                h = _unit(h)
                C = Q + (wn * _smooth_noise(s, 3 * p.internode_cm * 0.01 + 0.2, rng))[:, None] * h
                C[:, 2] = Q[:, 2] + p.stem_radius_mm * 0.001
                out = np.repeat(UP[None, :], len(C), 0)
            else:
                amp = wn * (0.3 if mode == ClimbingMode.CLINGING else 1.0)
                if side is not None:          # Meander within the plane of a wall-like guide
                    lat = _unit(np.cross(side, T))
                    C = Q + (amp * _smooth_noise(s, 4 * p.internode_cm * 0.01 + 0.2, rng))[:, None] * lat
                    C = C + side[None, :] * p.stem_radius_mm * 0.001
                    out = np.repeat(side[None, :], len(C), 0)
                else:
                    C = Q + (amp * _smooth_noise(s, 4 * p.internode_cm * 0.01 + 0.2, rng))[:, None] * N \
                        + (amp * _smooth_noise(s, 3.3 * p.internode_cm * 0.01 + 0.2, rng))[:, None] * B
                    hz = np.cross(T, UP)
                    hz = np.where(np.linalg.norm(hz, axis=1, keepdims=True) < 1e-6, N, hz)
                    out = _unit(hz)
            d_end = T[-1]
            start = C
        else:
            start = G[:1]
            d_end = _unit(G[1] - G[0])
            o0 = UP if mode == ClimbingMode.TRAILING else np.cross(d_end, UP)
            out = _unit((o0 if np.linalg.norm(o0) > 1e-6 else np.array([1.0, 0.0, 0.0]))[None, :])
        # Free searcher tip: continues the stem, curling with the apical hook (and staying on the soil
        # for trailing shoots)
        if tip > 0.005:
            n = max(3, int(tip / 0.01))
            d = d_end.copy()
            if mode == ClimbingMode.TRAILING:
                d[2] = 0.0
                d = _unit(d) if np.linalg.norm(d) > 1e-6 else np.array([1.0, 0.0, 0.0])
            axis = _unit(np.cross(d, out[-1])) if np.linalg.norm(np.cross(d, out[-1])) > 1e-6 else \
                _unit(np.cross(d, UP) + 1e-9)
            pts = [start[-1]]
            hook = p.tip_hook * math.radians(140.0)
            for k in range(1, n + 1):
                f = k / n
                ang = hook * (2.0 * f) / n * f                    # Curvature grows toward the apex
                d = _rotate(d, axis, ang)
                if mode != ClimbingMode.TRAILING:       # Long unsupported tips sag a little
                    d = _unit(d - 0.02 * f * UP)
                pts.append(pts[-1] + d * (tip / n))
            T1 = np.array(pts[1:])
            C = np.concatenate([start, T1])
            out = np.concatenate([out, np.repeat(out[-1:], len(T1), 0)])
        else:
            C = start
        if len(C) < 2:
            return None
        s = arc_length(C)
        L = s[-1]
        u = s / max(L, 1e-9)
        r0 = p.stem_radius_mm * 0.001
        R = r0 * (1.0 - p.stem_taper * u)
        dist_tip = L - s
        R = R * (0.35 + 0.65 * _smoothstep(0.0, max(tip, 0.05), dist_tip) ** 0.5)
        R = np.maximum(R, 0.00025)
        ground = float(G[0, 2]) if mode == ClimbingMode.TRAILING else None
        return _Shoot(C, s, R, out, True, ground, float(G[0, 2]))

    def _lateral(self, parent: _Shoot, s0: float, direction, length, rng) -> _Shoot | None:
        p = self.p
        if length < 0.03:
            return None
        n = max(3, int(length / 0.015))
        x = sample_at(parent.P, parent.s, s0)
        d = _unit(np.asarray(direction, float))
        pts = [x]
        trailing = parent.grounded(x[2])
        wall = parent.out[min(np.searchsorted(parent.s, s0), len(parent.s) - 1)] \
            if p.mode == ClimbingMode.CLINGING else None
        for k in range(n):
            if wall is not None:                             # Stay pressed to the support
                d = d + rng.normal(0, 0.03, 3) - p.branch_droop * 0.06 * UP
                d = _unit(d - (d @ wall) * wall)
            elif trailing:
                d = d.copy()
                d[2] = 0.0
                d = _rotate(_unit(d), UP, rng.normal(0, 0.06))
            else:
                d = _unit(d - p.branch_droop * 0.06 * UP + rng.normal(0, 0.03, 3))
            pts.append(pts[-1] + d * (length / n))
        P = np.array(pts)
        if trailing:
            P[:, 2] = parent.ground + p.stem_radius_mm * 0.001
        s = arc_length(P)
        R0 = float(np.interp(s0, parent.s, parent.R)) * 0.7
        R = np.maximum(R0 * (1.0 - 0.7 * s / s[-1]), 0.0002)
        # Leaves of a lateral face the same way as the parent's: away from the support (or up, on the soil)
        out = np.repeat(parent.out[min(np.searchsorted(parent.s, s0), len(parent.s) - 1)][None, :], len(P), 0)
        if trailing:
            out[:] = UP
        return _Shoot(P, s, R, out, False, parent.ground if trailing else None, parent.soil)

    # ----------------------------------------------------------------- nodes
    def _nodes(self, sh: _Shoot):
        """Arc lengths of the nodes: internodes shorten inside the expansion zone near the apex."""
        p = self.p
        L = sh.s[-1]
        out, s = [], p.internode_cm * 0.01 * 0.5
        zone = max(p.expansion_zone_cm * 0.01, 0.01)
        while s < L - 0.01:
            out.append(s)
            dist = L - s
            s += max(0.004, p.internode_cm * 0.01 * (0.2 + 0.8 * float(_smoothstep(0.0, zone, dist))))
        return np.array(out)

    def generate(self, guides, seed: int = 0, growth: float = 1.0, detail: float = 1.0,
                 leaf_density: float = 1.0, with_roots: bool = True, side_flip: bool = False,
                 fruit: FruitProfile | None = None) -> VineResult:
        p = self.p
        rng_stem = np.random.default_rng(seed)
        card = self.leaf_engine.generate_3d_leaf_mesh()
        card_len = float(np.max(card["vertices"][:, 1])) if len(card["vertices"]) else 0.05
        sides = max(4, int(round(7 * detail)))
        shoots: list[_Shoot] = []
        sides_of = []
        for gi, G in enumerate(guides):
            G = clean_polyline(G)
            side = guide_side(G)
            if side is not None and side_flip:
                side = -side
            sh = self._main_shoot(G, growth, np.random.default_rng(seed * 7919 + gi), side)
            if sh is not None:
                shoots.append(sh)
                sides_of.append((G, side))

        leaves_pos, leaves_y, leaves_z, leaves_s, leaves_rnd = [], [], [], [], []
        tendrils, roots, fruits, stems = [], [], [], []
        fl_pos, fl_dir = [], []
        n_nodes = n_tendrils = 0
        all_shoots = []
        rng_br = np.random.default_rng(seed + 11)
        zone = max(p.expansion_zone_cm * 0.01, 0.01)
        for sh, (G, side) in zip(shoots, sides_of):
            all_shoots.append((sh, G, side))
            if p.branch_probability > 0:
                ns = self._nodes(sh)
                for k, s0 in enumerate(ns):
                    dist = sh.s[-1] - s0
                    if rng_br.random() >= p.branch_probability or dist < zone:
                        continue
                    T = _tangent_at(sh, s0)
                    o = sh.out[min(np.searchsorted(sh.s, s0), len(sh.s) - 1)]
                    ang = math.radians(p.branch_angle_deg)
                    if p.mode == ClimbingMode.CLINGING:     # Root climbers spread over the support
                        side_dir = _unit(np.cross(o, T)) * (1.0 if rng_br.random() < 0.5 else -1.0)
                    else:
                        side_dir = _unit(o + rng_br.normal(0, 0.4, 3))
                    side_dir = _unit(side_dir - (side_dir @ T) * T)
                    d = math.cos(ang) * T + math.sin(ang) * side_dir
                    Lb = p.branch_length_cm * 0.01 * rng_br.uniform(0.5, 1.2) * float(_smoothstep(0, 3 * zone, dist))
                    lat = self._lateral(sh, s0, d, Lb, rng_br)
                    if lat is not None:
                        all_shoots.append((lat, G, side))

        rng_leaf = np.random.default_rng(seed + 23)
        rng_ten = np.random.default_rng(seed + 37)
        rng_root = np.random.default_rng(seed + 51)
        div = {LeafArrangement.ALTERNATE: math.radians(137.508), LeafArrangement.DISTICHOUS: math.pi,
               LeafArrangement.OPPOSITE: math.pi / 2}[p.leaf_arrangement]
        for sh, G, side in all_shoots:
            age = 1.0 - sh.s / max(sh.s[-1], 1e-9)
            woody = np.clip((age - (1.0 - p.woodiness)) / 0.15, 0.0, 1.0) if p.woodiness > 0 else np.zeros_like(age)
            stems.append(tube(sh.P, sh.R, sides if sh.main else max(4, sides - 2),
                              {"age": age, "woody": woody}))
            T_all, N_all, B_all = frames(sh.P)
            ns = self._nodes(sh)
            n_nodes += len(ns)
            L = sh.s[-1]
            for k, s0 in enumerate(ns):
                i = min(np.searchsorted(sh.s, s0), len(sh.s) - 1)
                x = sh.P[i]
                T, N, B = T_all[i], N_all[i], B_all[i]
                o = sh.out[i]
                r_here = sh.R[i]
                dist = L - s0
                young = float(_smoothstep(0.0, zone, dist))
                theta0 = k * div + (0.0 if sh.main else 1.0)
                thetas = [theta0, theta0 + math.pi] if p.leaf_arrangement == LeafArrangement.OPPOSITE else [theta0]
                lost = (s0 / max(L, 1e-9)) < p.basal_leaf_loss * (0.7 + 0.6 * rng_leaf.random()) or p.leafless
                if p.haustoria > 0 and dist > zone * 0.3 and rng_leaf.random() < p.haustoria:
                    stems.extend(self._haustorium(x, T, r_here, G))
                leaf_tip = None
                for th in thetas:
                    rdir = math.cos(th) * N + math.sin(th) * B
                    keep = rng_leaf.random() < leaf_density
                    scale = p.leaf_size * (p.young_leaf_size + (1 - p.young_leaf_size) * young) \
                        * rng_leaf.uniform(0.85, 1.15)
                    roll = rng_leaf.normal(0, 0.25)
                    if lost or not keep:
                        continue
                    y, z = self._leaf_frame(T, rdir, o if not sh.grounded(x[2]) or sh.ground is None else UP,
                                            sh.grounded(x[2]), roll)
                    leaves_pos.append(x + rdir * r_here)
                    leaves_y.append(y)
                    leaves_z.append(z)
                    leaves_s.append(scale)
                    leaves_rnd.append(rng_leaf.random())
                    if leaf_tip is None:
                        leaf_tip = (x + rdir * r_here + y * card_len * scale, y)
                # Tendrils
                if p.tendril_mode != TendrilMode.NONE and dist > 0.02:
                    if p.tendril_mode == TendrilMode.NODE:
                        rdir = math.cos(theta0 + math.pi * 0.8) * N + math.sin(theta0 + math.pi * 0.8) * B
                        a0, d0 = x + rdir * r_here, _unit(rdir + 0.5 * T)
                    elif leaf_tip is not None:
                        a0, d0 = leaf_tip
                    else:
                        a0 = None
                    if a0 is not None:
                        for b in range(max(1, p.tendril_branches)):
                            db = _unit(d0 + rng_ten.normal(0, 0.35 if b else 0.0, 3))
                            caught = rng_ten.random() < p.tendril_reach
                            ln = p.tendril_length_cm * 0.01 * (0.4 + 0.6 * young) * rng_ten.uniform(0.8, 1.2)
                            path = self._tendril(a0, db, ln, caught, G, rng_ten)
                            if path is not None and len(path) > 2:
                                tr = p.tendril_radius_mm * 0.001 * (0.6 + 0.4 * young)
                                Rt = np.maximum(tr * (1.0 - 0.6 * arc_length(path) / arc_length(path)[-1]), 0.00012)
                                tendrils.append(tube(path, Rt, 4, {"age": np.zeros(len(path)),
                                                                   "woody": np.zeros(len(path))}))
                                n_tendrils += 1
                # Nodal roots
                if with_roots and p.aerial_roots > 0 and dist > zone * 0.5:
                    if rng_root.random() < p.aerial_roots:
                        if sh.ground is None or sh.grounded(x[2]):
                            roots.extend(self._rootlets(x, T, o, sh.ground is not None, rng_root))
                # Flower sites (axils)
                if dist > 0.03:
                    fd = _unit(o * 0.6 + UP * 0.8 + 0.3 * (math.cos(theta0) * N + math.sin(theta0) * B))
                    fl_pos.append(x + _unit(fd) * r_here)
                    fl_dir.append(fd)

        # Fruits on the main stems
        fruit_parts = []
        n_fruit = 0
        if p.fruit_count > 0 and fruit is not None:
            rng_fr = np.random.default_rng(seed + 83)
            proto = hanging_fruit(fruit, detail, seed) if fruit.cluster_berries > 1 else \
                fruit_body_mesh(fruit, detail, seed)
            for sh, G, side in all_shoots:
                if not sh.main:
                    continue
                ns = self._nodes(sh)
                cand = ns[(sh.s[-1] - ns) > zone]
                if len(cand) == 0:
                    continue
                pick = rng_fr.choice(len(cand), min(p.fruit_count, len(cand)), replace=False)
                for s0 in np.sort(cand[pick]):
                    fruit_parts.extend(self._fruit(sh, s0, proto, rng_fr, fruit))
                    n_fruit += 1

        foliage = FoliageInstances(np.array(leaves_pos).reshape(-1, 3), np.array(leaves_y).reshape(-1, 3),
                                   np.array(leaves_z).reshape(-1, 3), np.array(leaves_s),
                                   np.array(leaves_rnd))
        leaf_mesh = BotanicalMeshEngine.build_foliage_mesh(foliage, card) if len(foliage) else MeshData.empty()
        stalks = [f for f in fruit_parts if "fruit_t" not in f.point_attributes]
        bodies = [f for f in fruit_parts if "fruit_t" in f.point_attributes]
        from .fruit import ATTRS as FRUIT_ATTRS
        stem_mesh = _merge(stems + stalks, ("age", "woody"))
        return VineResult(
            stem=stem_mesh,
            tendrils=_merge(tendrils, ("age", "woody")),
            leaves=leaf_mesh,
            fruits=_merge(bodies, FRUIT_ATTRS),
            roots=_merge(roots, ()),
            foliage=foliage, leaf_card=card, leaf_engine=self.leaf_engine,
            flower_pos=np.array(fl_pos).reshape(-1, 3), flower_dir=np.array(fl_dir).reshape(-1, 3),
            stem_count=len(shoots), node_count=n_nodes, leaf_count=len(foliage), tendril_count=n_tendrils,
            fruit_count=n_fruit,
            stats={"stem_length_m": float(sum(sh.s[-1] for sh, _, _ in all_shoots))})

    # ----------------------------------------------------------------- organs
    def _haustorium(self, x, T, r, G):
        """Haustorium of a parasitic twiner (Cuscuta): a peg leaving the inner side of a tight coil and
        pressed into the host, ending in a small pad (Teixeira-Costa 2021; hypertrophied host tissue)."""
        if G is None or len(G) < 2:
            return []
        i = int(np.argmin(((G - x) ** 2).sum(1)))
        to = G[i] - x
        to = to - (to @ T) * T
        d = float(np.linalg.norm(to))
        if d < 1e-4:
            return []
        u = to / d
        reach = max(0.0015, d - self.p.coil_radius_cm * 0.01 * 0.55)    # Host surface ~ just inside the coil
        a = x + u * r * 0.6
        b = a + u * min(reach, 0.006)
        P = np.linspace(a, b, 4)
        R = np.array([r * 0.9, r * 0.8, r * 1.2, r * 1.6])                 # Swelling at the host contact
        return [tube(P, R, 5, {"age": np.zeros(4), "woody": np.zeros(4)})]

    def _leaf_frame(self, T, rdir, out, trailing, roll):
        """Leaf axis (petiole -> apex) and adaxial normal. Petioles leave the stem at the petiole angle; the
        lamina then turns away from the support and toward the light by `leaf_facing`."""
        p = self.p
        pa = math.radians(self.leaf.petiole_angle_deg)
        y0 = _unit(math.cos(pa) * T + math.sin(pa) * rdir)
        a = math.radians(self.leaf.mean_leaf_angle_deg)
        if trailing:      # Long erect petioles hold the blades over the runner, facing up
            h = rdir - (rdir @ UP) * UP
            h = _unit(h) if np.linalg.norm(h) > 1e-6 else np.array([1.0, 0.0, 0.0])
            y_light = _unit(h * math.cos(a) + UP * math.sin(a) * 1.3)
            z_pref = UP
        else:
            o = out - (out @ T) * T
            o = _unit(o) if np.linalg.norm(o) > 1e-6 else rdir
            lean = _unit(o + 0.35 * rdir)
            y_light = _unit(lean * math.cos(a) - UP * math.sin(a) * 0.4)
            z_pref = _unit(UP * 0.6 + o)
        f = p.leaf_facing
        y = _unit((1 - f) * y0 + f * y_light)
        z = z_pref - (z_pref @ y) * y
        if np.linalg.norm(z) < 1e-6:
            z = np.cross(y, rdir)
        z = _unit(z)
        x = np.cross(y, z)
        c, s = math.cos(roll), math.sin(roll)
        return y, _unit(c * z + s * x)

    def _tendril(self, a0, d0, length, caught, G, rng):
        """Polyline of a tendril. Caught: straight base, then a coil between the base and an anchor on the
        support with a perversion halfway (the handedness reverses because both ends are fixed). Free: a
        straight part ending in a tight coil."""
        p = self.p
        if length < 0.01:
            return None
        straight = 0.3 * length
        a1 = a0 + d0 * straight
        rho = p.tendril_coil_mm * 0.001
        if caught and len(G) >= 2:
            sg = arc_length(G)
            j = int(np.argmin(np.linalg.norm(G - a1[None, :], axis=1)))
            anchor_s = float(np.clip(sg[j] + rng.uniform(0.0, 0.6) * length, 0, sg[-1]))
            b = sample_at(G, sg, anchor_s) + rng.normal(0, 0.004, 3)
            rest = length - straight
            D = float(np.linalg.norm(b - a1))
            if D > rest * 0.9:                               # Out of reach: stop short of the support
                b = a1 + (b - a1) * (rest * 0.9 / max(D, 1e-9))
                D = rest * 0.9
            axis = _unit(b - a1) if D > 1e-6 else d0
            coils = math.sqrt(max(rest * rest - D * D, 0.0)) / (2 * math.pi * rho)
            n = max(12, int(coils * 16) + 8)
            t = np.linspace(0.0, 1.0, n)
            phi = 2 * math.pi * coils * (0.5 - np.abs(t - 0.5))          # Perversion at t = 0.5
            env = _smoothstep(0.0, 0.08, t) * _smoothstep(0.0, 0.08, 1.0 - t)
            e1 = _unit(np.cross(axis, UP) if abs(axis @ UP) < 0.95 else np.cross(axis, [1.0, 0, 0]))
            e2 = np.cross(axis, e1)
            pts = a1[None, :] + D * t[:, None] * axis[None, :] + (rho * env)[:, None] * \
                (np.cos(phi)[:, None] * e1[None, :] + np.sin(phi)[:, None] * e2[None, :])
        else:
            rest = length - straight
            coils = max(p.tendril_coils, 0.5)
            n = max(12, int(coils * 14))
            t = np.linspace(0.0, 1.0, n)
            e1 = _unit(np.cross(d0, UP) if abs(d0 @ UP) < 0.95 else np.cross(d0, [1.0, 0, 0]))
            e2 = np.cross(d0, e1)
            # Arc length split: a gently curving reach, then a tightening spring at the tip
            reach = rest - 2 * math.pi * rho * coils * 0.7
            reach = max(reach, rest * 0.3)
            phi = 2 * math.pi * coils * t ** 2
            rr = rho * (1.0 - 0.5 * t) * _smoothstep(0.0, 0.3, t)
            pts = a1[None, :] + reach * t[:, None] * _unit(d0 - 0.2 * UP)[None, :] + rr[:, None] * \
                ((np.cos(phi) - 1)[:, None] * e1[None, :] + np.sin(phi)[:, None] * e2[None, :])
        return np.concatenate([np.stack([a0, a0 + d0 * straight * 0.5]), pts])

    def _rootlets(self, x, T, out, trailing, rng):
        p = self.p
        ln = p.rootlet_length_cm * 0.01
        parts = []
        k = 2 if trailing else 4
        for _ in range(k):
            if trailing:
                d = _unit(-UP + rng.normal(0, 0.5, 3))
                d[2] = -abs(d[2])
            else:
                d = _unit(-out + rng.normal(0, 0.45, 3))
            L = ln * rng.uniform(0.6, 1.2)
            n = 6
            pts = [x]
            for _ in range(n):
                d = _unit(d + rng.normal(0, 0.15, 3))
                pts.append(pts[-1] + d * L / n)
            parts.append(tube(np.array(pts), np.linspace(0.0006, 0.0002, n + 1), 3))
        return parts

    def _fruit(self, sh: _Shoot, s0: float, proto: MeshData, rng, fr: FruitProfile) -> list[MeshData]:
        """Stalk and fruit at a node. Fruits hang from climbing stems; on the soil (trailing runners, or a
        hanging fruit that would reach the ground) they rest on it, elongated ones on their side and round
        ones with the stalk end turned up toward the runner. The stalk always ends inside the fruit's stalk
        end (also when it is sunk, as in pumpkins). Bunches (grapes) hang from their peduncle."""
        x = sample_at(sh.P, sh.s, s0)
        T = _tangent_at(sh, s0)
        sc = rng.uniform(0.85, 1.12)
        if fr.cluster_berries > 1:
            Rm = _frame_rot(_unit(UP + rng.normal(0, 0.1, 3)), rng.uniform(0, 2 * math.pi))
            V = (proto.vertices.astype(float) * sc) @ Rm.T + x[None, :]
            return [MeshData(V.astype(np.float32), proto.loop_vertex.copy(), proto.loop_start.copy(),
                             proto.loop_total.copy(), proto.loop_uv.copy(),
                             {k: v.copy() for k, v in proto.point_attributes.items()})]
        L = fr.length_cm * 0.01 * sc
        Dm = fr.diameter_cm * 0.01 * sc
        stalk = max(fr.stalk_length_cm * 0.01, 0.005)
        side = _unit(np.cross(T, UP) if abs(T @ UP) < 0.95 else np.cross(T, [1.0, 0, 0]))
        side = side if rng.random() < 0.5 else -side
        hang_axis = _unit(-UP + rng.normal(0, 0.12, 3))
        hang_attach = x + side * stalk * 0.4 - UP * stalk * 0.8
        lowest = hang_attach[2] - L * 0.98
        on_soil = sh.grounded(x[2]) or lowest < sh.soil
        if on_soil:
            # Elongated fruits lie on their side; round / oblate ones tilt the stalk end up toward the runner
            tilt = 0.0 if L / max(Dm, 1e-6) > 1.15 else math.radians(rng.uniform(35, 60))
            h = _unit(side - (side @ UP) * UP) if np.linalg.norm(side - (side @ UP) * UP) > 1e-6 else side
            axis = _unit(h * math.cos(tilt) - UP * math.sin(tilt))
            attach = x + h * stalk * 0.6       # Provisional; the fruit is then dropped onto the soil
        else:
            axis, attach = hang_axis, hang_attach
        Rm = _frame_rot(axis, rng.uniform(0, 2 * math.pi))
        V = (proto.vertices.astype(float) * sc) @ Rm.T + attach[None, :]
        if on_soil:                            # Rest exactly on the soil (slightly settled into it)
            dz = sh.soil - 0.004 * sc - V[:, 2].min()
            V[:, 2] += dz
            attach = attach + np.array([0.0, 0.0, dz])
        end = attach + axis * (fruit_pole_depth(fr) * sc + 0.004)       # Inside the stalk end
        lift = max(end[2] - x[2], 0.0)
        mid = x + (end - x) * 0.5 + UP * (0.25 * lift + 0.01) if on_soil else x + side * stalk * 0.45
        pts = _bezier_q(np.array([x, mid, attach]), 12)
        pts = np.concatenate([pts, attach + np.outer(np.linspace(0.34, 1.0, 3), end - attach)])  # Along the axis
        rs = float(np.clip(fr.stalk_radius_mm * 0.001, 0.0008, 0.03))
        stalk_mesh = tube(pts, np.linspace(rs, rs * 1.3, len(pts)), 6, {"age": np.full(len(pts), 0.3),
                                                                       "woody": np.full(len(pts), 0.5)})
        body = MeshData(V.astype(np.float32), proto.loop_vertex.copy(), proto.loop_start.copy(),
                        proto.loop_total.copy(), proto.loop_uv.copy(),
                        {**{k: v.copy() for k, v in proto.point_attributes.items()},
                         "fruit_random": np.full(len(V), rng.random(), np.float32)})
        return [stalk_mesh, body]


def _frame_rot(z, roll):
    """Rotation whose third column is z (rolled by `roll` about it)."""
    z = _unit(np.asarray(z, float))
    e1 = _unit(np.cross(z, UP) if abs(z @ UP) < 0.95 else np.cross(z, [1.0, 0, 0]))
    e2 = np.cross(z, e1)
    e1, e2 = math.cos(roll) * e1 + math.sin(roll) * e2, -math.sin(roll) * e1 + math.cos(roll) * e2
    return np.stack([e1, e2, z], axis=1)


def _rotate(v, axis, ang):
    axis = _unit(np.asarray(axis, float))
    c, s = math.cos(ang), math.sin(ang)
    return v * c + np.cross(axis, v) * s + axis * (axis @ v) * (1 - c)


def _tangent_at(sh: _Shoot, s0):
    i = min(max(np.searchsorted(sh.s, s0), 1), len(sh.s) - 1)
    return _unit(sh.P[i] - sh.P[i - 1])


def _bezier_q(P, n):
    t = np.linspace(0, 1, n)[:, None]
    return (1 - t) ** 2 * P[0] + 2 * (1 - t) * t * P[1] + t ** 2 * P[2]
