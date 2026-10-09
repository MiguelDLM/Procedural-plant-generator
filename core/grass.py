"""
Grasses (Poaceae): cereals, forage and ornamental grasses, lawn tufts.

A grass shoot is a chain of phytomers (node, internode, leaf with sheath and blade, axillary bud) ending in an
inflorescence; tillers grow from the basal buds (Evers et al. 2005, New Phytol. 166: 801-812). The model
follows the architectural crop models ADEL-Maize / ADEL-Wheat (Fournier & Andrieu 1998, Ann. Bot. 81:
233-250) and the phytomer geometry of Wen et al. (2021, AoB Plants 13: plab055):

- Internodes lengthen up the culm; leaves alternate at 180 deg (distichous) with a slow drift.
- The sheath wraps the culm from its node; the blade starts at the ligule. Blade length and width follow a bell
  along the culm (longest leaves a little above mid-height, the flag leaf shorter).
- The blade midrib leaves the culm at an insertion angle and bends down under its weight, more toward the
  tip; the lamina is folded in a V along the midrib, twists slowly and, in maize, has wavy margins.
- Inflorescences: distichous spikes with awns (wheat, barley), panicles with whorled branches (oats, rice,
  sorghum), plumes (pampas and fountain grass), one-sided "flag" spikes (Bouteloua), and in maize a terminal
  tassel and lateral ears (cob with kernel rows, husk leaves, silks) on shanks.
- Collar: the ligule (a membrane or a fringe of hairs on the inner face, where the sheath meets the blade)
  and the auricles (paired lobes at the sheath margins), which identify cereals: wheat short blunt hairy
  auricles, barley long clasping ones, oats none and a tall toothed ligule, ryegrass claw-like, rice a long
  ligule and sickle-shaped auricles (extension keys: Oklahoma State, Victoria Agriculture, Oregon State).
- Nodes of sugarcane (Artschwager & Brandes 1958, USDA Handbook 122): root band with two staggered rows of
  root primordia (the upper one often irregular), growth ring swollen on one side, wax band below the node;
  each node varies a little (band widths, swelling, tint).
- Roots: fibrous crown roots; maize has brace (prop) roots from the lowest nodes.
- Ripeness turns green organs golden (cereals at harvest) and makes heavy heads nod.

Plant space: soil at z = 0.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from .mesh_engine import MeshData
from .vine import tube, frames, arc_length
from .vegetable import revolve

UP = np.array([0.0, 0.0, 1.0])
GOLDEN = math.radians(137.50776)
ATTRS = ("grass_u", "grass_v", "grass_part", "grass_node", "grass_nid")
FAR = 9.0          # grass_node of organs that are not the culm surface (no node colouring)
# grass_part: 0 blade, 1 culm / sheath, 2 spikelet, 3 awn / hair, 4 kernel, 5 husk, 6 silk, 7 root, 8 ligule,
# 9 auricle; grass_nid: random value of the nearest node (node-to-node variation)


class Ligule(str, Enum):
    MEMBRANE = "Membrane"        # Thin translucent membrane (most cereals)
    HAIRS = "Hairs"              # Fringe of hairs (many tropical and prairie grasses)
    NONE = "None"


class Auricles(str, Enum):
    NONE = "None"                # Oats, maize, sorghum
    BLUNT = "Blunt"              # Short, blunt, hairy (wheat)
    CLAW = "Claw"                # Slender claw-like (ryegrass, rice)
    CLASPING = "Clasping"        # Long, clasping and overlapping around the culm (barley)


class GrassHead(str, Enum):
    NONE = "None"                # Leaves only (lawn, sterile tufts)
    SPIKE = "Spike"              # Wheat, barley, rye, ryegrass
    PANICLE = "Panicle"          # Oats, rice, sorghum, many forage grasses
    PLUME = "Plume"              # Pampas grass, fountain grass: dense silky panicle
    ONE_SIDED = "One-sided"      # Bouteloua: comb-like spikes on one side of a rachis
    MAIZE = "Maize"              # Tassel at the top, ears on lateral shanks


@dataclass
class GrassProfile:
    """Habit of a grass. Lengths in cm unless noted (culm height in m)."""
    # Tillering
    tillers: int = 1                    # Main shoot + tillers
    tiller_spread_deg: float = 15.0     # Lean of the outermost tillers
    tiller_variation: float = 0.2       # Tillers are shorter than the main shoot by up to this fraction
    flowering_tillers: float = 1.0      # Share of tillers that elongate a culm and flower (others stay leafy)
    clump_radius_cm: float = 2.0

    # Culm (stem)
    culm_height_m: float = 1.0          # Base to the top of the culm (inflorescence excluded)
    culm_radius_mm: float = 2.0
    nodes: int = 6                      # Elongated internodes / leaves on the culm
    internode_gradient: float = 1.5     # >1: lower internodes short, upper long
    basal_leaves: int = 0               # Extra leaves crowded at the base (tufts, rosettes)
    culm_color: tuple = (0.45, 0.55, 0.25)

    # Nodes and internodes (culm surface)
    node_swell: float = 0.06            # Swelling of the culm at the nodes
    growth_ring: float = 0.0            # Narrow ring just above the node, slightly constricted (sugarcane)
    internode_barrel: float = 0.0       # Internodes bulging in the middle (sugarcane)
    bud_groove: float = 0.0             # Groove up each internode on the side of its axillary bud (maize, cane)
    bud_size_mm: float = 2.0            # Axillary bud ("eye") in the axil of each leaf
    zigzag_deg: float = 2.0             # The culm turns slightly at each node, alternately
    wax_band: float = 0.0               # Whitish waxy band just below each node (sugarcane, sorghum)
    node_color: tuple = (0.62, 0.62, 0.40)   # Node ring / root band

    # Leaves
    leaf_length_cm: float = 30.0        # Longest blade
    leaf_width_cm: float = 1.2          # Widest blade
    leaf_peak: float = 0.6              # Position along the culm of the longest leaf (0 base .. 1 flag leaf)
    leaf_angle_deg: float = 30.0        # Insertion angle between blade and culm
    leaf_droop: float = 0.5             # Bending of the blade under its weight (0 stiff .. 1 hanging)
    leaf_twist_deg: float = 60.0        # Twist along the blade
    leaf_fold: float = 0.3              # V-fold along the midrib
    margin_wave: float = 0.0            # Wavy margins (maize)
    sheath_fraction: float = 0.7        # Share of the internode wrapped by the sheath
    ligule: Ligule = Ligule.MEMBRANE    # Collar appendage on the inner face of the leaf
    ligule_mm: float = 2.0              # Ligule height
    auricles: Auricles = Auricles.NONE  # Paired lobes at the collar
    auricle_mm: float = 2.0             # Auricle length
    root_primordia: float = 0.0         # Relief of the root primordia rows in the root band (sugarcane)
    leaf_loss: float = 0.0              # Share of the nodes, from the base, whose leaves have died and fallen
    leaf_color: tuple = (0.20, 0.42, 0.12)
    midrib_color: tuple = (0.55, 0.65, 0.40)
    tip_dryness: float = 0.15           # Dry, straw-coloured tips

    # Inflorescence
    head: GrassHead = GrassHead.SPIKE
    head_length_cm: float = 9.0
    head_width_cm: float = 1.5
    peduncle_cm: float = 8.0            # Bare culm above the flag leaf
    spikelets: int = 20
    spikelet_mm: float = 10.0
    awn_cm: float = 0.0                 # Awn length (barley ~12, durum ~8, bread wheat 0-5)
    branches: int = 6                   # Panicle whorls / one-sided spikes / tassel branches
    branch_angle_deg: float = 40.0
    nod: float = 0.1                    # Head nodding (oats, rice, ripe barley)
    head_color: tuple = (0.45, 0.58, 0.25)
    awn_color: tuple = (0.70, 0.68, 0.45)

    # Maize ears
    ears: int = 1
    ear_node: float = 0.55              # Position of the (upper) ear along the culm
    ear_length_cm: float = 18.0
    ear_diameter_cm: float = 4.5
    kernel_rows: int = 16
    husk: float = 0.0                   # 0 closed in husk leaves .. 1 husk peeled back
    silk_cm: float = 10.0
    kernel_color: tuple = (0.95, 0.75, 0.15)
    silk_color: tuple = (0.75, 0.55, 0.30)

    # Roots and ripeness
    crown_roots: int = 12
    root_length_cm: float = 25.0
    brace_roots: int = 0                # Prop roots from the lowest nodes (maize)
    fine_roots: float = 1.0             # Density of the lateral roots on the crown roots
    ripeness: float = 0.0               # 0 green .. 1 golden, ripe


@dataclass
class GrassResult:
    leaves: MeshData
    culms: MeshData
    heads: MeshData
    ears: MeshData
    roots: MeshData
    stats: dict = field(default_factory=dict)


def _unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def _rotate(v, axis, ang):
    axis = _unit(np.asarray(axis, float))
    c, s = math.cos(ang), math.sin(ang)
    return v * c + np.cross(axis, v) * s + axis * (axis @ v) * (1 - c)


def _perp(d):
    p = np.cross(d, UP)
    if np.linalg.norm(p) < 1e-6:
        p = np.cross(d, [1.0, 0.0, 0.0])
    return _unit(p)


def _attrs(n, part, u=None, v=None, node=None):
    return {"grass_u": np.zeros(n) if u is None else u, "grass_v": np.zeros(n) if v is None else v,
            "grass_part": np.full(n, float(part)), "grass_node": np.full(n, FAR) if node is None else node,
            "grass_nid": np.zeros(n)}


def _merge(parts) -> MeshData:
    parts = [p for p in parts if len(p.vertices)]
    for p in parts:
        for k in ATTRS:
            default = np.full(len(p.vertices), FAR if k == "grass_node" else 0.0)
            p.point_attributes[k] = np.asarray(p.point_attributes.get(k, default), np.float32)
        for k in list(p.point_attributes):
            if k not in ATTRS:
                del p.point_attributes[k]
    return MeshData.concatenate(parts)


def _ellipsoid(length, width, n=6, m=6, part=2) -> MeshData:
    """Spikelet / kernel primitive along +Z from the origin."""
    t = 0.5 - 0.5 * np.cos(np.pi * np.linspace(0, 1, n + 2)[1:-1])
    return revolve(0.5 * width * np.sin(np.pi * t) ** 0.7, length * t, m,
                   {"grass_u": t, "grass_v": np.zeros(len(t)), "grass_part": np.full(len(t), float(part)),
                    "grass_node": np.full(len(t), FAR)})


def _place(m: MeshData, z_axis, origin, roll=0.0) -> MeshData:
    z = _unit(np.asarray(z_axis, float))
    e1 = _perp(z)
    e2 = np.cross(z, e1)
    c, s = math.cos(roll), math.sin(roll)
    e1, e2 = c * e1 + s * e2, -s * e1 + c * e2
    R = np.stack([e1, e2, z], axis=1)
    V = m.vertices.astype(float) @ R.T + np.asarray(origin)[None, :]
    return MeshData(V.astype(np.float32), m.loop_vertex.copy(), m.loop_start.copy(), m.loop_total.copy(),
                    m.loop_uv.copy(), {k: v.copy() for k, v in m.point_attributes.items()})


def _strip(P, W, side, normal, fold, wave=0.0, part=0, across=5) -> MeshData:
    """Ribbon along the curve P (n,3) with widths W, width directions `side` and normals `normal` (n,3);
    V-fold along the middle line and optional wavy margins."""
    n = len(P)
    v = np.linspace(-1.0, 1.0, across)
    s = arc_length(P)
    u = s / max(s[-1], 1e-9)
    off = (v[None, :, None] * 0.5 * W[:, None, None]) * side[:, None, :]
    lift = (fold * 0.35 * np.abs(v)[None, :] * W[:, None])[..., None] * normal[:, None, :]
    if wave > 0:
        # Ruffled margins (maize): gentle waves, only near the edges, growing from the base
        wav = wave * 0.07 * W[:, None] * np.sin(s[:, None] / 0.06 * 2 * math.pi) * (np.abs(v)[None, :] ** 4) \
            * np.clip(u[:, None] * 3, 0, 1)
        lift = lift + wav[..., None] * normal[:, None, :]
    V = (P[:, None, :] + off + lift).reshape(-1, 3)
    j, i = np.meshgrid(np.arange(n - 1), np.arange(across - 1), indexing="ij")
    i0 = (j * across + i).ravel()
    quads = np.stack([i0, i0 + 1, i0 + across + 1, i0 + across], axis=1)
    lv = quads.reshape(-1).astype(np.int32)
    lt = np.full(len(quads), 4, np.int32)
    ls = np.arange(len(quads), dtype=np.int32) * 4
    uu = np.repeat(u[:, None], across, 1).reshape(-1)
    vv = np.repeat(v[None, :], n, 0).reshape(-1)
    return MeshData(V.astype(np.float32), lv, ls, lt, np.stack([vv * 0.5 + 0.5, uu], 1)[lv].astype(np.float32),
                    {k: np.asarray(a, np.float32) for k, a in _attrs(len(V), part, uu, vv).items()})


class GrassEngine:
    def __init__(self, profile: GrassProfile = None):
        self.p = profile or GrassProfile()

    # ----------------------------------------------------------------- leaves
    def blade(self, start, d0, length, width, rng, detail=1.0, droop=None, part=0, twist=None) -> MeshData:
        """Free leaf blade (husk leaves): midrib bending toward the ground, width bell, pointed tip."""
        P, W, side, nrm = self._blade_rows(start, d0, length, width, rng, detail, droop, twist)
        return _strip(P, W, side, nrm, self.p.leaf_fold, self.p.margin_wave, part,
                      across=5 if detail >= 0.6 else 3)

    def _blade_rows(self, start, d0, length, width, rng, detail=1.0, droop=None, twist=None, base_width=None,
                    side0=None):
        """Midrib points, widths, width directions and normals of a blade. With `base_width` the blade starts
        as wide as the opening sheath (it clasps the culm) and widens to `width`."""
        p = self.p
        droop = p.leaf_droop if droop is None else droop
        twist = math.radians(p.leaf_twist_deg if twist is None else twist) * rng.uniform(0.6, 1.4)
        n = max(8, int(min(60, length / 0.012) * detail))
        step = length / n
        d = _unit(np.asarray(d0, float))
        total = math.radians(160.0) * droop * rng.uniform(0.8, 1.2)
        # The midrib bends in the vertical plane of d0 (rotation about cross(d, -UP) keeps it there), so the
        # sequential bending reduces to an angle from the vertical accumulated along the blade
        th0 = math.acos(float(np.clip(d @ UP, -1.0, 1.0)))
        horiz = d - (d @ UP) * UP
        h = _unit(horiz) if np.linalg.norm(horiz) > 1e-6 else _perp(d)
        f = (np.arange(n) + 0.5) / n
        inc = total * 2 * f / n
        th_max = math.pi - math.acos(0.97)
        th = np.empty(n + 1)
        th[0] = th0
        acc = th0
        for j in range(n):                                # Bending stops once the blade hangs down
            if acc < th_max:
                acc = acc + inc[j]
            th[j + 1] = acc
        D = np.cos(th)[:, None] * UP + np.sin(th)[:, None] * h
        if th0 < 1e-6:
            D[0] = d
        P = np.asarray(start, float) + np.concatenate([[np.zeros(3)], np.cumsum(D[1:] * step, axis=0)])
        u = np.linspace(0, 1, len(P))
        # Width: from the ligule (or the sheath opening) to the widest point about 40 % up, then to a point
        w0 = 0.45 * width if base_width is None else base_width
        g = np.sin(0.5 * np.pi * np.clip(u / 0.4, 0, 1))
        W = (w0 * (1 - g) + width * g) * (1 - u ** 2.5) ** 0.7
        W[-1] = width * 0.02
        # Width direction: given (continuing the sheath that clasps the culm) or horizontal; twisted along
        side0 = _perp(D[0]) if side0 is None else _unit(side0 - (side0 @ D[0]) * D[0])
        S = side0[None, :] - (D @ side0)[:, None] * D
        bad = np.linalg.norm(S, axis=1) < 1e-6
        S = S / np.maximum(np.linalg.norm(S, axis=1, keepdims=True), 1e-12)
        if bad.any():
            S[bad] = np.array([_perp(x) for x in D[bad]])
        a = (twist * u)[:, None]
        S = S * np.cos(a) + np.cross(D, S) * np.sin(a)        # Rodrigues (S is perpendicular to D)
        N = np.cross(S, D)
        N = N / np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-12)
        return P, W, S, N

    def leaf(self, at, radius, z0, z1, radial, L, Wd, ang, rng, detail) -> MeshData:
        """Sheath and blade as ONE surface. The sheath is the rolled leaf base: it wraps the culm from the node
        (margins overlapping, a slight pulvinus at its base) and opens progressively toward the collar, where
        it continues without a seam into the blade, which clasps the culm and then widens."""
        p = self.p
        A = 15 if detail >= 0.6 else 9                       # Enough to round the sheath around the culm
        v = np.linspace(-1.0, 1.0, A)
        rows, us, parts, vs = [], [], [], []
        ns = max(3, int(8 * detail))
        span0, span1 = 2 * math.pi * 1.08, math.pi * 1.05
        last = None
        if z1 - z0 > 0.002:
            for q in np.linspace(0.0, 1.0, ns):
                z = z0 + (z1 - z0) * q
                c, t = at(z)
                e1 = radial - (radial @ t) * t
                e1 = _unit(e1) if np.linalg.norm(e1) > 1e-6 else _perp(t)
                e2 = np.cross(t, e1)
                # Base attached at the node (flush with the swollen node), then a slight pulvinus and a
                # close-fitting sheath
                rz = radius(z)
                h = (z - z0) / max(rz, 1e-5)
                r = rz * (1.0 + (p.node_swell + 0.01) * math.exp(-(h / 0.6) ** 2)
                          + 0.05 * (1 - math.exp(-(h / 0.8) ** 2)) + 0.05 * math.exp(-((h - 1.5) / 1.0) ** 2))
                span = span0 + (span1 - span0) * q ** 0.8
                phi = v * span / 2
                # Overlapping margins: the outer margin lies a little farther out
                rr = r * (1.0 + 0.04 * np.clip(v, 0, 1))
                row = c + rr[:, None] * (np.cos(phi)[:, None] * e1 + np.sin(phi)[:, None] * e2)
                rows.append(row)
                us.append(0.0)
                parts.append(1.0)
                vs.append(v)
                last = (row, c, e1, r, span)
        c1, t1 = at(z1)
        e1 = radial - (radial @ t1) * t1
        e1 = _unit(e1) if np.linalg.norm(e1) > 1e-6 else _perp(t1)
        r1 = radius(z1) * 1.05
        chord = 2 * r1 * math.sin(min(span1 / 2, math.pi / 2)) if last is not None else None
        d0 = _unit(t1 * math.cos(ang) + e1 * math.sin(ang))
        e2_1 = np.cross(t1, e1)                              # Same order across as the sheath's arc
        P, W, side, nrm = self._blade_rows(c1 + e1 * r1, d0, L, Wd, rng, detail, base_width=chord, side0=e2_1)
        across = (v[None, :, None] * 0.5 * W[:, None, None]) * side[:, None, :]
        fold = (p.leaf_fold * 0.35 * np.abs(v)[None, :] * W[:, None])[..., None] * nrm[:, None, :]
        s = arc_length(P)
        u = s / max(s[-1], 1e-9)
        if p.margin_wave > 0:
            wav = p.margin_wave * 0.07 * W[:, None] * np.sin(s[:, None] / 0.06 * 2 * math.pi) * \
                (np.abs(v)[None, :] ** 4) * np.clip(u[:, None] * 3, 0, 1)
            fold = fold + wav[..., None] * nrm[:, None, :]
        flat = P[:, None, :] + across + fold
        if last is not None:
            # Collar: the rolled sheath section unrolls into the flat blade over the first rows
            arc_row = last[0]
            nt = min(4, len(P) - 1)
            for j in range(1, nt + 1):
                w = (j / nt) ** 1.5
                flat[j] = (1 - w) * (arc_row + (P[j] - P[0])) + w * flat[j]
            flat = flat[1:]
            u = u[1:]
        for j in range(len(flat)):
            rows.append(flat[j])
            us.append(u[j])
            parts.append(0.0)
            vs.append(v)
        V = np.array(rows)
        n = len(V)
        j, i = np.meshgrid(np.arange(n - 1), np.arange(A - 1), indexing="ij")
        i0 = (j * A + i).ravel()
        quads = np.stack([i0, i0 + 1, i0 + A + 1, i0 + A], axis=1)
        lv = quads.reshape(-1).astype(np.int32)
        uu = np.repeat(np.array(us)[:, None], A, 1).reshape(-1)
        vv = np.array(vs).reshape(-1)
        pp = np.repeat(np.array(parts)[:, None], A, 1).reshape(-1)
        return MeshData(V.reshape(-1, 3).astype(np.float32), lv, np.arange(len(quads), dtype=np.int32) * 4,
                        np.full(len(quads), 4, np.int32), np.stack([vv * 0.5 + 0.5, uu], 1)[lv].astype(np.float32),
                        {"grass_u": uu.astype(np.float32), "grass_v": vv.astype(np.float32),
                         "grass_part": pp.astype(np.float32), "grass_node": np.full(len(uu), FAR, np.float32)})

    def collar(self, at, radius, z1, radial, rng, detail) -> MeshData:
        """Ligule and auricles where the sheath meets the blade. The ligule is a membrane (or a fringe of
        hairs) on the inner face of the leaf, pressed against the culm above the collar; the auricles are two
        lobes at the sheath margins wrapping around the culm (short and blunt, claw-like or clasping)."""
        p = self.p
        parts = []
        c1, t1 = at(z1)
        e1 = radial - (radial @ t1) * t1
        e1 = _unit(e1) if np.linalg.norm(e1) > 1e-6 else _perp(t1)
        e2 = np.cross(t1, e1)
        r1 = radius(z1)
        half = math.pi * 1.05 / 2                                  # Half-span of the collar (sheath opening)
        if p.ligule == Ligule.MEMBRANE and p.ligule_mm > 0:
            h = p.ligule_mm * 0.001
            na, nz = max(6, int(14 * detail)), 3
            phi = np.linspace(-half * 0.92, half * 0.92, na)
            q = np.linspace(0.0, 1.0, nz)
            # Upper margin rounded at the ends, slightly toothed
            top = h * (1 - (phi / half) ** 6) * (1 + 0.08 * np.sin(phi * 23.0))
            rr = r1 * (1.0 + 0.012)
            V = np.array([[c1 + t1 * (top[a] * qq) + rr * (math.cos(phi[a]) * e1 + math.sin(phi[a]) * e2)
                           for a in range(na)] for qq in q])
            parts.append(_grid_rows(V, 8, q, phi / half))
        elif p.ligule == Ligule.HAIRS and p.ligule_mm > 0:
            for a in np.linspace(-half * 0.9, half * 0.9, max(5, int(16 * detail))):
                b = c1 + r1 * 1.01 * (math.cos(a) * e1 + math.sin(a) * e2)
                parts.append(_hair(b, b + _unit(t1 + rng.normal(0, 0.15, 3)) * p.ligule_mm * 0.001 * rng.uniform(0.6, 1.1),
                                   0.0003))
                parts[-1].point_attributes["grass_part"] = np.full(4, 8.0, np.float32)
        if p.auricles != Auricles.NONE and p.auricle_mm > 0:
            L = p.auricle_mm * 0.001
            wrap = {Auricles.BLUNT: 0.8, Auricles.CLAW: 1.0, Auricles.CLASPING: 1.4}[p.auricles] * L / max(r1, 1e-4)
            wid = {Auricles.BLUNT: 0.7, Auricles.CLAW: 0.3, Auricles.CLASPING: 0.4}[p.auricles] * L
            for sgn in (-1.0, 1.0):
                na = max(5, int(10 * detail))
                u = np.linspace(0.0, 1.0, na)
                phi = sgn * (half + wrap * u)
                prof = (1 - u) ** (0.6 if p.auricles == Auricles.BLUNT else 1.2)
                rr = r1 * (1.04 + 0.02 * u)
                q = np.linspace(-0.5, 0.5, 3)
                V = np.array([[c1 + t1 * (wid * prof[a] * qq + wid * 0.35 + (0.4 * L * u[a] ** 2 if
                                                                               p.auricles == Auricles.CLAW else 0.0))
                               + rr[a] * (math.cos(phi[a]) * e1 + math.sin(phi[a]) * e2) for a in range(na)]
                              for qq in q])
                parts.append(_grid_rows(V, 9, q + 0.5, u))
        return _merge(parts) if parts else MeshData.empty()

    def culm_mesh(self, C, cs, z_nodes, lens, radii_fn, bud_dirs, detail) -> MeshData:
        """Culm surface: radius varying along the culm (taper, node swelling, growth ring, barrel-shaped
        internodes) and around it (groove up each internode on the side of its bud). Attribute grass_node:
        signed distance to the nearest node in culm radii (for the node ring, wax band and root band)."""
        p = self.p
        sides = max(8, int(14 * detail))
        prim = False      # Root primordia are drawn by the material (bump), see blender.grasses
        # Rings: regular spacing plus extra rings around every node to resolve the node features
        s_list = list(np.linspace(0.0, cs[-1], max(8, int(cs[-1] / 0.02 * detail) + 2)))
        extra = (-2.0, -1.2, -0.6, -0.25, 0.0, 0.25, 0.6, 0.9, 1.2, 2.0) + \
            ((0.15, 0.3, 0.38, 0.47, 0.55, 0.68, 0.78) if prim else ())
        for zn in z_nodes:
            r = radii_fn(zn)
            s_list += [zn + k * r for k in extra]
        s = np.unique(np.clip(np.array(s_list), 0.0, cs[-1]))
        P = np.stack([np.interp(s, cs, C[:, k]) for k in range(3)], 1)
        T, N, B = frames(P)
        th = np.linspace(0.0, 2 * math.pi, sides + 1)
        zn_arr = np.asarray(z_nodes, float)
        rows, nodes, nids = [], [], []
        seed_c = float(C[0, 0] * 7.31 + C[0, 1] * 3.17)
        for i, z in enumerate(s):
            r = radii_fn(z)
            k = int(np.argmin(np.abs(zn_arr - z)))
            d = (z - zn_arr[k]) / max(r, 1e-5)                       # Signed distance to the nearest node
            hv = (k * 0.6180339 + seed_c) % 1.0                       # Each node differs a little
            f = 1.0 + p.node_swell * (0.75 + 0.5 * hv) * math.exp(-(d / 0.6) ** 2) \
                - p.growth_ring * 0.06 * math.exp(-((d - 0.9) / 0.25) ** 2)
            # Internode containing z: barrel shape and bud groove (groove runs up from the bud below)
            kb = int(np.searchsorted(zn_arr, z, side="right")) - 1
            ring = np.full(len(th), f)
            if 0 <= kb < len(lens) and lens[kb] > 1e-4:
                q = np.clip((z - zn_arr[kb]) / lens[kb], 0, 1)
                ring = ring + p.internode_barrel * math.sin(math.pi * q)
                bd = bud_dirs[kb] - (bud_dirs[kb] @ T[i]) * T[i]
                if np.linalg.norm(bd) > 1e-6 and p.bud_groove > 0:
                    bd = _unit(bd)
                    dirs = np.cos(th)[:, None] * N[i] + np.sin(th)[:, None] * B[i]
                    ang = np.arccos(np.clip(dirs @ bd, -1, 1))
                    ring = ring - p.bud_groove * np.exp(-(ang / 0.45) ** 2) * math.sin(math.pi * q) ** 0.4
            if p.growth_ring > 0:       # Growth ring swollen on one side (side turning with the node)
                ring = ring + p.growth_ring * 0.035 * math.exp(-((d - 0.9) / 0.3) ** 2) * \
                    np.maximum(np.cos(th - hv * 2 * math.pi), 0.0)
            R = r * ring
            rows.append(P[i] + R[:, None] * (np.cos(th)[:, None] * N[i] + np.sin(th)[:, None] * B[i]))
            nodes.append(np.clip(d, -FAR + 1, FAR - 1))
            nids.append(hv)
        V = np.array(rows)
        n, m = V.shape[0], V.shape[1]
        j, i = np.meshgrid(np.arange(n - 1), np.arange(sides), indexing="ij")
        i0 = (j * m + i).ravel()
        quads = np.stack([i0, i0 + 1, i0 + m + 1, i0 + m], axis=1)
        lv = quads.reshape(-1).astype(np.int32)
        uu = np.repeat((s / max(s[-1], 1e-9))[:, None], m, 1).reshape(-1)
        vv = np.repeat((th / (2 * math.pi))[None, :], n, 0).reshape(-1)
        node = np.repeat(np.array(nodes)[:, None], m, 1).reshape(-1)
        nid = np.repeat(np.array(nids)[:, None], m, 1).reshape(-1)
        return MeshData(V.reshape(-1, 3).astype(np.float32), lv, np.arange(len(quads), dtype=np.int32) * 4,
                        np.full(len(quads), 4, np.int32), np.stack([vv, uu * 10], 1)[lv].astype(np.float32),
                        {"grass_u": uu.astype(np.float32), "grass_v": vv.astype(np.float32),
                         "grass_part": np.ones(len(uu), np.float32), "grass_node": node.astype(np.float32),
                         "grass_nid": nid.astype(np.float32)})

    # ----------------------------------------------------------------- shoot
    def shoot(self, base, axis, height, rng, detail, main=True, fertile=True):
        """One culm with its phytomers and head. Returns (parts by group, culm top, culm points)."""
        p = self.p
        leaves, culm_parts, heads, ears = [], [], [], []
        r0 = p.culm_radius_mm * 0.001
        N = max(1, p.nodes)
        # Internode lengths: lower ones short (gradient), summing to the culm height
        w = (np.arange(N) + 1.0) ** p.internode_gradient
        lens = height * w / w.sum() if height > 0 else np.zeros(N)
        z_nodes = np.concatenate([[0.0], np.cumsum(lens)])
        # Leaf azimuths: distichous, with a slow drift
        az0 = rng.uniform(0, 2 * math.pi)
        n_leaves = N + p.basal_leaves
        azs = [az0 + k * math.pi + rng.normal(0, 0.25) for k in range(n_leaves)]
        side_ref = _perp(_unit(np.asarray(axis, float)))
        # Culm path: lean recovering toward the vertical (negative gravitropism), turning slightly at each
        # node away from the leaf (and bud) of that node, alternately: the zigzag of grass culms
        m = max(8, int(24 * detail))
        total = max(height, 1e-4) + (p.peduncle_cm * 0.01 if p.head != GrassHead.NONE and fertile else 0.0)
        s = np.linspace(0, total, m)
        s = np.unique(np.concatenate([s, np.clip(z_nodes, 0, total)]))
        d = _unit(np.asarray(axis, float))
        pts = [np.asarray(base, float)]
        bud_dirs = []
        node_i = 1
        for k in range(1, len(s)):
            d = _unit(d + 0.06 * UP * (1 - d @ UP) * (s[k] - s[k - 1]) / max(total / m, 1e-6)
                      + rng.normal(0, 0.006, 3))
            pts.append(pts[-1] + d * (s[k] - s[k - 1]))
            while node_i < len(z_nodes) - 1 and s[k] >= z_nodes[node_i] - 1e-9:
                rad = _unit(_rotate(side_ref, d, azs[p.basal_leaves + node_i]))
                axis_z = np.cross(d, rad)
                if np.linalg.norm(axis_z) > 1e-6:
                    d = _rotate(d, axis_z, math.radians(p.zigzag_deg) * rng.uniform(0.6, 1.2))
                node_i += 1
        C = np.array(pts)
        cs = arc_length(C)
        T, _, _ = frames(C)

        def at(z):
            z = min(max(z, 0.0), cs[-1])
            pos = np.array([np.interp(z, cs, C[:, k]) for k in range(3)])
            i = min(np.searchsorted(cs, z), len(cs) - 1)
            return pos, T[i]

        def radius(z):
            return r0 * (1.0 - 0.45 * min(max(z, 0.0), cs[-1]) / max(cs[-1], 1e-9))
        for k in range(N):
            _, tan = at(z_nodes[k])
            bud_dirs.append(_unit(_rotate(side_ref, tan, azs[p.basal_leaves + k])))
        if cs[-1] > 0.01:
            culm_parts.append(self.culm_mesh(C, cs, z_nodes[:N], lens, radius, bud_dirs, detail))
        # Leaves: sheath + blade in one surface, one per node; basal leaves crowded at the base
        bud = _ellipsoid(max(p.bud_size_mm, 0.1) * 0.001, max(p.bud_size_mm, 0.1) * 0.0007, 4, 6, part=1)
        for k in range(n_leaves):
            basal = k < p.basal_leaves
            if basal:
                zk, inter = 0.01 * k, 0.03
                f = 0.5
            else:
                kk = k - p.basal_leaves
                zk, inter = z_nodes[kk], lens[kk]
                f = (kk + 0.5) / N
            node, tan = at(zk)
            # Bell of blade length / width along the culm; flag leaf shorter
            bell = math.exp(-((f - p.leaf_peak) / 0.38) ** 2)
            L = p.leaf_length_cm * 0.01 * (0.35 + 0.65 * bell) * rng.uniform(0.85, 1.1) * (1.0 if main else 0.85)
            Wd = p.leaf_width_cm * 0.01 * (0.55 + 0.45 * bell)
            radial = _unit(_rotate(side_ref, tan, azs[k]))
            sheath_len = inter * min(p.sheath_fraction, 0.97) if not basal else 0.03 + 0.04 * rng.random()
            ang = math.radians(p.leaf_angle_deg + (25 if basal else 0)) * rng.uniform(0.8, 1.2)
            if basal or (k - p.basal_leaves) / N >= p.leaf_loss:    # Lower leaves may have fallen
                leaves.append(self.leaf(at, radius, zk, zk + sheath_len, radial, L, Wd, ang, rng, detail))
                if detail >= 0.5 and sheath_len > 0.002:
                    culm_parts.append(self.collar(at, radius, zk + sheath_len, radial, rng, detail))
            if not basal and p.bud_size_mm > 0 and cs[-1] > 0.01:      # Axillary bud ("eye") in the leaf axil
                rz = radius(zk) * (1 + p.node_swell)
                # The bud sits on the node, half sunk in the culm, pointing up along it (pressed to the culm)
                bpart = _place(bud, _unit(tan * 0.95 + radial * 0.12), node + radial * rz * 0.97 - tan * rz * 0.15,
                               rng.uniform(0, 6.28))
                bpart.point_attributes["grass_node"] = np.zeros(len(bpart.vertices), np.float32)
                culm_parts.append(bpart)
        top, top_tan = at(cs[-1])
        # Inflorescence
        if p.head != GrassHead.NONE and fertile:
            if p.head == GrassHead.MAIZE:
                heads.append(self.tassel(top, top_tan, rng, detail))
                for e in range(p.ears if main else 0):
                    # The ear (a reduced branch) grows from a node, in the axil of that node's leaf
                    ke = int(np.argmin(np.abs(z_nodes[:N] - height * max(0.1, p.ear_node - 0.12 * e))))
                    node, tan = at(z_nodes[ke])
                    radial = _unit(_rotate(side_ref, tan, azs[p.basal_leaves + ke]))
                    ears.append(self.ear(node, tan, radial, r0, rng, detail))
            else:
                heads.append(self.inflorescence(top, top_tan, rng, detail))
        return leaves, culm_parts, heads, ears, C

    # ----------------------------------------------------------------- heads
    def _rachis(self, start, d0, length, nod, n=12):
        """Rachis curve: continues the culm and bends down by `nod` (weight of the grains)."""
        d = _unit(np.asarray(d0, float))
        pts = [np.asarray(start, float)]
        total = math.radians(150.0) * nod
        for j in range(n):
            axis = np.cross(d, -UP)
            if np.linalg.norm(axis) > 1e-6 and d @ -UP < 0.97:
                d = _rotate(d, axis, total * 2 * (j + 0.5) / n / n)
            pts.append(pts[-1] + d * length / n)
        return np.array(pts)

    def _nod(self):
        p = self.p
        return float(np.clip(p.nod + 0.25 * p.ripeness * (p.head in (GrassHead.SPIKE, GrassHead.PANICLE)), 0, 1))

    def inflorescence(self, top, tan, rng, detail) -> MeshData:
        p = self.p
        parts = []
        Lh = p.head_length_cm * 0.01
        sl = p.spikelet_mm * 0.001
        prim = _ellipsoid(sl, sl * 0.45, 4 if detail < 0.8 else 6, 5 if detail < 0.8 else 6)
        r_rach = max(p.culm_radius_mm * 0.0004, 0.0004)
        if p.head == GrassHead.SPIKE:
            Pr = self._rachis(top, tan, Lh, self._nod())
            parts.append(tube(Pr, np.linspace(r_rach * 1.4, r_rach * 0.6, len(Pr)), 5, _attrs(len(Pr), 2)))
            sr = arc_length(Pr)
            Tr, Nr, _ = frames(Pr)
            plane = _perp(Tr[0])
            for k in range(p.spikelets):                 # Distichous: alternate sides of the rachis
                z = sr[-1] * (k + 0.5) / p.spikelets
                i = min(np.searchsorted(sr, z), len(sr) - 1)
                x = np.array([np.interp(z, sr, Pr[:, q]) for q in range(3)])
                side = plane if k % 2 else -plane
                d = _unit(Tr[i] * math.cos(math.radians(28)) + side * math.sin(math.radians(28)))
                size = 0.75 + 0.25 * math.sin(math.pi * (k + 0.5) / p.spikelets)
                for fl in range(3 if detail >= 0.6 else 2):  # Florets fanned in the spikelet
                    df = _unit(d + side * (fl - 1) * 0.25)
                    parts.append(_place(_scale(prim, size), df, x + side * r_rach, rng.uniform(0, 6.28)))
                    if p.awn_cm > 0:
                        a0 = x + df * sl * size
                        a1 = a0 + _unit(df + Tr[i] * 0.6) * p.awn_cm * 0.01 * rng.uniform(0.85, 1.1)
                        parts.append(tube(np.linspace(a0, a1, 4), np.full(4, 0.00018), 3, _attrs(4, 3)))
        elif p.head in (GrassHead.PANICLE, GrassHead.PLUME):
            plume = p.head == GrassHead.PLUME
            hairs = []
            Pr = self._rachis(top, tan, Lh, self._nod())
            parts.append(tube(Pr, np.linspace(r_rach * 1.5, r_rach * 0.5, len(Pr)), 5, _attrs(len(Pr), 2)))
            sr = arc_length(Pr)
            Tr, _, _ = frames(Pr)
            whorls = max(1, p.branches)
            per_branch = max(2, p.spikelets // (whorls * 3))
            for k in range(whorls):
                z = sr[-1] * (0.05 + 0.9 * k / whorls)
                i = min(np.searchsorted(sr, z), len(sr) - 1)
                x = np.array([np.interp(z, sr, Pr[:, q]) for q in range(3)])
                bl = (p.head_width_cm * 0.01 * 0.5) * (1.15 - 0.85 * k / whorls) + 0.005
                for b in range(3 if not plume else 5):
                    az = k * GOLDEN + b * 2 * math.pi / (3 if not plume else 5)
                    radial = _rotate(_perp(Tr[i]), Tr[i], az)
                    ang = math.radians(p.branch_angle_deg) * rng.uniform(0.8, 1.2)
                    d = _unit(Tr[i] * math.cos(ang) + radial * math.sin(ang))
                    Pb = self._rachis(x, d, bl, min(1.0, self._nod() + 0.3), 5)
                    parts.append(tube(Pb, np.full(len(Pb), r_rach * 0.45), 3, _attrs(len(Pb), 2)))
                    sb = arc_length(Pb)
                    for q in range(per_branch):
                        zz = sb[-1] * (0.3 + 0.7 * (q + 0.5) / per_branch)
                        y = np.array([np.interp(zz, sb, Pb[:, c]) for c in range(3)])
                        dd = _unit(Pb[-1] - Pb[0] + rng.normal(0, 0.3, 3) * np.linalg.norm(Pb[-1] - Pb[0]))
                        if plume:                        # Silky hairs instead of plump spikelets
                            h1 = y + _unit(dd + rng.normal(0, 0.4, 3)) * sl * rng.uniform(1.5, 3.0)
                            hairs.append((y, h1))
                        else:
                            parts.append(_place(prim, dd, y, rng.uniform(0, 6.28)))
                            if p.awn_cm > 0:
                                a0 = y + dd * sl
                                parts.append(tube(np.linspace(a0, a0 + _unit(dd - 0.3 * UP) * p.awn_cm * 0.01, 3),
                                                  np.full(3, 0.00015), 3, _attrs(3, 3)))
            if hairs:
                parts.append(_hairs(np.array([a for a, _ in hairs]), np.array([b for _, b in hairs]), 0.0006))
        elif p.head == GrassHead.ONE_SIDED:
            # Bouteloua: 1-3 spikes leaning to one side, spikelets crowded in two rows on the lower side
            for b in range(max(1, p.branches)):
                x = top - tan * (0.02 * b)
                az = rng.uniform(-0.4, 0.4)
                side = _rotate(_perp(tan), tan, az)
                d = _unit(tan * math.cos(math.radians(p.branch_angle_deg)) + side * math.sin(math.radians(p.branch_angle_deg)))
                Pb = self._rachis(x, d, Lh, 0.25 + self._nod(), 8)
                parts.append(tube(Pb, np.full(len(Pb), r_rach), 4, _attrs(len(Pb), 2)))
                sb = arc_length(Pb)
                Tb, _, _ = frames(Pb)
                down = _unit(np.cross(_perp(Tb[len(Tb) // 2]), Tb[len(Tb) // 2]))
                if down @ UP > 0:
                    down = -down
                for k in range(p.spikelets):
                    z = sb[-1] * (k + 0.5) / p.spikelets
                    i = min(np.searchsorted(sb, z), len(sb) - 1)
                    y = np.array([np.interp(z, sb, Pb[:, c]) for c in range(3)])
                    lat = _perp(Tb[i]) * (0.35 if k % 2 else -0.35)
                    dd = _unit(down + lat + Tb[i] * 0.3)
                    parts.append(_place(prim, dd, y, rng.uniform(0, 6.28)))
                    if p.awn_cm > 0:
                        a0 = y + dd * sl
                        parts.append(tube(np.linspace(a0, a0 + dd * p.awn_cm * 0.01, 3), np.full(3, 0.00012), 3,
                                          _attrs(3, 3)))
        return _merge(parts)

    def tassel(self, top, tan, rng, detail) -> MeshData:
        """Maize tassel: central spike and long spreading branches with paired spikelets."""
        p = self.p
        parts = []
        Lh = p.head_length_cm * 0.01
        sl = p.spikelet_mm * 0.001
        prim = _ellipsoid(sl, sl * 0.4, 4, 5)
        r = max(p.culm_radius_mm * 0.0003, 0.0008)
        Pc = self._rachis(top, tan, Lh, 0.1 + self._nod())
        parts.append(tube(Pc, np.linspace(r * 1.5, r * 0.6, len(Pc)), 5, _attrs(len(Pc), 2)))
        axes = [Pc]
        sc = arc_length(Pc)
        Tc, _, _ = frames(Pc)
        for b in range(max(0, p.branches)):
            z = sc[-1] * 0.05 + sc[-1] * 0.35 * b / max(p.branches, 1)
            i = min(np.searchsorted(sc, z), len(sc) - 1)
            x = np.array([np.interp(z, sc, Pc[:, q]) for q in range(3)])
            radial = _rotate(_perp(Tc[i]), Tc[i], b * GOLDEN)
            ang = math.radians(p.branch_angle_deg) * rng.uniform(0.8, 1.2)
            d = _unit(Tc[i] * math.cos(ang) + radial * math.sin(ang))
            Pb = self._rachis(x, d, Lh * rng.uniform(0.55, 0.85), 0.35 + self._nod(), 8)
            parts.append(tube(Pb, np.full(len(Pb), r * 0.6), 4, _attrs(len(Pb), 2)))
            axes.append(Pb)
        per = max(4, p.spikelets // max(len(axes), 1))
        for A in axes:
            sa = arc_length(A)
            for k in range(per):
                z = sa[-1] * (0.15 + 0.85 * (k + 0.5) / per)
                i = min(np.searchsorted(sa, z), len(sa) - 1)
                y = np.array([np.interp(z, sa, A[:, c]) for c in range(3)])
                t = _unit(A[min(i + 1, len(A) - 1)] - A[max(i - 1, 0)])
                for side in (-1, 1):                     # Paired spikelets hanging from the branch
                    dd = _unit(t * 0.6 + _perp(t) * 0.5 * side - UP * 0.4)
                    parts.append(_place(prim, dd, y, rng.uniform(0, 6.28)))
        return _merge(parts)

    def ear(self, node, tan, radial, r_culm, rng, detail) -> MeshData:
        """Maize ear on a short shank: cob with kernel rows, husk leaves (closed or peeled) and silks."""
        p = self.p
        parts = []
        L = p.ear_length_cm * 0.01
        D = p.ear_diameter_cm * 0.01
        axis = _unit(tan * math.cos(math.radians(28)) + radial * math.sin(math.radians(28)))
        shank0 = node + radial * r_culm
        base = shank0 + axis * 0.04
        parts.append(tube(np.linspace(shank0, base, 4), np.full(4, D * 0.18), 6, _attrs(4, 1)))
        # Cob: kernels in rows (grooves between rows and between kernels)
        n = max(24, int(60 * detail))
        nth = max(24, int(p.kernel_rows * 4 * max(detail, 0.6)))
        t = 0.5 - 0.5 * np.cos(np.pi * np.linspace(0, 1, n + 2)[1:-1])
        rho = 0.5 * D * (1 - t ** 3.0 * 0.55) * np.sin(np.pi * np.minimum(t / 0.08, 1) / 2) ** 0.5
        rho[-1] = rho[-2] * 0.6
        z = L * t
        kl = D * math.pi / max(p.kernel_rows, 4) * 0.9          # Kernel pitch along the row

        def radial_f(th, t=t):
            rows = np.abs(np.cos(p.kernel_rows * th / 2.0))[None, :]
            row_id = np.floor(p.kernel_rows * th / (2 * np.pi) + 0.5)[None, :]
            along = np.abs(np.cos(np.pi * (L * t[:, None]) / kl + 0.5 * np.pi * (row_id % 2)))   # Staggered
            return 1.0 - 0.07 * (1 - rows) ** 4 - 0.05 * (1 - along) ** 4
        cob = revolve(rho, z, nth, {"grass_u": t, "grass_v": np.zeros(len(t)), "grass_part": np.full(len(t), 4.0)},
                      radial_f)
        parts.append(_place(cob, axis, base))
        # Husk leaves: wrap the ear and extend past its tip; peeled back with `husk`
        k = 7
        for i in range(k):
            az = 2 * math.pi * i / k + rng.normal(0, 0.1)
            rad = _rotate(_perp(axis), axis, az)
            peel = p.husk * math.radians(150) * rng.uniform(0.8, 1.1)
            d0 = _unit(axis * math.cos(peel) + rad * math.sin(peel))
            start = base + rad * D * 0.45
            if p.husk < 0.3:                                    # Closed: hugging the cob, converging past the tip
                ln = L * 1.25
                pts = []
                for q in np.linspace(0, 1, 10):
                    rr = 0.5 * D * 1.1 * (1 - q ** 2.5 * 0.9) * min(1.0, 0.85 + q * 6)   # Covers the cob base
                    pts.append(base + axis * ln * q + rad * rr)
                P = np.array(pts)
                Wd = np.full(len(P), math.pi * D / k * 1.5) * (1 - np.linspace(0, 1, len(P)) ** 2 * 0.8)
                side = np.array([_unit(np.cross(axis, rad))] * len(P))
                nrm = np.array([rad] * len(P))
                parts.append(_strip(P, Wd, side, nrm, 0.0, 0.0, 5, 3))
            else:
                parts.append(self.blade(start, d0, L * 1.2, math.pi * D / k * 1.4, rng, detail, droop=0.4 * p.husk,
                                        part=5, twist=20))
        # Silks from the tip (out of the husk when closed)
        tip = base + axis * L * (1.25 if p.husk < 0.3 else 1.0)
        if p.silk_cm > 0:
            for i in range(max(6, int(30 * detail))):
                d = _unit(axis + rng.normal(0, 0.45, 3))
                pts = [tip]
                for _ in range(5):
                    d = _unit(d - 0.25 * UP + rng.normal(0, 0.15, 3))
                    pts.append(pts[-1] + d * p.silk_cm * 0.01 / 5)
                parts.append(tube(np.array(pts), np.full(6, 0.00025), 3, _attrs(6, 6)))
        return _merge(parts)

    # ----------------------------------------------------------------- roots
    def roots(self, rng, culm_paths, detail) -> MeshData:
        p = self.p
        parts = []
        from .root_architecture import RootBranching, grow_root, lateral_roots
        # Fibrous (fasciculate) system: crown (nodal) roots of similar girth leaving the base at steep
        # angles, the earliest (seminal) ones steepest; lateral roots densely spaced (CRootBox cereal
        # parameters: inter-branch distance ~1 cm, laterals much thinner and shorter). Every tiller roots
        # from its own basal nodes (Evers et al. 2005), the main shoot most.
        crown = []
        bases = [np.asarray(C[0], float) for C in culm_paths] or [np.zeros(3)]
        nt = len(bases)
        n_roots = max(p.crown_roots, 2 * nt) if p.crown_roots > 0 else 0
        for k in range(n_roots):
            a = k * GOLDEN
            b = bases[0] if k < max(3, p.crown_roots // max(nt, 1)) else bases[k % nt]
            dip = math.radians(rng.uniform(35, 80) if k >= 3 else rng.uniform(65, 85))
            d = np.array([math.cos(a) * math.cos(dip), math.sin(a) * math.cos(dip), -math.sin(dip)])
            ln = p.root_length_cm * 0.01 * rng.uniform(0.6, 1.1)
            r0 = p.culm_radius_mm * 0.0004 + 0.0003
            start = np.array([b[0], b[1], 0.005])
            P, R = grow_root(start, d, ln, r0, rng, gravitropism=0.2, tortuosity=0.35, surface=0.004, taper=0.3)
            P[0] = start
            crown.append((P, R))
            parts.append(tube(P, R, 4, _attrs(len(P), 7)))
        if p.fine_roots > 0 and detail >= 0.4:
            br = RootBranching(orders=2 if detail >= 0.8 else 1, interbranch_cm=1.5 / p.fine_roots ** 0.5,
                               basal_zone=0.1, apical_zone_cm=3.0, insertion_deg=70.0, length_ratio=0.25,
                               max_length_cm=12.0, radius_ratio=0.5, gravitropism=0.06, tortuosity=0.5,
                               min_radius_mm=0.08, budget=int(500 * p.fine_roots * detail))
            for P, R, *_ in lateral_roots(crown, br, rng, surface=0.004):
                parts.append(tube(P, np.maximum(R, 6e-5), 3, _attrs(len(P), 7)))
        # Brace roots from the two lowest above-ground nodes of the main culm, arching into the soil
        if p.brace_roots > 0 and culm_paths:
            C = culm_paths[0]
            cs = arc_length(C)
            N = max(1, p.nodes)
            w = (np.arange(N) + 1.0) ** p.internode_gradient
            zn = np.cumsum(p.culm_height_m * w / w.sum())
            for k in range(p.brace_roots):
                zz = zn[k % 2] * 0.5 if k % 2 else zn[0] * 0.35
                x = np.array([np.interp(zz, cs, C[:, q]) for q in range(3)])
                a = k * GOLDEN
                out = np.array([math.cos(a), math.sin(a), 0.0])
                start = x + out * p.culm_radius_mm * 0.001
                spread = x[2] * 0.8 + 0.05
                end = np.array([x[0], x[1], 0.0]) + out * spread - UP * 0.04
                mid = start + out * spread * 0.55 + UP * 0.01
                s = np.linspace(0, 1, 10)[:, None]
                P = (1 - s) ** 2 * start + 2 * (1 - s) * s * mid + s ** 2 * end
                parts.append(tube(P, np.linspace(p.culm_radius_mm * 0.00035, 0.0012, 10), 6, _attrs(10, 7)))
        return _merge(parts)

    # ----------------------------------------------------------------- plant
    def generate(self, seed: int = 0, detail: float = 1.0, with_roots: bool = True,
                 offset=(0.0, 0.0, 0.0)) -> GrassResult:
        p = self.p
        rng = np.random.default_rng(seed)
        leaves, culms, heads, ears, paths = [], [], [], [], []
        n = max(1, p.tillers)
        R = p.clump_radius_cm * 0.01
        for k in range(n):
            f = math.sqrt(k / max(n - 1, 1)) if n > 1 else 0.0
            a = k * GOLDEN + rng.uniform(0, 0.3)
            base = np.array([R * f * math.cos(a), R * f * math.sin(a), 0.0])
            lean = math.radians(p.tiller_spread_deg) * f * rng.uniform(0.6, 1.1)
            axis = np.array([math.cos(a) * math.sin(lean), math.sin(a) * math.sin(lean), math.cos(lean)])
            H = p.culm_height_m * (1.0 - p.tiller_variation * f * rng.uniform(0.5, 1.0))
            # Flowering tillers are spread through the clump (golden-angle order); the others stay leafy
            fertile = k == 0 or ((k * 0.618034) % 1.0) < p.flowering_tillers
            if not fertile:
                H = min(H, 0.02)
            lv, cp, hd, er, C = self.shoot(base, axis, H, np.random.default_rng(seed * 101 + k), detail, main=k == 0,
                                           fertile=fertile)
            if k > 0:
                # A tiller grows from an axillary bud of the crown (the lowest, underground nodes of the
                # mother shoot): a short underground stem joins its base to the crown
                r_t = p.culm_radius_mm * 0.001 * 0.75
                crown_pt = np.array([0.0, 0.0, -0.008])
                low = base - UP * 0.012
                seg = np.array([crown_pt + (low - crown_pt) * q for q in np.linspace(0, 1, 5)] +
                               [low + (base + UP * 0.004 - low) * q for q in np.linspace(0.25, 1, 4)])
                cp = cp + [tube(seg, np.full(len(seg), r_t), 6, _attrs(len(seg), 1))]
            leaves += lv
            culms += cp
            heads += hd
            ears += er
            paths.append(C)
        roots = self.roots(np.random.default_rng(seed + 9), paths, detail) if with_roots else MeshData.empty()
        res = GrassResult(_merge(leaves), _merge(culms), _merge(heads), _merge(ears), roots,
                          {"tillers": n, "leaves": len(leaves)})
        off = np.asarray(offset, np.float32)
        if np.any(off):
            for m in (res.leaves, res.culms, res.heads, res.ears, res.roots):
                if len(m.vertices):
                    m.vertices = m.vertices + off
        return res


def _grid_rows(V, part, u, v) -> MeshData:
    """Open quad grid from rows V (n, m, 3) with attributes u (rows) and v (columns)."""
    V = np.asarray(V, float)
    n, m = V.shape[:2]
    j, i = np.meshgrid(np.arange(n - 1), np.arange(m - 1), indexing="ij")
    i0 = (j * m + i).ravel()
    quads = np.stack([i0, i0 + 1, i0 + m + 1, i0 + m], axis=1)
    lv = quads.reshape(-1).astype(np.int32)
    uu = np.repeat(np.asarray(u, float)[:, None], m, 1).reshape(-1)
    vv = np.repeat(np.asarray(v, float)[None, :], n, 0).reshape(-1)
    return MeshData(V.reshape(-1, 3).astype(np.float32), lv, np.arange(len(quads), dtype=np.int32) * 4,
                    np.full(len(quads), 4, np.int32), np.stack([vv, uu], 1)[lv].astype(np.float32),
                    {k: np.asarray(a, np.float32) for k, a in _attrs(len(uu), part, uu, vv).items()})


def _hairs(A, B, width) -> MeshData:
    """Many thin flat hairs (one quad each) in one mesh."""
    d = B - A
    d = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
    w = np.cross(d, UP)
    small = np.linalg.norm(w, axis=1) < 1e-6
    w[small] = np.cross(d[small], [1.0, 0.0, 0.0])
    w = w / np.maximum(np.linalg.norm(w, axis=1, keepdims=True), 1e-12) * width * 0.5
    n = len(A)
    V = np.stack([A - w, A + w, B + w * 0.3, B - w * 0.3], 1).reshape(-1, 3)
    lv = np.arange(4 * n, dtype=np.int32)
    uv = np.tile(np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32), (n, 1))
    att = _attrs(4 * n, 3, np.tile([0, 0, 1, 1], n).astype(float), np.tile([-1, 1, 1, -1], n).astype(float))
    return MeshData(V.astype(np.float32), lv, np.arange(n, dtype=np.int32) * 4, np.full(n, 4, np.int32), uv,
                    {k: np.asarray(v, np.float32) for k, v in att.items()})


def _hair(a, b, width) -> MeshData:
    """A thin flat hair (one quad), far cheaper than a tube for the thousands of hairs of a plume."""
    d = _unit(b - a)
    w = _perp(d) * width * 0.5
    V = np.array([a - w, a + w, b + w * 0.3, b - w * 0.3], np.float32)
    lv = np.array([0, 1, 2, 3], np.int32)
    return MeshData(V, lv, np.array([0], np.int32), np.array([4], np.int32),
                    np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32),
                    {"grass_u": np.array([0, 0, 1, 1], np.float32), "grass_v": np.array([-1, 1, 1, -1], np.float32),
                     "grass_part": np.full(4, 3.0, np.float32)})


def _scale(m: MeshData, s: float) -> MeshData:
    return MeshData((m.vertices * s).astype(np.float32), m.loop_vertex, m.loop_start, m.loop_total, m.loop_uv,
                    m.point_attributes)


def surface_points(tris: np.ndarray, density_per_m2: float, seed: int = 0, min_dist: float = 0.0,
                   max_slope_deg: float = 60.0, max_points: int = 200000) -> tuple[np.ndarray, np.ndarray]:
    """Tuft positions on a terrain given as triangles (n, 3, 3): area-weighted random sampling thinned to a
    minimum spacing (Poisson disk, as Blender's Distribute Points on Faces; Deussen et al. 1998). Faces
    steeper than `max_slope_deg` are skipped. Returns points and the unit normals of their faces."""
    rng = np.random.default_rng(seed)
    tris = np.asarray(tris, float).reshape(-1, 3, 3)
    e1, e2 = tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]
    cr = np.cross(e1, e2)
    area = 0.5 * np.linalg.norm(cr, axis=1)
    nrm = cr / np.maximum(2 * area[:, None], 1e-12)
    nrm = np.where(nrm[:, 2:3] < 0, -nrm, nrm)
    ok = (area > 1e-12) & (nrm[:, 2] >= math.cos(math.radians(max_slope_deg)))
    if not ok.any():
        return np.zeros((0, 3)), np.zeros((0, 3))
    total = float(area[ok].sum())
    n = int(min(max_points, total * density_per_m2))
    if n <= 0:
        return np.zeros((0, 3)), np.zeros((0, 3))
    m = n * (3 if min_dist > 0 else 1)
    idx = rng.choice(np.nonzero(ok)[0], size=m, p=area[ok] / total)
    u, v = rng.random(m), rng.random(m)
    flip = u + v > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    P = tris[idx, 0] + u[:, None] * e1[idx] + v[:, None] * e2[idx]
    N = nrm[idx]
    if min_dist <= 0:
        return P[:n], N[:n]
    keep, grid = [], {}
    cell = min_dist
    for k, q in enumerate(P):
        key = (int(q[0] // cell), int(q[1] // cell), int(q[2] // cell))
        near = False
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    for j in grid.get((key[0] + dx, key[1] + dy, key[2] + dz), ()):
                        if ((P[j] - q) ** 2).sum() < min_dist * min_dist:
                            near = True
                            break
        if not near:
            keep.append(k)
            grid.setdefault(key, []).append(k)
            if len(keep) >= n:
                break
    keep = np.array(keep, int)
    return P[keep], N[keep]


def patch_points(size_m: float, density_per_m2: float, seed: int = 0, min_dist: float = 0.0) -> np.ndarray:
    """Tuft positions for a lawn / meadow patch: dart throwing with a minimum spacing (Poisson disk)."""
    rng = np.random.default_rng(seed)
    n = int(size_m * size_m * density_per_m2)
    pts = rng.uniform(-size_m / 2, size_m / 2, (n * 3, 2))
    if min_dist <= 0:
        return np.c_[pts[:n], np.zeros(n)]
    keep = []
    cell = min_dist
    grid = {}
    for q in pts:
        key = (int(q[0] // cell), int(q[1] // cell))
        ok = True
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for r in grid.get((key[0] + dx, key[1] + dy), ()):
                    if (r[0] - q[0]) ** 2 + (r[1] - q[1]) ** 2 < min_dist ** 2:
                        ok = False
                        break
        if ok:
            keep.append(q)
            grid.setdefault(key, []).append(q)
            if len(keep) >= n:
                break
    keep = np.array(keep).reshape(-1, 2)
    return np.c_[keep, np.zeros(len(keep))]
