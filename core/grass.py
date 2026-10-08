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
ATTRS = ("grass_u", "grass_v", "grass_part")
# grass_part: 0 blade, 1 culm / sheath, 2 spikelet, 3 awn / hair, 4 kernel, 5 husk, 6 silk, 7 root


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


def _attrs(n, part, u=None, v=None):
    return {"grass_u": np.zeros(n) if u is None else u, "grass_v": np.zeros(n) if v is None else v,
            "grass_part": np.full(n, float(part))}


def _merge(parts) -> MeshData:
    parts = [p for p in parts if len(p.vertices)]
    for p in parts:
        for k in ATTRS:
            p.point_attributes[k] = np.asarray(p.point_attributes.get(k, np.zeros(len(p.vertices))), np.float32)
        for k in list(p.point_attributes):
            if k not in ATTRS:
                del p.point_attributes[k]
    return MeshData.concatenate(parts)


def _ellipsoid(length, width, n=6, m=6, part=2) -> MeshData:
    """Spikelet / kernel primitive along +Z from the origin."""
    t = 0.5 - 0.5 * np.cos(np.pi * np.linspace(0, 1, n + 2)[1:-1])
    return revolve(0.5 * width * np.sin(np.pi * t) ** 0.7, length * t, m,
                   {"grass_u": t, "grass_v": np.zeros(len(t)), "grass_part": np.full(len(t), float(part))})


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
        """Leaf blade from the ligule: midrib bending toward the ground, width bell, pointed tip."""
        p = self.p
        droop = p.leaf_droop if droop is None else droop
        twist = math.radians(p.leaf_twist_deg if twist is None else twist) * rng.uniform(0.6, 1.4)
        n = max(8, int(min(60, length / 0.012) * detail))
        step = length / n
        d = _unit(np.asarray(d0, float))
        P = [np.asarray(start, float)]
        D = [d]
        total = math.radians(160.0) * droop * rng.uniform(0.8, 1.2)
        for j in range(n):
            f = (j + 0.5) / n
            ang = total * 2 * f / n                       # Bending grows toward the tip (weight)
            axis = np.cross(d, -UP)
            if np.linalg.norm(axis) > 1e-6 and d @ -UP < 0.97:
                d = _rotate(d, axis, ang)
            P.append(P[-1] + d * step)
            D.append(d)
        P, D = np.array(P), np.array(D)
        u = np.linspace(0, 1, len(P))
        # Width: narrow at the ligule, widest about 40 % up, tapering to a point
        W = width * (0.45 + 0.55 * np.sin(0.5 * np.pi * np.clip(u / 0.4, 0, 1))) * (1 - u ** 2.5) ** 0.7
        W[-1] = width * 0.02
        side0 = _perp(D[0])
        side, nrm = [], []
        for j in range(len(P)):
            s = side0 - (side0 @ D[j]) * D[j]
            s = _unit(s) if np.linalg.norm(s) > 1e-6 else _perp(D[j])
            s = _rotate(s, D[j], twist * u[j])
            side.append(s)
            nrm.append(_unit(np.cross(s, D[j])))
        return _strip(P, W, np.array(side), np.array(nrm), p.leaf_fold, p.margin_wave, part,
                      across=5 if detail >= 0.6 else 3)

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
        # Culm path: lean, recovering toward the vertical (negative gravitropism)
        m = max(8, int(24 * detail))
        s = np.linspace(0, max(height, 1e-4) + (p.peduncle_cm * 0.01 if p.head != GrassHead.NONE and fertile
                                                else 0.0), m)
        d = _unit(np.asarray(axis, float))
        pts = [np.asarray(base, float)]
        for k in range(1, m):
            d = _unit(d + 0.06 * UP * (1 - d @ UP) + rng.normal(0, 0.01, 3))
            pts.append(pts[-1] + d * (s[k] - s[k - 1]))
        C = np.array(pts)
        cs = arc_length(C)
        T, _, _ = frames(C)

        def at(z):
            z = min(max(z, 0.0), cs[-1])
            pos = np.array([np.interp(z, cs, C[:, k]) for k in range(3)])
            i = min(np.searchsorted(cs, z), len(cs) - 1)
            return pos, T[i]
        R = r0 * (1.0 - 0.45 * cs / max(cs[-1], 1e-9))
        if cs[-1] > 0.01:
            culm_parts.append(tube(C, R, 6 if detail >= 0.6 else 4, _attrs(len(C), 1)))
        # Leaves: one per node, alternating sides; basal leaves crowded at the base
        az0 = rng.uniform(0, 2 * math.pi)
        side_ref = _perp(T[0])
        n_leaves = N + p.basal_leaves
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
            az = az0 + k * math.pi + rng.normal(0, 0.25)
            radial = _unit(_rotate(side_ref, tan, az))
            sheath_len = inter * p.sheath_fraction if not basal else 0.03 + 0.04 * rng.random()
            lig, tan2 = at(zk + sheath_len)
            r_here = float(np.interp(zk, cs, R)) if cs[-1] > 0 else r0
            if sheath_len > 0.005:                                  # Sheath around the culm
                Sp = np.array([at(zk + sheath_len * q)[0] for q in np.linspace(0, 1, 5)])
                culm_parts.append(tube(Sp, np.full(5, r_here * 1.25 + 0.0004), 6 if detail >= 0.6 else 4,
                                       _attrs(5, 1, np.linspace(0, 1, 5))))
            ang = math.radians(p.leaf_angle_deg + (25 if basal else 0)) * rng.uniform(0.8, 1.2)
            d0 = _unit(tan * math.cos(ang) + radial * math.sin(ang))
            leaves.append(self.blade(lig + radial * r_here, d0, L, Wd, rng, detail))
        top, top_tan = at(cs[-1])
        # Inflorescence
        if p.head != GrassHead.NONE and fertile:
            if p.head == GrassHead.MAIZE:
                heads.append(self.tassel(top, top_tan, rng, detail))
                for e in range(p.ears if main else 0):
                    ez = height * max(0.1, p.ear_node - 0.12 * e)
                    node, tan = at(ez)
                    radial = _unit(_rotate(side_ref, tan, az0 + (round(ez / max(height, 1e-6) * N) % 2) * math.pi))
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
                            parts.append(_hair(y, h1, 0.0006))
                        else:
                            parts.append(_place(prim, dd, y, rng.uniform(0, 6.28)))
                            if p.awn_cm > 0:
                                a0 = y + dd * sl
                                parts.append(tube(np.linspace(a0, a0 + _unit(dd - 0.3 * UP) * p.awn_cm * 0.01, 3),
                                                  np.full(3, 0.00015), 3, _attrs(3, 3)))
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
        for k in range(p.crown_roots):
            a = k * GOLDEN
            d = _unit(np.array([math.cos(a), math.sin(a), -1.4 + rng.normal(0, 0.3)]))
            pts = [np.array([0.0, 0.0, 0.005])]
            ln = p.root_length_cm * 0.01 * rng.uniform(0.6, 1.1)
            for _ in range(8):
                d = _unit(d + rng.normal(0, 0.25, 3) - 0.15 * UP)
                d[2] = min(d[2], -0.2)
                d = _unit(d)
                pts.append(pts[-1] + d * ln / 8)
            parts.append(tube(np.array(pts), np.linspace(p.culm_radius_mm * 0.0004 + 0.0003, 0.0003, 9), 4,
                              _attrs(9, 7)))
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
