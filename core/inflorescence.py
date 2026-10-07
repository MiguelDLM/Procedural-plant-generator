"""
Inflorescences: arrangement of flowers on a branching axis, and flower sites on plants.

Types follow the classical typology (Weberling 1989, *Morphology of Flowers and
Inflorescences*) and the developmental view of Prusinkiewicz et al. (2007,
Science 316: 1452-1456), in which racemes, panicles and cymes differ by how
long each meristem stays vegetative before turning into a flower:

- SOLITARY: one terminal flower on a peduncle.
- RACEME: pedicellate flowers along an indeterminate rachis, opening acropetally
  (old open flowers below, buds at the tip).
- SPIKE: sessile flowers (pedicel 0) along the rachis; CATKIN is a pendulous spike
  of reduced flowers (amentum: Quercus, Betula, Salix, Juglans).
- UMBEL: pedicels of equal length from one point (Prunus avium, Agave branches).
- CORYMB: pedicels lengthening basipetally so that flowers form a flat top (Malus).
- PANICLE: a branched raceme; laterals shorten acropetally (pyramidal thyrse of
  Aesculus, the paniculate scape of Agave with umbellate clusters).

Flowers are produced at a few discrete opening stages and transformed into place,
so dense inflorescences cost little more than one flower per stage.
"""

from dataclasses import dataclass
from enum import Enum
import math
import numpy as np

from .flower import FlowerEngine, FlowerProfile, _Builder, tube_mesh, STEM, GOLDEN, _normalize
from .mesh_engine import MeshData

UP = np.array([0.0, 0.0, 1.0])
STAGES = (0.12, 0.4, 0.7, 1.0)


class InflorescenceType(str, Enum):
    SOLITARY = "Solitary"
    RACEME = "Raceme"
    SPIKE = "Spike"
    CATKIN = "Catkin"
    UMBEL = "Umbel"
    CORYMB = "Corymb"
    PANICLE = "Panicle"


class FlowerSiteMode(str, Enum):
    TERMINAL = "Terminal"     # At shoot / stem apices
    AXILLARY = "Axillary"     # Along the shoots (leaf axils, areoles, old wood)


@dataclass
class InflorescenceProfile:
    kind: InflorescenceType = InflorescenceType.SOLITARY
    flower_count: int = 1
    peduncle_cm: float = 5.0         # Stalk below the first flower (scape)
    rachis_cm: float = 0.0           # Flower-bearing axis
    pedicel_cm: float = 1.0
    pedicel_angle_deg: float = 45.0  # From the rachis
    divergence_deg: float = 137.5
    branches: int = 0                # Panicle laterals
    branch_start: float = 0.0        # Fraction of the rachis below the first lateral
    branch_length_ratio: float = 0.4
    branch_angle_deg: float = 45.0
    branch_umbels: bool = False      # Laterals end in umbellate clusters (Agave)
    maturation: float = 0.6          # Acropetal opening gradient (0 all open .. 1 apical buds)
    nodding: float = 0.0             # Flowers turn downward (0 facing out/up .. 1 pendent)
    droop: float = 0.0               # Whole axis arches / hangs (catkins, Robinia, Echeveria)
    stem_radius_mm: float = 1.5
    stem_color: tuple = (0.30, 0.42, 0.18)
    site_mode: FlowerSiteMode = FlowerSiteMode.TERMINAL
    orientation: float = 0.5         # On plants: 0 along the shoot .. 1 vertical (upright candles)


@dataclass
class InflorescenceResult:
    mesh: MeshData
    flower_count: int
    height_m: float
    width_m: float


def _frame(z, y_hint):
    z = _normalize(np.asarray(z, float))
    y = np.asarray(y_hint, float) - z * float(np.dot(y_hint, z))
    if np.linalg.norm(y) < 1e-6:
        y = np.array([0.0, 1.0, 0.0]) - z * z[1]
        if np.linalg.norm(y) < 1e-6:
            y = np.array([1.0, 0.0, 0.0])
    y = _normalize(y)
    return np.stack([np.cross(y, z), y, z], 1)     # Columns: x, y (dorsal), z (floral axis)


def _bezier(c, n):
    t = np.linspace(0.0, 1.0, n)[:, None]
    c0, c1, c2, c3 = c
    return (1 - t) ** 3 * c0 + 3 * (1 - t) ** 2 * t * c1 + 3 * (1 - t) * t ** 2 * c2 + t ** 3 * c3


class InflorescenceEngine:
    def __init__(self, flower: FlowerProfile, infl: InflorescenceProfile):
        self.f = flower
        self.p = infl

    def generate(self, seed: int = 0, detail: float = 1.0, flower_scale: float = 1.0, bloom: float = 0.5,
                 max_flowers: int = 400) -> InflorescenceResult:
        """bloom: phenological stage of the whole inflorescence (0 buds .. 0.5 peak .. 1 all open)."""
        p = self.p
        rng = np.random.default_rng(seed)
        b = _Builder()
        r_st = p.stem_radius_mm * 1e-3
        sides = max(4, int(6 * detail))
        # Main axis: peduncle + rachis as one curve, arching by `droop`
        Lp = p.peduncle_cm * 0.01
        Lr = p.rachis_cm * 0.01 if p.kind not in (InflorescenceType.SOLITARY, InflorescenceType.UMBEL) else 0.0
        Lt = max(Lp + Lr, 1e-3)
        side = np.array([1.0, 0.0, 0.0])
        dr = p.droop
        end_dir = _normalize(UP * (1.0 - 2.0 * dr) + side * (1.2 * dr * (1 - dr) + 0.3 * dr))
        c = [np.zeros(3), UP * Lt * 0.35, UP * Lt * 0.35 * (1 - dr) + side * Lt * 0.3 * dr + UP * Lt * 0.3 * (1 - dr),
             None]
        c[3] = c[2] + end_dir * Lt * 0.45
        axis = _bezier(c, 48)
        seg = np.linalg.norm(np.diff(axis, axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(seg)])
        axis = axis * (Lt / max(s[-1], 1e-9))
        s = s * (Lt / max(s[-1], 1e-9))
        radii = r_st * (1.0 - 0.6 * (s / Lt))
        tube_mesh(b, axis, radii, sides, STEM, 0.0, 1.0, cap=True)

        def at(sv):
            pos = np.stack([np.interp(sv, s, axis[:, k]) for k in range(3)], -1)
            i = np.clip(np.searchsorted(s, sv) - 1, 0, len(s) - 2)
            tan = _normalize(axis[i + 1] - axis[i])
            return pos, tan

        sites = []      # (position, floral axis, dorsal hint, openness)
        n = max(1, min(p.flower_count, max_flowers))
        kind = p.kind
        top, top_t = at(np.array(Lt))

        def openness(rel):
            # Acropetal maturation shifted by the bloom stage
            o = 1.0 - p.maturation * rel + (bloom - 0.5) * 1.6
            return float(np.clip(o + rng.normal(0, 0.06), 0.0, 1.0))

        def pedicel(base, tan, az, length, angle, o, ref):
            u = _normalize(np.cross(tan, ref if abs(np.dot(tan, ref)) < 0.95 else np.array([0.0, 1.0, 0.0])))
            v = np.cross(tan, u)
            radial = math.cos(az) * u + math.sin(az) * v
            d = _normalize(math.cos(math.radians(angle)) * tan + math.sin(math.radians(angle)) * radial)
            if length > 1e-4:
                tip = base + d * length
                mid = base + d * length * 0.5 + UP * length * 0.12 * (1 - p.nodding) - UP * length * 0.25 * p.nodding
                pts = _bezier([base, base + d * length * 0.4, mid, tip], 5)
                tube_mesh(b, pts, np.linspace(r_st * 0.45, r_st * 0.3, 5), max(3, sides - 2), STEM, 0.0, 1.0, cap=False)
                fa = _normalize(d * (1 - p.nodding) - UP * p.nodding + 0.25 * (pts[-1] - pts[-2]) / max(length, 1e-6))
            else:
                tip = base + radial * r_st * 0.8
                fa = _normalize(d * (1 - p.nodding) - UP * p.nodding)
            sites.append((tip, fa, tan, o))

        if kind == InflorescenceType.SOLITARY:
            sites.append((top, _normalize(top_t * (1 - p.nodding) - UP * p.nodding + side * 0.05), UP, openness(0.0)))
        elif kind == InflorescenceType.UMBEL:
            for k in range(n):
                pedicel(top, top_t, k * GOLDEN, p.pedicel_cm * 0.01 * rng.uniform(0.9, 1.1),
                        p.pedicel_angle_deg * math.sqrt((k + 0.5) / n) * 1.3, openness(rng.random() * 0.5), side)
        elif kind in (InflorescenceType.RACEME, InflorescenceType.SPIKE, InflorescenceType.CATKIN,
                      InflorescenceType.CORYMB):
            sv = Lp + Lr * (np.arange(n) + 0.5) / n
            pos, tan = at(sv)
            for k in range(n):
                rel = (sv[k] - Lp) / max(Lr, 1e-6)
                ln = 0.0 if kind in (InflorescenceType.SPIKE, InflorescenceType.CATKIN) else p.pedicel_cm * 0.01
                if kind == InflorescenceType.CORYMB:   # Lower pedicels longer: flowers reach a common level
                    ln = p.pedicel_cm * 0.01 + (Lt - sv[k]) * 1.05
                pedicel(pos[k], tan[k], k * math.radians(p.divergence_deg), ln, p.pedicel_angle_deg,
                        openness(rel), side)
        elif kind == InflorescenceType.PANICLE:
            nb = max(1, p.branches)
            per = max(1, n // nb)
            sb = Lp + Lr * (p.branch_start + (1 - p.branch_start) * (np.arange(nb) + 0.5) / nb)
            pos, tan = at(sb)
            for k in range(nb):
                rel = (sb[k] - Lp) / max(Lr, 1e-6)
                bl = Lr * p.branch_length_ratio * (1.0 - 0.75 * rel) * rng.uniform(0.85, 1.1)
                az = k * math.radians(p.divergence_deg)
                u = _normalize(np.cross(tan[k], side if abs(np.dot(tan[k], side)) < 0.95 else np.array([0, 1.0, 0])))
                v = np.cross(tan[k], u)
                radial = math.cos(az) * u + math.sin(az) * v
                a = math.radians(p.branch_angle_deg)
                d = _normalize(math.cos(a) * tan[k] + math.sin(a) * radial)
                bend = UP * bl * 0.25 * (1 - p.droop) - UP * bl * 0.3 * p.droop
                pts = _bezier([pos[k], pos[k] + d * bl * 0.4, pos[k] + d * bl * 0.75 + bend * 0.5,
                               pos[k] + d * bl + bend], 8)
                tube_mesh(b, pts, np.linspace(radii[np.searchsorted(s, sb[k]) - 1] * 0.6, r_st * 0.25, 8),
                          max(3, sides - 1), STEM, 0.0, 1.0, cap=True)
                bt = _normalize(pts[-1] - pts[-2])
                if p.branch_umbels:
                    for j in range(per):
                        pedicel(pts[-1], bt, j * GOLDEN, p.pedicel_cm * 0.01 * rng.uniform(0.85, 1.1),
                                p.pedicel_angle_deg * math.sqrt((j + 0.5) / per) * 1.4, openness(rel), side)
                else:
                    seg2 = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
                    for j in range(per):
                        f = 0.25 + 0.75 * (j + 0.5) / per
                        q = np.array([np.interp(f * seg2[-1], seg2, pts[:, i]) for i in range(3)])
                        ii = min(len(pts) - 2, int(np.searchsorted(seg2, f * seg2[-1])))
                        pedicel(q, _normalize(pts[ii + 1] - pts[ii]), j * GOLDEN, p.pedicel_cm * 0.01,
                                p.pedicel_angle_deg, openness(min(1.0, rel * 0.6 + f * 0.4)), side)
            # Terminal flower(s) of the main axis
            if not p.branch_umbels:
                for j in range(max(1, per // 2)):
                    pedicel(top - top_t * Lr * 0.04 * j, top_t, j * GOLDEN, p.pedicel_cm * 0.01,
                            p.pedicel_angle_deg * 0.6, openness(1.0), side)

        # Flowers: one mesh per opening stage, transformed to each site
        eng = FlowerEngine(self.f)
        stage_mesh = {}
        parts = [b.mesh()]
        stage_detail = detail * (1.0 if len(sites) <= 3 else (0.7 if len(sites) <= 30 else 0.5))
        for pos, fa, dorsal_hint, o in sites:
            st = min(STAGES, key=lambda x: abs(x - o))
            if st not in stage_mesh:
                stage_mesh[st] = eng.generate(openness=st, detail=stage_detail, seed=seed).mesh
            m = stage_mesh[st]
            # Dorsal side away from the bearing axis, toward the sky: zygomorphic flowers present their lip
            R = _frame(fa, UP + 0.3 * _normalize(dorsal_hint))
            roll = rng.uniform(-0.25, 0.25) * (1.0 - self.f.zygomorphy) + (rng.uniform(0, 2 * math.pi)
                                                                             if self.f.zygomorphy < 0.2 else 0.0)
            cr, sr = math.cos(roll), math.sin(roll)
            Rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1.0]])
            sc = flower_scale * rng.uniform(0.92, 1.08)
            # The pedicel (or the areole) carries the base of the inferior ovary / hypanthium
            lift = np.array([0.0, 0.0, self.f.hypanthium_cm * 0.01 if not self.f.capitulum else 0.25 * self.f.disc_radius_cm * 0.01])
            V = ((m.vertices.astype(np.float64) + lift) @ (R @ Rz).T) * sc + pos
            parts.append(MeshData(V.astype(np.float32), m.loop_vertex, m.loop_start, m.loop_total, m.loop_uv,
                                  dict(m.point_attributes)))
        mesh = MeshData.concatenate(parts)
        V = mesh.vertices
        return InflorescenceResult(mesh, len(sites), float(V[:, 2].max()) if len(V) else 0.0,
                                   2.0 * float(np.hypot(V[:, 0], V[:, 1]).max()) if len(V) else 0.0)


# -----------------------------------------------------------------------------
# Flower sites on generated plants
# -----------------------------------------------------------------------------
@dataclass
class FlowerSites:
    positions: np.ndarray     # (N, 3)
    directions: np.ndarray    # (N, 3) inflorescence axis
    scales: np.ndarray        # (N,)

    def __len__(self):
        return len(self.positions)


def _orient(D, infl: InflorescenceProfile):
    up = np.repeat(UP[None, :], len(D), 0)
    if infl.kind == InflorescenceType.CATKIN or infl.droop > 0.6:
        target = -up                                      # Pendulous
    else:
        target = up
    w = np.clip(infl.orientation, 0, 1)
    return _normalize(D * (1 - w) + target * w)


def tree_flower_sites(graph, infl: InflorescenceProfile, count: int, seed: int = 0) -> FlowerSites:
    """Terminal sites at the tips of the youngest shoots, or axillary sites along them (leaf axils,
    spurs). `count` caps the number of inflorescences (sites are subsampled evenly)."""
    rng = np.random.default_rng(seed + 977)
    axes = [a for a in graph.axes if len(a.positions) >= 2]
    if not axes or count <= 0:
        z = np.zeros((0, 3))
        return FlowerSites(z, z, np.zeros(0))
    max_order = max(a.order for a in axes)
    P, D = [], []
    if infl.site_mode == FlowerSiteMode.TERMINAL:
        for a in axes:
            if a.order >= max(1, max_order - 1) or (max_order == 0):
                P.append(a.positions[-1])
                D.append(a.positions[-1] - a.positions[-2])
    else:
        for a in axes:
            if a.order < max(1, max_order - 1):
                continue
            s = a.arc_length
            if s[-1] < 1e-3:
                continue
            k = max(1, int(s[-1] / 0.08))
            sv = np.sort(rng.uniform(0.2, 0.95, k)) * s[-1]
            pos = np.stack([np.interp(sv, s, a.positions[:, i]) for i in range(3)], 1)
            i = np.clip(np.searchsorted(s, sv) - 1, 0, len(s) - 2)
            tan = _normalize(a.positions[i + 1] - a.positions[i])
            out = _normalize(np.cross(tan, rng.normal(size=(len(sv), 3))))
            P.append(pos)
            D.append(_normalize(tan * 0.4 + out))
        P = [np.vstack(P)] if P else []
        D = [np.vstack(D)] if D else []
    if not P:
        z = np.zeros((0, 3))
        return FlowerSites(z, z, np.zeros(0))
    P = np.vstack(P) if P[0].ndim == 2 else np.array(P)
    D = _normalize(np.vstack(D) if D[0].ndim == 2 else np.array(D))
    if len(P) > count:
        idx = rng.choice(len(P), count, replace=False)
        P, D = P[idx], D[idx]
    D = _orient(D, infl)
    return FlowerSites(P, D, rng.uniform(0.85, 1.15, len(P)))


def surface_flower_sites(pos, normal, weight, count: int, seed: int = 0, lean: float = 0.6,
                         min_dist: float = 0.0, sink: float = 0.0) -> FlowerSites:
    """Sites on a stem surface (cactus areoles): sampled by weight (e.g. near the apex), facing
    outward-and-up, at least `min_dist` apart (flowers must not overlap)."""
    rng = np.random.default_rng(seed + 541)
    w = np.asarray(weight, float)
    ok = np.flatnonzero(w > 1e-3)
    if len(ok) == 0 or count <= 0:
        z = np.zeros((0, 3))
        return FlowerSites(z, z, np.zeros(0))
    pr = w[ok] / w[ok].sum()
    k = min(count, len(ok))
    if min_dist > 0:
        order = ok[rng.choice(len(ok), len(ok), replace=False, p=pr)]
        chosen = []
        P = np.asarray(pos)
        for i in order:
            if all(np.linalg.norm(P[i] - P[j]) >= min_dist for j in chosen):
                chosen.append(i)
                if len(chosen) >= k:
                    break
        idx = np.array(chosen, dtype=int)
        k = len(idx)
    else:
        idx = ok[rng.choice(len(ok), k, replace=False, p=pr)]
    N = np.asarray(normal)[idx]
    D = _normalize(N * (1 - lean) + UP * lean)
    # The flower base is sunk slightly into the areole so its rim never hangs over the curved stem
    return FlowerSites(np.asarray(pos)[idx] - N * sink, D, rng.uniform(0.85, 1.1, k))
