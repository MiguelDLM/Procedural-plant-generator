"""
Guests on generated trees: mistletoes (parasites) and epiphytic orchids, placed on the branches.

Epiphyte placement follows Johansson's (1974) host-tree zones: 1 base of the trunk, 2 lower trunk, 3 base
of the main branches (inner third), 4 their middle third, 5 the outer twigs; canopy epiphytes, mostly orchids
and ferns, concentrate in zones 3-5 (Krömer et al. 2007; Zotz 2016, Plants on Plants), on branches wide
enough to hold them, on their upper side. Mistletoes are dispersed by birds onto young, sunlit branches of
the outer crown (zones 4-5), where they attach on the upper side.
"""

from __future__ import annotations

import copy
import math

import numpy as np

from .mesh_engine import MeshData

UP = np.array([0.0, 0.0, 1.0])
ZONES = {3: (0.0, 1 / 3), 4: (1 / 3, 2 / 3), 5: (2 / 3, 1.0)}


def branch_sites(skeleton, n, zones=(3, 4), min_radius=0.02, max_radius=1.0, seed=0, max_order=2):
    """Points on the upper side of branches: (surface point, axis point, unit tangent, radius, up normal)."""
    rng = np.random.default_rng(seed)
    cands = []
    for a in skeleton.axes:
        if a.order < 1 or a.order > max_order or len(a.positions) < 3:
            continue
        P, R = a.positions, a.radii
        s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
        f = s / max(s[-1], 1e-9)
        for z in zones:
            lo, hi = ZONES[z]
            for i in np.nonzero((f >= lo) & (f < hi) & (R >= min_radius) & (R <= max_radius))[0]:
                cands.append((i, P, R))
    if not cands:
        return []
    out = []
    for k in rng.choice(len(cands), size=min(n, len(cands)), replace=False):
        i, P, R = cands[k]
        i = max(1, min(i, len(P) - 2))
        t = P[i + 1] - P[i - 1]
        t = t / max(np.linalg.norm(t), 1e-9)
        up = UP - (UP @ t) * t
        if np.linalg.norm(up) < 1e-3:                              # Vertical branch: any side
            up = np.cross(t, [1.0, 0.0, 0.0])
        up = up / np.linalg.norm(up)
        side = np.cross(t, up)
        a = rng.normal(0, 0.35)                                   # Mostly on top, sometimes on a flank
        up = up * math.cos(a) + side * math.sin(a)
        up = up / np.linalg.norm(up)
        out.append((P[i] + up * R[i], P[i].copy(), t, float(R[i]), up))
    return out


def place(m: MeshData, R, origin) -> MeshData:
    V = m.vertices.astype(float) @ R.T + np.asarray(origin)[None, :]
    return MeshData(V.astype(np.float32), m.loop_vertex, m.loop_start, m.loop_total, m.loop_uv,
                    dict(m.point_attributes))


def epiphytic_orchids(profile, skeleton, n, seed=0, detail=0.6, scale=1.0):
    """Orchid plants on the upper side of the main branches (zones 3-4): each grown as mounted on a branch of
    the host's diameter, its virtual bough aligned with the host branch, so the velamen roots wrap the
    real bark. Returns one merged MeshData (orchid attributes) without the virtual bough."""
    from .orchid import OrchidEngine, OrchidMount, OrchidHabit
    parts = []
    sites = branch_sites(skeleton, n, zones=(3, 4), min_radius=0.025, seed=seed + 5)
    for k, (surf, axis_pt, t, r, up) in enumerate(sites):
        prof = copy.deepcopy(profile)
        if prof.habit == OrchidHabit.CLIMBING:
            continue
        prof.mount = OrchidMount.BRANCH
        prof.branch_diameter_cm = 2 * r * 100 / max(scale, 1e-6)
        res = OrchidEngine(prof).generate(seed=seed * 31 + k, detail=detail)
        ms = [m for m in (res.leaves, res.stems, res.roots, res.flowers, res.spikes) if len(m.vertices)]
        if not ms:
            continue
        m = MeshData.concatenate(ms)
        # Local frame of the generated plant: X along its bough, Z up; bough top surface at the origin
        x = t
        z = up - (up @ x) * x
        z = z / max(np.linalg.norm(z), 1e-9)
        y = np.cross(z, x)
        Rm = np.stack([x, y, z], 1) * scale
        parts.append(place(m, Rm, surf))
    return MeshData.concatenate(parts) if parts else MeshData.empty()


def mistletoes(profile, skeleton, n, seed=0, detail=0.6):
    """Mistletoe clumps on sunlit young branches of the outer crown (zones 4-5), on their upper side."""
    from .mistletoe import mistletoe
    parts = []
    sites = branch_sites(skeleton, n, zones=(4, 5), min_radius=0.01, max_radius=0.08, seed=seed + 9, max_order=3)
    for k, (surf, axis_pt, t, r, up) in enumerate(sites):
        d = up * 0.8 + t * 0.3
        parts.append(mistletoe(profile, surf, d / np.linalg.norm(d), t, r, seed=seed * 13 + k, detail=detail))
    return MeshData.concatenate(parts) if parts else MeshData.empty()
