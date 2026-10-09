"""
Mistletoes: hemiparasitic shrubs growing on tree branches (Viscum album; Phoradendron in the Americas).

- Attachment: the seedling's holdfast becomes a haustorium that enters the host branch and spreads an
  endophyte inside it (Teixeira-Costa 2021, Braz. J. Bot. 44: 809; X-ray tomography of Viscum album on
  Aesculus); the host branch swells into a spindle around the attachment.
- Architecture of Viscum album: sympodial pseudo-dichotomous branching, each year's shoot one internode
  ending in a pair of leaves and two (or more) shoots at right angles to the previous pair, so the clump is a
  ball whose size grows with its age (one generation per year); leaves opposite, leathery, obovate-oblong;
  white berries in the forks. Phoradendron branches less regularly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .mesh_engine import MeshData
from .vine import tube
from .vegetable import revolve

UP = np.array([0.0, 0.0, 1.0])
ATTRS = ("mst_part", "mst_u")      # mst_part: 0 stem, 1 leaf, 2 berry, 3 host swelling (bark)


@dataclass
class MistletoeProfile:
    generations: int = 6               # Years of growth (one internode generation per year)
    internode_cm: float = 4.0
    stem_radius_mm: float = 2.5        # At the base of the clump
    fork_angle_deg: float = 35.0       # Half-angle between the two shoots of a fork
    regularity: float = 1.0            # 1 strictly dichotomous and decussate (Viscum) .. 0 irregular
    leaf_length_cm: float = 5.0
    leaf_width_cm: float = 1.5
    berries: int = 2                   # Per fork of the last generations
    berry_mm: float = 8.0
    stem_color: tuple = (0.50, 0.58, 0.22)
    leaf_color: tuple = (0.55, 0.62, 0.25)
    berry_color: tuple = (0.95, 0.95, 0.88)
    swelling: float = 1.5              # Host branch swelling around the haustorium (radius factor)


@dataclass
class MistletoePreset:
    scientific_name: str
    common_name: str
    family: str
    notes: str
    profile: MistletoeProfile


MISTLETOE_CATALOG = {
    "viscum_album": MistletoePreset(
        "Viscum album", "European mistletoe (muérdago)", "Santalaceae",
        "Yellow-green ball of forked stems with paired leathery leaves and white berries.",
        MistletoeProfile()),
    "phoradendron_leucarpum": MistletoePreset(
        "Phoradendron leucarpum", "American mistletoe (injerto, muérdago)", "Santalaceae",
        "Smaller, darker, less regularly branched clumps with broader leaves and white berries.",
        MistletoeProfile(generations=5, internode_cm=3.5, stem_radius_mm=2.0, regularity=0.4, leaf_length_cm=3.5,
                         leaf_width_cm=1.8, berries=3, berry_mm=5.0, stem_color=(0.38, 0.45, 0.18),
                         leaf_color=(0.30, 0.42, 0.15))),
}


def _unit(v):
    return v / max(float(np.linalg.norm(v)), 1e-12)


def _perp(d):
    p = np.cross(d, UP)
    if np.linalg.norm(p) < 1e-6:
        p = np.cross(d, [1.0, 0.0, 0.0])
    return _unit(p)


def _rot(v, axis, ang):
    axis = _unit(axis)
    return v * math.cos(ang) + np.cross(axis, v) * math.sin(ang) + axis * (axis @ v) * (1 - math.cos(ang))


def _attrs(n, part, u=0.0):
    return {"mst_part": np.full(n, float(part), np.float32), "mst_u": np.broadcast_to(u, (n,)).astype(np.float32)}


def _leaf(base, d, side, L, W, rng):
    nu = 7
    u = np.linspace(0, 1, nu)
    hw = 0.5 * W * np.sin(np.pi * np.clip(u ** 0.8, 0, 1)) ** 0.8 * (1 - 0.2 * u)
    nrm = _unit(np.cross(side, d))
    mid = base[None, :] + (u * L)[:, None] * d[None, :] - (0.15 * L * u ** 2)[:, None] * nrm   # Slight curl
    V = np.stack([mid - hw[:, None] * side, mid, mid + hw[:, None] * side], 1)
    n, m = V.shape[:2]
    j, i = np.meshgrid(np.arange(n - 1), np.arange(m - 1), indexing="ij")
    i0 = (j * m + i).ravel()
    q = np.stack([i0, i0 + 1, i0 + m + 1, i0 + m], 1)
    return MeshData(V.reshape(-1, 3).astype(np.float32), q.reshape(-1).astype(np.int32),
                    np.arange(len(q), dtype=np.int32) * 4, np.full(len(q), 4, np.int32),
                    np.zeros((len(q) * 4, 2), np.float32), _attrs(n * m, 1, np.repeat(u, m)))


def _ball(c, r, part):
    t = np.linspace(0, 1, 7)[1:-1]
    m = revolve(r * np.sin(np.pi * t), r * (2 * t - 1) * 1.0, 8, {"mst_part": np.full(len(t), float(part)),
                                                                  "mst_u": t})
    m.vertices = (m.vertices + np.asarray(c, np.float32)).astype(np.float32)
    return m


def mistletoe(profile: MistletoeProfile, base, direction, host_tangent, host_radius, seed=0, detail=1.0) -> MeshData:
    """One clump attached at `base` on the host branch surface, growing along `direction`."""
    p = profile
    rng = np.random.default_rng(seed)
    parts = []
    # Host swelling: spindle around the branch at the attachment
    t = _unit(host_tangent)
    d0 = _unit(direction)
    axis_c = np.asarray(base, float) - d0 * host_radius
    L_sw = host_radius * 5.0
    zz = np.linspace(-0.5, 0.5, 9)
    rr = host_radius * (1.0 + (p.swelling - 1.0) * np.cos(np.pi * zz) ** 2) * 1.01
    sw = revolve(rr, zz * L_sw, 12, {"mst_part": np.full(9, 3.0), "mst_u": zz})
    e1 = _perp(t)
    e2 = np.cross(t, e1)
    sw.vertices = (axis_c + sw.vertices[:, 0:1] * e1 + sw.vertices[:, 1:2] * e2 + sw.vertices[:, 2:3] * t
                   ).astype(np.float32)
    parts.append(sw)
    sides = 5 if detail >= 0.6 else 4
    # Generations of forks (decussate: each fork plane turned 90 degrees)
    tips = [(np.asarray(base, float) - d0 * host_radius * 0.3, d0, _perp(d0), p.stem_radius_mm * 0.001)]
    for g in range(max(1, p.generations)):
        nxt = []
        last = g == p.generations - 1
        for x, d, plane, r in tips:
            L = p.internode_cm * 0.01 * rng.uniform(0.8, 1.2) * (1.0 if g else 1.3)
            bend = rng.normal(0, 0.12, 3) * (1.2 - p.regularity)
            d = _unit(d + bend)
            P = np.array([x + d * L * k / 4 for k in range(5)])
            parts.append(tube(P, np.linspace(r, r * 0.85, 5), sides, _attrs(5, 0, np.full(5, g / p.generations))))
            end = P[-1]
            # Pair of leaves at the end of the youngest generations
            if g >= p.generations - 3:                         # Leaves live about three years
                for sgn in (-1, 1):
                    ld = _unit(d * 0.6 + plane * sgn * 0.8)
                    parts.append(_leaf(end, ld, _unit(np.cross(ld, d)) if np.linalg.norm(np.cross(ld, d)) > 1e-6
                                       else _perp(ld), p.leaf_length_cm * 0.01 * rng.uniform(0.8, 1.15),
                                       p.leaf_width_cm * 0.01, rng))
            if last:
                for _ in range(p.berries if rng.random() < 0.6 else 0):
                    off = _unit(rng.normal(0, 1, 3)) * p.berry_mm * 0.0006
                    parts.append(_ball(end + off, p.berry_mm * 0.0005, 2))
                continue
            n_child = 2 if rng.random() < 0.6 + 0.4 * p.regularity else 3
            new_plane = _unit(np.cross(d, plane)) if p.regularity > 0.5 else _perp(_unit(d + rng.normal(0, 0.3, 3)))
            for c in range(n_child):
                ang = math.radians(p.fork_angle_deg) * (1 if c % 2 else -1) * rng.uniform(0.8, 1.2)
                if c == 2:
                    ang = 0.0
                dc = _rot(d, new_plane, ang)
                nxt.append((end, dc, new_plane, max(r * 0.85, 0.0007)))
        tips = nxt
    out = MeshData.concatenate([m for m in parts if len(m.vertices)])
    return out
