"""
Vegetables: storage roots, tubers and inflorescence heads, with their leaves.

- Storage taproots (carrot, radish, beet, turnip): a body of revolution of the swollen root and hypocotyl,
  from the crown down: rounded shoulder, widest point, then a conical (carrot) or rounded (beet, radish)
  taper into a thin tail. Part of it may stand above the soil (beet, turnip, radish) and take another colour
  there (green or purple shoulders). Lateral rootlets leave the root in vertical ranks along the xylem poles
  (two poles in the diarch roots of carrot and beet give two or four ranks); horizontal growth rings and
  rootlet scars mark the skin.
- Tubers (potato): stolons grow from the underground nodes of the stems and swell at the tip into tubers
  (Solanum tuberosum); the eyes (axillary buds in the axils of scale leaves) sit on a spiral of about 2/5 and
  crowd toward the apical "rose" end, opposite the stolon ("heel") end.
- Heads (cauliflower, broccoli, Romanesco): the curd is an inflorescence whose meristems keep branching
  instead of making flowers, repeating the same spiral (golden-angle) arrangement at every scale (Azpeitia et
  al. 2021, Science 373: 192-197; Kieffer et al. 1998). Cauliflower: rounded meristem domes packed into a curd;
  broccoli: clusters of flower buds on green branches; Romanesco: self-similar cones on cones.

The leaves use the leaf engine of the trees: a basal rosette on short internodes (root crops, brassicas) or
leaves along one or several erect stems (potato); in brassicas the inner leaves curl up around the head.
Plant space: the soil surface is z = 0; underground organs are below it.
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
from .vine import tube, frames, arc_length

UP = np.array([0.0, 0.0, 1.0])
GOLDEN = math.radians(137.50776)


class StorageOrgan(str, Enum):
    TAPROOT = "Taproot"      # Carrot, radish, beet, turnip, parsnip
    TUBERS = "Tubers"        # Potato
    HEAD = "Head"            # Cauliflower, broccoli, Romanesco
    NONE = "None"            # Leaves only


class HeadType(str, Enum):
    CURD = "Curd"            # Cauliflower: packed meristem domes
    BUDS = "Buds"            # Broccoli: flower-bud clusters on green branches
    CONES = "Cones"          # Romanesco: self-similar cones


@dataclass
class VegetableProfile:
    """Habit and storage organ of a vegetable. Sizes in cm unless noted."""
    organ: StorageOrgan = StorageOrgan.TAPROOT

    # Shoot and leaves
    stem_count: int = 1                # Erect stems (potato 3-6); 1 for rosettes
    stem_height_cm: float = 0.0        # 0 = basal rosette
    stem_radius_mm: float = 4.0
    leaf_count: int = 10               # Per stem
    leaf_size: float = 1.0             # Scale of the leaves relative to the leaf blade length
    elevation_outer_deg: float = 40.0  # Oldest (outer / lower) leaves
    elevation_inner_deg: float = 75.0  # Youngest (inner / upper) leaves
    head_wrap: float = 0.0             # Brassicas: inner leaves curl up around the head
    stem_color: tuple = (0.45, 0.55, 0.30)

    # Storage taproot
    root_length_cm: float = 18.0
    root_diameter_cm: float = 3.2
    widest_position: float = 0.08      # 0 crown .. 1 tip (carrot near the top, beet ~0.5)
    shoulder: float = 0.45             # Roundness of the top: 0.2 square .. 1 pointed
    taper: float = 1.0                 # Below the widest point: 0.4 rounded (beet) .. 1 conical (carrot) .. 2 slender
    exposure: float = 0.05             # Fraction of the root standing above the soil
    tail_cm: float = 4.0               # Thin terminal root
    rings: float = 0.3                 # Horizontal growth rings / wrinkles
    rootlets: float = 0.3              # Density of lateral rootlets
    rootlet_ranks: int = 4             # Vertical rows of rootlets
    root_color: tuple = (0.95, 0.45, 0.08)
    shoulder_color: tuple = (0.40, 0.48, 0.15)   # Above-ground part
    shoulder_tint: float = 0.4
    tip_color: tuple = (0.95, 0.90, 0.80)        # Lower part (white-tipped radish)
    tip_tint: float = 0.0

    # Tubers
    tuber_count: int = 8
    tuber_length_cm: float = 8.0
    tuber_diameter_cm: float = 6.0
    tuber_depth_cm: float = 12.0
    stolon_length_cm: float = 15.0
    eyes: int = 9
    eye_depth: float = 0.08
    tuber_color: tuple = (0.75, 0.60, 0.40)
    tuber_dots: float = 0.3

    # Inflorescence head
    head_type: HeadType = HeadType.CURD
    head_diameter_cm: float = 16.0
    head_height_ratio: float = 0.45    # Dome height / diameter (Romanesco ~0.9: a cone)
    head_levels: int = 3               # Orders of branching shown
    florets: int = 21                  # Branches of the first order (Fibonacci numbers suit the spirals)
    floret_scale: float = 0.33         # Size of a branch relative to its parent
    bud_size_mm: float = 2.0           # Broccoli buds
    head_color: tuple = (0.95, 0.92, 0.80)
    branch_color: tuple = (0.80, 0.85, 0.60)


@dataclass
class VegetableResult:
    root: MeshData           # Storage root or tubers (+ rootlets, stolons)
    stems: MeshData
    head: MeshData
    leaves: MeshData
    foliage: FoliageInstances
    leaf_engine: LeafMorphologyEngine
    stats: dict = field(default_factory=dict)


def _unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def _ss(e0, e1, x):
    t = np.clip((np.asarray(x, float) - e0) / max(e1 - e0, 1e-12), 0, 1)
    return t * t * (3 - 2 * t)


def revolve(rho, z, nth, attrs=None, radial=None) -> MeshData:
    """Closed surface of revolution around Z: rings (rho[i], z[i]); poles at both ends by fans with one vertex
    per triangle. `radial(theta_grid, i)` may scale ring radii per angle. attrs: per-ring arrays."""
    rows = len(rho)
    th = np.linspace(0.0, 2 * math.pi, nth + 1)
    R = np.repeat(np.asarray(rho, float)[:, None], nth + 1, 1)
    if radial is not None:
        R = R * radial(th)
    V = np.stack([R * np.cos(th)[None, :], R * np.sin(th)[None, :], np.repeat(np.asarray(z, float)[:, None],
                                                                              nth + 1, 1)], -1).reshape(-1, 3)
    m = nth + 1
    j, i = np.meshgrid(np.arange(rows - 1), np.arange(nth), indexing="ij")
    i0 = (j * m + i).ravel()
    quads = np.stack([i0, i0 + 1, i0 + m + 1, i0 + m], axis=1)
    top, bot = len(V), len(V) + nth
    poles = np.concatenate([np.repeat([[0.0, 0.0, z[0]]], nth, 0), np.repeat([[0.0, 0.0, z[-1]]], nth, 0)])
    ring0, ringN = np.arange(nth), (rows - 1) * m + np.arange(nth)
    fan0 = np.stack([top + ring0, ring0, ring0 + 1], axis=1)
    fanN = np.stack([ringN + 1, ringN, bot + ring0], axis=1)
    verts = np.concatenate([V, poles])
    lv = np.concatenate([quads.reshape(-1), fan0.reshape(-1), fanN.reshape(-1)]).astype(np.int32)
    lt = np.concatenate([np.full(len(quads), 4), np.full(2 * nth, 3)]).astype(np.int32)
    ls = np.concatenate([[0], np.cumsum(lt)[:-1]]).astype(np.int32)
    um = (th[:-1] + th[1:]) / (4 * math.pi)
    u = np.concatenate([np.repeat((th / (2 * math.pi))[None, :], rows, 0).reshape(-1), um, um])
    vv = np.concatenate([np.repeat(np.linspace(0, 1, rows)[:, None], m, 1).reshape(-1), np.zeros(nth),
                         np.ones(nth)])
    out = {}
    for k, a in (attrs or {}).items():
        a = np.asarray(a, float)
        out[k] = np.concatenate([np.repeat(a[:, None], m, 1).reshape(-1), np.full(nth, a[0]), np.full(nth, a[-1])])
    return MeshData(verts.astype(np.float32), lv, ls, lt, np.stack([u, vv], 1)[lv].astype(np.float32),
                    {k: v.astype(np.float32) for k, v in out.items()})


ROOT_ATTRS = ("veg_t", "veg_above", "veg_ring", "veg_part")   # veg_part: 0 body, 1 rootlet / stolon


def _merge(parts, names) -> MeshData:
    parts = [p for p in parts if len(p.vertices)]
    for p in parts:
        for k in names:
            p.point_attributes[k] = np.asarray(p.point_attributes.get(k, np.zeros(len(p.vertices))), np.float32)
        for k in list(p.point_attributes):
            if k not in names:
                del p.point_attributes[k]
    return MeshData.concatenate(parts)


def _rotation_to(z, roll=0.0):
    z = _unit(np.asarray(z, float))
    e1 = _unit(np.cross(z, UP) if abs(z @ UP) < 0.95 else np.cross(z, [1.0, 0.0, 0.0]))
    e2 = np.cross(z, e1)
    c, s = math.cos(roll), math.sin(roll)
    e1, e2 = c * e1 + s * e2, -s * e1 + c * e2
    return np.stack([e1, e2, z], axis=1)


def _place(m: MeshData, R, t, scale=1.0, extra=None) -> MeshData:
    V = (m.vertices.astype(float) * scale) @ np.asarray(R).T + np.asarray(t)[None, :]
    attrs = {k: v.copy() for k, v in m.point_attributes.items()}
    for k, v in (extra or {}).items():
        attrs[k] = np.full(len(V), v, np.float32)
    return MeshData(V.astype(np.float32), m.loop_vertex.copy(), m.loop_start.copy(), m.loop_total.copy(),
                    m.loop_uv.copy(), attrs)


class VegetableEngine:
    def __init__(self, profile: VegetableProfile = None, leaf: LeafMorphologyProfile = None,
                 venation: VenationProfile = None):
        self.p = profile or VegetableProfile()
        self.leaf = leaf or LeafMorphologyProfile()
        self.venation = venation or VenationProfile()
        self.leaf_engine = LeafMorphologyEngine(self.leaf)

    # ----------------------------------------------------------------- taproot
    def taproot(self, rng, detail=1.0) -> tuple[MeshData, float]:
        """Swollen root and hypocotyl along -Z; returns the mesh and the height of the crown (top)."""
        p = self.p
        L = p.root_length_cm * 0.01
        R = 0.5 * p.root_diameter_cm * 0.01
        top = p.exposure * L
        n = max(24, int(60 * detail))
        v = np.linspace(0, 1, n)
        t = v ** 1.15                                      # Finer near the crown
        w = float(np.clip(p.widest_position, 0.02, 0.9))
        upper = np.sin(0.5 * np.pi * np.clip(t / w, 0, 1)) ** max(p.shoulder, 0.05)
        lower = np.clip((1 - t) / (1 - w), 0, 1) ** max(p.taper, 0.05)
        rho = R * np.where(t < w, upper, lower)
        rho = np.maximum(rho, R * 0.02)
        rho[0] = 0.0
        z = top - L * t
        # Growth rings: shallow irregular constrictions
        ring_n = max(4, int(L / 0.006))
        ring = np.abs(np.sin(np.pi * ring_n * t + rng.uniform(0, 6.28))) ** 6
        rings = 1.0 - p.rings * 0.06 * ring * (rng.uniform(0.5, 1.0, n))
        rho = rho * rings
        nth = max(10, int(24 * detail))
        wob = rng.normal(0, 1, 3)

        def radial(th):          # Slight lumpiness
            return 1.0 + 0.03 * (wob[0] * np.cos(th + wob[1]) + 0.5 * wob[2] * np.cos(2 * th))[None, :]

        body = revolve(rho[1:], z[1:], nth, {"veg_t": t[1:], "veg_above": (z[1:] > 0).astype(float),
                                             "veg_ring": 1 - rings[1:], "veg_part": np.zeros(n - 1)}, radial)
        parts = [body]
        # Tail
        if p.tail_cm > 0:
            tl = p.tail_cm * 0.01
            k = 10
            pts = [np.array([0.0, 0.0, z[-1] + 0.002])]
            d = np.array([0.0, 0.0, -1.0])
            for _ in range(k):
                d = _unit(d + rng.normal(0, 0.15, 3))
                d[2] = -abs(d[2])
                pts.append(pts[-1] + d * tl / k)
            P = np.array(pts)
            r0 = max(float(rho[-2]), 0.0008)
            parts.append(tube(P, np.linspace(r0, 0.0003, len(P)), 5, {"veg_t": np.ones(len(P)),
                                                                       "veg_part": np.ones(len(P))}))
        # Rootlets in vertical ranks, below the soil
        if p.rootlets > 0:
            ranks = max(1, p.rootlet_ranks)
            phase = rng.uniform(0, 2 * math.pi)
            for k in range(int(p.rootlets * 30 * L / 0.18) + 1):
                tt = rng.uniform(max(0.15, p.exposure + 0.05), 0.95)
                i = min(int(tt ** (1 / 1.15) * (n - 1)), n - 1)
                if z[i] > -0.005:
                    continue
                a = phase + 2 * math.pi * rng.integers(ranks) / ranks + rng.normal(0, 0.12)
                radial_dir = np.array([math.cos(a), math.sin(a), 0.0])
                base = radial_dir * rho[i] * 0.95 + UP * z[i]
                ln = rng.uniform(0.3, 1.0) * (0.02 + 0.6 * rho[i])
                pts = [base]
                d = _unit(radial_dir - 0.4 * UP)
                for _ in range(5):
                    d = _unit(d + rng.normal(0, 0.25, 3) - 0.15 * UP)
                    pts.append(pts[-1] + d * ln / 5)
                parts.append(tube(np.array(pts), np.linspace(0.0005, 0.0002, 6), 3,
                                  {"veg_t": np.full(6, tt), "veg_part": np.ones(6)}))
        return _merge(parts, ROOT_ATTRS), top

    # ----------------------------------------------------------------- tubers
    def tubers(self, rng, stem_bases, detail=1.0) -> MeshData:
        """Seed (mother) tuber at planting depth; an underground stem rises from it to every aerial stem;
        stolons and fibrous roots leave the underground nodes; tubers swell at the stolon tips."""
        p = self.p
        parts = []
        depth0 = p.tuber_depth_cm * 0.01 * 0.8
        seed_c = np.array([0.0, 0.0, -depth0])
        sd = max(p.tuber_diameter_cm * 0.01 * 0.6, 0.02)
        t0 = 0.5 - 0.5 * np.cos(np.pi * np.linspace(0, 1, 12)[1:-1])
        mother = revolve(0.5 * sd * np.sin(np.pi * t0) ** 0.6, sd * 1.2 * t0 - sd * 0.6, 12,
                         {"veg_t": t0, "veg_part": np.zeros(len(t0)), "veg_ring": np.full(len(t0), 0.3)},
                         lambda th: 1.0 + 0.08 * np.cos(3 * th + 1.0)[None, :])
        parts.append(_place(mother, _rotation_to(np.array([1.0, 0.2, 0.1])), seed_c))
        nodes = []
        r_st = p.stem_radius_mm * 0.001
        for base in stem_bases:
            under = np.linspace(seed_c + _unit(base - seed_c) * sd * 0.3, base + UP * 0.004, 6)
            parts.append(tube(under, np.full(6, r_st * 0.9), 6, {"veg_part": np.zeros(6), "veg_t": np.zeros(6)}))
            nodes.append(under[1:-1])
            for q in under[1:-1]:                           # Fibrous roots from the underground nodes
                for _ in range(2):
                    d = _unit(rng.normal(0, 1, 3) * np.array([1, 1, 0.4]) - 0.6 * UP)
                    pts = [q]
                    for _ in range(6):
                        d = _unit(d + rng.normal(0, 0.25, 3) - 0.1 * UP)
                        d[2] = min(d[2], -0.15)                 # Roots grow down or sideways, never up
                        d = _unit(d)
                        pts.append(pts[-1] + d * 0.025)
                    parts.append(tube(np.array(pts), np.linspace(0.0009, 0.0003, 7), 3,
                                      {"veg_part": np.ones(7), "veg_t": np.ones(7)}))
        nodes = np.concatenate(nodes)
        L = p.tuber_length_cm * 0.01
        D = p.tuber_diameter_cm * 0.01
        depth = p.tuber_depth_cm * 0.01
        n_t = max(12, int(22 * detail))
        nth = max(12, int(20 * detail))
        for k in range(p.tuber_count):
            base = nodes[rng.integers(len(nodes))]            # Stolons leave the underground stem nodes
            a = 2 * math.pi * k / max(p.tuber_count, 1) + rng.normal(0, 0.4)
            h = np.array([math.cos(a), math.sin(a), 0.0])
            sl = p.stolon_length_cm * 0.01 * rng.uniform(0.5, 1.2)
            pts = [base]
            d = _unit(h - 0.2 * UP)
            for _ in range(10):
                d = _unit(d + rng.normal(0, 0.2, 3) * np.array([1, 1, 0.3]))
                pts.append(pts[-1] + d * sl / 10)
            P = np.array(pts)
            P[:, 2] = np.minimum(P[:, 2], -0.02)
            parts.append(tube(P, np.full(len(P), 0.0015), 4, {"veg_part": np.ones(len(P))}))
            # Tuber: heel at the stolon tip, rose (apical) end away from it; eyes on a 2/5 spiral
            sc = rng.uniform(0.6, 1.15)
            t = 0.5 - 0.5 * np.cos(np.pi * np.linspace(0, 1, n_t)[1:-1])
            rho = 0.5 * D * sc * np.sin(np.pi * t) ** 0.6
            z = L * sc * t
            eyes_t = np.sort(1 - rng.beta(1.2, 2.2, max(p.eyes, 0)))       # Crowded toward the rose end
            eyes_a = np.arange(len(eyes_t)) * 2 * math.pi * 2 / 5 + rng.uniform(0, 6.28)
            lumps = rng.normal(0, 1, 4)

            def radial(th, eyes_t=eyes_t, eyes_a=eyes_a, t=t, z=z, rho=rho, lumps=lumps, D=D * sc):
                f = 1.0 + 0.05 * (lumps[0] * np.cos(th + lumps[1]) + lumps[2] * np.cos(2 * th + lumps[3]))[None, :]
                f = np.repeat(f, len(t), 0)
                # Eye dimples: Gaussian in the distance measured on the surface (not in angle, which would
                # stretch the dimples into wedges near the poles)
                P = np.stack([rho[:, None] * np.cos(th)[None, :], rho[:, None] * np.sin(th)[None, :],
                              np.repeat(z[:, None], len(th), 1)], -1)
                sig = 0.11 * D
                for et, ea in zip(eyes_t, eyes_a):
                    ie = min(int(np.searchsorted(t, et)), len(t) - 1)
                    E = np.array([rho[ie] * math.cos(ea), rho[ie] * math.sin(ea), z[ie]])
                    d2 = np.sum((P - E) ** 2, -1) / sig ** 2
                    f = f * (1.0 - p.eye_depth * np.exp(-d2))
                return f
            m = revolve(rho, z, nth, {"veg_t": t, "veg_part": np.zeros(len(t)), "veg_ring": np.zeros(len(t))},
                        radial)
            axis = _unit(d + rng.normal(0, 0.2, 3))
            parts.append(_place(m, _rotation_to(axis, rng.uniform(0, 6.28)), P[-1] - axis * 0.004))
        return _merge(parts, ROOT_ATTRS)

    # ----------------------------------------------------------------- heads
    def head(self, rng, center, detail=1.0, stem_top=None, stem_radius=0.01) -> MeshData:
        """Self-similar inflorescence head. A base dome (curd, buds) or cone (Romanesco) is displaced by nested
        height fields: at every order the branch meristems sit in a golden-angle spiral, the first order over
        the whole head and each further order around every meristem of the previous one, each order smaller
        by `floret_scale`. A meristem raises a rounded cap (curd), a bead (buds) or a cone (Romanesco); caps
        of one order merge where they overlap, so the head stays one continuous surface with crevices
        between the lobes (attribute veg_h). Broccoli shows its branches under the dome."""
        from .spatial import nearest_points
        p = self.p
        c0 = np.asarray(center, float)
        R = 0.5 * p.head_diameter_cm * 0.01
        H = max(p.head_height_ratio * 2 * R, 0.01)
        cones = p.head_type == HeadType.CONES
        a_max = math.radians(100.0)
        # Base surface (u: 0 apex .. 1 rim, b: azimuth); apex at c0 + H
        nu = max(40, int(120 * detail))
        nb = max(60, int(260 * detail))
        u = np.linspace(0.0, 1.0, nu)[1:]
        b = np.linspace(0.0, 2 * math.pi, nb + 1)
        U, Bt = np.meshgrid(u, b, indexing="ij")
        if cones:
            radial = R * U
            up = H * (1.0 - U)
            nr, nz = H, R                                      # Outward normal of the cone side
        else:
            al = U * a_max
            radial = R * np.sin(al)
            up = H * np.cos(al)
            nr, nz = np.sin(al) / R, np.cos(al) / H
        cb, sb = np.cos(Bt), np.sin(Bt)
        P = np.stack([radial * cb, radial * sb, up], -1) + c0
        N = np.stack([nr * cb, nr * sb, nz * np.ones_like(U)], -1)
        N = N / np.linalg.norm(N, axis=-1, keepdims=True)
        # Area of the head and seeds of the first order (area-uniform spiral)
        area = math.pi * R * math.hypot(R, H) if cones else 2 * math.pi * R * max(R, H) * (1 - math.cos(a_max))

        def base_point(uk, bk):
            if cones:
                return c0 + np.stack([R * uk * np.cos(bk), R * uk * np.sin(bk), H * (1 - uk)], -1)
            a = uk * a_max
            return c0 + np.stack([R * np.sin(a) * np.cos(bk), R * np.sin(a) * np.sin(bk), H * np.cos(a)], -1)

        n1 = max(3, int(p.florets))
        # Caps of the curd overlap (merged lobes); cones meet at their bases (sharp valleys between them).
        # Every order includes a meristem at the centre of its parent, so each lobe or cone keeps its tip.
        overlap = 1.0 if cones else 1.3
        k = np.arange(n1) + (0.0 if cones else 0.5)
        if cones:
            uk = np.sqrt(k / n1)
        else:
            uk = 2.0 / a_max * np.arcsin(np.sqrt(k / n1) * math.sin(a_max / 2))
        bk = k * GOLDEN + rng.uniform(0, 6.28)
        seeds = [base_point(uk, bk)]
        if cones:
            sn = np.stack([H * np.cos(bk), H * np.sin(bk), np.full(len(bk), R)], -1)
        else:
            a = uk * a_max
            sn = np.stack([np.sin(a) / R * np.cos(bk), np.sin(a) / R * np.sin(bk), np.cos(a) / H], -1)
        normals = [_unit(sn)]                     # Each bump grows along its meristem's axis
        radii = [math.sqrt(area / (math.pi * n1)) * overlap]
        for lv in range(1, max(1, p.head_levels)):
            r_prev = radii[-1]
            r_new = max(r_prev * p.floret_scale, 0.0008)
            if p.head_type == HeadType.BUDS and lv == p.head_levels - 1:
                r_new = max(p.bud_size_mm * 0.001, 0.0005)
            m = max(3, int(round(overlap * 1.1 * (0.85 * r_prev / r_new) ** 2)))
            kk = np.arange(m) + (0.0 if cones else 0.5)
            f = np.sqrt(kk / m)
            ang = kk * GOLDEN
            out, out_n = [], []
            for sd, nrm in zip(seeds[-1], normals[-1]):
                e1 = _unit(np.cross(nrm, UP) if abs(nrm @ UP) < 0.95 else np.cross(nrm, [1.0, 0, 0]))
                e2 = np.cross(nrm, e1)
                out.append(sd + (r_prev * 0.85 * f)[:, None] * (np.cos(ang)[:, None] * e1 + np.sin(ang)[:, None] * e2))
                # Children lean outward from their parent's axis, like the branches of the curd
                spread = (np.cos(ang)[:, None] * e1 + np.sin(ang)[:, None] * e2) * (0.6 * f)[:, None]
                out_n.append(_unit(nrm[None, :] + spread))
            seeds.append(np.concatenate(out))
            normals.append(np.concatenate(out_n))
            radii.append(r_new)
        # Height fields: union of caps (or cones) per order, summed over orders
        Q = P.reshape(-1, 3)
        h = np.zeros(len(Q))
        disp = np.zeros_like(Q)
        shape_amp = {HeadType.CURD: 0.55, HeadType.BUDS: 0.45, HeadType.CONES: 1.4}[p.head_type]
        last = len(seeds) - 1
        for lv, (S, SN, r) in enumerate(zip(seeds, normals, radii)):
            d, idx = nearest_points(Q, S, r * 1.05)
            x = np.clip(d / r, 0, 1)
            amp = shape_amp * r * (1.3 if cones and lv == 0 else 1.0)
            if cones:
                bump = np.where(np.isfinite(d), 1 - x, 0.0)
            else:
                bump = np.where(np.isfinite(d), np.sqrt(np.clip(1 - x * x, 0, 1)), 0.0)
            if lv == last and lv > 0 and p.head_type != HeadType.CONES:
                amp = (0.9 if p.head_type == HeadType.BUDS else 0.8) * r      # Granules / buds stand out
            h += amp * bump
            disp += (amp * bump)[:, None] * np.where((idx >= 0)[:, None], SN[np.maximum(idx, 0)], N.reshape(-1, 3))
        rim_fade = _ss(1.0, 0.85, U.reshape(-1))                  # The rim joins the base without a step
        h *= 0.4 + 0.6 * rim_fade
        disp *= (0.4 + 0.6 * rim_fade)[:, None]
        V = Q + disp
        hn = h / max(h.max(), 1e-9)
        rows, cols = len(u), nb + 1
        j, i = np.meshgrid(np.arange(rows - 1), np.arange(nb), indexing="ij")
        i0 = (j * cols + i).ravel()
        quads = np.stack([i0, i0 + 1, i0 + cols + 1, i0 + cols], axis=1)
        # Apex fan (one pole vertex per triangle)
        apex = len(V)
        apex_pt = c0 + np.array([0.0, 0.0, H + (h[:cols].mean())])
        Vs = [V, np.repeat(apex_pt[None, :], nb, 0)]
        fan = np.stack([apex + np.arange(nb), np.arange(nb), np.arange(nb) + 1], axis=1)
        # Underside: from the rim back to a ring around the stem top (branch colour)
        rim = V[(rows - 1) * cols:(rows) * cols]
        r_in = max(stem_radius, 0.003) * 0.95             # The underside closes onto the stem top
        base = np.asarray(stem_top, float) if stem_top is not None else c0 - np.array([0.0, 0.0, H * 0.15])
        under = []
        for s in np.linspace(0.0, 1.0, 6)[1:]:
            ring = np.stack([base[0] + (rim[:, 0] - c0[0]) * (1 - s) + r_in * s * np.cos(b),
                             base[1] + (rim[:, 1] - c0[1]) * (1 - s) + r_in * s * np.sin(b),
                             rim[:, 2] * (1 - s) + base[2] * s - 0.05 * H * math.sin(math.pi * s)], -1)
            under.append(ring)
        U2 = np.concatenate(under)
        off = len(V) + nb
        uq = []
        prev = (rows - 1) * cols
        for r_i in range(len(under)):
            cur = off + r_i * cols
            jj = np.arange(nb)
            uq.append(np.stack([prev + jj, prev + jj + 1, cur + jj + 1, cur + jj], 1))
            prev = cur
        uq = np.concatenate(uq)
        verts = np.concatenate(Vs + [U2])
        lv = np.concatenate([quads.reshape(-1), fan.reshape(-1), uq.reshape(-1)]).astype(np.int32)
        lt = np.concatenate([np.full(len(quads), 4), np.full(nb, 3), np.full(len(uq), 4)]).astype(np.int32)
        ls = np.concatenate([[0], np.cumsum(lt)[:-1]]).astype(np.int32)
        uvs = np.concatenate([np.stack([Bt.reshape(-1) / (2 * math.pi), U.reshape(-1)], 1), np.zeros((nb, 2)),
                              np.zeros((len(U2), 2))])
        n = len(verts)
        veg_h = np.concatenate([hn, np.ones(nb), np.zeros(len(U2))])
        branch = np.concatenate([np.zeros(len(V) + nb), np.ones(len(U2))])
        head = MeshData(verts.astype(np.float32), lv, ls, lt, uvs[lv].astype(np.float32),
                        {"veg_h": veg_h.astype(np.float32), "veg_branch": branch.astype(np.float32)})
        parts = [head]
        # Broccoli: the first-order branches fan out under the dome
        if p.head_type == HeadType.BUDS:
            top = base + np.array([0.0, 0.0, -H * 0.05])
            for sd in seeds[0]:
                tip = c0 + (sd - c0) * 0.75 - np.array([0, 0, H * 0.1])
                if tip[2] > c0[2] + H * 0.55:
                    continue
                P2 = np.linspace(top, tip, 5)
                parts.append(tube(P2, np.linspace(radii[0] * 0.28, radii[0] * 0.18, 5), 6,
                                  {"veg_h": np.zeros(5), "veg_branch": np.ones(5)}))
        return _merge(parts, ("veg_h", "veg_branch"))

    # ----------------------------------------------------------------- shoot
    def generate(self, seed: int = 0, detail: float = 1.0, leaf_density: float = 1.0,
                 with_roots: bool = True, lift: float = 0.0) -> VegetableResult:
        p = self.p
        rng = np.random.default_rng(seed)
        card = self.leaf_engine.generate_3d_leaf_mesh()
        root = MeshData.empty()
        crown = 0.0
        if p.organ == StorageOrgan.TAPROOT and with_roots:
            root, crown = self.taproot(np.random.default_rng(seed + 1), detail)
        elif p.organ == StorageOrgan.TAPROOT:
            crown = p.exposure * p.root_length_cm * 0.01
        # Stems
        stems, bases, tops, stem_axes = [], [], [], []
        n_st = max(1, p.stem_count)
        H = p.stem_height_cm * 0.01
        for k in range(n_st):
            a = 2 * math.pi * k / n_st + rng.normal(0, 0.3)
            lean = math.radians(rng.uniform(15, 40)) if n_st > 1 else math.radians(rng.uniform(0, 4))
            d = _unit(np.array([math.cos(a) * math.sin(lean), math.sin(a) * math.sin(lean), math.cos(lean)]))
            base = np.array([math.cos(a), math.sin(a), 0.0]) * (0.02 if n_st > 1 else 0.0) + UP * crown
            bases.append(base)
            if H > 0.005:
                m = 12
                pts = [base]
                dd = d.copy()
                for _ in range(m):           # Leaning stems curve up again (negative gravitropism)
                    dd = _unit(dd + rng.normal(0, 0.04, 3) + (0.06 if n_st > 1 else 0.03) * UP)
                    pts.append(pts[-1] + dd * H / m)
                P = np.array(pts)
                r = p.stem_radius_mm * 0.001
                stems.append(tube(P, np.linspace(r, r * 0.55, len(P)), 8, {"veg_part": np.zeros(len(P))}))
                stem_axes.append(P)
                tops.append(P[-1])
            else:
                stem_axes.append(np.array([base, base + UP * 0.01]))
                tops.append(base)
        # Head on the main stem
        head = MeshData.empty()
        if p.organ == StorageOrgan.HEAD:
            Rh = 0.5 * p.head_diameter_cm * 0.01
            r_top = p.stem_radius_mm * 0.001 * 0.55
            head = self.head(np.random.default_rng(seed + 2), tops[0] + UP * Rh * 0.15, detail,
                             stem_top=tops[0] - UP * 0.005, stem_radius=r_top)
        # Tubers
        if p.organ == StorageOrgan.TUBERS and with_roots:
            root = self.tubers(np.random.default_rng(seed + 3), bases, detail)
        # Leaves: rosette (no stem) or spiral along each stem; brassica inner leaves curl over the head
        pos, ys, zs, sc, rnd = [], [], [], [], []
        rng_l = np.random.default_rng(seed + 4)
        head_r = 0.5 * p.head_diameter_cm * 0.01 if p.organ == StorageOrgan.HEAD else 0.0
        for k, P in enumerate(stem_axes):
            s = arc_length(P)
            T, N, B = frames(P)
            nl = p.leaf_count
            for i in range(nl):
                f = i / max(nl - 1, 1)                      # 0 oldest (outer / lowest) .. 1 youngest
                if H > 0.005:
                    si = s[-1] * (0.08 + 0.88 * f ** 0.9)
                else:
                    si = 0.0
                j = min(np.searchsorted(s, si), len(s) - 1)
                x = P[j]
                a = i * GOLDEN + k * 1.3
                radial = _unit(math.cos(a) * N[j] + math.sin(a) * B[j])
                h = radial - (radial @ UP) * UP
                h = _unit(h) if np.linalg.norm(h) > 1e-6 else np.array([1.0, 0.0, 0.0])
                elev = math.radians(p.elevation_outer_deg + (p.elevation_inner_deg - p.elevation_outer_deg) * f)
                if head_r > 0:
                    elev = elev + p.head_wrap * math.radians(25) * f
                y = _unit(h * math.cos(elev) + UP * math.sin(elev) + rng_l.normal(0, 0.06, 3))
                zpref = _unit(UP - h * (0.6 if head_r > 0 else 0.3))
                z = zpref - (zpref @ y) * y
                z = _unit(z if np.linalg.norm(z) > 1e-6 else np.cross(y, h))
                size = p.leaf_size * (0.55 + 0.45 * (1 - f) ** 0.6 if H <= 0.005 else 0.7 + 0.3 * math.sin(np.pi * f))
                if rng_l.random() > leaf_density:
                    continue
                r_here = p.stem_radius_mm * 0.001 * (1.0 - 0.45 * si / max(s[-1], 1e-9)) if H > 0.005 else 0.002
                offset = h * r_here * 0.8 - UP * (0.002 if H <= 0.005 else 0.0)
                pos.append(x + offset)
                ys.append(y)
                zs.append(z)
                sc.append(size * rng_l.uniform(0.85, 1.15))
                rnd.append(rng_l.random())
        foliage = FoliageInstances(np.array(pos).reshape(-1, 3), np.array(ys).reshape(-1, 3),
                                   np.array(zs).reshape(-1, 3), np.array(sc), np.array(rnd))
        leaves = BotanicalMeshEngine.build_foliage_mesh(foliage, card) if len(foliage) else MeshData.empty()
        stem_mesh = _merge(stems, ("veg_part",))
        if lift:
            shift = np.array([0.0, 0.0, lift], np.float32)
            for m in (root, stem_mesh, head, leaves):
                if len(m.vertices):
                    m.vertices = m.vertices + shift
            foliage.positions = foliage.positions + shift
        return VegetableResult(root, stem_mesh, head, leaves, foliage, self.leaf_engine,
                               {"leaves": len(foliage), "crown_m": crown})
