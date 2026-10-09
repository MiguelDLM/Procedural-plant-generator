"""
Fruits and infructescences.

One parametric body covers most fleshy fruits. Its outline follows the shape descriptors of Tomato Analyzer
(Brewer et al. 2006, Plant Physiology 141: 15-25): length and width, the position of the widest point (pear-
vs club-shaped), proximal and distal end shapes (blunt or pointed), the depth of the stalk cavity and of the
distal (calyx) basin, a neck (pears, bottle gourds), ribs or lobes, and lopsidedness. The persistent calyx
forms a crown at the distal end (pomegranate, apple, medlar) or a star on the stalk end (tomato, aubergine).

Kinds (Spjut 1994 terminology): pome (apple, pear: the flesh is the floral cup, sepals persist in the calyx
basin), drupe (cherry, olive), berry (grape, tomato), hesperidium (citrus: oil glands dot the rind), balausta
(pomegranate: leathery rind with a crown of calyx lobes) and pepo (cucurbits, see also the vines).

Grapes and other bunches are panicles of berries: a peduncle and rachis with laterals that shorten toward
the tip (a cone), the basal ones sometimes enlarged into shoulders (wings), and berries packed more or less
tightly (compactness, Tello & Ibáñez 2018, Aust. J. Grape Wine Res. 24: 6-23).

Fruit prototypes hang from their stalk: the stalk top is the origin and the fruit hangs along -Z, ready to
be instanced on the branches of trees and vines.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np

from .mesh_engine import MeshData

UP = np.array([0.0, 0.0, 1.0])
ATTRS = ("fruit_t", "fruit_u", "fruit_rib", "fruit_crown", "fruit_stalk", "fruit_random")


class FruitKind(str, Enum):
    BERRY = "Berry"              # Grape, tomato, blueberry
    DRUPE = "Drupe"              # Cherry, olive, plum, peach
    POME = "Pome"                # Apple, pear, quince
    HESPERIDIUM = "Hesperidium"  # Citrus
    BALAUSTA = "Balausta"        # Pomegranate
    PEPO = "Pepo"                # Pumpkin, cucumber, melon, watermelon
    POD = "Pod"                  # Legume pods


class CrownPosition(str, Enum):
    DISTAL = "Distal"            # Calyx at the blossom end (pome, pomegranate)
    PROXIMAL = "Proximal"        # Calyx on the stalk end (tomato, aubergine, persimmon)


@dataclass
class FruitProfile:
    """Shape and surface of a fleshy fruit (sizes in cm, along the stalk-to-blossom axis)."""
    kind: FruitKind = FruitKind.POME
    length_cm: float = 7.0              # Stalk end to blossom end
    diameter_cm: float = 7.5            # Largest width
    widest_position: float = 0.45       # 0 stalk end .. 1 blossom end (pear-shaped > 0.5)
    bluntness: float = 0.55             # End shape exponent: 0.3 boxy .. 1 pointed
    distal_point: float = 0.0           # Extra point at the blossom end (lemon, chilli)
    neck: float = 0.0                   # Constriction toward the stalk (pear, bottle gourd)
    stalk_cavity: float = 0.0           # Depth of the stalk cavity relative to the length (apple ~0.15)
    calyx_basin: float = 0.0            # Depth of the distal basin relative to the length (apple ~0.08)
    ribs: int = 0                       # Lobes / ribs (pumpkin 10, tomato 6, apple 5 faint)
    rib_depth: float = 0.0              # Groove depth relative to the radius
    lopsided: float = 0.0               # One side larger than the other (apples)

    # Persistent calyx
    crown_lobes: int = 0
    crown_length_cm: float = 0.5
    crown_flare_deg: float = 30.0       # 0 = along the axis (closed) .. 90 = spread flat over the fruit
    crown_position: CrownPosition = CrownPosition.DISTAL
    crown_color: tuple = (0.35, 0.28, 0.16)

    # Stalk
    stalk_length_cm: float = 2.5
    stalk_radius_mm: float = 1.2
    stalk_color: tuple = (0.38, 0.30, 0.18)

    # Bunch (infructescence); 1 = a single fruit
    cluster_berries: int = 1
    cluster_length_cm: float = 15.0     # Rachis length
    cluster_width_cm: float = 10.0      # Width at the shoulders
    shoulders: float = 0.3              # Enlarged basal laterals (wings)
    compactness: float = 0.6            # 0 loose, straggly .. 1 tightly packed
    pedicel_cm: float = 0.6
    peduncle_cm: float = 4.0

    # Surface (sRGB 0..1)
    color: tuple = (0.75, 0.70, 0.20)
    blush_color: tuple = (0.70, 0.08, 0.06)
    blush: float = 0.0                  # Sun-side over-colour (apples, peaches)
    stripes: float = 0.0                # Streaks / stripes along the fruit
    stripe_count: int = 14              # Around the fruit (apple streaks ~30, watermelon ~16)
    stripe_width: float = 0.5           # Share of each period covered: 0.15 fine streaks .. 0.5 broad bands
    stripe_color: tuple = (0.55, 0.05, 0.05)
    dots: float = 0.0                   # Lenticels, oil glands, speckles
    dot_color: tuple = (0.90, 0.85, 0.60)
    russet: float = 0.0                 # Corky brown patches (pears, some apples)
    bloom: float = 0.0                  # Waxy whitish bloom (grapes, plums, blueberries)
    gloss: float = 0.5


def _unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


# -----------------------------------------------------------------------------
# Single fruit body: +Z from the stalk end (origin) to the blossom end
# -----------------------------------------------------------------------------
def pole_depth(p: FruitProfile) -> float:
    """How far the stalk-end pole is sunk into the fruit (m): the stalk must reach it."""
    return p.stalk_cavity * p.length_cm * 0.01 * 0.97


def body_mesh(p: FruitProfile, detail: float = 1.0, seed: int = 0, length_scale: float = 1.0) -> MeshData:
    L = p.length_cm * 0.01 * length_scale
    D = p.diameter_cm * 0.01 * length_scale
    rng = np.random.default_rng(seed)
    nt = max(14, int(34 * detail))
    nth = max(12, int(max(24, p.ribs * 6) * detail))
    w = float(np.clip(p.widest_position, 0.15, 0.85))
    e_prox = max(p.bluntness, 0.05)
    e_dist = e_prox + 1.6 * p.distal_point

    def outline(t):
        tw = np.where(t < w, 0.5 * t / w, 0.5 + 0.5 * (t - w) / (1 - w))
        rho = 0.5 * D * np.where(t < w, np.sin(np.pi * tw) ** e_prox, np.sin(np.pi * tw) ** e_dist)
        rho = rho * (1.0 - p.neck * 0.75 * np.exp(-((t - 0.22) / 0.16) ** 2))
        # Stalk cavity and calyx basin: dimples around the axis (a function of the distance to the axis, so
        # the shoulders keep their shape); the profile curls back inward at each end
        rn = rho / max(0.5 * D, 1e-9)
        z = L * t + np.where(t < 0.5, p.stalk_cavity * L * np.exp(-(rn / 0.32) ** 2),
                             -p.calyx_basin * L * np.exp(-(rn / 0.28) ** 2))
        return rho, z
    # Rings evenly spaced along the outline (not along the axis): blunt ends and cavities get as many rings
    # as the flanks, so the poles close with small triangles
    tf = np.linspace(0.0, 1.0, 4001)
    rf, zf = outline(tf)
    sl = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(rf), np.diff(zf)))])
    t = np.interp(np.linspace(0.0, sl[-1], nt + 1)[1:-1], sl, tf)
    rho, z = outline(t)
    th = np.linspace(0.0, 2 * math.pi, nth + 1)
    if p.ribs > 0:
        rib = np.abs(np.sin(p.ribs * th / 2.0))
        groove = 1.0 - p.rib_depth * (1.0 - rib) ** 3
    else:
        rib = np.ones_like(th)
        groove = np.ones_like(th)
    phase = rng.uniform(0, 2 * math.pi)
    lop = 1.0 + p.lopsided * np.cos(th - phase)[None, :] * np.sin(np.pi * t)[:, None]
    r = rho[:, None] * groove[None, :] * lop
    V = np.stack([r * np.cos(th)[None, :], r * np.sin(th)[None, :], np.repeat(z[:, None], nth + 1, 1)], -1)
    V = V.reshape(-1, 3)
    m, rows = nth + 1, len(t)
    j, i = np.meshgrid(np.arange(rows - 1), np.arange(nth), indexing="ij")
    i0 = (j * m + i).ravel()
    quads = np.stack([i0, i0 + m, i0 + m + 1, i0 + 1], axis=1)
    # Poles: one vertex per fan triangle (with the mid angle of its sector) so attributes converge cleanly
    top, bot = len(V), len(V) + nth
    pz0 = pole_depth(p) * length_scale
    pz1 = L - p.calyx_basin * L * 0.97
    um = (th[:-1] + th[1:]) / (4 * math.pi)
    poles = np.concatenate([np.repeat([[0.0, 0.0, pz0]], nth, 0), np.repeat([[0.0, 0.0, pz1]], nth, 0)])
    ring0, ringN = np.arange(nth), (rows - 1) * m + np.arange(nth)
    fan0 = np.stack([top + ring0, ring0 + 1, ring0], axis=1)
    fanN = np.stack([ringN, ringN + 1, bot + ring0], axis=1)
    verts = np.concatenate([V, poles])
    tt = np.concatenate([np.repeat(t[:, None], m, 1).reshape(-1), np.zeros(nth), np.ones(nth)])
    uu = np.concatenate([np.repeat((th / (2 * math.pi))[None, :], rows, 0).reshape(-1), um, um])
    rr = np.concatenate([np.repeat(rib[None, :], rows, 0).reshape(-1), np.ones(2 * nth)])
    lv = np.concatenate([quads.reshape(-1), fan0.reshape(-1), fanN.reshape(-1)]).astype(np.int32)
    lt = np.concatenate([np.full(len(quads), 4), np.full(2 * nth, 3)]).astype(np.int32)
    ls = np.concatenate([[0], np.cumsum(lt)[:-1]]).astype(np.int32)
    n = len(verts)
    body = MeshData(verts.astype(np.float32), lv, ls, lt, np.stack([uu, tt], 1)[lv].astype(np.float32),
                    {"fruit_t": tt, "fruit_u": uu, "fruit_rib": rr, "fruit_crown": np.zeros(n),
                     "fruit_stalk": np.zeros(n), "fruit_random": np.zeros(n)})
    parts = [body]
    if p.crown_lobes > 0 and p.crown_length_cm > 0:
        parts.append(_crown(p, L, rho, t, z, length_scale, rng))
    return _concat(parts)


def _crown(p: FruitProfile, L, rho, t, z, k, rng) -> MeshData:
    """Persistent calyx lobes: tapering strips from a ring near the pole, flaring outward."""
    n = p.crown_lobes
    lobe = p.crown_length_cm * 0.01 * k
    distal = p.crown_position == CrownPosition.DISTAL
    idx = -3 if distal else 2
    base_r = float(rho[idx]) * (0.55 if distal else 0.9)
    base_z = float(z[idx]) if distal else float(z[idx]) - 0.002
    axis = UP if distal else -UP
    flare = math.radians(p.crown_flare_deg)
    if not distal:                                  # Calyx star lies over the shoulders of the fruit
        flare = math.radians(90.0 + 0.4 * p.crown_flare_deg)
    segs = 4
    parts = []
    for i in range(n):
        a = 2 * math.pi * i / n + rng.normal(0, 0.08)
        radial = np.array([math.cos(a), math.sin(a), 0.0])
        tang = np.array([-math.sin(a), math.cos(a), 0.0])
        root = radial * base_r + UP * base_z
        width = 2 * math.pi * base_r / n * 0.8 + 0.25 * lobe
        pts, wid = [], []
        for s in np.linspace(0.0, 1.0, segs + 1):
            ang = flare * (0.6 + 0.6 * s)               # Lobes curl outward toward their tips
            d = _unit(axis * math.cos(ang) + radial * math.sin(ang))
            pts.append(root + d * lobe * s if s == 0 else pts[-1] + d * lobe / segs)
            wid.append(width * (1.0 - s) ** 0.8 * 0.5)
        P = np.array(pts)
        W = np.array(wid)
        V = np.concatenate([P - tang * W[:, None], P + tang * W[:, None]])
        q = []
        for s in range(segs):
            q.append([s, s + 1, segs + 2 + s, segs + 1 + s])
        q = np.array(q, np.int32)
        m = len(V)
        parts.append(MeshData(V.astype(np.float32), q.reshape(-1), np.arange(len(q), dtype=np.int32) * 4,
                              np.full(len(q), 4, np.int32), np.zeros((len(q) * 4, 2), np.float32),
                              {"fruit_t": np.full(m, 1.0 if distal else 0.0), "fruit_u": np.zeros(m),
                               "fruit_rib": np.ones(m), "fruit_crown": np.ones(m), "fruit_stalk": np.zeros(m),
                               "fruit_random": np.zeros(m)}))
    return _concat(parts)


def _concat(parts) -> MeshData:
    parts = [q for q in parts if len(q.vertices)]
    for q in parts:
        for k in ATTRS:
            q.point_attributes.setdefault(k, np.zeros(len(q.vertices)))
            q.point_attributes[k] = np.asarray(q.point_attributes[k], np.float32)
    return MeshData.concatenate(parts)


def _tube(P, R, sides=6, attrs=None) -> MeshData:
    from .vine import tube
    m = tube(P, R, sides, attrs)
    return m


def _stalk_attrs(n):
    return {"fruit_t": np.zeros(n), "fruit_u": np.zeros(n), "fruit_rib": np.ones(n), "fruit_crown": np.zeros(n),
            "fruit_stalk": np.ones(n), "fruit_random": np.zeros(n)}


def _transform(m: MeshData, R, t, rnd=None) -> MeshData:
    V = m.vertices.astype(float) @ np.asarray(R).T + np.asarray(t)[None, :]
    attrs = {k: v.copy() for k, v in m.point_attributes.items()}
    if rnd is not None:
        attrs["fruit_random"] = np.full(len(V), rnd, np.float32)
    return MeshData(V.astype(np.float32), m.loop_vertex.copy(), m.loop_start.copy(), m.loop_total.copy(),
                    m.loop_uv.copy(), attrs)


def _frame_to(z, roll=0.0):
    """Rotation whose third column is z."""
    z = _unit(np.asarray(z, float))
    e1 = _unit(np.cross(z, UP) if abs(z @ UP) < 0.95 else np.cross(z, [1.0, 0.0, 0.0]))
    e2 = np.cross(z, e1)
    c, s = math.cos(roll), math.sin(roll)
    e1, e2 = c * e1 + s * e2, -s * e1 + c * e2
    return np.stack([e1, e2, z], axis=1)


# -----------------------------------------------------------------------------
# Hanging prototypes (stalk top at the origin)
# -----------------------------------------------------------------------------
def hanging_fruit(p: FruitProfile, detail: float = 1.0, seed: int = 0) -> MeshData:
    """A fruit or a bunch hanging from its stalk; the stalk top is the origin."""
    if p.cluster_berries > 1:
        return bunch(p, detail, seed)
    rng = np.random.default_rng(seed)
    body = body_mesh(p, detail, seed)
    stalk = p.stalk_length_cm * 0.01
    sink = pole_depth(p) + 0.002
    # Fruit axis points down; its stalk end (origin of the body) sits at the end of the stalk
    R = _frame_to(-UP + rng.normal(0, 0.05, 3), rng.uniform(0, 2 * math.pi))
    end = np.array([0.0, 0.0, -stalk])
    parts = [_transform(body, R, end - R[:, 2] * 0.0)]
    # Stalk: gently curved from the origin to the stalk end, then straight along the fruit axis to the bottom
    # of the cavity, so even thick stalks (pumpkin) enter the fruit and do not end beside it
    tip = end + R[:, 2] * sink
    bend = rng.normal(0, 0.15, 3) * stalk
    bend[2] = 0
    s = np.linspace(0, 1, 10)[:, None]
    ctrl = end - R[:, 2] * stalk * 0.45 + bend * 0.5      # Tangent along the fruit axis at the stalk end
    P = (1 - s) ** 2 * np.zeros(3) + 2 * (1 - s) * s * ctrl + s ** 2 * end
    P = np.concatenate([P, end + np.outer(np.linspace(0.25, 1.0, 4), tip - end)])
    r = p.stalk_radius_mm * 0.001
    parts.append(_tube(P, np.linspace(r * 0.9, r * 1.2, len(P)), 6, _stalk_attrs(len(P))))
    return _concat(parts)


def bunch(p: FruitProfile, detail: float = 1.0, seed: int = 0) -> MeshData:
    """A panicle of berries (grape bunch): peduncle and rachis down -Z, laterals shortening toward the tip,
    shoulders at the base, berries packed in a cone by dart throwing with a compactness-dependent spacing."""
    rng = np.random.default_rng(seed)
    n = int(max(1, p.cluster_berries))
    Lr = max(p.cluster_length_cm * 0.01, 0.01)
    W = max(p.cluster_width_cm * 0.01, 0.005)
    pd = p.peduncle_cm * 0.01
    bd = p.diameter_cm * 0.01
    spacing = bd * (1.25 - 0.3 * np.clip(p.compactness, 0, 1))
    z0 = -pd

    def radius_at(h):              # h: 0 top of the rachis .. 1 tip
        cone = 0.5 * W * (1.0 - h) ** 0.85 + bd * 0.4
        wing = p.shoulders * 0.5 * W * np.exp(-((h - 0.12) / 0.12) ** 2)
        return cone + wing

    centers, tries = [], 0
    while len(centers) < n and tries < n * 400:
        tries += 1
        h = rng.random() ** 0.8
        rr = radius_at(h) * math.sqrt(rng.random())
        a = rng.uniform(0, 2 * math.pi)
        c = np.array([rr * math.cos(a), rr * math.sin(a), z0 - p.pedicel_cm * 0.01 - h * Lr])
        if all(np.linalg.norm(c - q) > spacing for q in centers[-400:]):
            centers.append(c)
    C = np.array(centers)
    parts = []
    stem_r = max(p.stalk_radius_mm * 0.001, 0.0008)
    rachis = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, z0], [0.0, 0.0, z0 - Lr * 0.95]])
    rachis = np.concatenate([rachis[:1], np.linspace(rachis[1], rachis[2], 8)])
    parts.append(_tube(rachis, np.linspace(stem_r * 1.4, stem_r * 0.6, len(rachis)), 6, _stalk_attrs(len(rachis))))
    # Laterals: berries grouped by height; each group hangs from a lateral branching off the rachis
    k = max(1, int(round(math.sqrt(len(C)))))
    order = np.argsort(-C[:, 2])
    groups = np.array_split(order, k)
    proto = body_mesh(p, max(0.35, 0.4 * detail), seed, length_scale=1.0)
    for g in groups:
        if len(g) == 0:
            continue
        G = C[g]
        cen = G.mean(0)
        zr = float(np.clip(cen[2] + bd, z0 - Lr * 0.95, z0))     # On the rachis
        j0 = np.array([0.0, 0.0, zr])
        j1 = j0 + (np.array([cen[0], cen[1], 0.0])) * 0.65 + np.array([0, 0, -bd * 0.3])
        lat = np.linspace(j0, j1, 4)
        parts.append(_tube(lat, np.linspace(stem_r * 0.8, stem_r * 0.5, 4), 5, _stalk_attrs(4)))
        for c in G:
            d = _unit(c - j1 + np.array([0, 0, 1e-6]))
            top = c - d * bd * 0.45 + UP * (p.pedicel_cm * 0.01 * 0.2)
            ped = np.linspace(j1, top, 4)
            parts.append(_tube(ped, np.full(4, stem_r * 0.45), 4, _stalk_attrs(4)))
            # Berry axis along the pedicel direction, stalk end at the pedicel tip
            R = _frame_to(_unit(c - top), rng.uniform(0, 2 * math.pi))
            parts.append(_transform(proto, R, top, rng.random()))
    return _concat(parts)
