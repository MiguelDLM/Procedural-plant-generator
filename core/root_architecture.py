"""
Fine root architecture: lateral roots grown on the coarse root frame of any growth form.

Botanical root-system types (Cannon 1949, revised by Zobel & Waisel 2010; Kutschera & Lichtenegger root
atlases) and where the generator builds them:

    Taproot / allorhizic   a dominant primary root with laterals (dicots, gymnosperms)      core.roots TAPROOT
    Heart                  several oblique roots of similar girth, hemispherical "cage"     core.roots HEART
    Plate / sinker         shallow laterals carrying vertical sinkers (Picea, Populus)      core.roots PLATE
    Buttress               plank roots continuing up the stem (tropical trees)              core.roots BUTTRESS
    Stilt / prop           shoot-borne roots arching from the lower stem into the soil      core.roots STILT
                           (Rhizophora, Pandanus, banyan figs; maize brace roots)
    Fibrous / fasciculate  homorhizic: many shoot-borne roots of similar girth (monocots:   core.roots FIBROUS,
                           palms, agaves, grasses, orchids)                                 core.grass, core.orchid
    Napiform / conical /   storage taproots (turnip, carrot, radish; Lophophora)            core.roots TUBEROUS,
    fusiform                                                                                core.vegetable
    Tuberous cluster       fasciculate storage roots: swollen adventitious roots from the   core.roots TUBER_CLUSTER,
                           stem base (Dahlia, sweet potato, cassava, Peniocereus striatus)  core.vegetable
    Aerial / velamen       epiphytic and climbing roots (orchids, ivy)                      core.orchid, core.vine

Lateral roots follow the structural rules of root architecture models:

- CRootBox / CPlantBox (Schnepf et al. 2018, Ann. Bot. 121: 1033; Leitner et al. 2010, Plant Soil 332:
  177): each root type has an unbranched basal zone (lb), branched zone with an inter-branch distance (ln),
  an unbranched apical zone (la), a maximal length, an insertion angle and a tropism (gravitropism strength
  plus a random deflection, sigma). Laterals are placed at ln along the mother root and grown by small steps
  whose direction is pulled by gravity and perturbed at random.
- ArchiSimple (Pages et al. 2014, Ecol. Model. 290: 76): root developmental traits scale with tip
  diameter; a lateral's diameter is a fraction of its mother's (dIDm ~0.3-0.5), its length grows with its
  diameter, and the finest roots (Dmin) are short.
- Desert cacti: primary and lateral roots of Sonoran cacti grow determinately (Dubrovsky 1997, Planta 203:
  85; Shishkova et al. 2013, Ann. Bot. 112: 239) and short, dense "rain roots" emerge from the laterals after
  rainfall (Nobel 1988; Snyman 2005), so cactus laterals are short and close together.
- Shape of the whole system by growth form (Schenk & Jackson 2002, J. Ecol. 90: 480): the ratio of lateral
  spread to rooting depth is ~4.5 for stem succulents (Cactaceae mean depth 0.29 m), ~3 for trees, ~1 for
  shrubs, ~0.5 for semi-shrubs and 0.3-0.35 for herbaceous plants (grasses ~1 m deep).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

UP = np.array([0.0, 0.0, 1.0])


@dataclass
class RootBranching:
    """Lateral root rules (one set, scaled for each successive order)."""
    orders: int = 2                  # Lateral orders grown on the coarse frame (0 = none)
    interbranch_cm: float = 8.0      # ln: distance between first-order laterals along a mother root
    basal_zone: float = 0.08         # lb: unbranched share of the mother root near its base
    apical_zone_cm: float = 6.0      # la: unbranched zone behind the mother's tip
    insertion_deg: float = 65.0      # Insertion angle between lateral and mother root
    length_ratio: float = 0.3        # Lateral length relative to the remaining mother length
    max_length_cm: float = 120.0     # Longest first-order lateral
    radius_ratio: float = 0.4        # Lateral / mother diameter (ArchiSimple dIDm)
    gravitropism: float = 0.25       # + bends down, 0 keeps its direction (plagiotropic), - bends up
    tortuosity: float = 0.35         # Random deflection per unit length (sigma)
    determinate_cm: float = 0.0      # > 0: laterals stop at this length (desert cacti rain roots)
    min_radius_mm: float = 0.4       # Finest roots drawn (Dmin)
    budget: int = 1200               # Maximum number of lateral roots


def _unit(v):
    v = np.asarray(v, float)
    return v / max(float(np.linalg.norm(v)), 1e-12)


def _perp(d):
    p = np.cross(d, UP)
    if np.linalg.norm(p) < 1e-6:
        p = np.cross(d, [1.0, 0.0, 0.0])
    return _unit(p)


def _rotate(v, axis, ang):
    axis = _unit(axis)
    c, s = math.cos(ang), math.sin(ang)
    return v * c + np.cross(axis, v) * s + axis * (axis @ v) * (1 - c)


def grow_root(start, d0, length, r0, rng, gravitropism=0.25, tortuosity=0.35, step=None, surface=0.0,
              floor=None, taper=0.5, aerial=False):
    """Root axis grown by small steps: direction pulled toward gravity (gravitropism > 0), away from it
    (< 0), and deflected at random (tortuosity). Stays below `surface` unless aerial; stops at `floor`.
    Returns points (n, 3) and radii (n,)."""
    step = step or float(np.clip(length / 10.0, 0.004, 0.12))
    n = max(2, int(math.ceil(length / step)))
    d = _unit(d0)
    pts = [np.asarray(start, float)]
    for i in range(n):
        d = _unit(d - UP * gravitropism * 0.12 + rng.normal(0.0, tortuosity * 0.25, 3))
        nxt = pts[-1] + d * length / n
        if not aerial and nxt[2] > surface - r0:
            nxt[2] = surface - r0
            d[2] = -abs(d[2]) - 0.05
            d = _unit(d)
        if floor is not None and nxt[2] < floor:
            pts.append(nxt)
            break
        pts.append(nxt)
    P = np.array(pts)
    t = np.linspace(0.0, 1.0, len(P))
    R = r0 * (1.0 - taper * t)
    return P, R


def lateral_roots(mothers, br: RootBranching, rng, surface=0.0, floor=None, start_order=1):
    """Laterals (and their laterals) on mother roots given as (points, radii). Returns a list of
    (points, radii, order, mother index, sample index) with mother indices into `mothers` + the output."""
    out = []
    if br.orders <= 0 or br.budget <= 0:
        return out
    queue = [(np.asarray(P, float), np.asarray(R, float), start_order - 1, -1 - i) for i, (P, R) in enumerate(mothers)]
    rmin = br.min_radius_mm * 0.001
    q = 0
    while q < len(queue) and len(out) < br.budget:
        P, R, order, mid = queue[q]
        q += 1
        if order - (start_order - 1) >= br.orders or len(P) < 3:
            continue
        seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(seg)])
        L = s[-1]
        k_ord = order - (start_order - 1)                      # 0 for laterals on the coarse frame
        ln = min(br.interbranch_cm * 0.01 * (0.55 ** k_ord), L / 5.0)    # Short roots branch too
        la = br.apical_zone_cm * 0.01 * (0.6 ** k_ord)
        s0, s1 = max(br.basal_zone * L, 0.01), L - la
        if s1 <= s0 or ln <= 0:
            continue
        z = s0 + rng.uniform(0, ln)
        while z < s1 and len(out) < br.budget:
            i = min(int(np.searchsorted(s, z)), len(P) - 2)
            x = np.array([np.interp(z, s, P[:, c]) for c in range(3)])
            r_m = float(np.interp(z, s, R))
            if x[2] > surface - r_m:                              # No laterals in the air (stilts, collar)
                z += ln
                continue
            r = r_m * br.radius_ratio * rng.uniform(0.8, 1.2)
            if r < rmin:
                break
            t = _unit(P[i + 1] - P[i])
            ang = math.radians(br.insertion_deg) * rng.uniform(0.8, 1.15)
            side = _rotate(_perp(t), t, rng.uniform(0, 2 * math.pi))
            d0 = _unit(t * math.cos(ang) + side * math.sin(ang))
            rem = L - z
            ll = min(br.length_ratio * rem * rng.uniform(0.6, 1.3), br.max_length_cm * 0.01 * (0.4 ** k_ord))
            if br.determinate_cm > 0:
                ll = min(ll, br.determinate_cm * 0.01 * rng.uniform(0.5, 1.2) * (0.6 ** k_ord))
            ll = min(ll, 900 * r)                                # Thin roots stay short (ArchiSimple)
            if ll > 0.005:
                Pl, Rl = grow_root(x - t * 0.0 + side * r_m * 0.3, d0, ll, r, rng, br.gravitropism, br.tortuosity,
                                   surface=surface, floor=floor, taper=0.45)
                out.append((Pl, Rl, order + 1, mid, i))
                queue.append((Pl, Rl, order + 1, len(out) - 1))
            z += ln * rng.uniform(0.6, 1.4)
    return out


def root_tubes(axes, sides=4, attrs=None):
    """Meshes of root axes as thin tubes (vine.tube). `attrs(order, n)` returns the per-point attribute
    dict of an axis."""
    from .vine import tube
    from .mesh_engine import MeshData
    parts = []
    for P, R, order, *_ in axes:
        if len(P) < 2:
            continue
        parts.append(tube(P, np.maximum(R, 1e-4), sides, attrs(order, len(P)) if attrs else None))
    return MeshData.concatenate(parts) if parts else MeshData.empty()
