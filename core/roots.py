"""
Coarse root system architecture.

Root system types follow the classical classification of forest-tree root
systems (Köstler, Brückner & Bibelriether 1968, "Die Wurzeln der Waldbäume"):

    TAPROOT   a dominant vertical root plus laterals (Quercus robur, Pinus, Juglans)
    HEART     several oblique roots of similar girth from the stump (Fagus, Tilia, Betula, Acer)
    PLATE     shallow horizontal laterals carrying vertical sinkers (Picea, Populus, Salix)
    BUTTRESS  superficial laterals continuing up the stem as plank buttresses, with sinkers
              at their ends (Crook, Ennos & Banks 1997, J. Exp. Bot. 48: 1703)
    FIBROUS   many adventitious roots of near-constant girth from the stem base (palms)

Quantitative rules:
- Root collar: the stem cross-section is shared among the main roots following
  Leonardo's rule (r_stem^D = sum r_i^D), which holds in coarse root systems
  (Oppelt, Kurth & Godbold 2001, Tree Physiol. 21: 117).
- Zone of rapid taper (ZRT): main roots lose most of their girth within a radius
  of about 2.2 x DBH; beyond it they taper mainly by giving off branches.
- Vertical distribution: cumulative root fraction Y(d) = 1 - beta^d (d in cm),
  with beta fitted per biome by Jackson et al. (1996, Oecologia 108: 389), e.g.
  boreal forest 0.943, temperate deciduous 0.966, temperate coniferous 0.976,
  tropical evergreen 0.962. Depths of laterals and sinkers are sampled from it.
- Maximum rooting depth by vegetation type (Canadell et al. 1996, Oecologia
  108: 583): trees 7.0 +- 1.2 m on average; temperate coniferous forest 3.9 m.
- Lateral spread is expressed relative to crown radius; laterals commonly
  extend well beyond the drip line.
- Near the stem, main laterals have vertically elongated (plank / I-beam like)
  cross-sections; in buttressed species this is the buttress itself.
"""

from dataclasses import dataclass
from enum import Enum
import math
import numpy as np

from .architecture import Axis, BranchingGraph, refine_base


class RootSystemType(str, Enum):
    TAPROOT = "Taproot"
    HEART = "Heart"
    PLATE = "Plate"
    BUTTRESS = "Buttress"
    FIBROUS = "Fibrous"


# Jackson et al. (1996), Table 1: fitted depth coefficient beta per biome
JACKSON_BETA = {
    "boreal_forest": 0.943,
    "temperate_coniferous": 0.976,
    "temperate_deciduous": 0.966,
    "temperate_grassland": 0.943,
    "sclerophyllous_shrubs": 0.964,
    "tropical_deciduous": 0.961,
    "tropical_evergreen": 0.962,
    "tropical_savanna": 0.972,
    "desert": 0.975,
    "tundra": 0.914,
}


@dataclass
class RootProfile:
    system: RootSystemType = RootSystemType.HEART
    lateral_count: int = 6            # Main structural laterals (= buttress count when buttressed)
    spread_crown_ratio: float = 1.4   # Lateral reach / crown radius
    max_depth_m: float = 3.0          # Species/biome maximum rooting depth
    beta: float = 0.966               # Jackson vertical distribution coefficient
    taproot_share: float = 0.15       # Fraction of the collar pipe flow taken by the taproot / obliques
    zrt_dbh_ratio: float = 2.2        # Zone of rapid taper radius / DBH
    sinker_spacing_m: float = 1.2     # Mean distance between sinkers along laterals
    branch_spacing_m: float = 0.9     # Mean distance between second-order laterals
    surface_exposure: float = 0.15    # 0 buried .. 1 laterals ride on the soil surface near the stem
    plank: float = 0.5                # Vertical elongation of lateral cross-sections near the stem
    buttress_height_dbh: float = 0.6  # Height on the stem where laterals/buttresses merge (x DBH)
    tortuosity: float = 0.25
    knees: int = 0                    # Pneumatophores ("cypress knees") on shallow laterals
    fibrous_count: int = 60           # Adventitious roots (FIBROUS)
    fibrous_radius_m: float = 0.006


def sample_depth(beta: float, u: np.ndarray | float) -> np.ndarray:
    """Inverse of Y(d) = 1 - beta^d: depth (m) below which a fraction (1 - u) of roots lies."""
    u = np.clip(np.asarray(u, dtype=float), 1e-4, 0.9999)
    return np.log(1.0 - u) / math.log(beta) / 100.0


class RootSystemEngine:
    """Generates coarse root axes attached to the stem base."""

    MIN_RADIUS = 0.003

    def __init__(self, profile: RootProfile = None):
        self.profile = profile or RootProfile()

    def generate(self, dbh_m: float, crown_radius_m: float, collar_radius_m: float, pipe_delta: float = 2.3,
                 seed: int = 42, display_depth_m: float = 3.0, azimuths: np.ndarray | None = None) -> BranchingGraph:
        prof = self.profile
        rng = np.random.default_rng(seed + 977)
        graph = BranchingGraph()
        self._delta = pipe_delta
        self._display = max(0.2, display_depth_m)
        R = collar_radius_m
        dbh = dbh_m
        if prof.system == RootSystemType.FIBROUS:
            self._fibrous(graph, R, rng)
            return graph

        n = max(1, int(prof.lateral_count))
        weights = rng.lognormal(0.0, 0.35, n)
        n_main = n
        if azimuths is None:
            az0 = rng.uniform(0, 2 * math.pi)
            azimuths = az0 + 2 * math.pi * np.arange(n) / n + rng.normal(0, 0.18, n)
        else:
            # Main roots under the stem flutes, any further laterals (thinner) between them
            azimuths = np.asarray(azimuths, dtype=float)
            m = len(azimuths)
            if n > m:
                extra = azimuths[np.arange(n - m) % m] + math.pi / m + rng.normal(0, 0.15, n - m)
                azimuths = np.concatenate([azimuths, extra])
                weights[m:] *= 0.45
            else:
                n = m
                weights = weights[:m]
            n_main = m
        lat_shares = (1.0 - prof.taproot_share) * weights / weights.sum()
        zrt = prof.zrt_dbh_ratio * dbh
        spread = prof.spread_crown_ratio * crown_radius_m

        for i, (az, share) in enumerate(zip(azimuths, lat_shares)):
            r0 = R * share ** (1.0 / pipe_delta)
            depth = float(np.clip(sample_depth(prof.beta, rng.uniform(0.15, 0.45)), 0.08, 0.8))
            length = spread * rng.uniform(0.7, 1.15)
            self._lateral(graph, -1, -1, az, r0, R, length, depth, zrt, dbh, rng, order=1,
                          minor=i >= n_main)

        if prof.system in (RootSystemType.TAPROOT, RootSystemType.HEART) and prof.taproot_share > 0:
            self._vertical_roots(graph, R, dbh, zrt, rng)
        return graph

    # ------------------------------------------------------------------
    def _lateral(self, graph, parent_axis, parent_sample, az, r0, R, length, depth, zrt, dbh, rng, order,
                 start=None, minor=False):
        prof = self.profile
        primary = start is None
        n_pts = int(np.clip(length / 0.12, 12, 48))
        if primary:
            # Dense sampling near the stem where girth, height and section shape change fastest
            s_h = length * np.linspace(0.0, 1.0, n_pts) ** 1.8
            start = np.array([0.25 * R * math.cos(az), 0.25 * R * math.sin(az), 0.0])
        else:
            s_h = np.linspace(0.0, length, max(4, n_pts // 2))
        ds = np.diff(s_h, prepend=0.0)
        heading = az + np.cumsum(rng.normal(0.0, prof.tortuosity * 0.12, len(s_h)) * np.sqrt(ds / 0.3 + 1e-9))
        x = start[0] + np.cumsum(np.cos(heading) * ds)
        y = start[1] + np.cumsum(np.sin(heading) * ds)

        if primary:
            # Girth: the root base matches the stem surface; rapid taper starts where it leaves the stem
            # Most of the girth is lost within the zone of rapid taper (radius zrt from the stem)
            s_out = np.maximum(0.0, s_h - R)
            r = r0 * (0.3 + 0.7 * np.exp(-2.5 * s_out / max(0.05, zrt)))
            r = np.maximum(r, 0.5 * R * np.exp(-2.0 * s_out / max(0.05, zrt)))
            aspect = 1.0 + prof.plank * (0.3 if minor else 1.0) * np.exp(-s_out / max(0.05, 0.45 * zrt))
            r_v = r * np.sqrt(aspect)  # Vertical half-height of the (plank) section
            # Height: concave top edge descending from the collar height to the soil, then settle at depth
            # Minor laterals between buttresses merge low on the stem and stay buried
            h0 = max(prof.buttress_height_dbh * dbh * (0.25 if minor else 1.0), r_v[0])
            top = h0 * np.exp(-s_out / max(0.05, 0.35 * zrt))
            settle = 1.0 - np.exp(-s_out / max(0.1, 0.7 * zrt))
            z_surface = -r_v * (1.0 - prof.surface_exposure * (0.3 if minor else 1.0))
            z_far = -depth - 0.4 * r
            z = np.maximum(z_surface * (1.0 - settle) + z_far * settle, top - r_v)
        else:
            r = r0 * (0.55 + 0.45 * np.exp(-s_h / max(0.05, 0.3 * length)))
            aspect = None
            t = np.clip(s_h / max(1e-3, length), 0.0, 1.0)
            z = start[2] + (-depth - start[2]) * (1.0 - np.exp(-9.0 * t))
        z = z + 0.03 * depth * np.sin(s_h * 2.1 + az)  # Soil heterogeneity
        pts = np.column_stack([x, y, z])
        pts[0] = start
        keep = np.nonzero(pts[:, 2] < -self._display)[0]
        cut = int(keep[0]) + 1 if len(keep) else len(pts)
        pts = pts[:max(2, cut)]
        r = r[:len(pts)]
        s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
        # Taper by branching (pipe model) toward the tip
        radii = np.maximum(r * np.clip(1.0 - 0.75 * s / max(1e-3, s[-1]), 0.15, 1.0), self.MIN_RADIUS)
        axis = Axis(pts, radii, order, parent_axis, parent_sample, az, frame_up=True)
        if aspect is not None:
            axis.aspect = aspect[:len(pts)]
        aid = graph.add_axis(axis)

        if order >= 3:
            return aid
        # Sinkers and second-order laterals leave from the lower edge of the (plank) section
        half_height = radii * np.sqrt(axis.aspect) if axis.aspect is not None else radii
        self._sinkers(graph, aid, rng, dense_end=(prof.system == RootSystemType.BUTTRESS and primary),
                      min_dist=(zrt + R) if primary else 0.0)
        n_br = int(s[-1] / max(0.2, prof.branch_spacing_m * (1.6 if order == 2 else 1.0)))
        for _ in range(n_br):
            # Branches arise beyond the zone of rapid taper, from the lower flank of the parent
            j_min = int(np.searchsorted(s, min(s[-1] * 0.5, zrt * 0.8 + R)))
            j = int(rng.uniform(0.0, 1.0) * (len(pts) - 1 - j_min)) + j_min
            j = max(1, min(j, len(pts) - 2))
            side = rng.choice([-1.0, 1.0])
            child_az = az + side * math.radians(rng.uniform(50.0, 90.0))   # herringbone-like branching
            child_r = max(self.MIN_RADIUS, float(radii[j]) * rng.uniform(0.35, 0.6))
            child_len = (s[-1] - s[j]) * rng.uniform(0.35, 0.7)
            if child_len < 0.25:
                continue
            self._lateral(graph, aid, j, child_az, child_r, R, child_len, depth * rng.uniform(0.9, 1.4),
                          zrt, dbh, rng, order + 1, start=pts[j] - np.array([0.0, 0.0, 0.85 * half_height[j]]))
        if prof.knees > 0 and order == 1:
            self._knees(graph, aid, rng)
        return aid

    def _sinkers(self, graph, aid, rng, dense_end=False, min_dist=0.0):
        prof = self.profile
        axis = graph.axes[aid]
        pts, radii = axis.positions, axis.radii
        s = axis.arc_length
        # Sinkers descend from laterals beyond the zone of rapid taper
        f0 = float(np.clip(max(0.2, min_dist / max(1e-3, s[-1])), 0.0, 0.95))
        n = int(s[-1] * (0.95 - f0) / max(0.2, prof.sinker_spacing_m))
        positions = list(rng.uniform(f0, 0.95, n))
        if dense_end:
            positions += [0.97, 0.99]  # Sinkers at buttress ends (Crook et al. 1997)
        for f in positions:
            j = max(1, min(len(pts) - 2, int(np.searchsorted(s, f * s[-1]))))
            top = pts[j] - np.array([0.0, 0.0, 0.85 * float(radii[j] * (np.sqrt(axis.aspect[j])
                                                                       if axis.aspect is not None else 1.0))])
            target = float(np.clip(sample_depth(prof.beta, rng.uniform(0.7, 0.97)), 0.4, prof.max_depth_m))
            depth = min(target, self._display) - (-top[2])
            if depth < 0.15:
                continue
            k = max(3, int(depth / 0.15))
            z = top[2] - np.linspace(0.0, depth, k + 1)
            wob = rng.normal(0.0, 0.03, (k + 1, 2)).cumsum(axis=0)
            p = np.column_stack([top[0] + wob[:, 0], top[1] + wob[:, 1], z])
            r0 = max(self.MIN_RADIUS, float(radii[j]) * rng.uniform(0.35, 0.55))
            r = np.maximum(self.MIN_RADIUS, r0 * np.linspace(1.0, 0.3, k + 1))
            graph.add_axis(Axis(p, r, 3, aid, j, 0.0))

    def _vertical_roots(self, graph, R, dbh, zrt, rng):
        """Taproot (TAPROOT) or a fan of oblique heart roots (HEART)."""
        prof = self.profile
        if prof.system == RootSystemType.TAPROOT:
            dirs = [(0.0, 0.0)]
            shares = [prof.taproot_share]
        else:
            k = int(rng.integers(3, 6))
            dirs = [(rng.uniform(0, 2 * math.pi), math.radians(rng.uniform(25, 55))) for _ in range(k)]
            w = rng.lognormal(0.0, 0.3, k)
            shares = list(prof.taproot_share * w / w.sum())
        for (az, tilt), share in zip(dirs, shares):
            r0 = R * share ** (1.0 / self._delta)
            depth = min(self._display, prof.max_depth_m * rng.uniform(0.6, 1.0) *
                        (1.0 if prof.system == RootSystemType.TAPROOT else 0.6))
            length = depth / max(0.3, math.cos(tilt))
            k = max(4, int(length / 0.15))
            t = np.linspace(0.0, 1.0, k + 1)
            horiz = math.sin(tilt) * length * t
            wob = rng.normal(0.0, prof.tortuosity * 0.04, (k + 1, 2)).cumsum(axis=0)
            p = np.column_stack([horiz * math.cos(az) + wob[:, 0], horiz * math.sin(az) + wob[:, 1],
                                 0.2 * dbh - depth * t])
            r = r0 * (0.3 + 0.7 * np.exp(-t * length / max(0.1, 0.5 * zrt))) * (1.0 - 0.6 * t)
            r[0] = max(r[0], 0.5 * R)
            p, r = refine_base(p, np.maximum(r, self.MIN_RADIUS), 0.5 * zrt)
            aid = graph.add_axis(Axis(p, r, 1, -1, -1, az))
            self._sinkers_from_taproot(graph, aid, rng)

    def _sinkers_from_taproot(self, graph, aid, rng):
        """Short laterals branching from the taproot at depth."""
        axis = graph.axes[aid]
        for _ in range(int(rng.integers(2, 5))):
            j = max(2, int(rng.uniform(0.3, 0.85) * (len(axis.radii) - 1)))
            az = rng.uniform(0, 2 * math.pi)
            top = axis.positions[j]
            L = rng.uniform(0.4, 1.2)
            k = 6
            t = np.linspace(0.0, 1.0, k + 1)
            p = np.column_stack([top[0] + math.cos(az) * L * t, top[1] + math.sin(az) * L * t,
                                 top[2] - 0.25 * L * t ** 1.5])
            r = np.maximum(self.MIN_RADIUS, axis.radii[j] * 0.45 * (1.0 - 0.6 * t))
            graph.add_axis(Axis(p, r, 2, aid, j, az))

    def _knees(self, graph, aid, rng):
        """Pneumatophore-like conical knees rising from shallow laterals."""
        axis = graph.axes[aid]
        for _ in range(self.profile.knees):
            j = max(2, int(rng.uniform(0.3, 0.9) * (len(axis.radii) - 1)))
            base = axis.positions[j]
            h = rng.uniform(0.2, 0.9)
            t = np.linspace(0.0, 1.0, 6)
            p = np.column_stack([np.full(6, base[0]), np.full(6, base[1]), base[2] + (h - base[2]) * t])
            r = np.maximum(self.MIN_RADIUS, float(axis.radii[j]) * 1.4 * (1.0 - 0.75 * t ** 1.5))
            graph.add_axis(Axis(p, r, 2, aid, j, 0.0))

    def _fibrous(self, graph, R, rng):
        """Monocot root system: adventitious roots of constant girth from the stem-base root zone."""
        prof = self.profile
        n = max(4, int(prof.fibrous_count))
        for i in range(n):
            az = i * math.radians(137.508) + rng.normal(0, 0.1)
            h = rng.uniform(-0.05, 0.25) * R * 2.0
            dip = math.radians(rng.uniform(15.0, 70.0))
            L = min(self._display / max(0.2, math.sin(dip)), rng.uniform(1.0, 3.0))
            k = max(4, int(L / 0.12))
            t = np.linspace(0.0, 1.0, k + 1)
            start = np.array([R * 0.85 * math.cos(az), R * 0.85 * math.sin(az), h])
            d = np.array([math.cos(az) * math.cos(dip), math.sin(az) * math.cos(dip), -math.sin(dip)])
            bend = np.column_stack([np.zeros(k + 1), np.zeros(k + 1), -0.15 * L * t ** 2])
            p = start + (L * t)[:, None] * d + bend
            r = np.full(k + 1, prof.fibrous_radius_m * rng.uniform(0.8, 1.2))
            graph.add_axis(Axis(p, r, 2, -1, -1, az))
