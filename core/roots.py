"""
Coarse root system architecture.

Root system types follow the classical classification of forest-tree root
systems (Köstler, Brückner & Bibelriether 1968, "Die Wurzeln der Waldbäume"):

    TAPROOT   a dominant vertical root plus laterals (Quercus robur, Pinus, Juglans)
    HEART     several oblique roots of similar girth from the stump (Fagus, Tilia, Betula, Acer)
    PLATE     shallow horizontal laterals carrying vertical sinkers (Picea, Populus, Salix)
    BUTTRESS  superficial laterals continuing up the stem as plank buttresses, with sinkers
              at their ends (Crook, Ennos & Banks 1997, J. Exp. Bot. 48: 1703)
    FIBROUS   many adventitious roots of near-constant girth from the stem base (palms, agaves,
              aloes, crassulaceous rosettes)
    TUBEROUS  a napiform (turnip/carrot-shaped) storage taproot carrying fine laterals, as in some
              globose cacti (Lophophora, Ariocarpus)
    TUBER_CLUSTER  fasciculate storage roots: several swollen adventitious roots radiating from the stem
              base, each with a neck and fine roots (Dahlia, sweet potato, cassava, Peniocereus striatus)
    STILT     shoot-borne prop roots arching from the lower stem into the soil, branching where they
              enter it (Rhizophora, Pandanus, banyan figs)

The coarse frame is completed by the 3D "cage" of mature trees (Danjon, Fourcaud & Bert 2005, New Phytol.
167: 917): taproot or obliques, the zone of rapid taper of the surface roots and numerous sinkers and deep
roots near the stem enclose a mass of soil, guyed by long horizontal surface roots. Sinkers near the stem
therefore reach a large share of the maximum rooting depth (`deep_roots`), farther ones stay shallower.
Heart systems are a hemispherical cage of several oblique roots that fork as they descend. Fine laterals of
the next orders are grown on the frame by core.root_architecture (CRootBox / ArchiSimple rules).

Succulents: most cacti are shallow-rooted, with laterals rarely deeper than 15-30 cm but extending
up to ~10 m from a saguaro (Cannon 1911, The Root Habits of Desert Plants, Carnegie Inst. Publ. 131);
cultivated Opuntia concentrate roots in the top 100-150 mm and spread 1.6-1.7 m in one season,
producing ephemeral "rain roots" after wetting (Snyman 2005, J. Prof. Assoc. Cactus Dev. 7: 1).
Shallow placement is critical for desert agaves: lowering an Agave deserti root system by 0.24 m
cut simulated water uptake by ~25% (Franco & Nobel 1990, Oecologia 82: 151,
doi:10.1007/BF00323528).

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
    TUBEROUS = "Tuberous"   # Napiform storage taproot (Lophophora, Ariocarpus) with fine laterals
    TUBER_CLUSTER = "Tuberous cluster"   # Fasciculate storage roots (Dahlia, sweet potato, cassava)
    STILT = "Stilt"         # Prop roots arching from the lower stem (Rhizophora, Pandanus, banyan figs)


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
    fibrous_spread_m: float = 0.0     # Horizontal reach of fibrous roots (0 = 1-3 m, palms)
    tuber_length_m: float = 0.12      # TUBEROUS storage root
    tuber_radius_m: float = 0.03
    deep_roots: float = 0.6           # Depth of sinkers and obliques near the stem, share of max_depth_m (cage)
    heart_roots: int = 7              # Oblique roots of a heart system
    fine_roots: float = 1.0           # Density of fine lateral roots on the frame (0 = coarse frame only)
    fine_orders: int = 2              # Orders of fine laterals
    stilt_height_dbh: float = 3.0     # Height on the stem of the highest stilt roots (x DBH)
    tuber_count: int = 6              # Storage roots of a tuberous cluster
    drop_roots: int = 0               # Aerial roots dropping from the branches to the soil (Rhizophora, banyans)
    pneumatophores: int = 0           # Pencil-like breathing roots per lateral, rising from the mud (Avicennia)


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
        if prof.system == RootSystemType.TUBEROUS:
            self._tuber(graph, R, rng)
            return graph
        if prof.system == RootSystemType.TUBER_CLUSTER:
            self._tuber_cluster(graph, R, rng)
            return graph
        self._dbh = dbh
        self._cage = prof.zrt_dbh_ratio * dbh * 1.5 + R

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
            # Structural laterals run 10-50 cm deep (Danjon et al. 2005: horizontal surface roots)
            depth = float(np.clip(sample_depth(prof.beta, rng.uniform(0.35, 0.8)), 0.1, 0.9))
            length = spread * rng.uniform(0.7, 1.15)
            self._lateral(graph, -1, -1, az, r0, R, length, depth, zrt, dbh, rng, order=1,
                          minor=i >= n_main)

        if prof.system in (RootSystemType.TAPROOT, RootSystemType.HEART) and prof.taproot_share > 0:
            self._vertical_roots(graph, R, dbh, zrt, rng)
        if prof.system == RootSystemType.STILT:
            self._stilts(graph, R, dbh, rng)
        return graph

    # ------------------------------------------------------------------
    def branching(self):
        """Fine-lateral rules for this system (core.root_architecture), scaled by `fine_roots`."""
        from .root_architecture import RootBranching
        prof = self.profile
        sysm = prof.system
        # Woody laterals are mostly plagiotropic: they spread in all directions, few turn down
        base = dict(orders=max(0, int(prof.fine_orders)), interbranch_cm=18.0, insertion_deg=60.0,
                    length_ratio=0.3, max_length_cm=150.0, radius_ratio=0.35, gravitropism=0.04,
                    tortuosity=0.5, min_radius_mm=1.2, budget=int(900 * prof.fine_roots))
        if sysm == RootSystemType.PLATE:
            base.update(gravitropism=0.02, insertion_deg=70.0)
        elif sysm in (RootSystemType.FIBROUS, RootSystemType.TUBER_CLUSTER):
            base.update(interbranch_cm=10.0, insertion_deg=80.0, gravitropism=0.15, radius_ratio=0.4,
                        min_radius_mm=0.6, max_length_cm=60.0)
        elif sysm == RootSystemType.TUBEROUS:
            base.update(interbranch_cm=4.0, max_length_cm=25.0, min_radius_mm=0.4, radius_ratio=0.5)
        if prof.fine_roots <= 0:
            base["orders"] = 0
        else:
            base["interbranch_cm"] /= max(prof.fine_roots, 0.1) ** 0.5
        return RootBranching(**base)

    def fine_roots(self, graph: BranchingGraph, seed: int = 42, display_depth_m: float = 3.0,
                   branching=None) -> BranchingGraph:
        """Fine lateral roots of the next orders grown on the coarse frame (CRootBox / ArchiSimple rules)."""
        from .root_architecture import lateral_roots
        br = branching or self.branching()
        out = BranchingGraph()
        if br.orders <= 0 or not graph.axes:
            return out
        rng = np.random.default_rng(seed + 4049)
        mothers = [(a.positions, a.radii) for a in graph.axes if len(a.positions) >= 3]
        for P, R, order, _, _ in lateral_roots(mothers, br, rng, surface=0.0, floor=-max(0.2, display_depth_m)):
            out.add_axis(Axis(P, np.maximum(R, br.min_radius_mm * 0.0005), 3 + order, -1, -1, 0.0))
        return out

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
        # Soil heterogeneity: the root dips and rises around obstacles and through soil layers
        z = z + (0.03 + 0.12 * np.clip(s_h / max(zrt + R, 0.1) - 1.0, 0.0, 1.0)) * depth * \
            np.sin(s_h * 1.3 + az * 3.0) + 0.06 * depth * np.sin(s_h * 3.7 + az)
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
                      min_dist=(0.45 * zrt + R) if primary else 0.0)
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
        if prof.pneumatophores > 0 and order == 1:
            self._pneumatophores(graph, aid, rng)
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
        cage = getattr(self, "_cage", 0.0)
        if min_dist > 0 and cage > min_dist:          # Extra sinkers inside the cage near the stem
            f_cage = float(np.clip(cage / max(1e-3, s[-1]), f0, 0.95))
            positions += list(rng.uniform(f0, f_cage, int(rng.integers(1, 3))))
        if dense_end:
            positions += [0.97, 0.99]  # Sinkers at buttress ends (Crook et al. 1997)
        for f in positions:
            j = max(1, min(len(pts) - 2, int(np.searchsorted(s, f * s[-1]))))
            top = pts[j] - np.array([0.0, 0.0, 0.85 * float(radii[j] * (np.sqrt(axis.aspect[j])
                                                                       if axis.aspect is not None else 1.0))])
            dist = float(np.hypot(top[0], top[1]))
            if dist < getattr(self, "_cage", 0.0):          # Cage near the stem (Danjon et al. 2005)
                target = prof.max_depth_m * prof.deep_roots * rng.uniform(0.55, 1.15)
            else:
                target = sample_depth(prof.beta, rng.uniform(0.7, 0.97)) * (1.0 + prof.deep_roots)
            target = float(np.clip(target, 0.4, prof.max_depth_m))
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
            k = max(3, int(prof.heart_roots))
            az0 = rng.uniform(0, 2 * math.pi)
            dirs = [(az0 + 2 * math.pi * i / k + rng.normal(0, 0.25), math.radians(rng.uniform(25, 60)))
                    for i in range(k)]
            w = rng.lognormal(0.0, 0.3, k)
            shares = list(prof.taproot_share * w / w.sum())
        heart = prof.system != RootSystemType.TAPROOT
        for (az, tilt), share in zip(dirs, shares):
            r0 = R * share ** (1.0 / self._delta)
            depth = min(self._display, prof.max_depth_m * rng.uniform(0.6, 1.0) *
                        (1.0 if not heart else prof.deep_roots * rng.uniform(0.7, 1.1)))
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
            if heart or depth > 1.2:
                self._fork(graph, aid, az, tilt, rng)

    def _fork(self, graph, aid, az, tilt, rng):
        """Obliques and deep taproots fork as they descend (heart cage; taproot splitting at depth)."""
        axis = graph.axes[aid]
        n = len(axis.radii)
        j = max(2, int(rng.uniform(0.3, 0.55) * (n - 1)))
        top = axis.positions[j]
        rest = max(0.2, (top[2] + self._display) * rng.uniform(0.6, 0.95))
        a2 = az + rng.choice([-1.0, 1.0]) * math.radians(rng.uniform(30, 70))
        t2 = min(math.radians(80), tilt + math.radians(rng.uniform(5, 25)))
        L = rest / max(0.3, math.cos(t2))
        k = max(4, int(L / 0.15))
        t = np.linspace(0.0, 1.0, k + 1)
        wob = rng.normal(0.0, self.profile.tortuosity * 0.04, (k + 1, 2)).cumsum(axis=0)
        p = np.column_stack([top[0] + math.sin(t2) * L * t * math.cos(a2) + wob[:, 0],
                             top[1] + math.sin(t2) * L * t * math.sin(a2) + wob[:, 1], top[2] - rest * t])
        r = np.maximum(self.MIN_RADIUS, float(axis.radii[j]) * 0.75 * (1.0 - 0.7 * t))
        graph.add_axis(Axis(p, r, 2, aid, j, a2))

    def _stilts(self, graph, R, dbh, rng):
        """Prop roots: leave the stem between the soil and stilt_height_dbh x DBH, arch outward and down
        into the soil, where they branch into a few anchoring roots."""
        prof = self.profile
        n = max(3, int(prof.lateral_count))
        H = prof.stilt_height_dbh * dbh
        for i in range(n):
            az = 2 * math.pi * i / n + rng.normal(0, 0.2)
            h = H * rng.uniform(0.25, 1.0)
            out = np.array([math.cos(az), math.sin(az), 0.0])
            reach = h * rng.uniform(0.8, 1.5) + R
            start = out * R * 0.6 + np.array([0.0, 0.0, h])
            mid = out * (R + reach * 0.45) + np.array([0.0, 0.0, h * 1.05])
            end = out * (R + reach) - np.array([0.0, 0.0, 0.25])
            t = np.linspace(0.0, 1.0, 16)[:, None]
            p = (1 - t) ** 2 * start + 2 * (1 - t) * t * mid + t ** 2 * end
            r = np.maximum(self.MIN_RADIUS, R * rng.uniform(0.12, 0.2) * (1.0 - 0.3 * t[:, 0]))
            aid = graph.add_axis(Axis(p, r, 1, -1, -1, az))
            for _ in range(int(rng.integers(2, 5))):              # Anchors where it enters the soil
                a2 = az + rng.normal(0, 0.9)
                L = reach * rng.uniform(0.3, 0.7)
                k = 6
                u = np.linspace(0.0, 1.0, k + 1)
                q = np.column_stack([end[0] + math.cos(a2) * L * u, end[1] + math.sin(a2) * L * u,
                                     end[2] - (0.3 + 0.5 * rng.random()) * L * u])
                graph.add_axis(Axis(q, np.maximum(self.MIN_RADIUS, r[-1] * 0.6 * (1 - 0.6 * u)), 2, aid, 15, a2))

    def _tuber_cluster(self, graph, R, rng):
        """Tuberous cluster: storage roots radiating from the stem base (neck, swollen spindle body, thin
        tail), between thinner fibrous roots."""
        prof = self.profile
        n = max(1, int(prof.tuber_count))
        for i in range(n):
            az = 2 * math.pi * i / n + rng.normal(0, 0.35)
            dip = math.radians(rng.uniform(25, 70))
            L = min(self._display, prof.tuber_length_m * rng.uniform(0.7, 1.2))
            k = 20
            t = np.linspace(0.0, 1.0, k + 1)
            d = np.array([math.cos(az) * math.cos(dip), math.sin(az) * math.cos(dip), -math.sin(dip)])
            wob = rng.normal(0.0, 0.01, (k + 1, 3)).cumsum(axis=0) * L
            p = np.array([R * 0.4 * math.cos(az), R * 0.4 * math.sin(az), -0.01]) + (L * t)[:, None] * d + \
                wob * t[:, None] - np.column_stack([np.zeros(k + 1), np.zeros(k + 1), 0.15 * L * t ** 2])
            # Neck, swollen body (widest about 40 % along), thin tail
            body = np.sin(np.pi * np.clip((t - 0.08) / 0.8, 0, 1)) ** 0.8
            r = prof.tuber_radius_m * rng.uniform(0.7, 1.15) * np.maximum(body, 0.08)
            r = np.maximum(r, self.MIN_RADIUS)
            graph.add_axis(Axis(p, r, 1, -1, -1, az))
        self._fibrous(graph, R, rng)

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

    def _pneumatophores(self, graph, aid, rng):
        """Pencil-like pneumatophores: first-order laterals of the horizontal cable roots growing straight up
        (negatively gravitropic), ~0.5-0.7 cm thick, 10-30 cm above the mud (Avicennia)."""
        axis = graph.axes[aid]
        s = axis.arc_length
        for _ in range(self.profile.pneumatophores):
            f = rng.uniform(0.15, 0.95)
            j = max(1, min(len(s) - 2, int(np.searchsorted(s, f * s[-1]))))
            base = axis.positions[j] + rng.normal(0, 0.05, 3) * np.array([1, 1, 0])
            h = rng.uniform(0.1, 0.3)
            t = np.linspace(0.0, 1.0, 6)
            p = np.column_stack([base[0] + rng.normal(0, 0.01) * t, base[1] + rng.normal(0, 0.01) * t,
                                 base[2] + (h - base[2]) * t])
            r = np.maximum(self.MIN_RADIUS, rng.uniform(0.0025, 0.0035) * (1.0 - 0.6 * t ** 2))
            graph.add_axis(Axis(p, r, 3, aid, j, 0.0))

    def drop_roots(self, skeleton: BranchingGraph, total_height: float, crown_radius: float,
                   seed: int = 42) -> BranchingGraph:
        """Aerial roots that leave the lower branches, hang down to the soil and anchor there, thickening
        into props (Rhizophora drop roots; banyan figs)."""
        out = BranchingGraph()
        n = int(self.profile.drop_roots)
        if n <= 0:
            return out
        rng = np.random.default_rng(seed + 811)
        cands = []
        for a in skeleton.axes:
            if a.order < 1 or a.order > 2:
                continue
            P = a.positions
            dist = np.hypot(P[:, 0], P[:, 1])
            ok = np.nonzero((P[:, 2] > 0.2 * total_height) & (P[:, 2] < 0.75 * total_height) &
                            (dist > 0.2 * crown_radius))[0]
            cands += [(P[i], float(a.radii[i])) for i in ok[::3]]
        if not cands:
            return out
        for k in rng.choice(len(cands), size=min(n, len(cands)), replace=False):
            top, rb = cands[k]
            m = max(6, int(top[2] / 0.25))
            t = np.linspace(0.0, 1.0, m + 1)
            sway = rng.normal(0, 0.03, (m + 1, 2)).cumsum(0) * t[:, None]
            p = np.column_stack([top[0] + sway[:, 0], top[1] + sway[:, 1], top[2] - (top[2] + 0.15) * t])
            r = np.clip(rb * 0.5, 0.008, 0.05) * (0.6 + 0.4 * t)          # Thicker where it props
            aid = out.add_axis(Axis(p, r, 1, -1, -1, 0.0))
            for _ in range(int(rng.integers(2, 4))):                    # Anchors into the soil
                az = rng.uniform(0, 2 * math.pi)
                L = rng.uniform(0.3, 0.8)
                u = np.linspace(0.0, 1.0, 6)
                q = np.column_stack([p[-1, 0] + math.cos(az) * L * u, p[-1, 1] + math.sin(az) * L * u,
                                     p[-1, 2] - 0.4 * L * u])
                out.add_axis(Axis(q, np.maximum(self.MIN_RADIUS, r[-1] * 0.6 * (1 - 0.6 * u)), 2, aid, m, az))
        return out

    def _fibrous(self, graph, R, rng):
        """Monocot root system: adventitious roots of constant girth from the stem-base root zone."""
        prof = self.profile
        n = max(4, int(prof.fibrous_count))
        for i in range(n):
            az = i * math.radians(137.508) + rng.normal(0, 0.1)
            h = rng.uniform(-0.05, 0.25) * R * 2.0
            if prof.fibrous_spread_m > 0.0:
                # Reach set by the spread; dip so that roots end within the rooting depth (beta model)
                L = prof.fibrous_spread_m * rng.uniform(0.5, 1.05)
                depth = float(np.clip(sample_depth(prof.beta, rng.uniform(0.3, 0.9)), 0.02, prof.max_depth_m))
                dip = math.atan2(depth, L)
            else:
                dip = math.radians(rng.uniform(15.0, 70.0))
                L = rng.uniform(1.0, 3.0)
            L = min(self._display / max(0.05, math.sin(dip)), L)
            k = max(4, int(L / 0.12))
            t = np.linspace(0.0, 1.0, k + 1)
            start = np.array([R * 0.85 * math.cos(az), R * 0.85 * math.sin(az), h])
            d = np.array([math.cos(az) * math.cos(dip), math.sin(az) * math.cos(dip), -math.sin(dip)])
            sag = (0.15 * L) if prof.fibrous_spread_m <= 0.0 else (0.2 * L * math.tan(dip))
            bend = np.column_stack([np.zeros(k + 1), np.zeros(k + 1), -sag * t ** 2])
            # Sinuous course through the soil (cumulative random deflection, larger toward the tip)
            wobble = np.cumsum(rng.normal(0.0, 0.035 * L / np.sqrt(k), (k + 1, 3)), axis=0) * t[:, None]
            wobble[:, 2] *= 0.4
            p = start + (L * t)[:, None] * d + bend + wobble
            r = np.full(k + 1, prof.fibrous_radius_m * rng.uniform(0.8, 1.2))
            graph.add_axis(Axis(p, r, 2, -1, -1, az))

    def _tuber(self, graph, R, rng):
        """Napiform storage taproot: swollen below the collar, tapering to a thin tip, with fine laterals."""
        prof = self.profile
        L = min(self._display, prof.tuber_length_m)
        k = 24
        t = np.linspace(0.0, 1.0, k + 1)
        wob = rng.normal(0.0, 0.004, (k + 1, 2)).cumsum(axis=0) * L
        p = np.column_stack([wob[:, 0] * t, wob[:, 1] * t, 0.02 * L - L * t])
        # Napiform profile: widest a little below the collar, then a long taper
        r = prof.tuber_radius_m * (1.0 + 0.25 * np.sin(np.pi * np.clip(t / 0.35, 0, 1))) * (1.0 - t) ** 0.9
        r = np.maximum(r, self.MIN_RADIUS)
        aid = graph.add_axis(Axis(p, r, 1, -1, -1, 0.0))
        for _ in range(max(3, int(prof.fibrous_count))):
            j = int(rng.uniform(0.15, 0.9) * k)
            az = rng.uniform(0, 2 * math.pi)
            top = p[j]
            Lr = (prof.fibrous_spread_m or 3 * L) * rng.uniform(0.3, 1.0)
            n = 8
            u = np.linspace(0.0, 1.0, n + 1)
            q = np.column_stack([top[0] + math.cos(az) * Lr * u, top[1] + math.sin(az) * Lr * u,
                                 top[2] - 0.3 * Lr * u ** 1.5])
            rr = np.maximum(self.MIN_RADIUS * 0.5, prof.fibrous_radius_m * (1.0 - 0.6 * u))
            graph.add_axis(Axis(q, rr, 2, aid, j, az))
