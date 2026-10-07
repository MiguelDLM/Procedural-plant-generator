"""
Leaf Venation Architecture.

Builds a hierarchical vein network on top of the blade decomposition of a leaf
(see leaf_morphology). Each blade contributes its midvein; secondary veins
follow the categories of the Manual of Leaf Architecture (Ellis et al., 2009):

    craspedodromous  secondaries end at the margin (Fagus, Castanea, Betula)
    brochidodromous  secondaries join in loops inside the margin (Ficus, Magnolia)
    eucamptodromous  secondaries upturn and fade before the margin (Cornus)
    actinodromous    3+ primaries radiate from the base (Acer, Tilia, Cercis)
    parallelodromous parallel primaries converge at the apex (needles, monocots)
    flabellate       repeated dichotomies fan out from the base (Ginkgo)

Tertiary veins are percurrent cross-veins between adjacent secondaries; the
minor (quaternary+) reticulum and areoles are produced at raster time
(leaf_texture), sized from vein length per area (VLA): for a regular polygonal
areole network of mean width d, VLA ~ 2 / d.

Vein radii follow a hydraulic (Murray-type) hierarchy given in millimetres.
"""

from dataclasses import dataclass, field
from enum import Enum
import math
import numpy as np

from .leaf_morphology import Blade, LeafShapeModel


class VenationPattern(str, Enum):
    CRASPEDODROMOUS = "Craspedodromous"
    BROCHIDODROMOUS = "Brochidodromous"
    EUCAMPTODROMOUS = "Eucamptodromous"
    ACTINODROMOUS = "Actinodromous"
    PARALLELODROMOUS = "Parallelodromous"
    FLABELLATE = "Flabellate"


@dataclass
class VenationProfile:
    """Quantitative venation traits."""
    pattern: VenationPattern = VenationPattern.CRASPEDODROMOUS

    vla_mm_per_mm2: float = 6.8         # Vein length per area (typ. 2 - 16 mm/mm^2)
    secondary_vein_pairs: int = 9       # Pairs along the main lamina
    divergence_angle_deg: float = 46.0  # Secondary-to-midvein angle (mid-blade)
    divergence_angle_std: float = 4.2   # Random variation
    basal_angle_increase_deg: float = 12.0  # Secondaries diverge wider near the base

    primary_radius_mm: float = 0.85
    secondary_radius_mm: float = 0.28
    tertiary_radius_mm: float = 0.08

    reticulation_density: float = 0.65  # 0 = regular percurrent ladders, 1 = irregular reticulum
    areole_diameter_mm: float = 0.0     # 0 = derived from VLA
    secondary_curvature: float = 0.45   # 0 = straight, 1 = strongly arcuate toward the apex
    vein_contrast: float = 0.6          # Visual prominence of veins in the texture


@dataclass
class VeinPolyline:
    """A vein path in blade length units (BLU)."""
    points: np.ndarray   # (k, 2)
    radii: np.ndarray    # (k,) BLU
    order: int


@dataclass
class LeafVeinNetwork:
    polylines: list[VeinPolyline] = field(default_factory=list)
    minor_spacing_blu: float = 0.01

    @property
    def nodes(self) -> list[np.ndarray]:
        return [pt for pl in self.polylines for pt in pl.points]

    def count(self, order: int) -> int:
        return sum(1 for pl in self.polylines if pl.order == order)

    def total_length(self) -> float:
        return float(sum(np.sum(np.linalg.norm(np.diff(pl.points, axis=0), axis=1)) for pl in self.polylines))


class VenationEngine:
    """Constructs venation networks for every lamina of a LeafShapeModel."""

    def __init__(self, profile: VenationProfile = None):
        self.profile = profile or VenationProfile()

    def generate(self, model: LeafShapeModel, blade_length_mm: float, seed: int = 7) -> LeafVeinNetwork:
        prof = self.profile
        rng = np.random.default_rng(seed)
        net = LeafVeinNetwork()
        mm = 1.0 / max(1e-3, blade_length_mm)  # BLU per millimetre

        areole_mm = prof.areole_diameter_mm if prof.areole_diameter_mm > 0 else 2.0 / max(0.5, prof.vla_mm_per_mm2)
        net.minor_spacing_blu = areole_mm * mm

        # Stalk tissue (petiole / rachis) as order-1 veins
        for blade in model.blades:
            if blade.is_axis:
                t = np.linspace(0.0, blade.length, 24)
                pts = blade.local_to_global(t, np.zeros_like(t))
                net.polylines.append(VeinPolyline(pts, np.full(len(t), blade.max_half_width * 0.6), 1))

        laminae = [model.blades[i] for i in model.laminae]
        largest = max((b.length for b in laminae), default=1.0)
        for blade in laminae:
            if blade.role == "needle":
                self._needle(blade, net, mm)
                continue
            if prof.pattern == VenationPattern.FLABELLATE:
                self._flabellate(blade, net, mm)
                continue
            if prof.pattern == VenationPattern.PARALLELODROMOUS:
                self._parallel(blade, net, mm)
                continue
            order = blade.vein_order
            r_mid = (prof.primary_radius_mm if order == 1 else prof.secondary_radius_mm) * mm
            if blade.role in ("palm",):
                continue
            mid = self._midvein(blade, r_mid, order)
            net.polylines.append(mid)

            pairs = blade.secondary_pairs
            if pairs < 0:
                pairs = max(2, int(round(prof.secondary_vein_pairs * (blade.length / largest) ** 0.8)))
                if blade.role == "lobe" and order == 2:
                    pairs = max(1, int(round(pairs * 0.4)))
            if pairs == 0:
                continue
            sec_order = order + 1
            r_sec = (prof.secondary_radius_mm if sec_order == 2 else prof.tertiary_radius_mm * 1.6) * mm
            sides = self._secondaries(blade, pairs, r_sec, sec_order, rng)
            for side_paths in sides:
                net.polylines.extend(side_paths)
            if prof.pattern == VenationPattern.BROCHIDODROMOUS:
                for side_paths in sides:
                    self._brochidodromous_loops(side_paths, net, r_sec * 0.7, sec_order)
            self._tertiaries(blade, mid, sides, net, prof.tertiary_radius_mm * mm, sec_order + 1, rng)
        return net

    # Backwards-compatible entry point (simple elliptic lamina)
    def generate_network(self, blade_length_m: float, blade_width_m: float, boundary_pts=None) -> LeafVeinNetwork:
        from .leaf_morphology import LeafMorphologyProfile, LeafShapeBuilder, MarginType
        prof = LeafMorphologyProfile(aspect_ratio=blade_length_m / max(1e-6, blade_width_m),
                                     margin_type=MarginType.ENTIRE, petiole_length_ratio=0.0)
        model = LeafShapeBuilder(prof).build()
        return self.generate(model, blade_length_m * 1000.0)

    # ------------------------------------------------------------------
    def _midvein(self, blade: Blade, r0: float, order: int) -> VeinPolyline:
        t = np.linspace(0.0, blade.length * (0.97 - 0.2 * blade.notch), 40)
        pts = blade.local_to_global(t, np.zeros_like(t))
        tn = t / blade.length
        radii = r0 * np.sqrt(np.clip(1.0 - 0.85 * tn, 0.05, 1.0))
        return VeinPolyline(pts, radii, order)

    def _march(self, blade: Blade, t_start: float, side: float, angle0: float, stop_fraction: float,
               r0: float, curvature: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Integrates a secondary vein in blade-local coordinates; returns global points, radii, local (t, s)."""
        L = blade.length
        step = L / 90.0
        t, s = t_start, 0.0
        ts, ss = [t], [s]
        for _ in range(400):
            tn = t / L
            w = float(blade.width_at(np.array([min(max(tn, 0.0), 1.0)]))[0])
            if w <= 1e-4 or tn >= 0.995:
                break
            frac = abs(s) / w
            if frac >= stop_fraction:
                break
            ang = angle0 * (1.0 - curvature * min(1.0, frac) ** 1.4)
            t += step * math.cos(ang)
            s += side * step * math.sin(ang)
            ts.append(t)
            ss.append(s)
        ts, ss = np.array(ts), np.array(ss)
        prog = np.linspace(0.0, 1.0, len(ts))
        radii = r0 * (1.0 - 0.65 * prog)
        return blade.local_to_global(ts, ss), radii, np.stack([ts, ss], axis=1)

    def _secondaries(self, blade: Blade, pairs: int, r_sec: float, order: int, rng) -> list[list[VeinPolyline]]:
        prof = self.profile
        pattern = prof.pattern
        stop = {
            VenationPattern.CRASPEDODROMOUS: 0.97,
            VenationPattern.ACTINODROMOUS: 0.95,
            VenationPattern.BROCHIDODROMOUS: 0.80,
            VenationPattern.EUCAMPTODROMOUS: 0.74,
        }.get(pattern, 0.9)
        curv = float(np.clip(prof.secondary_curvature, 0.0, 1.0))
        if pattern == VenationPattern.EUCAMPTODROMOUS:
            curv = max(curv, 0.75)
        positions = np.linspace(0.07, 0.86, pairs) if pairs > 1 else np.array([0.4])
        out: list[list[VeinPolyline]] = [[], []]
        for k, side in enumerate((-1.0, 1.0)):
            for i, pos in enumerate(positions):
                tn = float(pos + rng.normal(0.0, 0.012))
                base_boost = prof.basal_angle_increase_deg * (1.0 - tn)
                ang = math.radians(np.clip(prof.divergence_angle_deg + base_boost +
                                           rng.normal(0.0, prof.divergence_angle_std), 10.0, 88.0))
                o = order
                r = r_sec
                stop_i = stop
                if pattern == VenationPattern.ACTINODROMOUS and i == 0 and blade.vein_order == 1 \
                        and blade.role == "lamina":
                    # Basal pair of lateral primaries
                    tn, ang, o, r, stop_i = 0.01, math.radians(prof.divergence_angle_deg * 0.95), order - 1, r_sec * 1.8, 0.93
                pts, radii, local = self._march(blade, tn * blade.length, side, ang, stop_i, r, curv)
                if len(pts) < 3:
                    continue
                pl = VeinPolyline(pts, radii, max(1, o))
                pl.local = local
                out[k].append(pl)
        return out

    def _brochidodromous_loops(self, paths: list[VeinPolyline], net: LeafVeinNetwork, r: float, order: int):
        for a, b in zip(paths[:-1], paths[1:]):
            p0 = a.points[-1]
            j = int(len(b.points) * 0.82)
            p2 = b.points[min(j, len(b.points) - 1)]
            mid = 0.5 * (p0 + p2)
            outward = p0 - a.points[0]
            outward = outward / max(1e-9, np.linalg.norm(outward))
            p1 = mid + outward * np.linalg.norm(p2 - p0) * 0.25
            u = np.linspace(0.0, 1.0, 14)[:, None]
            pts = (1 - u) ** 2 * p0 + 2 * (1 - u) * u * p1 + u ** 2 * p2
            net.polylines.append(VeinPolyline(pts, np.full(len(pts), r), order))

    def _tertiaries(self, blade: Blade, mid: VeinPolyline, sides, net: LeafVeinNetwork, r3: float,
                    order: int, rng):
        prof = self.profile
        spacing = max(net.minor_spacing_blu * 4.0, blade.length * 0.025)
        irregular = float(np.clip(prof.reticulation_density, 0.0, 1.0))
        for paths in sides:
            for a, b in zip(paths[:-1], paths[1:]):
                la = np.linalg.norm(np.diff(a.points, axis=0), axis=1).sum()
                n = int(np.clip(la / spacing, 1, 40))
                for f in np.linspace(0.12, 0.92, n):
                    fa = np.clip(f + rng.normal(0.0, 0.05 * irregular), 0.0, 1.0)
                    fb = np.clip(f + rng.normal(0.0, 0.08 * irregular), 0.0, 1.0)
                    pa = a.points[int(fa * (len(a.points) - 1))]
                    pb = b.points[int(fb * (len(b.points) - 1))]
                    d = pb - pa
                    nrm = np.array([-d[1], d[0]])
                    ctrl = 0.5 * (pa + pb) + nrm * rng.normal(0.0, 0.10 + 0.25 * irregular)
                    u = np.linspace(0.0, 1.0, 7)[:, None]
                    pts = (1 - u) ** 2 * pa + 2 * (1 - u) * u * ctrl + u ** 2 * pb
                    net.polylines.append(VeinPolyline(pts, np.full(len(pts), r3), order))

    def _parallel(self, blade: Blade, net: LeafVeinNetwork, mm: float):
        prof = self.profile
        n = max(3, int(round(blade.max_half_width * 2.0 / max(1e-4, net.minor_spacing_blu * 3.0))))
        n = min(n, 31) | 1
        t = np.linspace(0.0, blade.length * 0.99, 40)
        w = blade.width_at(t / blade.length)
        for k, sig in enumerate(np.linspace(-0.8, 0.8, n)):
            central = abs(sig) < 1e-6
            r = (prof.primary_radius_mm if central else prof.secondary_radius_mm) * mm
            pts = blade.local_to_global(t, sig * w)
            net.polylines.append(VeinPolyline(pts, np.full(len(t), r), 1 if central else 2))

    def _needle(self, blade: Blade, net: LeafVeinNetwork, mm: float):
        t = np.linspace(0.0, blade.length * 0.98, 16)
        pts = blade.local_to_global(t, np.zeros_like(t))
        net.polylines.append(VeinPolyline(pts, np.full(len(t), blade.max_half_width * 0.25), 1))

    def _flabellate(self, blade: Blade, net: LeafVeinNetwork, mm: float):
        """Dichotomous fan: each final vein follows the mean of its descendants' targets."""
        prof = self.profile
        levels = 5
        n_final = 2 ** levels
        targets = np.linspace(-0.93, 0.93, n_final)
        split_t = np.array([0.0, 0.10, 0.24, 0.42, 0.62, 0.80])
        t = np.linspace(0.0, blade.length * 0.985, 60)
        tn = t / blade.length
        w = blade.width_at(tn)
        for i in range(n_final):
            sig_levels = []
            for lvl in range(levels + 1):
                group = n_final // (2 ** lvl)
                g0 = (i // group) * group
                sig_levels.append(targets[g0:g0 + group].mean())
            sig = np.interp(tn, split_t, sig_levels)
            pts = blade.local_to_global(t, sig * w)
            r = prof.secondary_radius_mm * mm * (1.0 - 0.4 * tn)
            net.polylines.append(VeinPolyline(pts, r, 2))
