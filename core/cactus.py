"""
Cactaceae (stem succulents).

Stems are parametric surfaces of revolution around an axis, sculpted by ribs
and/or tubercles, carrying areoles with spines (Mauseth 2006, Ann. Bot. 98: 901,
doi:10.1093/aob/mcl133: ribs are podaria joined end to end, tubercles are free
podaria, with intermediates).

Geometry
- Generatrix (rho(t), z(t)): basal taper, body, apical dome (superellipse) and an
  optional apical depression; columnar, barrel and globose habits are points of
  this family.
- Ribs: cross-section r = rho * (1 - depth * (1 - |cos(m theta' / 2)|^p)),
  m ribs, p < 1 rounded crests / narrow grooves, p > 1 sharp crests; theta'
  includes an optional helical twist. Rib depth fades toward the apex.
  Rib numbers default to Fibonacci numbers, as measured in barrel cacti
  (Robberecht & Nobel 1983, Ann. Bot. 51: 153, doi:10.1093/oxfordjournals.aob.a086440).
- Areoles: on rib crests at a fixed spacing, offset by half a step on
  neighbouring ribs; or, for tuberculate species, on an equal-area spiral
  lattice: areole n sits where the cumulative lateral area equals n areas per
  areole, at azimuth n x 137.508 deg (Vogel 1979, Math. Biosci. 44: 179,
  generalised from the disc to a surface of revolution). Each areole raises a
  Gaussian tubercle.
- Spines: radial spines spread around the areole close to the surface, central
  spines point outward; curvature by gravity or a terminal hook; areolar wool.
- Branching: arms (saguaro, candelabra), basal offsets (clumps) and, for
  Opuntia, chains of flattened cladodes with spirally arranged areoles.
"""

from dataclasses import dataclass, field
from enum import Enum
import math
import numpy as np

from .mesh_engine import MeshData
from .roots import RootSystemType

GOLDEN = math.radians(137.50776)
UP = np.array([0.0, 0.0, 1.0])


class CactusHabit(str, Enum):
    COLUMNAR = "Columnar"   # Carnegiea, Pachycereus, Cereus
    BARREL = "Barrel"       # Ferocactus, Echinocactus
    GLOBOSE = "Globose"     # Mammillaria, Astrophytum, Gymnocalycium
    CLADODE = "Cladode"     # Opuntia (flattened stem segments)


class AreoleArrangement(str, Enum):
    RIBS = "Ribs"           # Areoles in rows on rib crests
    SPIRAL = "Spiral"       # Areoles on spiral tubercles (parastichies)


@dataclass
class CactusProfile:
    habit: CactusHabit = CactusHabit.COLUMNAR
    arrangement: AreoleArrangement = AreoleArrangement.RIBS

    # Stem body
    height_m: float = 3.0
    diameter_m: float = 0.35
    base_taper: float = 0.15        # Narrowing toward the ground (fraction of radius)
    apex_dome: float = 1.0          # Dome height in stem radii
    apex_roundness: float = 2.0     # Superellipse exponent of the dome (2 = elliptic, >2 = flat-topped)
    apex_depression: float = 0.0    # Sunken apex (fraction of radius)

    # Ribs and tubercles
    rib_count: int = 13
    rib_depth: float = 0.18         # Groove depth relative to radius
    rib_sharpness: float = 0.7      # < 1 rounded crests, > 1 sharp crests
    rib_twist_deg_per_m: float = 0.0
    tubercle_height: float = 0.0    # Relative bump raised by each areole
    areole_spacing_cm: float = 2.5

    # Spines
    radial_spines: int = 10
    radial_length_cm: float = 1.5
    central_spines: int = 3
    central_length_cm: float = 4.0
    spine_thickness_mm: float = 0.8
    spine_curvature: float = 0.1    # Gravity curvature (0 straight .. 1 strongly bent)
    central_hook: float = 0.0       # Terminal hook on centrals (Ferocactus wislizeni, Mammillaria)
    radial_lift_deg: float = 15.0   # Angle of radials above the surface
    spine_jitter: float = 0.25
    wool: float = 0.3               # Areolar wool / felt size (relative to spacing)
    apical_wool: float = 0.0        # Extra wool on the apex (Echinocactus grusonii, cephalia)

    # Branching
    arm_count: int = 0
    arm_height_min: float = 0.35    # Relative height range of arm insertions
    arm_height_max: float = 0.65
    arm_radius_ratio: float = 0.8
    arm_reach_m: float = 0.5        # Horizontal elbow before turning upward
    arm_length_ratio: float = 0.5   # Vertical rise of an arm relative to the main stem height
    arm_lean_deg: float = 3.0       # Outward lean of the erect part of each arm
    arm_branching: float = 0.0
    crown_fill: float = 0.0          # Fraction of arms whose columns fill the crown disc (area-uniform) instead of a ring      # Mean secondary arms per arm (dense candelabra crowns)
    offsets: int = 0                # Basal offsets (clumping)
    offset_scale: float = 0.7

    # Cladodes (Opuntia)
    pad_length_cm: float = 30.0
    pad_width_ratio: float = 0.65
    pad_thickness_ratio: float = 0.08
    pad_levels: int = 4
    pad_branching: float = 1.6      # Mean daughter pads per pad

    # Colour (sRGB)
    stem_color: tuple = (0.24, 0.40, 0.20)
    groove_color: tuple = (0.15, 0.28, 0.13)
    spine_color: tuple = (0.85, 0.80, 0.62)
    spine_tip_color: tuple = (0.35, 0.25, 0.18)
    wool_color: tuple = (0.92, 0.90, 0.85)
    glaucous: float = 0.2           # Waxy bloom
    flecks: float = 0.0             # White trichome flecks (Astrophytum)

    # Roots (Cannon 1911; Snyman 2005): shallow laterals, optional taproot or napiform tuber
    root_system: RootSystemType = RootSystemType.PLATE
    root_count: int = 10
    root_spread_ratio: float = 1.0  # Lateral reach relative to plant height
    root_depth_m: float = 0.3       # 90% of roots above this depth
    taproot_share: float = 0.0
    taproot_depth_m: float = 0.5
    root_core_ratio: float = 0.25   # Vascular cylinder / stem radius (root collar)
    tuber_length_cm: float = 12.0
    tuber_radius_ratio: float = 0.8 # Tuber radius relative to stem radius


@dataclass
class CactusResult:
    stem: MeshData
    spines: MeshData
    areole_count: int
    spine_count: int
    height_m: float
    roots: MeshData = None
    flower_pos: np.ndarray = None      # Areoles able to flower, with outward normals and weights
    flower_normal: np.ndarray = None
    flower_weight: np.ndarray = None


# -----------------------------------------------------------------------------
def _normalize(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def _quads_grid(rows: int, cols: int, offset: int = 0) -> np.ndarray:
    """Quad indices of a (rows x cols) grid wrapped around in columns."""
    r = np.arange(rows - 1)[:, None] * cols
    c = np.arange(cols)[None, :]
    c1 = (c + 1) % cols
    return (np.stack([r + c, r + c1, r + cols + c1, r + cols + c], axis=-1).reshape(-1, 4) + offset)


def _mesh(verts, quads=None, tris=None, ngons=None, uvs=None, attrs=None) -> MeshData:
    """MeshData from quad / triangle arrays (vectorized) plus optional n-gons."""
    loops, totals = [], []
    for block, k in ((quads, 4), (tris, 3)):
        if block is not None and len(block):
            arr = np.asarray(block, dtype=np.int32).reshape(-1, k)
            loops.append(arr.reshape(-1))
            totals.append(np.full(len(arr), k, dtype=np.int32))
    for f in (ngons or []):
        loops.append(np.asarray(f, dtype=np.int32))
        totals.append(np.array([len(f)], dtype=np.int32))
    loops = np.concatenate(loops) if loops else np.zeros(0, np.int32)
    totals = np.concatenate(totals) if totals else np.zeros(0, np.int32)
    starts = (np.concatenate([[0], np.cumsum(totals[:-1])]).astype(np.int32) if len(totals)
              else np.zeros(0, np.int32))
    loop_uv = (np.zeros((len(loops), 2), np.float32) if uvs is None
               else np.asarray(uvs, dtype=np.float32)[loops])
    return MeshData(np.asarray(verts, np.float32), loops, starts, totals, loop_uv, attrs or {})


class Generatrix:
    """Meridian curve (rho, z) of a stem, densely sampled and parameterised by arc length."""

    def __init__(self, length: float, radius: float, prof: CactusProfile, globose: bool):
        R = radius
        L = max(length, 0.2 * R)
        dome = min(prof.apex_dome * R, 0.95 * L) if not globose else min(L * 0.55, prof.apex_dome * R)
        z_dome = L - dome
        pts = []
        # Below-ground start and basal taper (globose bodies are rounded at the base too)
        z0 = -0.15 * R
        n_body = 60
        for z in np.linspace(z0, z_dome, n_body):
            u = np.clip((z - z0) / max(1e-6, z_dome - z0), 0.0, 1.0)
            base = 1.0 - prof.base_taper * (1.0 - u) ** 3
            if globose:
                # Barrel/globose: elliptic lower shoulder
                bottom = math.sqrt(max(0.0, 1.0 - (1.0 - min(1.0, (z - z0) / max(1e-6, 0.45 * (L - z0)))) ** 2))
                base *= 0.55 + 0.45 * bottom
            pts.append((R * base, z))
        p = max(1.2, prof.apex_roundness)
        rho_body = pts[-1][0]
        for tau in np.linspace(0.0, math.pi / 2, 60)[1:]:
            c, s = math.cos(tau), math.sin(tau)
            rho = rho_body * (abs(c) ** (2.0 / p))
            z = z_dome + dome * (abs(s) ** (2.0 / p))
            pts.append((rho, z))
        pts = np.array(pts)
        # Apical depression: the centre sinks below the rim of the dome
        if prof.apex_depression > 0.0:
            rd = 0.45 * R
            near = pts[:, 0] < rd
            w = (1.0 - pts[near, 0] / rd) ** 2
            pts[near, 1] -= prof.apex_depression * R * w
        self.rho = pts[:, 0]
        self.z = pts[:, 1]
        seg = np.hypot(np.diff(self.rho), np.diff(self.z))
        self.sigma = np.concatenate([[0.0], np.cumsum(seg)])
        self.length = float(self.sigma[-1])
        self.R = R
        # Cumulative lateral area for the equal-area areole lattice
        self.area = np.concatenate([[0.0], np.cumsum(2 * math.pi * 0.5 * (self.rho[1:] + self.rho[:-1]) * seg)])

    def at(self, sigma: np.ndarray):
        s = np.clip(sigma, 0.0, self.length)
        return np.interp(s, self.sigma, self.rho), np.interp(s, self.sigma, self.z)

    def sigma_of_area(self, a: np.ndarray):
        return np.interp(a, self.area, self.sigma)


class StemAxis:
    """Axis curve with projection frames; height h along the axis maps to a point and a frame."""

    def __init__(self, points: np.ndarray):
        self.P = np.asarray(points, float)
        seg = np.linalg.norm(np.diff(self.P, axis=0), axis=1)
        self.s = np.concatenate([[0.0], np.cumsum(seg)])
        T = np.empty_like(self.P)
        T[1:-1] = self.P[2:] - self.P[:-2]
        T[0] = self.P[1] - self.P[0]
        T[-1] = self.P[-1] - self.P[-2]
        self.T = _normalize(T)
        ref = np.array([1.0, 0.0, 0.0]) if abs(self.T[0, 2]) > 0.9 else UP
        n0 = _normalize(np.cross(self.T[0], ref))
        N = n0[None, :] - np.sum(n0[None, :] * self.T, axis=1, keepdims=True) * self.T
        self.N = _normalize(N)
        self.B = np.cross(self.T, self.N)

    @property
    def length(self):
        return float(self.s[-1])

    def frame(self, h: np.ndarray):
        h = np.asarray(h, float)
        hc = np.clip(h, 0.0, self.length)
        P = np.stack([np.interp(hc, self.s, self.P[:, k]) for k in range(3)], axis=-1)
        extra = (h - hc)[..., None]
        T = _normalize(np.stack([np.interp(hc, self.s, self.T[:, k]) for k in range(3)], axis=-1))
        N = np.stack([np.interp(hc, self.s, self.N[:, k]) for k in range(3)], axis=-1)
        N = _normalize(N - np.sum(N * T, axis=-1, keepdims=True) * T)
        B = np.cross(T, N)
        return P + extra * T, T, N, B


class CactusEngine:
    def __init__(self, profile: CactusProfile):
        self.p = profile

    # ------------------------------------------------------------------
    def generate(self, seed: int = 7, detail: float = 1.0, spine_budget: int = 60000, with_roots: bool = False,
                 root_display_depth: float = 3.0) -> CactusResult:
        p = self.p
        rng = np.random.default_rng(seed)
        stems, spines = [], []
        n_areoles = 0
        if p.habit == CactusHabit.CLADODE:
            parts = self._opuntia(rng, detail)
            stems += [m for m, _ in parts]
            for _, ar in parts:
                spines.append(ar)
        else:
            R = 0.5 * p.diameter_m
            globose = p.habit in (CactusHabit.BARREL, CactusHabit.GLOBOSE)
            main_axis = StemAxis(np.array([[0.0, 0.0, 0.0], [0.0, 0.0, p.height_m]]) +
                                 np.array([[0, 0, 0], [rng.normal(0, 0.01), rng.normal(0, 0.01), 0]]) * p.height_m)
            specs = [(main_axis, p.height_m, R, globose, rng.uniform(0, 2 * math.pi))]
            specs += self._arms(main_axis, R, rng)
            specs += self._offsets(R, globose, rng)
            for axis, length, radius, glob, phase in specs:
                stem, areoles = self._stem(axis, length, radius, glob, phase, rng, detail)
                stems.append(stem)
                spines.append(areoles)
        stem = MeshData.concatenate(stems)
        ar_pos = np.concatenate([a["pos"] for a in spines]) if spines else np.zeros((0, 3))
        ar_nrm = np.concatenate([a["normal"] for a in spines]) if spines else np.zeros((0, 3))
        ar_tan = np.concatenate([a["tangent"] for a in spines]) if spines else np.zeros((0, 3))
        ar_scale = np.concatenate([a["scale"] for a in spines]) if spines else np.zeros(0)
        ar_apex = np.concatenate([a["apex"] for a in spines]) if spines else np.zeros(0)
        ar_flw = np.concatenate([a.get("flower", np.zeros(len(a["pos"]))) for a in spines]) if spines else np.zeros(0)
        spine_mesh, n_sp = self._spines(ar_pos, ar_nrm, ar_tan, ar_scale, ar_apex, rng, spine_budget)
        height = float(max(stem.vertices[:, 2].max(), 0.0)) if len(stem.vertices) else 0.0
        roots = None
        if with_roots:
            from .succulent_roots import succulent_roots
            R0 = 0.5 * (p.diameter_m if p.habit != CactusHabit.CLADODE else p.pad_length_cm * 0.01 * p.pad_thickness_ratio * 2)
            roots = succulent_roots(
                p.root_system, R0, max(0.05, p.root_spread_ratio * max(height, 0.05)), p.root_depth_m, p.root_count,
                core_ratio=p.root_core_ratio, taproot_share=p.taproot_share, taproot_depth_m=p.taproot_depth_m,
                root_radius_mm=max(0.5, 0.06 * R0 * 1000 * p.root_core_ratio * 4),
                tuber_length_m=p.tuber_length_cm * 0.01, tuber_radius_m=p.tuber_radius_ratio * R0,
                display_depth_m=root_display_depth, seed=seed)
        return CactusResult(stem, spine_mesh, len(ar_pos), n_sp, height, roots, ar_pos, ar_nrm, ar_flw)

    # ------------------------------------------------------------------
    def _arm_collides(self, ax1: StemAxis, r1: float, ax2: StemAxis, r2: float, start1: float = 0.10) -> bool:
        """Test whether two arm trajectories penetrate each other (ax1 sampled from `start1` of its length)."""
        s1 = np.linspace(start1 * ax1.length, ax1.length, 30)
        s2 = np.linspace(0.10 * ax2.length, ax2.length, 30)
        pts1 = ax1.frame(s1)[0]
        pts2 = ax2.frame(s2)[0]
        dists = np.linalg.norm(pts1[:, None, :] - pts2[None, :, :], axis=-1)
        return bool(dists.min() < (r1 + r2) * 0.95)

    def _arm_trunk_collides(self, ax: StemAxis, r_arm: float, R_trunk: float) -> bool:
        """Test whether an arm penetrates the main trunk outside the insertion base."""
        s = np.linspace(0.20 * ax.length, ax.length, 30)
        pts = ax.frame(s)[0]
        d_trunk = np.hypot(pts[:, 0], pts[:, 1])
        below = pts[:, 2] < self.p.height_m           # Above the trunk's apex there is no trunk to hit
        return bool(np.any((d_trunk < (R_trunk + r_arm) * 0.95) & below))

    def _arms(self, main: StemAxis, R: float, rng) -> list:
        p = self.p
        specs = []
        if p.arm_count <= 0:
            return specs
        az0 = rng.uniform(0, 2 * math.pi)
        num_arms = p.arm_count
        n_fill = int(round(p.crown_fill * num_arms))
        disc = p.arm_reach_m + R                    # Radius of the crown disc filled by the columns
        for k in range(num_arms):
            base_az = az0 + 2 * math.pi * k / num_arms
            fill = k < n_fill
            if fill:
                # Vogel spiral: column k sits at radius disc*sqrt((k+.5)/n), golden-angle azimuth, so the
                # vertical parts of the arms cover the crown disc with equal area per column (no hollow centre).
                d_col = disc * math.sqrt((k + 0.5) / n_fill)
                base_az = az0 + k * GOLDEN
            # In dense candelabra branching (e.g. Garambullo, Euphorbia),
            # branches naturally organize into tiered reaches (staggered concentric layers)
            # where outer arms reach further out, preventing vertical columns from merging.
            tier = (k % 2) if num_arms >= 6 else 0
            tier_reach_mult = (1.25 if tier == 1 else 0.85) if num_arms >= 6 else 1.0
            h_frac = 0.2 + 0.6 * ((k * 3) % num_arms) / max(1, num_arms - 1) if num_arms > 1 else 0.5
            base_h = p.height_m * (p.arm_height_min + (p.arm_height_max - p.arm_height_min) * h_frac)
            min_reach = R + (R * p.arm_radius_ratio) + 0.04
            base_reach = max(min_reach, p.arm_reach_m * tier_reach_mult)
            if fill:
                # Inner columns branch from near the apex, outer ones lower down (dome of insertions)
                rel = d_col / disc
                base_h = p.height_m * (p.arm_height_max - (p.arm_height_max - p.arm_height_min) * rel)
                base_reach = max(min_reach, d_col)
                tier, tier_reach_mult = 0, 1.0
            for attempt in range(16 if fill else 12):
                az_nudge = 0.0 if attempt == 0 else rng.choice([-1, 1]) * (0.05 + 0.04 * attempt)
                az = base_az + rng.normal(0, 0.04) + az_nudge
                out = np.array([math.cos(az), math.sin(az), 0.0])
                h = base_h + rng.normal(0, 0.02 * p.height_m)
                h = np.clip(h, p.height_m * p.arm_height_min, p.height_m * p.arm_height_max)
                r_arm = R * p.arm_radius_ratio * rng.uniform(0.88, 1.04)
                reach_attempt_mult = 1.0
                if attempt > 1:
                    reach_attempt_mult = (1.18 if (attempt % 2 == 0) else 0.88)
                reach = max(R + r_arm + 0.03, base_reach * reach_attempt_mult * rng.uniform(0.95, 1.05))
                if fill:     # Short elbows for inner columns so they stand close to the axis
                    reach = max(0.2 * R, min(reach, d_col * rng.uniform(0.9, 1.05)))
                    if reach < R + r_arm:        # Central columns sprout from the trunk apex
                        h = p.height_m * 0.99
                rise = p.arm_length_ratio * p.height_m * rng.uniform(0.75, 1.05) + 2 * r_arm
                base = main.frame(np.array(h))[0]
                lean = math.tan(math.radians((p.arm_lean_deg + 3.0 * tier) * (rel ** 1.5 if fill else 1.0)))
                ctrl = [base,
                        base + out * reach * 0.6 + UP * reach * 0.15,
                        base + out * reach + UP * reach * 0.6,
                        base + out * (reach + lean * rise) + UP * (reach * 0.6 + rise)]
                t = np.linspace(0.0, 1.0, 40)[:, None]
                c0, c1, c2, c3 = [np.asarray(c) for c in ctrl]
                pts = ((1 - t) ** 3) * c0 + 3 * ((1 - t) ** 2) * t * c1 + 3 * (1 - t) * t ** 2 * c2 + t ** 3 * c3
                cand_axis = StemAxis(pts)
                if self._arm_trunk_collides(cand_axis, r_arm, R):
                    continue
                collides = False
                for prev_ax, _, prev_r, _, _ in specs:
                    if self._arm_collides(cand_axis, r_arm, prev_ax, prev_r):
                        collides = True
                        break
                if not collides:
                    specs.append((cand_axis, cand_axis.length, r_arm, False, rng.uniform(0, 2 * math.pi)))
                    break
        if p.arm_branching > 0.0:
            specs += self._secondary_arms(list(specs), R, rng)
        return specs

    def _secondary_arms(self, arms: list, R: float, rng) -> list:
        """Arms on arms (dense candelabra crowns such as Pachycereus weberi), with the same collision tests."""
        p = self.p
        out_specs = []
        everything = list(arms)
        for ax, length, r_par, _, _ in arms:
            for _ in range(int(rng.poisson(p.arm_branching))):
                for attempt in range(10):
                    s0 = length * rng.uniform(0.45, 0.75)
                    P0, T0, N0, B0 = ax.frame(np.array(s0))
                    az = rng.uniform(0, 2 * math.pi)
                    side = _normalize(math.cos(az) * N0 + math.sin(az) * B0)
                    side = _normalize(side - UP * float(side @ UP))
                    r_sub = r_par * rng.uniform(0.8, 0.95)
                    reach = (r_par + r_sub) * rng.uniform(1.4, 2.2)
                    rise = (length - s0) * rng.uniform(0.5, 0.8)
                    lean = math.tan(math.radians(p.arm_lean_deg))
                    ctrl = [P0, P0 + side * reach * 0.6 + UP * reach * 0.15, P0 + side * reach + UP * reach * 0.6,
                            P0 + side * (reach + lean * rise) + UP * (reach * 0.6 + rise)]
                    t = np.linspace(0.0, 1.0, 30)[:, None]
                    c0, c1, c2, c3 = [np.asarray(c) for c in ctrl]
                    pts = ((1 - t) ** 3) * c0 + 3 * ((1 - t) ** 2) * t * c1 + 3 * (1 - t) * t ** 2 * c2 + t ** 3 * c3
                    cand = StemAxis(pts)
                    cand.parent = ax
                    if self._arm_trunk_collides(cand, r_sub, R):
                        continue
                    if any(self._arm_collides(cand, r_sub, a2, r2) for a2, _, r2, _, _ in everything if a2 is not ax):
                        continue
                    # Its own parent arm: only the part beyond the elbow must stay clear
                    if self._arm_collides(cand, r_sub, ax, r_par, start1=0.35):
                        continue
                    spec = (cand, cand.length, r_sub, False, rng.uniform(0, 2 * math.pi))
                    out_specs.append(spec)
                    everything.append(spec)
                    break
        return out_specs

    def _offsets(self, R: float, globose: bool, rng) -> list:
        p = self.p
        specs = []
        if p.offsets <= 0:
            return specs
        az0 = rng.uniform(0, 2 * math.pi)
        for i in range(p.offsets):
            scale = p.offset_scale * rng.uniform(0.6, 1.0)
            r_off = R * scale * rng.uniform(0.8, 1.0)
            h = p.height_m * scale * (rng.uniform(0.5, 1.0) if not globose else 1.0)
            lean = math.radians(rng.uniform(0, 12))
            base_az = az0 + i * GOLDEN
            base_d = R * rng.uniform(1.6, 2.6) * (1 + 0.3 * (i // 6))
            for attempt in range(8):
                az = base_az + rng.normal(0, 0.2) + (attempt * 0.15 if attempt > 0 else 0)
                d = max(R + r_off + 0.02, base_d * (1.0 + 0.1 * (attempt // 2)))
                base = np.array([d * math.cos(az), d * math.sin(az), 0.0])
                col = False
                for prev_ax, _, prev_r, _, _ in specs:
                    prev_base = prev_ax.P[0]
                    if np.linalg.norm(base[:2] - prev_base[:2]) < (r_off + prev_r) * 1.02:
                        col = True
                        break
                if not col:
                    top = base + h * np.array([math.sin(lean) * math.cos(az), math.sin(lean) * math.sin(az), math.cos(lean)])
                    pts = np.linspace(base, top, 12)
                    axis = StemAxis(pts)
                    specs.append((axis, axis.length, r_off, globose, rng.uniform(0, 2 * math.pi)))
                    break
        return specs

    # ------------------------------------------------------------------
    def _rib_factor(self, theta, rho_rel):
        p = self.p
        m = max(1, int(p.rib_count))
        if p.arrangement == AreoleArrangement.SPIRAL or p.rib_depth <= 0.0:
            return np.ones_like(theta), np.ones_like(theta)
        c = np.abs(np.cos(0.5 * m * theta))
        crest = c ** max(0.05, p.rib_sharpness)
        depth = p.rib_depth * np.clip(rho_rel, 0.0, 1.0) ** 0.7
        return 1.0 - depth * (1.0 - crest), crest

    def _areoles(self, gen: Generatrix, phase: float, rng, apex_sigma: float):
        """Areole lattice in (sigma, theta) on the generatrix surface."""
        p = self.p
        a = max(0.3, p.areole_spacing_cm) * 0.01
        if p.arrangement == AreoleArrangement.SPIRAL:
            per = a * a * 0.866
            n = int(gen.area[-1] / per)
            k = np.arange(n)
            sig = gen.sigma_of_area((k + 0.5) * per)
            theta = phase + k * GOLDEN
            return sig, np.mod(theta, 2 * math.pi)
        m = max(1, int(p.rib_count))
        sigs, ths = [], []
        start = 0.08 * gen.R + 0.02
        for j in range(m):
            off = 0.5 * a * (j % 2)
            s = np.arange(start + off, gen.length - 0.15 * a, a)
            sigs.append(s)
            ths.append(np.full(len(s), phase + 2 * math.pi * j / m))
        return np.concatenate(sigs), np.concatenate(ths)

    def _surface(self, axis: StemAxis, gen: Generatrix, sigma, theta, ar_sig, ar_th, R):
        """Surface points for (sigma, theta) arrays (any shape)."""
        p = self.p
        rho, z = gen.at(sigma)
        twist = math.radians(p.rib_twist_deg_per_m) * z
        th = theta + twist
        F, crest = self._rib_factor(th - self._phase, rho / max(1e-6, R))
        bump = np.zeros_like(rho)
        if p.tubercle_height > 0.0 and len(ar_sig):
            a = max(0.3, p.areole_spacing_cm) * 0.01
            sb = 0.38 * a
            flat_s = sigma.reshape(-1)
            flat_t = th.reshape(-1)
            flat_r = rho.reshape(-1)
            out = np.zeros(flat_s.shape)
            ar_rho, ar_z = gen.at(ar_sig)
            ar_t = ar_th + math.radians(p.rib_twist_deg_per_m) * ar_z
            for i0 in range(0, len(flat_s), 4096):
                ds = flat_s[i0:i0 + 4096, None] - ar_sig[None, :]
                dt = np.angle(np.exp(1j * (flat_t[i0:i0 + 4096, None] - ar_t[None, :])))
                d2 = ds ** 2 + (dt * np.maximum(flat_r[i0:i0 + 4096, None], 1e-4)) ** 2
                out[i0:i0 + 4096] = np.exp(-d2.min(axis=1) / (2 * sb * sb))
            bump = out.reshape(rho.shape)
        r = rho * F * (1.0 + p.tubercle_height * bump)
        C, T, N, Bv = axis.frame(z)
        pos = C + r[..., None] * (np.cos(theta)[..., None] * N + np.sin(theta)[..., None] * Bv)
        return pos, crest, bump

    def _stem(self, axis: StemAxis, length: float, R: float, globose: bool, phase: float, rng, detail: float):
        p = self.p
        self._phase = phase
        gen = Generatrix(length, R, p, globose)
        ar_sig, ar_th = self._areoles(gen, phase, rng, gen.length)
        # Areoles follow twisted ribs: their position angle undoes the helical twist
        ar_th = ar_th - math.radians(p.rib_twist_deg_per_m) * gen.at(ar_sig)[1]
        m = max(1, int(p.rib_count))
        cols = int(max(48, (m * 8 if p.arrangement == AreoleArrangement.RIBS else 96)) * detail)
        if p.arrangement == AreoleArrangement.RIBS:
            cols = max(m * max(6, int(8 * detail)), cols - cols % m)
        a = max(0.3, p.areole_spacing_cm) * 0.01
        row_step = min(a / (2.5 if p.tubercle_height > 0 else 1.5), R / 4.0) / max(0.25, detail)
        rows = int(np.clip(gen.length / row_step, 40, 1500))
        sig = np.linspace(0.0, gen.length, rows)
        sig = sig[:-1]  # Apex closed by a pole vertex
        th = np.linspace(0.0, 2 * math.pi, cols, endpoint=False)
        S, TH = np.meshgrid(sig, th, indexing="ij")
        pos, crest, bump = self._surface(axis, gen, S, TH, ar_sig, ar_th, R)
        verts = pos.reshape(-1, 3)
        quads = _quads_grid(len(sig), cols)
        apex_rho, apex_z = gen.at(np.array([gen.length]))
        apex = axis.frame(apex_z)[0][0]
        apex_idx = len(verts)
        last = (len(sig) - 1) * cols
        tris = np.stack([last + np.arange(cols), last + (np.arange(cols) + 1) % cols, np.full(cols, apex_idx)], 1)
        base_cap = [list(range(cols))[::-1]]
        verts = np.vstack([verts, apex[None, :]])
        crest = np.concatenate([crest.reshape(-1), [1.0]])
        bump = np.concatenate([bump.reshape(-1), [0.0]])
        rel_h = np.concatenate([np.repeat(gen.at(sig)[1] / max(1e-6, length), cols), [1.0]])
        uv = np.concatenate([np.stack([np.tile(th / (2 * math.pi), len(sig)), np.repeat(sig, cols)], 1), [[0.5, gen.length]]])
        stem = _mesh(verts, quads, tris, base_cap, uv,
                     {"rib": crest.astype(np.float32), "tubercle": bump.astype(np.float32),
                      "height_rel": rel_h.astype(np.float32)})

        # Areole positions, normals and tangents (finite differences on the surface)
        if len(ar_sig):
            eps_s, eps_t = 1e-3, 1e-3
            P0, _, _ = self._surface(axis, gen, ar_sig, ar_th, ar_sig, ar_th, R)
            Ps, _, _ = self._surface(axis, gen, ar_sig + eps_s, ar_th, ar_sig, ar_th, R)
            Pt, _, _ = self._surface(axis, gen, ar_sig, ar_th + eps_t, ar_sig, ar_th, R)
            ts = _normalize(Ps - P0)
            tt = _normalize(Pt - P0)
            nrm = _normalize(np.cross(tt, ts))
            centre = axis.frame(gen.at(ar_sig)[1])[0]
            flip = np.sum(nrm * (P0 - centre), axis=1) < 0
            nrm[flip] *= -1
            apex_w = np.clip((ar_sig / gen.length - 0.85) / 0.15, 0.0, 1.0)
            # Flowering zone: areoles of the youngest growth, in a ring just below the apex (not the very tip)
            flw = apex_w ** 0.7 * (1.0 - 0.7 * apex_w ** 4)
            areoles = {"pos": P0, "normal": nrm, "tangent": ts, "scale": 1.0 - 0.6 * apex_w, "apex": apex_w,
                       "flower": flw}
        else:
            z = np.zeros((0, 3))
            areoles = {"pos": z, "normal": z, "tangent": z, "scale": np.zeros(0), "apex": np.zeros(0),
                       "flower": np.zeros(0)}
        return stem, areoles

    # ------------------------------------------------------------------
    @staticmethod
    def _pad_inside(pts: np.ndarray, geom, inflate: float = 1.05) -> np.ndarray:
        """Exact inside test against a cladode's volume (same obovate outline and thickness profile)."""
        centre, e1, e2, n, L, W, Th, base = geom
        d = pts - centre
        x, y, z = d @ e1, d @ e2, d @ n
        phi = np.arctan2(y / (0.5 * L), x / (0.5 * W))
        obov = 1.0 + 0.12 * np.sin(phi)
        r = np.hypot(x / (0.5 * W * obov), y / (0.5 * L))
        th = 0.5 * Th * np.clip(1.0 - np.minimum(r, 1.0) ** 2.5, 0.0, 1.0) ** 0.5
        return (r < inflate) & (np.abs(z) < th * inflate + 0.02 * Th)

    @staticmethod
    def _pad_surface_points(geom, n_r: int = 7, n_phi: int = 28) -> np.ndarray:
        """Points on both faces and the rim of a cladode."""
        centre, e1, e2, n, L, W, Th, base = geom
        rr = np.linspace(0.15, 1.0, n_r)
        ph = np.linspace(0.0, 2 * math.pi, n_phi, endpoint=False)
        R_, P_ = np.meshgrid(rr, ph, indexing="ij")
        obov = 1.0 + 0.12 * np.sin(P_)
        x = 0.5 * W * R_ * np.cos(P_) * obov
        y = 0.5 * L * R_ * np.sin(P_)
        t = 0.5 * Th * np.clip(1.0 - R_ ** 2.5, 0.0, 1.0) ** 0.5
        pts = [centre + x[..., None] * e1 + y[..., None] * e2 + (s * t)[..., None] * n for s in (1.0, -1.0)]
        return np.concatenate([q.reshape(-1, 3) for q in pts])

    def _pads_overlap(self, new_geom, old_geom, is_parent: bool) -> bool:
        """Two-way exact penetration test; around a daughter's insertion on its parent, contact is allowed."""
        a = self._pad_surface_points(new_geom)
        b = self._pad_surface_points(old_geom)
        if is_parent:
            base, L_new = new_geom[7], new_geom[4]
            a = a[np.linalg.norm(a - base, axis=1) > 0.22 * L_new]
            b = b[np.linalg.norm(b - base, axis=1) > 0.22 * L_new]
        return bool(self._pad_inside(a, old_geom).any() or self._pad_inside(b, new_geom).any())

    def _pad_frame(self, base, up, roll, L, parent_n=None):
        """Compute coordinate frame and dimensions for a cladode."""
        p = self.p
        W = L * p.pad_width_ratio
        Th = L * p.pad_thickness_ratio
        e2 = _normalize(up)
        if parent_n is None:
            side = np.cross(e2, UP)
            if np.linalg.norm(side) < 1e-6:
                side = np.array([1.0, 0.0, 0.0])
            side = _normalize(side)
            e1 = _normalize(math.cos(roll) * side + math.sin(roll) * np.cross(e2, side))
        else:
            # Align width axis with the margin tangent in the parent's plane,
            # using roll as the natural dihedral twist angle.
            side = np.cross(parent_n, e2)
            if np.linalg.norm(side) < 1e-4:
                side = np.cross(UP, e2)
                if np.linalg.norm(side) < 1e-4:
                    side = np.array([1.0, 0.0, 0.0])
            side = _normalize(side)
            n0 = _normalize(np.cross(e2, side))
            e1 = _normalize(math.cos(roll) * side + math.sin(roll) * n0)
        n = np.cross(e1, e2)
        centre = base + e2 * (0.5 * L * 0.92)
        return centre, e1, e2, n, L, W, Th, base

    def _pad_sample_points(self, pad_geom):
        """Dense volume and surface sample points for collision checking."""
        centre, e1, e2, n, L, W, Th, base = pad_geom
        pts = [centre, base]
        for t in np.linspace(-0.45, 0.45, 9):
            pts.append(centre + t * L * e2)
        for v_rel in [-0.3, -0.15, 0.0, 0.15, 0.3]:
            obov = 1.0 + 0.12 * v_rel
            w_max = 0.5 * W * obov * math.sqrt(max(0.0, 1.0 - (v_rel / 0.5) ** 2))
            for u_rel in np.linspace(-0.85, 0.85, 5):
                pts.append(centre + u_rel * w_max * e1 + v_rel * L * e2)
        for phi in np.linspace(0.0, 2 * math.pi, 16, endpoint=False):
            obov = 1.0 + 0.12 * math.sin(phi)
            x = 0.5 * W * math.cos(phi) * obov * 0.95
            y = 0.5 * L * math.sin(phi) * 0.95
            pts.append(centre + x * e1 + y * e2)
        for phi in np.linspace(0.0, 2 * math.pi, 8, endpoint=False):
            for sgn in [-1.0, 1.0]:
                x = 0.25 * W * math.cos(phi)
                y = 0.25 * L * math.sin(phi)
                z = sgn * 0.35 * Th
                pts.append(centre + x * e1 + y * e2 + z * n)
        return np.array(pts)

    def _is_inside_pad(self, pts, pad_geom, margin=0.92):
        """Test whether 3D points lie inside a cladode's obovate volume."""
        centre, e1, e2, n, L, W, Th, _ = pad_geom
        d = pts - centre
        u = np.dot(d, e1)
        v = np.dot(d, e2)
        w = np.dot(d, n)
        v_norm = np.clip(v / (0.5 * L + 1e-9), -1.0, 1.0)
        obov = 1.0 + 0.12 * v_norm
        r_uv = np.hypot(u / (0.5 * W * obov + 1e-9), v_norm)
        thick = 0.5 * Th * np.sqrt(np.clip(1.0 - r_uv ** 2.5, 0.0, 1.0))
        inside = (r_uv < margin) & (np.abs(w) < (thick * margin + 1e-4))
        return inside

    def _pads_collide(self, p1_geom, p2_geom, is_parent=False):
        """Test whether two cladodes penetrate each other (excluding natural base joint)."""
        c1, _, _, _, L1, _, _, _ = p1_geom
        c2, _, e2_2, _, L2, _, _, b2 = p2_geom
        if np.linalg.norm(c1 - c2) > (0.55 * L1 + 0.55 * L2):
            return False
        pts1 = self._pad_sample_points(p1_geom)
        pts2 = self._pad_sample_points(p2_geom)
        if is_parent:
            # Child (p2) body penetrating parent (p1)
            v2 = np.dot(pts2 - c2, e2_2)
            pts2_body = pts2[v2 > -0.32 * L2]
            if np.any(self._is_inside_pad(pts2_body, p1_geom)):
                return True
            # Parent (p1) penetrating child body away from insertion base
            d_base = np.linalg.norm(pts1 - b2, axis=1)
            pts1_away = pts1[d_base > 0.18 * L2]
            if np.any(self._is_inside_pad(pts1_away, p2_geom)):
                return True
            return False
        else:
            if np.any(self._is_inside_pad(pts1, p2_geom)):
                return True
            if np.any(self._is_inside_pad(pts2, p1_geom)):
                return True
            return False

    def _opuntia(self, rng, detail):
        """
        Chains of flattened cladodes (Opuntia).
        Biological principles:
        - Areoles along the upper margin develop daughter pads with lateral inhibition
          (apical dominance) preventing duplicate or overlapping insertions.
        - Daughter pads align their planar width axis with the parent margin, fanning
          outward with controlled dihedral twist.
        - Negative autotropism and phototropism adjust growth trajectories to seek free space.
        - Steric hindrance / density-dependent bud dormancy suppresses buds that cannot
          grow without colliding with existing pads.
        """
        p = self.p
        L = p.pad_length_cm * 0.01
        parts = []
        base0 = np.array([0.0, 0.0, -0.1 * L])
        az0 = rng.uniform(0, 2 * math.pi)
        mesh0, ar0, rim0 = self._pad(base0, UP.copy(), az0, L, rng, detail, parent_n=None)
        geom0 = self._pad_frame(base0, UP.copy(), az0, L, parent_n=None)
        parts.append((mesh0, ar0))

        placed_pads = [{'idx': 0, 'parent': None, 'level': 0, 'geom': geom0, 'scale': 1.0}]
        queue = [placed_pads[0]]

        while queue and len(placed_pads) < 200:
            parent = queue.pop(0)
            level = parent['level']
            if level + 1 >= p.pad_levels:
                continue
            n_child = int(rng.poisson(p.pad_branching))
            if p.pad_branching >= 1.0:
                # Young plants keep branching: at least two daughters on the basal pad, one above
                n_child = max(n_child, 2 if level == 0 else (1 if level + 2 < p.pad_levels else 0))
            if n_child <= 0:
                continue
            n_child = min(4, n_child)

            p_centre, p_e1, p_e2, p_n, p_L, p_W, p_Th, _ = parent['geom']

            # Stratified margin areoles (lateral inhibition along the upper rim)
            if n_child == 1:
                base_phis = [math.pi * 0.5 + rng.uniform(-0.06, 0.06)]
            elif n_child == 2:
                base_phis = [math.pi * 0.32 + rng.uniform(-0.04, 0.04),
                             math.pi * 0.68 + rng.uniform(-0.04, 0.04)]
            elif n_child == 3:
                base_phis = [math.pi * 0.25 + rng.uniform(-0.03, 0.03),
                             math.pi * 0.50 + rng.uniform(-0.03, 0.03),
                             math.pi * 0.75 + rng.uniform(-0.03, 0.03)]
            else:
                base_phis = [math.pi * (0.2 + 0.6 * (i + 0.5) / n_child) + rng.uniform(-0.02, 0.02)
                             for i in range(n_child)]

            for b_phi in base_phis:
                child_scale = parent['scale'] * rng.uniform(0.78, 0.95)
                child_L = L * child_scale
                accepted = False

                # Phototropism / collision avoidance: test candidate trajectories
                for attempt in range(12):
                    phi = b_phi + (rng.uniform(-0.08, 0.08) * (1 + attempt // 3) if attempt > 0 else 0.0)
                    pos = (p_centre + 0.5 * p_W * 0.9 * math.cos(phi) * p_e1 * 1.1 +
                           0.5 * p_L * 0.92 * math.sin(phi) * p_e2)
                    outward = _normalize(pos - p_centre)
                    up_weight = rng.uniform(0.4, 0.8) + (attempt * 0.12)
                    child_up = _normalize(outward * 0.65 + UP * up_weight)
                    twist = rng.normal(0.0, math.radians(16.0)) + (attempt * rng.choice([-0.25, 0.25]))

                    cand_geom = self._pad_frame(pos, child_up, twist, child_L, parent_n=p_n)

                    # Collision test against all previously placed cladodes
                    collides = False
                    for ex in placed_pads:
                        is_par = (ex['idx'] == parent['idx'])
                        if self._pads_overlap(cand_geom, ex['geom'], is_par):
                            collides = True
                            break
                    if not collides:
                        mesh, ar, rim = self._pad(pos, child_up, twist, child_L, rng, detail, parent_n=p_n)
                        pad_entry = {'idx': len(placed_pads), 'parent': parent['idx'], 'level': level + 1,
                                     'geom': cand_geom, 'scale': child_scale}
                        placed_pads.append(pad_entry)
                        parts.append((mesh, ar))
                        queue.append(pad_entry)
                        accepted = True
                        break

        return parts

    def _pad(self, base, up, roll, L, rng, detail, parent_n=None):
        p = self.p
        centre, e1, e2, n, L, W, Th, base = self._pad_frame(base, up, roll, L, parent_n)
        nr, nf = int(10 * detail) + 4, int(32 * detail) + 8
        rr = np.linspace(0.0, 1.0, nr)[1:]
        ph = np.linspace(0.0, 2 * math.pi, nf, endpoint=False)
        Rr, Ph = np.meshgrid(rr, ph, indexing="ij")
        obov = 1.0 + 0.12 * np.sin(Ph)  # Obovate: wider toward the top
        x = 0.5 * W * Rr * np.cos(Ph) * obov
        y = 0.5 * L * Rr * np.sin(Ph)
        thick = 0.5 * Th * np.clip(1.0 - Rr ** 2.5, 0.0, 1.0) ** 0.5
        verts, uvs = [], []
        for sgn in (1.0, -1.0):
            pts = centre + x[..., None] * e1 + y[..., None] * e2 + (sgn * thick)[..., None] * n
            verts.append(np.vstack([(centre + sgn * 0.5 * Th * n)[None, :], pts.reshape(-1, 3)]))
        top, bot = verts
        # Build: top centre(0) + top rings, bottom centre + bottom rings, sharing the rim ring
        top_rings = top[1:].reshape(nr - 1, nf, 3)
        bot_rings = bot[1:].reshape(nr - 1, nf, 3)[:-1]
        V = np.vstack([top[:1], top_rings.reshape(-1, 3), bot[:1], bot_rings.reshape(-1, 3)])
        t0, tr = 0, 1
        b0 = 1 + (nr - 1) * nf
        br = b0 + 1
        faces_q, faces_t = [], []
        for j in range(nf):
            faces_t.append([t0, tr + j, tr + (j + 1) % nf])
        faces_q += list(_quads_grid(nr - 1, nf, tr))
        rim = tr + (nr - 2) * nf
        nb = nr - 2
        for j in range(nf):
            faces_t.append([b0, br + (j + 1) % nf, br + j])
        for i in range(nb - 1):
            for j in range(nf):
                a0 = br + i * nf + j
                a1 = br + i * nf + (j + 1) % nf
                faces_q.append([a0, a1, a1 + nf, a0 + nf][::-1])
        last_b = br + (nb - 1) * nf
        for j in range(nf):
            faces_q.append([last_b + j, rim + j, rim + (j + 1) % nf, last_b + (j + 1) % nf])
        attrs = {"rib": np.ones(len(V), np.float32), "tubercle": np.zeros(len(V), np.float32),
                 "height_rel": np.full(len(V), 0.5, np.float32)}
        mesh = _mesh(V, faces_q, faces_t, None, None, attrs)

        # Areoles: equal-area spiral on both faces (disc model of Vogel 1979)
        a = max(0.3, p.areole_spacing_cm) * 0.01
        area = math.pi * 0.25 * L * W
        N = int(area / (a * a))
        k = np.arange(N)
        rk = np.sqrt((k + 0.5) / max(1, N)) * 0.9
        pk = k * GOLDEN + rng.uniform(0, 2 * math.pi)
        xs = 0.5 * W * rk * np.cos(pk) * (1 + 0.12 * np.sin(pk))
        ys = 0.5 * L * rk * np.sin(pk)
        tk = 0.5 * Th * np.clip(1.0 - rk ** 2.5, 0.0, 1.0) ** 0.5
        pos, nrm, tan = [], [], []
        for sgn in (1.0, -1.0):
            pos.append(centre + xs[:, None] * e1 + ys[:, None] * e2 + (sgn * tk)[:, None] * n)
            nrm.append(np.repeat((sgn * n)[None, :], N, axis=0))
            tan.append(np.repeat(e2[None, :], N, axis=0))
        rim_pts = []
        for phi in np.linspace(0.15 * math.pi, 0.85 * math.pi, 9):
            q = centre + 0.5 * W * 0.9 * math.cos(phi) * e1 * 1.1 + 0.5 * L * 0.92 * math.sin(phi) * e2
            rim_pts.append((q, _normalize(q - centre)))
        # Opuntia flowers arise from areoles on the distal margin of the cladode
        fw = np.clip((rk - 0.65) / 0.25, 0.0, 1.0) * np.clip(ys / (0.5 * L) * 1.4, 0.0, 1.0)
        ar = {"pos": np.vstack(pos), "normal": np.vstack(nrm), "tangent": np.vstack(tan),
              "scale": np.ones(2 * N), "apex": np.zeros(2 * N), "flower": np.concatenate([fw, fw])}
        return mesh, ar, rim_pts

    # ------------------------------------------------------------------
    def _spines(self, pos, nrm, tan, scale, apex, rng, budget: int = 60000):
        """Radial + central spines (3-sided tapered tubes, 3 segments) and areolar wool cushions."""
        p = self.p
        n_ar = len(pos)
        if n_ar == 0:
            return MeshData.empty(), 0
        bit = np.cross(nrm, tan)
        dirs, lens, kinds, owners = [], [], [], []
        nr, nc = max(0, int(p.radial_spines)), max(0, int(p.central_spines))
        # Budget: keep every areole and its centrals; thin radials evenly when over budget
        if n_ar * (nr + nc) > budget and nr > 0:
            nr = int(np.clip((budget / max(1, n_ar)) - nc, min(nr, 2), nr))
        lift = math.radians(p.radial_lift_deg)
        for k in range(nr):
            ang = 2 * math.pi * k / max(1, nr) + rng.normal(0, 0.15 * p.spine_jitter, n_ar)
            d = np.cos(lift) * (np.cos(ang)[:, None] * tan + np.sin(ang)[:, None] * bit) + np.sin(lift) * nrm
            dirs.append(d)
            lens.append(p.radial_length_cm * 0.01 * scale * rng.uniform(1 - p.spine_jitter, 1 + p.spine_jitter, n_ar))
            kinds.append(np.zeros(n_ar))
            owners.append(np.arange(n_ar))
        for k in range(nc):
            spread = math.radians(25.0) if nc > 1 else 0.0
            ang = 2 * math.pi * k / max(1, nc) + rng.normal(0, 0.3, n_ar)
            el = math.pi / 2 - spread * rng.uniform(0.5, 1.0, n_ar)
            d = (np.cos(el)[:, None] * (np.cos(ang)[:, None] * tan + np.sin(ang)[:, None] * bit)
                 + np.sin(el)[:, None] * nrm)
            # Many centrals point slightly downward (protecting the stem)
            d = _normalize(d - 0.25 * tan)
            dirs.append(d)
            lens.append(p.central_length_cm * 0.01 * scale * rng.uniform(1 - p.spine_jitter, 1 + p.spine_jitter, n_ar))
            kinds.append(np.ones(n_ar))
            owners.append(np.arange(n_ar))
        if not dirs:
            D = np.zeros((0, 3))
            Ls = np.zeros(0)
            K = np.zeros(0)
            O = np.zeros(0, int)
        else:
            D = _normalize(np.vstack(dirs))
            Ls = np.concatenate(lens)
            K = np.concatenate(kinds)
            O = np.concatenate(owners)
        keep = Ls > 1e-4
        D, Ls, K, O = D[keep], Ls[keep], K[keep], O[keep]
        n_sp = len(D)
        parts = []
        if n_sp:
            u = np.array([0.0, 0.35, 0.7, 1.0])
            base = pos[O] + nrm[O] * (0.15 * p.wool * p.areole_spacing_cm * 0.01)
            down = -UP[None, :] - np.sum(-UP[None, :] * D, axis=1, keepdims=True) * D
            down = _normalize(down + 1e-9)
            hook_dir = _normalize(np.cross(D, bit[O]) + 1e-9)
            curv = p.spine_curvature * Ls
            hook = p.central_hook * Ls * K
            centre = (base[:, None, :] + Ls[:, None, None] * u[None, :, None] * D[:, None, :]
                      + (curv[:, None] * u[None, :] ** 2)[..., None] * down[:, None, :] * 0.5
                      + (hook[:, None] * np.clip((u[None, :] - 0.6) / 0.4, 0, 1) ** 2)[..., None] * hook_dir[:, None, :] * 0.4)
            r0 = 0.5 * p.spine_thickness_mm * 0.001 * (1.0 + 0.6 * K) * scale[O]
            ref = np.where(np.abs(D[:, 2:3]) > 0.9, np.array([1.0, 0.0, 0.0]), UP)
            e1 = _normalize(np.cross(D, ref))
            e2 = np.cross(D, e1)
            ang = np.array([0.0, 2 * math.pi / 3, 4 * math.pi / 3])
            ring_off = np.cos(ang)[None, :, None] * e1[:, None, :] + np.sin(ang)[None, :, None] * e2[:, None, :]
            radius = r0[:, None] * np.array([1.0, 0.7, 0.35])[None, :]
            rings = centre[:, :3, None, :] + radius[..., None, None] * ring_off[:, None, :, :]  # (n, 3, 3, 3)
            tip = centre[:, 3, :]
            V = np.concatenate([rings.reshape(n_sp, 9, 3), tip[:, None, :]], axis=1).reshape(-1, 3)
            per = 10
            off = (np.arange(n_sp) * per)[:, None, None]
            q = []
            for r in range(2):
                for a in range(3):
                    q.append([r * 3 + a, r * 3 + (a + 1) % 3, (r + 1) * 3 + (a + 1) % 3, (r + 1) * 3 + a])
            q = (np.array(q)[None] + off).reshape(-1, 4)
            t = (np.array([[6, 7, 9], [7, 8, 9], [8, 6, 9]])[None] + off).reshape(-1, 3)
            spine_t = np.tile(np.array([0, 0, 0, 0.45, 0.45, 0.45, 0.8, 0.8, 0.8, 1.0]), n_sp)
            parts.append(_mesh(V, q, t, None, None, {"spine_t": spine_t.astype(np.float32),
                                                     "wool": np.zeros(len(V), np.float32)}))
        # Wool cushions: low hexagonal domes (plus extra apical wool)
        w = (p.wool + p.apical_wool * apex) * p.areole_spacing_cm * 0.01 * 0.3 * scale
        has = w > 1e-4
        if np.any(has):
            c = pos[has]
            nn = nrm[has]
            tt = tan[has]
            bb = np.cross(nn, tt)
            ww = w[has]
            ang = np.linspace(0.0, 2 * math.pi, 6, endpoint=False)
            ring = c[:, None, :] + ww[:, None, None] * (np.cos(ang)[None, :, None] * tt[:, None, :]
                                                        + np.sin(ang)[None, :, None] * bb[:, None, :])
            top = c + nn * ww[:, None] * 0.6
            V = np.concatenate([ring, top[:, None, :]], axis=1).reshape(-1, 3)
            per = 7
            off = (np.arange(len(c)) * per)[:, None, None]
            t = (np.array([[a, (a + 1) % 6, 6] for a in range(6)])[None] + off).reshape(-1, 3)
            parts.append(_mesh(V, None, t, None, None,
                               {"spine_t": np.zeros(len(V), np.float32), "wool": np.ones(len(V), np.float32)}))
        return MeshData.concatenate(parts), n_sp
