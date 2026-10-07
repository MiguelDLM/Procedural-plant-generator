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
    arm_length_ratio: float = 0.5   # Arm length relative to the stem above its insertion
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


@dataclass
class CactusResult:
    stem: MeshData
    spines: MeshData
    areole_count: int
    spine_count: int
    height_m: float


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
    faces = []
    if quads is not None and len(quads):
        faces += [list(q) for q in quads]
    if tris is not None and len(tris):
        faces += [list(t) for t in tris]
    if ngons:
        faces += [list(n) for n in ngons]
    totals = np.array([len(f) for f in faces], dtype=np.int32)
    loops = np.concatenate([np.asarray(f, dtype=np.int32) for f in faces]) if faces else np.zeros(0, np.int32)
    starts = np.concatenate([[0], np.cumsum(totals[:-1])]).astype(np.int32) if len(totals) else np.zeros(0, np.int32)
    if uvs is None:
        loop_uv = np.zeros((len(loops), 2), np.float32)
    else:
        loop_uv = np.asarray(uvs, dtype=np.float32)[loops]
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
    def generate(self, seed: int = 7, detail: float = 1.0, spine_budget: int = 60000) -> CactusResult:
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
        spine_mesh, n_sp = self._spines(ar_pos, ar_nrm, ar_tan, ar_scale, ar_apex, rng, spine_budget)
        return CactusResult(stem, spine_mesh, len(ar_pos), n_sp,
                            float(max(stem.vertices[:, 2].max(), 0.0)) if len(stem.vertices) else 0.0)

    # ------------------------------------------------------------------
    def _arms(self, main: StemAxis, R: float, rng) -> list:
        p = self.p
        specs = []
        if p.arm_count <= 0:
            return specs
        az0 = rng.uniform(0, 2 * math.pi)
        for k in range(p.arm_count):
            h = p.height_m * rng.uniform(p.arm_height_min, p.arm_height_max)
            az = az0 + 2 * math.pi * k / p.arm_count + rng.normal(0, 0.35)
            out = np.array([math.cos(az), math.sin(az), 0.0])
            r_arm = R * p.arm_radius_ratio * rng.uniform(0.85, 1.05)
            reach = max(R + r_arm, p.arm_reach_m * rng.uniform(0.8, 1.2))
            rise = p.arm_length_ratio * (p.height_m - h) * rng.uniform(0.6, 1.1) + 2 * r_arm
            base = main.frame(np.array(h))[0]
            # Elbow: out and slightly up, then turning vertical (saguaro / candelabra arms)
            ctrl = [base, base + out * reach * 0.6 + UP * reach * 0.15, base + out * reach + UP * reach * 0.6,
                    base + out * reach * 1.05 + UP * (reach * 0.6 + rise)]
            t = np.linspace(0.0, 1.0, 40)[:, None]
            c0, c1, c2, c3 = [np.asarray(c) for c in ctrl]
            pts = ((1 - t) ** 3) * c0 + 3 * ((1 - t) ** 2) * t * c1 + 3 * (1 - t) * t ** 2 * c2 + t ** 3 * c3
            axis = StemAxis(pts)
            specs.append((axis, axis.length, r_arm, False, rng.uniform(0, 2 * math.pi)))
        return specs

    def _offsets(self, R: float, globose: bool, rng) -> list:
        p = self.p
        specs = []
        for k in range(p.offsets):
            az = k * GOLDEN + rng.normal(0, 0.3)
            d = R * rng.uniform(1.6, 2.6) * (1 + 0.3 * (k // 6))
            scale = p.offset_scale * rng.uniform(0.6, 1.0)
            h = p.height_m * scale * (rng.uniform(0.5, 1.0) if not globose else 1.0)
            lean = math.radians(rng.uniform(0, 12))
            base = np.array([d * math.cos(az), d * math.sin(az), 0.0])
            top = base + h * np.array([math.sin(lean) * math.cos(az), math.sin(lean) * math.sin(az), math.cos(lean)])
            pts = np.linspace(base, top, 12)
            axis = StemAxis(pts)
            specs.append((axis, axis.length, R * scale * rng.uniform(0.8, 1.0), globose, rng.uniform(0, 2 * math.pi)))
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
            areoles = {"pos": P0, "normal": nrm, "tangent": ts, "scale": 1.0 - 0.6 * apex_w, "apex": apex_w}
        else:
            z = np.zeros((0, 3))
            areoles = {"pos": z, "normal": z, "tangent": z, "scale": np.zeros(0), "apex": np.zeros(0)}
        return stem, areoles

    # ------------------------------------------------------------------
    def _opuntia(self, rng, detail):
        """Chains of flattened cladodes; areoles on an equal-area spiral on both faces."""
        p = self.p
        L = p.pad_length_cm * 0.01
        parts = []
        queue = [(np.array([0.0, 0.0, -0.1 * L]), UP.copy(), rng.uniform(0, 2 * math.pi), 1.0, 0)]
        count = 0
        while queue and count < 200:
            base, up, roll, scale, level = queue.pop(0)
            count += 1
            mesh, ar, rim = self._pad(base, up, roll, L * scale, rng, detail)
            parts.append((mesh, ar))
            if level + 1 >= p.pad_levels:
                continue
            n_child = rng.poisson(p.pad_branching)
            for _ in range(min(4, n_child)):
                # Daughter pads arise from upper-margin areoles
                a = rng.uniform(0.15, 0.85)
                j = int(a * (len(rim) - 1))
                pos, outward = rim[j]
                child_up = _normalize(outward * 0.6 + UP * rng.uniform(0.5, 1.0))
                queue.append((pos, child_up, rng.uniform(0, 2 * math.pi), scale * rng.uniform(0.75, 1.0), level + 1))
        return parts

    def _pad(self, base, up, roll, L, rng, detail):
        p = self.p
        W = L * p.pad_width_ratio
        Th = L * p.pad_thickness_ratio
        e2 = _normalize(up)
        side = np.cross(e2, UP)
        if np.linalg.norm(side) < 1e-6:
            side = np.array([1.0, 0.0, 0.0])
        side = _normalize(side)
        e1 = _normalize(math.cos(roll) * side + math.sin(roll) * np.cross(e2, side))
        n = np.cross(e1, e2)
        centre = base + e2 * (0.5 * L * 0.92)
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
        ar = {"pos": np.vstack(pos), "normal": np.vstack(nrm), "tangent": np.vstack(tan),
              "scale": np.ones(2 * N), "apex": np.zeros(2 * N)}
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
