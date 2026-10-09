"""
Procedural flowers from floral diagrams.

A flower is described the way botanists describe it, by a floral diagram and a
few geometric descriptors per organ (Ijiri et al. 2005, "Floral diagrams and
inflorescences", ACM TOG 24(3); Ronse De Craene 2010, *Floral Diagrams*):

- Organ identity by concentric zones (ABC model, Coen & Meyerowitz 1991): calyx,
  corolla, androecium, gynoecium, on a receptacle.
- Arrangement: whorled with merosity m (organs of successive whorls alternate by
  pi/m) or spiral at the golden angle (Magnolia, Nymphaea, cactus flowers, double
  roses). Whorls emerge from sequential spiral initiation with repulsion
  (Kitazawa & Fujimoto 2015, PLoS Comput Biol 11: e1004145), so both share one
  generator: organ k sits at azimuth k * divergence.
- Symmetry: actinomorphic or zygomorphic, as a continuous dorsiventral modulation
  of organ size, elevation and azimuth along d = cos(theta - theta_dorsal)
  (Endress 2001, Curr Opin Plant Biol 4: 86-91).
- Petal outline: the leaf half-width model (base/apex angles and curvatures) plus
  claw, apex truncation, emargination and fringing; the petal grows along its
  proximodistal axis (Rolland-Lagan et al. 2003, Nature 422: 161-163), so shape
  is written as width along that axis.
- Sympetaly: a corolla tube (surface of revolution) carrying the free lobes.
- Capitula (Asteraceae): florets on Vogel's (1979) spiral r = c sqrt(k), theta =
  k * 137.508 deg, ray florets (ligules) on the rim or over the whole head (double).

Each vertex carries `organ`, `pu` (proximodistal position along the perianth),
`pv` (across the organ, -1..1), `whorl` (0 outer .. 1 inner) and `orand`, which
drive the shader. Flowers are built in a local frame: receptacle at the origin,
floral axis +Z, dorsal side +Y.
"""

from dataclasses import dataclass, field
from enum import Enum
import math
import numpy as np

from .leaf_morphology import half_width_profile, T_GRID
from .mesh_engine import MeshData

GOLDEN = math.radians(137.50776)

# Organ codes (vertex attribute `organ`)
SEPAL, PETAL, FILAMENT, ANTHER, PISTIL, RECEPTACLE, FLORET, STEM = range(8)


class Arrangement(str, Enum):
    WHORLED = "Whorled"
    SPIRAL = "Spiral"


@dataclass
class FlowerProfile:
    # Floral diagram
    arrangement: Arrangement = Arrangement.WHORLED
    merosity: int = 5                # Organs per whorl
    petal_whorls: int = 1            # > 1: double flowers (extra petal whorls replace stamens)
    spiral_tepals: int = 0           # Perianth organs when spiral (Magnolia, cactus, double rose)
    zygomorphy: float = 0.0          # 0 radial symmetry .. 1 strongly bilateral
    lip_bias: float = 0.0            # + dorsal organs larger (standard), - ventral lip larger
    petals_visible: bool = True      # False: perianth reduced or shed (Eucalyptus operculum, catkin flowers)

    # Perianth (petal or tepal) geometry
    petal_length_cm: float = 2.0
    petal_aspect: float = 1.2        # Length / width
    widest_position: float = 0.6
    base_angle_deg: float = 60.0
    apex_angle_deg: float = 140.0
    apex_curvature: float = 0.6
    claw: float = 0.0                # Narrow stalk-like base (unguiculate petals), fraction of length
    truncation: float = 0.0          # Blunt, wide apex (0 pointed .. 1 truncate)
    notch: float = 0.0               # Emarginate apex (cherry, Silene)
    fringe: float = 0.0              # Laciniate / toothed apex (Dianthus, ligule teeth)
    fringe_count: int = 0
    undulation: float = 0.0          # Wavy margins
    opening_deg: float = 15.0        # Elevation of the organ base above the floral plane (90 = erect)
    reflex_deg: float = -10.0        # Bending along the organ (+ incurved, - reflexed)
    cup: float = 0.25                # Transverse concavity of the blade
    twist_deg: float = 0.0           # Twist along the organ (convolute, pinwheel)
    inner_scale: float = 0.7         # Size of innermost perianth organ relative to outer (spiral / double)
    inner_opening_deg: float = 60.0  # Elevation of innermost perianth organs (spiral / double)

    # Corolla tube (sympetaly)
    tube_length_cm: float = 0.0
    tube_radius_cm: float = 0.2      # At the base
    tube_flare: float = 2.0          # Exponent of the tube profile (1 conical, >1 trumpet)
    limb_fusion: float = 0.0         # Fused limb: lobes widen to fill their sector (1 = entire rim, Ipomoea)

    # Calyx
    sepal_length_ratio: float = 0.5  # Relative to petal length (0 = no visible calyx)
    sepal_aspect: float = 2.0
    sepal_opening_deg: float = 20.0
    hypanthium_cm: float = 0.3       # Floral cup / inferior ovary below the perianth (cactus pericarpel)
    hypanthium_scales: int = 0       # Bract scales on it (cactus pericarpel)

    # Androecium
    stamen_count: int = 10
    stamen_length_ratio: float = 0.35
    stamen_spread_deg: float = 25.0
    anther_size_mm: float = 1.5
    staminal_column: float = 0.0     # Monadelphous column (Hibiscus): fraction of stamen length fused
    stamen_declination: float = 0.0  # Stamens curve toward the dorsal side (zygomorphic flowers)

    # Gynoecium and receptacle
    carpels: int = 5                 # Stigma lobes
    style_length_ratio: float = 0.3
    stigma_size_mm: float = 1.2
    receptacle_radius_cm: float = 0.2
    receptacle_height_cm: float = 0.1   # > radius: conical receptacle (Magnolia gynoecium)

    # Capitulum (Asteraceae head)
    capitulum: bool = False
    disc_florets: int = 0
    disc_radius_cm: float = 1.0
    disc_dome: float = 0.15
    ray_count: int = 0               # Ray florets; spread over the whole head when ray_fill > 0 (double)
    ray_fill: float = 0.0            # 0 one rim row .. 1 rays fill the head (Dahlia, Tagetes)
    involucre_bracts: int = 0

    # Colour (sRGB)
    petal_color: tuple = (0.95, 0.85, 0.90)
    tip_color: tuple = (0.95, 0.85, 0.90)
    tip_start: float = 0.6           # Where the tip colour starts along the petal
    eye_color: tuple = (0.95, 0.85, 0.3)
    eye_size: float = 0.0            # Contrasting base zone (0..1 of the perianth length)
    guide_lines: int = 0             # Nectar guides per petal
    guide_contrast: float = 0.0
    spots: float = 0.0               # Dark spots on the inner perianth (Lilium, Aesculus)
    spot_color: tuple = (0.35, 0.08, 0.05)
    outer_color: tuple = (0.45, 0.55, 0.30)   # Outer tepals / sepaloid gradient
    outer_tint: float = 0.0
    sepal_color: tuple = (0.30, 0.45, 0.18)
    stamen_color: tuple = (0.95, 0.92, 0.75)
    anther_color: tuple = (0.95, 0.75, 0.15)
    pistil_color: tuple = (0.80, 0.85, 0.55)
    disc_color: tuple = (0.30, 0.17, 0.07)
    disc_center_color: tuple = (0.30, 0.35, 0.10)
    sheen: float = 0.3               # Velvety epidermis (conical cells)
    translucency: float = 0.3


@dataclass
class FlowerResult:
    mesh: MeshData
    diameter_m: float
    height_m: float


# -----------------------------------------------------------------------------
def _normalize(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


class _Builder:
    """Accumulates vertices, faces and per-vertex attributes."""
    KEYS = ("organ", "pu", "pv", "whorl", "orand")

    def __init__(self):
        self.V, self.F, self.T, self.A = [], [], [], {k: [] for k in self.KEYS}
        self.n = 0

    def grid(self, P, organ, pu, pv, whorl, orand, close_v=False):
        """P: (nu, nv, 3) grid; attribute arrays broadcastable to (nu, nv)."""
        nu, nv = P.shape[:2]
        idx = self.n + np.arange(nu * nv).reshape(nu, nv)
        if close_v:
            idx2 = np.concatenate([idx, idx[:, :1]], axis=1)
        else:
            idx2 = idx
        a, b = idx2[:-1, :-1], idx2[1:, :-1]
        c, d = idx2[1:, 1:], idx2[:-1, 1:]
        self.F.append(np.stack([a, b, c, d], -1).reshape(-1, 4))
        self._push(P.reshape(-1, 3), organ, pu, pv, whorl, orand, (nu, nv))

    def tris(self, P, T, organ, pu, pv, whorl, orand):
        self.T.append(np.asarray(T) + self.n)
        self._push(P, organ, pu, pv, whorl, orand, (len(P),))

    def _push(self, P, organ, pu, pv, whorl, orand, shape):
        self.V.append(np.asarray(P, np.float64))
        for k, val in zip(self.KEYS, (organ, pu, pv, whorl, orand)):
            self.A[k].append(np.broadcast_to(np.asarray(val, np.float64), shape).reshape(-1))
        self.n += len(P)

    def mesh(self) -> MeshData:
        V = np.vstack(self.V) if self.V else np.zeros((0, 3))
        loops, totals = [], []
        for blk, k in ((self.F, 4), (self.T, 3)):
            if blk:
                arr = np.vstack(blk).astype(np.int32)
                loops.append(arr.reshape(-1))
                totals.append(np.full(len(arr), k, np.int32))
        loops = np.concatenate(loops) if loops else np.zeros(0, np.int32)
        totals = np.concatenate(totals) if totals else np.zeros(0, np.int32)
        starts = np.concatenate([[0], np.cumsum(totals[:-1])]).astype(np.int32) if len(totals) else totals
        attrs = {k: np.concatenate(v).astype(np.float32) if v else np.zeros(0, np.float32) for k, v in self.A.items()}
        return MeshData(V.astype(np.float32), loops, starts, totals, np.zeros((len(loops), 2), np.float32), attrs)


def tube_mesh(b: _Builder, pts, radii, sides, organ, pu0=0.0, pu1=1.0, cap=True, orand=0.0):
    """Generalized cylinder along a polyline (rotation-minimizing frames)."""
    pts = np.asarray(pts, float)
    n = len(pts)
    T = _normalize(np.gradient(pts, axis=0))
    ref = np.array([1.0, 0.0, 0.0]) if abs(T[0, 0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    N = np.zeros_like(T)
    N[0] = _normalize(np.cross(T[0], ref))
    for i in range(1, n):
        v = N[i - 1] - T[i] * float(N[i - 1] @ T[i])
        N[i] = _normalize(v) if np.linalg.norm(v) > 1e-9 else N[i - 1]
    B = np.cross(T, N)
    th = np.linspace(0, 2 * math.pi, sides, endpoint=False)
    ring = np.cos(th)[None, :, None] * N[:, None, :] + np.sin(th)[None, :, None] * B[:, None, :]
    P = pts[:, None, :] + np.asarray(radii, float).reshape(-1, 1, 1) * ring
    pu = np.linspace(pu0, pu1, n)[:, None]
    b.grid(P, organ, pu, np.cos(th)[None, :], 0.0, orand, close_v=True)
    if cap:
        start = b.n - n * sides
        tip = pts[-1] + T[-1] * float(np.ravel(radii)[-1]) * 0.6
        T3 = np.array([[start + (n - 1) * sides + k, start + (n - 1) * sides + (k + 1) % sides, b.n]
                       for k in range(sides)]) - b.n
        b.tris(tip[None, :], T3, organ, pu1, 0.0, 0.0, orand)


def ellipsoid(b: _Builder, centre, axis, rx, rz, organ, nu=5, nv=6, orand=0.0, pu=1.0):
    """Closed low-poly ellipsoid (anthers, stigma lobes, buds) along `axis`."""
    axis = _normalize(np.asarray(axis, float))
    ref = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = _normalize(np.cross(axis, ref))
    e2 = np.cross(axis, e1)
    phi = np.linspace(0.15, math.pi - 0.15, nu)
    th = np.linspace(0, 2 * math.pi, nv, endpoint=False)
    s, c = np.sin(phi)[:, None], np.cos(phi)[:, None]
    P = (centre + (-c * rz)[..., None] * axis + (s * rx * np.cos(th))[..., None] * e1
         + (s * rx * np.sin(th))[..., None] * e2)
    b.grid(P, organ, pu, 0.0, 0.0, orand, close_v=True)
    start = b.n - nu * nv
    bot = centre - axis * rz
    top = centre + axis * rz
    T = [[start + (k + 1) % nv, start + k, b.n] for k in range(nv)]
    T += [[start + (nu - 1) * nv + k, start + (nu - 1) * nv + (k + 1) % nv, b.n + 1] for k in range(nv)]
    b.tris(np.vstack([bot, top]), np.array(T) - b.n, organ, pu, 0.0, 0.0, orand)


class FlowerEngine:
    NU = 16
    NV = 9

    def __init__(self, profile: FlowerProfile):
        self.p = profile
        p = profile
        hw = half_width_profile(max(0.3, p.petal_aspect), p.widest_position, p.base_angle_deg, p.apex_angle_deg,
                                0.0, p.apex_curvature)
        hw = np.maximum(np.asarray(hw, float), 0.0)
        self._hw_t = T_GRID
        self._hw = hw / max(1e-6, hw.max())

    # ------------------------------------------------------------------
    def _half_width(self, y, base_min=0.0):
        """Normalised half-width at proximodistal position y in [0, 1] (1 = widest)."""
        p = self.p
        ut = 1.0 - 0.35 * p.truncation           # Truncation: stop the outline before it closes
        w = np.interp(np.clip(y, 0, 1) * ut, self._hw_t, self._hw)
        if p.claw > 0:
            c = np.clip(y / max(p.claw, 1e-3), 0.0, 1.0)
            w = w * (0.18 + 0.82 * c * c * (3 - 2 * c))
        if base_min > 0:
            wb = 1.0 - np.clip(y / 0.25, 0.0, 1.0)
            w = np.maximum(w, base_min * wb ** 0.5)
        return w

    def organ(self, b, base, radial, up, L, W, elev, reflex, cup, twist, und, organ, whorl, orand,
              detail, base_min=0.0, elev_start=None, pu0=0.0, pu1=1.0, outline=True, rng=None, fuse=0.0,
              sectors=5, axis_pt=None):
        """
        One perianth organ (petal, tepal, sepal, ligule, bract) as a curved, cupped blade.
        The midline turns from `elev_start` to `elev` (degrees above the floral plane) over the basal
        fifth, then bends by `reflex` toward the tip. Cross-sections are parabolic (cup) and twisted.
        """
        p = self.p
        nu = int(self.NU * detail) + 3
        nv = 2 * (int(self.NV * detail) // 2) + 3
        u = np.linspace(0.0, 1.0, nu)
        v = np.linspace(-1.0, 1.0, nv)
        U, Vv = np.meshgrid(u, v, indexing="ij")
        # Distal outline: apex notch, rounded lobes and fringe teeth shorten the columns near the tip
        y_end = np.ones_like(v)
        if outline:
            y_end -= p.notch * (1.0 - v ** 2) ** 2
            if p.fringe > 0 and p.fringe_count > 0:
                y_end -= p.fringe * 0.5 * (1.0 - np.cos(p.fringe_count * math.pi * (v + 1.0)))
        Y = U * np.clip(y_end, 0.05, 1.0)[None, :]
        # Midline on a fine table, evaluated at Y
        tf = np.linspace(0.0, 1.0, 64)
        e0 = math.radians(elev if elev_start is None else elev_start)
        e1 = math.radians(elev)
        sm = np.clip(tf / 0.2, 0.0, 1.0)
        ang = e0 + (e1 - e0) * sm * sm * (3 - 2 * sm) + math.radians(reflex) * tf ** 1.5
        dirs = np.cos(ang)[:, None] * radial + np.sin(ang)[:, None] * up
        mid = base + np.concatenate([[np.zeros(3)], np.cumsum((dirs[:-1] + dirs[1:]) * 0.5 * L / 63, axis=0)])
        lateral = _normalize(np.cross(up, radial))
        nrm = _normalize(np.cross(dirs, lateral))
        M = np.stack([np.interp(Y, tf, mid[:, k]) for k in range(3)], -1)
        Nn = _normalize(np.stack([np.interp(Y, tf, nrm[:, k]) for k in range(3)], -1))
        hw = 0.5 * W * self._half_width(Y, base_min)
        if fuse > 0:
            # Sympetalous limb: each lobe covers at least its sector 2*pi*r/m out to `fuse` of its length
            ap = np.zeros(3) if axis_pt is None else axis_pt
            r_row = np.hypot(M[..., 0] - ap[0], M[..., 1] - ap[1])
            mask = 1.0 - np.clip((Y - fuse) / 0.12 + 1.0, 0.0, 1.0) if fuse < 1 else np.ones_like(Y)
            mask = mask * mask * (3 - 2 * mask)
            hw = np.maximum(hw, mask * math.pi * r_row / max(1, sectors) * 1.03)
        X = hw * Vv
        Z = cup * hw * Vv ** 2
        if und > 0:
            ph = 0.0 if rng is None else rng.uniform(0, 2 * math.pi)
            Z = Z + und * W * 0.25 * np.sin(2 * math.pi * (Y * L / max(W, 1e-6)) * 1.3 + ph) * np.abs(Vv) ** 2
        if twist:
            a = math.radians(twist) * Y
            X, Z = X * np.cos(a) - Z * np.sin(a), X * np.sin(a) + Z * np.cos(a)
        P = M + X[..., None] * lateral + Z[..., None] * Nn
        b.grid(P, organ, pu0 + (pu1 - pu0) * Y, Vv, whorl, orand)
        return mid[-1]

    # ------------------------------------------------------------------
    def generate(self, openness: float = 1.0, detail: float = 1.0, seed: int = 0) -> FlowerResult:
        """openness: 0 closed bud .. 1 fully open (anthesis)."""
        p = self.p
        rng = np.random.default_rng(seed)
        o = float(np.clip(openness, 0.0, 1.0))
        b = _Builder()
        up = np.array([0.0, 0.0, 1.0])
        L = p.petal_length_cm * 0.01 * (0.45 + 0.55 * o)
        W = L / max(0.3, p.petal_aspect)
        r_rec = p.receptacle_radius_cm * 0.01 * (0.7 + 0.3 * o)
        sides = max(10, int(18 * detail))

        def openness_elev(e_open, closed=86.0):
            return closed + (e_open - closed) * o ** 0.8

        def openness_reflex(r_open):
            return r_open * o + 25.0 * (1.0 - o)     # Buds: tips incurved over the centre

        # Hypanthium / inferior ovary / pericarpel below the perianth
        z_top = 0.0
        if p.hypanthium_cm > 0 and not p.capitulum:
            hl = p.hypanthium_cm * 0.01
            t = np.linspace(0.0, 1.0, 10)
            # Obovoid ovary / pericarpel with a rounded, closed base where it joins the stem or pedicel
            r = np.maximum(r_rec * (0.12 + 0.88 * np.sin(t * math.pi * 0.5) ** 0.6), 1e-4)
            r = np.maximum(r, r_rec * 0.8 if p.tube_length_cm <= 0 else r)
            pts = np.stack([np.zeros(10), np.zeros(10), -hl + hl * t], 1)
            tube_mesh(b, pts, r, sides, SEPAL, 0.0, 0.0, cap=False)
            for k in range(p.hypanthium_scales):      # Spiral bract scales (cactus pericarpel)
                f = (k + 0.5) / p.hypanthium_scales
                zz = -hl + hl * f
                rr = float(np.interp(f, t, r))
                az = k * GOLDEN
                rad = np.array([math.cos(az), math.sin(az), 0.0])
                self.organ(b, np.array([0, 0, zz]) + rad * rr * 0.95, rad, up, hl * 0.35, hl * 0.3,
                           70.0, 10.0, 0.2, 0.0, 0.0, SEPAL, 0.0, rng.random(), 0.5, outline=False)

        if p.capitulum:
            return self._head(b, o, detail, rng, up)

        # Corolla tube
        r_top = r_rec
        tube_ang = None
        Lt = p.tube_length_cm * 0.01 * (0.6 + 0.4 * o)
        if Lt > 0:
            r0 = p.tube_radius_cm * 0.01
            r1 = max(r0 * 1.2, r0 + Lt * 0.25 * (0.4 + 0.6 * o))
            t = np.linspace(0.0, 1.0, max(6, int(10 * detail)))
            rt = r0 + (r1 - r0) * t ** max(0.3, p.tube_flare)
            zt = Lt * t
            th = np.linspace(0, 2 * math.pi, max(3 * p.merosity, sides * 2), endpoint=False)
            P = np.stack([rt[:, None] * np.cos(th)[None, :], rt[:, None] * np.sin(th)[None, :],
                          np.repeat(zt[:, None], len(th), 1)], -1)
            lobe_az = np.arange(p.merosity) * 2 * math.pi / p.merosity + math.pi / 2
            d = np.angle(np.exp(1j * (th[:, None] - lobe_az[None, :])))
            pv = d[np.arange(len(th)), np.argmin(np.abs(d), axis=1)] / (math.pi / p.merosity)
            frac = Lt / (Lt + L)
            b.grid(P, PETAL, (frac * t)[:, None], pv[None, :], 0.0, 0.5, close_v=True)
            r_top, z_top = r1, Lt
            tube_ang = math.degrees(math.atan2(zt[-1] - zt[-2], rt[-1] - rt[-2]))
        frac = Lt / (Lt + L) if Lt > 0 else 0.0

        # Perianth: calyx then corolla (whorled) or one spiral sequence of tepals
        m = max(1, p.merosity)
        organs = []          # (azimuth, scale, elevation, insertion radius, height, whorl t, organ)
        if p.sepal_length_ratio > 0:
            for k in range(m):
                organs.append((k * 2 * math.pi / m + math.pi / 2 + math.pi / m, p.sepal_length_ratio,
                               p.sepal_opening_deg, r_rec * 0.9, z_top * 0.0, 0.0, SEPAL))
        if p.arrangement == Arrangement.SPIRAL and p.spiral_tepals > 0:
            n = p.spiral_tepals
            for k in range(n):
                t = k / max(1, n - 1)
                organs.append((k * GOLDEN, 1.0 - (1.0 - p.inner_scale) * t ** 1.2,
                               p.opening_deg + (p.inner_opening_deg - p.opening_deg) * t ** 1.3,
                               r_top * (1.0 - 0.55 * t), z_top + r_rec * 0.25 * t, t, PETAL))
        else:
            nw = max(1, p.petal_whorls)
            for w in range(nw):
                t = w / max(1, nw - 1) if nw > 1 else 0.0
                for k in range(m):
                    az = k * 2 * math.pi / m + math.pi / 2 + (math.pi / m) * (w % 2) + 0.12 * w
                    organs.append((az, 1.0 - (1.0 - p.inner_scale) * t,
                                   p.opening_deg + (p.inner_opening_deg - p.opening_deg) * t,
                                   r_top * (1.0 - 0.45 * t), z_top + r_rec * 0.2 * t, t, PETAL))

        tips = []
        z_ = p.zygomorphy
        for az, s, elev, r_ins, z_ins, wt, kind in organs:
            d = math.sin(az)                                  # +1 dorsal (+Y), -1 ventral
            if z_ > 0:
                az = az - 0.3 * z_ * math.sin(2 * (az - math.pi / 2)) * 0.5
                s = s * (1.0 + 0.45 * z_ * p.lip_bias * d)
                elev = elev + z_ * (30.0 * max(d, 0.0) - 25.0 * max(-d, 0.0))
            rad = np.array([math.cos(az), math.sin(az), 0.0])
            base = rad * r_ins + up * z_ins
            if kind == SEPAL:
                Ls = L * s / max(0.45 + 0.55 * o, 1e-3) * (0.6 + 0.4 * o)
                self.organ(b, base, rad, up, Ls, Ls / max(0.3, p.sepal_aspect), openness_elev(elev, 75.0),
                           openness_reflex(-10.0), 0.3, 0.0, 0.0, SEPAL, 0.0, rng.random(), detail * 0.6,
                           outline=False)
                continue
            if not p.petals_visible:
                continue
            Lp = L * s * rng.uniform(0.95, 1.05)
            base_min = (math.pi * r_top / m) / max(0.5 * W * s, 1e-6) * 1.1 if Lt > 0 else 0.0
            e = openness_elev(elev)
            tips.append(self.organ(b, base, rad, up, Lp, W * s, e, openness_reflex(p.reflex_deg * (1.0 - 0.5 * wt)),
                                   p.cup * (1.0 + 0.8 * wt), p.twist_deg, p.undulation, PETAL, wt, rng.random(),
                                   detail, base_min=min(base_min, 1.2),
                                   elev_start=(tube_ang if tube_ang is not None else e), pu0=frac, pu1=1.0, rng=rng,
                                   # Limb fusion only exists in sympetalous corollas (with a tube)
                                   fuse=(p.limb_fusion * (1.0 if p.zygomorphy < 0.5 else 0.6)) if Lt > 0 else 0.0,
                                   sectors=m))

        # Receptacle (dome or cone)
        h_rec = p.receptacle_height_cm * 0.01
        if r_rec > 0:
            t = np.linspace(0.0, 1.0, 6)
            if h_rec <= r_rec:        # Flat or domed receptacle
                rr = 0.9 * r_rec * np.sqrt(np.maximum(1.0 - t ** 2, 0.0))
            else:                     # Elongated conical receptacle carrying spiral carpels (Magnolia)
                rr = r_rec * (1.0 - t) ** 0.6
            rr = np.maximum(rr, 1e-4)
            pts = np.stack([np.zeros(6), np.zeros(6), z_top * 0.2 + h_rec * t], 1)
            tube_mesh(b, pts, rr, sides, RECEPTACLE, 0.0, 0.0, cap=False)

        # Androecium (hidden in tight buds)
        so = np.clip((o - 0.35) / 0.65, 0.0, 1.0)
        tube = (Lt, p.tube_radius_cm * 0.01, r_top, max(0.3, p.tube_flare)) if Lt > 0 else None
        if p.stamen_count > 0 and so > 0:
            self._stamens(b, L, r_rec, r_top, z_top, h_rec, so, detail, rng, up, tube)
        # Gynoecium
        if p.style_length_ratio > 0 and o > 0.3:
            self._pistil(b, L, z_top, h_rec, o, detail, rng, up, tube)

        mesh = b.mesh()
        V = mesh.vertices
        diam = 2.0 * float(np.hypot(V[:, 0], V[:, 1]).max()) if len(V) else 0.0
        return FlowerResult(mesh, diam, float(V[:, 2].max()) if len(V) else 0.0)

    # ------------------------------------------------------------------
    def _stamens(self, b, L, r_rec, r_top, z_top, h_rec, so, detail, rng, up, tube=None):
        p = self.p
        n = p.stamen_count
        Ls = L * p.stamen_length_ratio / (0.45 + 0.55 * 1.0)
        fr = max(0.15e-3, 0.012 * Ls)
        ar = p.anther_size_mm * 1e-3
        z0 = z_top * 0.15 + h_rec * 0.3
        nseg = 5
        if p.staminal_column > 0:
            # Monadelphous column (Malvaceae): filaments fused into a tube around the style
            Lc = Ls * p.staminal_column * so
            pts = np.stack([np.zeros(6), np.zeros(6), z0 + np.linspace(0, Lc, 6)], 1)
            tube_mesh(b, pts, np.linspace(fr * 3.0, fr * 2.0, 6), 6, FILAMENT, 0.0, 1.0, cap=False)
            for k in range(n):
                f = 0.55 + 0.45 * (k + 0.5) / n
                az = k * GOLDEN
                rad = np.array([math.cos(az), math.sin(az), 0.0])
                base = np.array([0, 0, z0 + Lc * f]) + rad * fr * 2.2
                tip = base + (rad * 0.8 + up * 0.6) * Ls * 0.08
                tube_mesh(b, np.linspace(base, tip, 3), np.full(3, fr * 0.6), 3, FILAMENT, 0.0, 1.0, cap=False)
                ellipsoid(b, tip, rad, ar * 0.45, ar * 0.5, ANTHER, 3, 4, rng.random())
            return
        many = n > 2 * max(1, p.merosity)
        r_in, r_out = (r_rec * 0.45, r_top * 0.95) if many else (r_rec * 0.6, r_rec * 0.6)
        if tube is not None:
            # Inserted inside the corolla / receptacular tube: they rise through it and exceed it by Ls
            Lt, r0, r1, _ = tube
            r_in, r_out = (r0 * 0.25, r0 * 0.85) if many else (r0 * 0.5, r0 * 0.5)
            z0 = Lt * 0.05
        for k in range(n):
            if many:      # Dense androecium: equal-area spiral on an annulus
                f = (k + 0.5) / n
                rr = math.sqrt(r_in ** 2 + (r_out ** 2 - r_in ** 2) * f)
                az = k * GOLDEN
                spread = math.radians(p.stamen_spread_deg) * (0.4 + 0.6 * (rr - r_in) / max(r_out - r_in, 1e-6))
            else:
                rr = r_in
                az = k * 2 * math.pi / n + math.pi / 2 + (math.pi / n if n == p.merosity else 0.0)
                spread = math.radians(p.stamen_spread_deg)
            rad = np.array([math.cos(az), math.sin(az), 0.0])
            ln = Ls * so * rng.uniform(0.85, 1.1)
            if tube is not None:
                ln = ln + tube[0] * 0.9
                spread = min(spread, math.atan2(max(tube[2] - rr, 0.0), tube[0]) * 0.9)
            t = np.linspace(0, 1, nseg)[:, None]
            dirv = math.sin(spread) * rad + math.cos(spread) * up
            bend = rad * (0.25 * ln * t ** 2) * math.sin(spread)
            if p.stamen_declination > 0:     # Declinate: lie along the ventral side, tips curving up
                dirv = _normalize(dirv * (1 - p.stamen_declination) + np.array([0, -0.8, 0.6]) * p.stamen_declination)
                bend = bend + np.array([0, 0, 1.0]) * 0.3 * ln * t ** 2 * p.stamen_declination
            base = rad * rr + up * z0
            pts = base + dirv * ln * t + bend
            if tube is not None:       # Keep the filament inside the trumpet-shaped tube
                Lt_, r0_, r1_, fl_ = tube
                zt = np.clip(pts[:, 2] / Lt_, 0.0, 1.0)
                r_lim = 0.85 * (r0_ + (r1_ - r0_) * zt ** fl_)
                rr_ = np.hypot(pts[:, 0], pts[:, 1])
                k_ = np.where((pts[:, 2] < Lt_) & (rr_ > r_lim), r_lim / np.maximum(rr_, 1e-9), 1.0)
                pts[:, :2] *= k_[:, None]
            tube_mesh(b, pts, np.full(nseg, fr), 3 if detail < 1.5 else 4, FILAMENT, 0.0, 1.0, cap=False,
                      orand=rng.random())
            ax = _normalize(pts[-1] - pts[-2])
            ellipsoid(b, pts[-1] + ax * ar * 0.4, np.cross(ax, rad) if p.zygomorphy < 0.3 else ax,
                      min(ar * 0.35, 1.2e-3), ar * 0.6, ANTHER, 3, 4, rng.random())     # Large anthers are linear

    def _pistil(self, b, L, z_top, h_rec, o, detail, rng, up, tube=None):
        p = self.p
        Ls = L * p.style_length_ratio / (0.45 + 0.55 * 1.0)
        z0 = z_top * 0.15 + h_rec
        if tube is not None:
            z0 = tube[0] * 0.05
            Ls = Ls + tube[0] * 0.95 / max(o, 0.3)
        sr = float(np.clip(0.02 * Ls, 0.2e-3, 1.5e-3))
        t = np.linspace(0, 1, 5)[:, None]
        tilt = np.array([0.0, -0.25, 0.0]) * p.stamen_declination
        pts = up * z0 + (up + tilt) * Ls * o * t + np.array([0, 0, 1.0]) * 0.15 * Ls * p.stamen_declination * t ** 2
        tube_mesh(b, pts, np.linspace(sr * 1.3, sr, 5), 5, PISTIL, 0.0, 1.0, cap=False)
        tip = pts[-1]
        sg = p.stigma_size_mm * 1e-3
        nc = max(1, p.carpels)
        if nc == 1:
            ellipsoid(b, tip, up, sg * 0.6, sg * 0.6, PISTIL, 4, 6, 0.0)
            return
        for k in range(nc):      # Radiating stigma lobes
            az = k * 2 * math.pi / nc
            rad = np.array([math.cos(az), math.sin(az), 0.0])
            ax = _normalize(rad + up * 0.6)
            ellipsoid(b, tip + ax * sg * 0.6, ax, sg * 0.25, sg * 0.6, PISTIL, 3, 4, 0.0)

    # ------------------------------------------------------------------
    def _head(self, b, o, detail, rng, up):
        """Capitulum: involucre, ray florets (ligules) and disc florets on Vogel's spiral."""
        p = self.p
        R = p.disc_radius_cm * 0.01 * (0.6 + 0.4 * o)
        dome = p.disc_dome * R
        # Receptacle disc (underside) as a shallow dish
        t = np.linspace(0.0, 1.0, 6)
        rr = np.maximum(R * np.sin(t * math.pi * 0.5) ** 0.7 * 1.02, 1e-4)
        pts = np.stack([np.zeros(6), np.zeros(6), -0.25 * R + 0.25 * R * t], 1)
        tube_mesh(b, pts, rr, max(10, int(16 * detail)), SEPAL, 0.0, 0.0, cap=False)
        # Involucral bracts (phyllaries): spiral, imbricate under the rim
        # Phyllaries sit under the rim, spreading flat or bent down, hidden behind the rays when open
        for k in range(p.involucre_bracts):
            f = k / max(1, p.involucre_bracts - 1)
            az = k * GOLDEN
            rad = np.array([math.cos(az), math.sin(az), 0.0])
            Lb = R * (0.22 + 0.12 * f) * (1.6 - 0.6 * o)
            self.organ(b, rad * R * (0.55 + 0.35 * f) + up * (-0.22 * R + 0.1 * R * f), rad, up, Lb, Lb * 0.5,
                       (-15.0 + 25.0 * f) * o + 80.0 * (1 - o), -15.0 * o, 0.25, 0.0, 0.0, SEPAL, 0.0, rng.random(),
                       0.5, outline=False)
        # Ray florets (ligules)
        n = p.ray_count
        L = p.petal_length_cm * 0.01 * (0.4 + 0.6 * o)
        W = L / max(0.3, p.petal_aspect)
        for k in range(n):
            f = (k + 0.5) / max(1, n)
            # Rim row (single) or spread inward over the head (double heads): Vogel annulus from outside in
            r_k = R * math.sqrt(max(0.0, 1.0 - p.ray_fill * f * 0.92))
            wt = p.ray_fill * f
            s = 1.0 - (1.0 - p.inner_scale) * wt ** 1.1
            az = k * GOLDEN
            rad = np.array([math.cos(az), math.sin(az), 0.0])
            zr = dome * (1.0 - (r_k / max(R, 1e-6)) ** 2)
            elev = p.opening_deg + (p.inner_opening_deg - p.opening_deg) * wt ** 1.2
            e = 86.0 + (elev - 86.0) * o ** 0.8
            self.organ(b, rad * r_k * 0.95 + up * zr, rad, up, L * s * rng.uniform(0.92, 1.06), W * s, e,
                       p.reflex_deg * o + 20 * (1 - o), p.cup * (1 + wt), p.twist_deg * rng.uniform(0.5, 1.0),
                       p.undulation, PETAL, wt, rng.random(), detail * (0.8 if n > 60 else 1.0), rng=rng)
        # Disc florets: Vogel spiral, maturing centripetally (outer open, inner buds)
        nd = int(p.disc_florets)
        if nd > 0:
            k = np.arange(nd)
            rk = R * np.sqrt((k + 0.5) / nd) * (0.97 if n else 1.0)
            ak = k * GOLDEN
            fr = R * math.sqrt(1.0 / nd) * 0.85                  # Floret radius: equal area per floret
            frac = rk / max(R, 1e-6)
            zk = dome * (1.0 - frac ** 2)
            ht = fr * (0.6 + 1.6 * np.clip((frac - (1 - o)) * 2.0, 0.0, 1.0))
            six = np.linspace(0, 2 * math.pi, 6, endpoint=False)
            cx = rk * np.cos(ak)
            cy = rk * np.sin(ak)
            ring = np.stack([np.cos(six), np.sin(six)], 1) * fr
            bot = np.stack([cx[:, None] + ring[None, :, 0], cy[:, None] + ring[None, :, 1],
                            np.repeat(zk[:, None], 6, 1) - fr], -1)
            top = bot.copy()
            top[..., 2] = (zk + ht)[:, None]
            apex = np.stack([cx, cy, zk + ht + fr * 0.7], 1)
            V = np.concatenate([bot, top, apex[:, None, :]], axis=1)       # (nd, 13, 3)
            base = (np.arange(nd) * 13)[:, None]
            q = np.arange(6)
            quads = np.stack([q, (q + 1) % 6, 6 + (q + 1) % 6, 6 + q], 1)
            tri = np.stack([6 + q, 6 + (q + 1) % 6, np.full(6, 12)], 1)
            off = b.n
            b.F.append((quads[None] + base[:, :, None] + off).reshape(-1, 4))
            b.T.append((tri[None] + base[:, :, None] + off).reshape(-1, 3))
            b._push(V.reshape(-1, 3), FLORET, np.repeat(frac, 13), 0.0,
                    np.repeat(np.clip((frac - (1 - o)) * 2.0, 0, 1), 13), np.repeat(rng.random(nd), 13), (nd * 13,))
        mesh = b.mesh()
        Vx = mesh.vertices
        return FlowerResult(mesh, 2.0 * float(np.hypot(Vx[:, 0], Vx[:, 1]).max()), float(Vx[:, 2].max()))
