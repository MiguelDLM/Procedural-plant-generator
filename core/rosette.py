"""
Leaf succulents with rosette habit (Crassulaceae, Asphodelaceae, Agavaceae).

A rosette is a shoot with near-zero internodes: leaves are inserted on a short
stem at the golden divergence angle (137.5 deg; distichous in some Gasteria and
Haworthia), the oldest outermost and lowest, the youngest at the centre. Leaf
elevation, size and curvature change with leaf age (index), which is what
distinguishes a flat Echeveria from a vase-shaped Agave or an open Aloe.

Each leaf is a closed volume, not a card: a midline that bends along the leaf,
a half-width profile (the lamina descriptors of leaf_morphology: aspect, widest
point, base/apex angles and curvatures), a thickness profile, and a
cross-section given by a superellipse with an adaxial channel (concave upper
face) and an abaxial keel. Marginal teeth and a terminal spine (Agave), and
leaf colouring (glaucous bloom, blushed margins and tips, spots and bands) are
parameters of the same profile.
"""

from dataclasses import dataclass
from enum import Enum
import math
import numpy as np

from .leaf_morphology import half_width_profile, T_GRID
from .mesh_engine import MeshData
from .cactus import _mesh, _normalize, _quads_grid
from .roots import RootSystemType

GOLDEN = math.radians(137.50776)
UP = np.array([0.0, 0.0, 1.0])


class RosettePhyllotaxis(str, Enum):
    SPIRAL = "Spiral"
    DISTICHOUS = "Distichous"


@dataclass
class RosetteProfile:
    phyllotaxis: RosettePhyllotaxis = RosettePhyllotaxis.SPIRAL
    leaf_count: int = 40
    stem_height_m: float = 0.0       # Visible stem below the rosette (Aeonium: tall)
    stem_radius_m: float = 0.012
    rosette_height_m: float = 0.02   # Vertical spread of leaf insertions along the stem

    # Leaf size and age gradients (index 0 = oldest, outermost)
    leaf_length_cm: float = 5.0
    leaf_aspect: float = 2.0         # Length / width
    leaf_thickness: float = 0.35     # Thickness / width (succulence)
    thickness_taper: float = 0.6     # Thinner toward the tip
    size_gradient: float = 0.55      # Youngest leaves smaller by this fraction
    elevation_outer_deg: float = 15.0
    elevation_inner_deg: float = 75.0
    elevation_power: float = 1.5     # How quickly leaves become erect toward the centre
    curvature_deg: float = 10.0      # Bending along the leaf (+ incurved upward, - recurved)

    # Outline (lamina descriptors)
    widest_position: float = 0.6
    base_angle_deg: float = 50.0
    apex_angle_deg: float = 70.0
    base_curvature: float = 0.0
    apex_curvature: float = 0.3

    # Cross-section
    channel: float = 0.15            # Adaxial concavity
    keel: float = 0.3                # Abaxial keel
    section_exponent: float = 2.2    # Superellipse exponent (2 elliptic, >2 boxy)

    # Leaf base and young leaves
    base_width: float = 0.3          # Half-width at the insertion relative to the widest point
    clasp: float = 0.0               # How much the base wraps around the stem (sheathing / clasping)
    base_swell: float = 0.0          # Extra thickness of the fleshy leaf base
    furl: float = 0.0                # Rolling of young central leaves into the spike (cogollo)

    # Stem body and old leaves
    dead_leaves: int = 0             # Persistent dry leaves below the living rosette (skirt)
    dead_color: tuple = (0.55, 0.46, 0.34)
    stem_color: tuple = (0.50, 0.45, 0.36)   # Corky stem below the leaves
    stem_scars: float = 0.0          # Visibility of crescent leaf scars on the stem
    scar_spacing_mm: float = 6.0     # Internode length along the bare stem

    # Armature
    terminal_spine_cm: float = 0.0
    teeth_count: int = 0             # Per margin
    teeth_size_cm: float = 0.0
    teeth_hook: float = 0.4

    # Offsets (pups)
    offsets: int = 0
    offset_scale: float = 0.45

    # Colour (sRGB)
    leaf_color: tuple = (0.45, 0.58, 0.55)
    blush_color: tuple = (0.75, 0.35, 0.40)
    blush_amount: float = 0.0        # Margin / tip anthocyanin
    glaucous: float = 0.6            # Epicuticular wax bloom
    spots: float = 0.0               # Pale spots (Aloe)
    bands: float = 0.0               # Transverse tubercle bands (Haworthiopsis)
    armature_color: tuple = (0.30, 0.20, 0.15)
    blush_tip: float = 1.0           # Weight of the tip in the blush (0 = margins only)
    margin_band: float = 0.0         # Horny dark margin (Agave)
    striation: float = 0.0           # Fine longitudinal lines
    imprints: float = 0.0            # Bud imprints of neighbouring leaves' teeth and outline (Agave)

    # Shallow fibrous roots (Franco & Nobel 1990 for Agave deserti)
    root_system: RootSystemType = RootSystemType.FIBROUS
    root_count: int = 25
    root_spread_ratio: float = 1.0  # Reach relative to rosette diameter
    root_depth_m: float = 0.2
    root_radius_mm: float = 1.5


@dataclass
class RosetteResult:
    leaves: MeshData
    armature: MeshData
    stem: MeshData
    leaf_count: int
    diameter_m: float
    roots: MeshData = None
    terminal_sites: tuple = None     # (positions, directions): rosette apices (Agave scape)
    axillary_sites: tuple = None     # (positions, directions): axils of mature leaves (Echeveria, Dudleya)


class RosetteEngine:
    NU = 22   # Samples along a leaf
    NV = 16   # Samples around the cross-section

    def __init__(self, profile: RosetteProfile):
        self.p = profile

    def generate(self, seed: int = 7, detail: float = 1.0, with_roots: bool = False,
                 root_display_depth: float = 3.0) -> RosetteResult:
        rng = np.random.default_rng(seed)
        leaves, arm, stems = [], [], []
        centres = [(np.zeros(3), 1.0)]
        p = self.p
        for k in range(p.offsets):
            az = k * GOLDEN + rng.uniform(0, 0.5)
            d = p.leaf_length_cm * 0.01 * rng.uniform(0.9, 1.4)
            centres.append((np.array([d * math.cos(az), d * math.sin(az), 0.0]), p.offset_scale * rng.uniform(0.7, 1.1)))
        n_total = 0
        diam = 0.0
        self._term, self._axil = [], []
        for ci, (c, scale) in enumerate(centres):
            lv, ar, st, n, dm = self._rosette(c, scale, rng, detail, primary=ci == 0)
            leaves.append(lv)
            arm.append(ar)
            stems.append(st)
            n_total += n
            diam = max(diam, dm + 2 * float(np.hypot(c[0], c[1])))
        roots = None
        if with_roots:
            from .succulent_roots import succulent_roots
            roots = succulent_roots(p.root_system, p.stem_radius_m, max(0.05, p.root_spread_ratio * diam),
                                    p.root_depth_m, p.root_count, core_ratio=0.9,
                                    root_radius_mm=p.root_radius_mm, display_depth_m=root_display_depth, seed=seed)
        def pack(lst):
            if not lst:
                return np.zeros((0, 3)), np.zeros((0, 3))
            return np.array([a for a, _ in lst]), np.array([d for _, d in lst])
        return RosetteResult(MeshData.concatenate(leaves), MeshData.concatenate(arm), MeshData.concatenate(stems),
                             n_total, diam, roots, pack(self._term), pack(self._axil))

    # ------------------------------------------------------------------
    def _rosette(self, centre, scale, rng, detail, primary):
        p = self.p
        n = max(1, int(round(p.leaf_count * (1.0 if primary else 0.6))))
        nu = int(self.NU * detail) + 4
        nv = int(self.NV * detail) + 4
        hw_tab = half_width_profile(p.leaf_aspect, p.widest_position, p.base_angle_deg, p.apex_angle_deg,
                                    p.base_curvature, p.apex_curvature)
        u = np.linspace(0.0, 1.0, nu)
        hw = np.interp(u, T_GRID, hw_tab)
        hw = np.maximum(hw, 0.004)
        hw[-1] = 0.0
        hw /= max(1e-6, hw.max())
        # Broad sheathing base: blend the lamina outline toward `base_width` over the basal ~18%
        wb = 1.0 - np.clip(u / 0.18, 0.0, 1.0)
        wb = wb * wb * (3.0 - 2.0 * wb)
        hw = hw + (max(p.base_width, 0.004) - hw[0]) * wb
        hw[-1] = 0.0
        hw = np.maximum(hw, 0.0)
        # Leaves are inserted along the top of the stem: the oldest at its base, above ground
        stem_top = centre + UP * (max(p.stem_height_m, 0.0) + p.rosette_height_m) * scale
        verts, quads, tris, attrs = [], [], [], {"leaf_u": [], "leaf_edge": [], "leaf_top": [], "leaf_rand": []}
        arm_parts = []
        diam = 0.0
        step = GOLDEN if p.phyllotaxis == RosettePhyllotaxis.SPIRAL else math.pi
        az0 = rng.uniform(0, 2 * math.pi)
        attrs["leaf_dead"] = []
        dz = p.rosette_height_m * scale / max(1, n - 1)
        n_dead = p.dead_leaves if primary else p.dead_leaves // 2
        ground = centre[2] + 0.004
        for i in range(-n_dead, n):
            dead = i < 0
            age = max(0.0, i / max(1, n - 1))            # 0 oldest .. 1 youngest
            L = p.leaf_length_cm * 0.01 * scale * (1.0 - p.size_gradient * age ** 1.3) * rng.uniform(0.92, 1.06)
            W = L / max(0.5, p.leaf_aspect)               # Full width
            elev = math.radians(p.elevation_outer_deg + (p.elevation_inner_deg - p.elevation_outer_deg)
                                * age ** p.elevation_power + rng.normal(0, 3))
            az = az0 + i * step + rng.normal(0, 0.04)
            r_ins = p.stem_radius_m * scale * (1.0 - 0.7 * age)
            z_ins = -p.rosette_height_m * scale * (1.0 - age)
            if dead:
                # Withered leaves: shrunken, papery, hanging from below the living whorl
                L *= rng.uniform(0.7, 0.95)
                W *= rng.uniform(0.55, 0.8)
                elev = -math.radians(rng.uniform(20, 70))
                z_ins = max(-p.rosette_height_m * scale + i * max(dz, 0.002), ground - stem_top[2] + 0.004)
            radial = np.array([math.cos(az), math.sin(az), 0.0])
            lateral = np.array([-math.sin(az), math.cos(az), 0.0])
            base = stem_top + radial * r_ins + UP * z_ins
            # Midline: angle from horizontal changes along the leaf by the curvature
            # Bending concentrated in the distal half (rigid, fibre-reinforced base; recurving tips)
            ang = elev + math.radians(p.curvature_deg) * u ** 2 * (1.0 - 0.6 * age)
            if dead:
                ang = ang - math.radians(rng.uniform(10, 50)) * u ** 2
            ds = L / (nu - 1)
            dirs = np.cos(ang)[:, None] * radial + np.sin(ang)[:, None] * UP
            mid = base + np.concatenate([[np.zeros(3)], np.cumsum(dirs[:-1] * ds, axis=0)])
            if dead:          # Rest on the ground instead of passing through it
                mid[:, 2] = np.maximum(mid[:, 2], ground + 0.002 * (n_dead + i + 1) * scale)
            normal = _normalize(np.cross(lateral[None, :], dirs))       # Adaxial (upper) normal
            half_w = 0.5 * W * hw
            thick = p.leaf_thickness * W * (1.0 - p.thickness_taper * u) * np.clip(hw * 1.3, 0.25, 1.0)
            thick = thick * (1.0 + p.base_swell * (1.0 - u) ** 6)
            if dead:
                thick = thick * 0.12
            v = np.linspace(0.0, 2 * math.pi, nv, endpoint=False)
            e = 2.0 / max(1.2, p.section_exponent)
            cx = np.sign(np.cos(v)) * np.abs(np.cos(v)) ** e
            sz = np.sign(np.sin(v)) * np.abs(np.sin(v)) ** e
            top = sz > 0
            X = half_w[:, None] * cx[None, :]
            Z = 0.5 * thick[:, None] * sz[None, :]
            xr = np.abs(cx)[None, :]
            Z = np.where(top[None, :], Z - p.channel * thick[:, None] * (1.0 - xr ** 2),
                         Z - p.keel * thick[:, None] * (1.0 - xr))
            # Young central leaves roll about their long axis (furled spike / cogollo)
            furl = 0.0 if dead else p.furl * np.clip((age - 0.55) / 0.45, 0.0, 1.0) ** 1.5
            if furl > 1e-3:
                amax = furl * math.pi * 0.85
                wrow = np.maximum(half_w, 1e-6)[:, None]
                a = amax * X / wrow
                rho = wrow / amax
                X, Z = rho * np.sin(a) - Z * np.sin(a), rho * (1.0 - np.cos(a)) + Z * np.cos(a)
            P = mid[:, None, :] + X[..., None] * lateral[None, None, :] + Z[..., None] * normal[:, None, :]
            # Clasping base: near the insertion the lateral spread follows an arc around the stem axis
            if p.clasp > 0.0:
                wc = p.clasp * (1.0 - np.clip(u / 0.22, 0.0, 1.0)) ** 2
                Rrow = np.maximum(np.hypot(*(mid[:, :2] - stem_top[:2]).T), 0.3 * p.stem_radius_m * scale + 1e-4)
                phi = X / Rrow[:, None]
                delta = (Rrow[:, None, None] * (np.cos(phi) - 1.0)[..., None] * radial[None, None, :]
                         + (Rrow[:, None] * np.sin(phi) - X)[..., None] * lateral[None, None, :])
                P = P + wc[:, None, None] * delta
            off = sum(len(vv) for vv in verts)
            verts.append(P.reshape(-1, 3))
            quads.append(_quads_grid(nu, nv, off))
            attrs["leaf_u"].append(np.repeat(u, nv))
            attrs["leaf_edge"].append(np.tile(np.abs(cx), nu))
            attrs["leaf_top"].append(np.tile(top.astype(float), nu))
            attrs["leaf_rand"].append(np.full(nu * nv, rng.random()))
            attrs["leaf_dead"].append(np.full(nu * nv, 1.0 if dead else 0.0))
            diam = max(diam, 2 * float(np.hypot(*(mid[-1] - centre)[:2])))
            if not dead:
                arm_parts.append(self._armature(mid, dirs, lateral, normal, half_w, L, rng))
                if 0.15 < age < 0.6:      # Axil of a mature leaf: lateral inflorescences emerge here
                    self._axil.append((base + UP * 0.01 * scale, _normalize(radial * 0.6 + UP)))
        V = np.vstack(verts)
        Q = np.vstack(quads)
        attrs = {k: np.concatenate(vv).astype(np.float32) for k, vv in attrs.items()}
        leaves = _mesh(V, Q, None, None, None, attrs)
        self._term.append((stem_top.copy(), UP.copy()))
        w0 = 0.5 * p.leaf_length_cm * 0.01 * scale / max(0.5, p.leaf_aspect) * hw[0]
        stem = self._stem(centre, stem_top, scale, az0, step, w0, rng)
        return leaves, MeshData.concatenate([a for a in arm_parts if a is not None]), stem, n, diam

    def _armature(self, mid, dirs, lateral, normal, half_w, L, rng):
        """Terminal spine and marginal teeth as small cones."""
        p = self.p
        cones = []
        if p.terminal_spine_cm > 0:
            tip = mid[-1]
            length = p.terminal_spine_cm * 0.01
            # Spine base about a tenth of its length, never wider than the leaf just below the tip
            j = max(0, len(mid) - 3)
            r = min(0.1 * length + 0.0003, 0.8 * float(half_w[j]))
            cones.append((tip - dirs[-1] * 0.4 * length, dirs[-1], 1.4 * length, r))
        if p.teeth_count > 0 and p.teeth_size_cm > 0:
            nu = len(mid)
            for side in (-1.0, 1.0):
                for k in range(p.teeth_count):
                    f = 0.15 + 0.75 * (k + 0.5) / p.teeth_count
                    j = int(f * (nu - 1))
                    pos = mid[j] + side * lateral * half_w[j]
                    d = _normalize(side * lateral + dirs[j] * p.teeth_hook)
                    size = p.teeth_size_cm * 0.01 * (0.6 + 0.4 * math.sin(math.pi * f))
                    cones.append((pos, d, size, size * 0.35))
        if not cones:
            return None
        verts, tris = [], []
        for base, d, length, r in cones:
            d = _normalize(d)
            ref = np.array([1.0, 0.0, 0.0]) if abs(d[2]) > 0.9 else UP
            e1 = _normalize(np.cross(d, ref))
            e2 = np.cross(d, e1)
            off = len(verts)
            for a in np.linspace(0, 2 * math.pi, 5, endpoint=False):
                verts.append(base + r * (math.cos(a) * e1 + math.sin(a) * e2))
            verts.append(base + d * length)
            for a in range(5):
                tris.append([off + a, off + (a + 1) % 5, off + 5])
        V = np.array(verts)
        return _mesh(V, None, np.array(tris), None, None, {"leaf_u": np.ones(len(V), np.float32)})

    def _stem(self, centre, top, scale, az0, step, w_base, rng):
        """Stem as a body of revolution: flared root crown, bare corky internodes with crescent leaf scars
        following the phyllotactic spiral, then the insertion zone tapering into the apical dome, where the
        radius matches the leaf insertion radius r_ins = r0 (1 - 0.7 age)."""
        p = self.p
        r0 = p.stem_radius_m * scale
        rh = p.rosette_height_m * scale
        zb, zt = centre[2] - 0.03 * max(scale, 0.3), top[2]
        z_zone = zt - rh                                   # Oldest living leaf
        bare = max(0.0, z_zone - centre[2])
        dz_s = max(0.001, p.scar_spacing_mm * 1e-3 * scale) if bare > 0.05 * scale else max(rh / max(1, p.leaf_count), 1e-3)
        nseg = 32
        n_bare = int(np.clip(bare / (dz_s / 4.0), 4, 600))
        z = np.concatenate([np.linspace(zb, z_zone, n_bare, endpoint=False), np.linspace(z_zone, zt, 12)])
        t_zone = np.clip((z - z_zone) / max(rh, 1e-6), 0.0, 1.0)
        r = r0 * (1.0 - 0.7 * t_zone) * 0.92
        r = r * (1.0 + 0.35 * np.exp(-np.maximum(z - centre[2], 0.0) / (0.8 * r0)))     # Root-crown flare
        r[-1] = 0.15 * r0
        th = np.linspace(0, 2 * math.pi, nseg, endpoint=False)
        V = np.stack([centre[0] + r[:, None] * np.cos(th)[None, :], centre[1] + r[:, None] * np.sin(th)[None, :],
                      np.repeat(z[:, None], nseg, 1)], -1).reshape(-1, 3)
        nr = len(z)
        Q = _quads_grid(nr, nseg)
        apex = len(V)
        V = np.vstack([V, [centre[0], centre[1], zt + 0.05 * r0]])
        T = np.array([[(nr - 1) * nseg + k, (nr - 1) * nseg + (k + 1) % nseg, apex] for k in range(nseg)])
        # Crescent leaf scars below the living whorl (leaf -j sits at azimuth az0 - j*step)
        Z = np.repeat(z, nseg)
        TH = np.tile(th, nr)
        R = np.repeat(r, nseg)
        k = (z_zone - Z) / dz_s
        scar = np.zeros_like(Z)
        w_a = np.clip(w_base / np.maximum(R, 1e-4), 0.3, 1.6)       # Crescent half-angle (< ~90 deg)
        for off in (-1, 0, 1, 2):
            j = np.floor(k) + off
            ok = j >= 1
            zj = z_zone - j * dz_s
            d_th = np.angle(np.exp(1j * (TH - (az0 - j * step))))
            q = np.clip(np.abs(d_th) / w_a, 0.0, 1.0)
            zc = zj + 0.3 * dz_s * q ** 2
            sig = max(0.14 * dz_s, 0.6 * (bare / max(n_bare, 1)))
            val = np.exp(-((Z - zc) / sig) ** 2) * (1.0 - q ** 4)
            scar = np.maximum(scar, np.where(ok, val, 0.0))
        # Scars sit on slightly raised leaf cushions
        cushion = (1.0 + 0.05 * scar * (Z < z_zone))[:, None]
        V[:-1, :2] = centre[:2] + (V[:-1, :2] - centre[:2]) * cushion
        scar = np.concatenate([scar, [0.0]])
        h = np.concatenate([np.clip((Z - centre[2]) / max(zt - centre[2], 1e-6), 0, 1), [1.0]])
        zone = np.concatenate([t_zone.repeat(nseg), [1.0]])
        return _mesh(V, Q, T, [list(range(nseg))[::-1]], None,
                     {"leaf_u": np.zeros(len(V), np.float32), "scar": scar.astype(np.float32),
                      "stem_h": h.astype(np.float32), "stem_zone": zone.astype(np.float32)})
