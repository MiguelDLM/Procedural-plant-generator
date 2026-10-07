"""
Procedural leaf texture synthesis (pure NumPy).

Rasterizes a leaf shape model and its vein network into:
- colour + alpha (sRGB, row 0 = bottom, matching Blender's pixel order),
- a height map for bump/normal shading (veins and areole relief).

Colour model:
- mesophyll: adaxial chlorophyll colour with low-frequency mottling,
- veins: less chlorophyll (lighter, yellower), contrast scaled per vein order,
- minor reticulum: Voronoi areoles sized from VLA (d ~ 2 / VLA), clamped to the
  smallest spacing the texture resolution can show,
- senescence: carotenoid/anthocyanin colour invading the lamina from the margin
  and the intercostal areas first while tissue along the veins stays green
  longest.
"""

from dataclasses import dataclass
import math
import numpy as np

from .leaf_morphology import LeafMorphologyProfile, LeafShapeBuilder, LeafShapeModel, smooth_min
from .leaf_venation import VenationEngine, VenationProfile, LeafVeinNetwork

# Visual weight and minimum drawn radius (pixels) per vein order
ORDER_INTENSITY = {1: 1.0, 2: 0.78, 3: 0.5, 4: 0.38}
ORDER_MIN_RADIUS_PX = {1: 1.3, 2: 0.85, 3: 0.55, 4: 0.45}
ORDER_HEIGHT = {1: 1.0, 2: 0.7, 3: 0.4, 4: 0.3}


@dataclass
class LeafTexture:
    color: np.ndarray    # (H, W, 4) float32, sRGB + alpha
    height: np.ndarray   # (H, W) float32 in [0, 1]
    bounds: tuple        # (x0, y0, x1, y1) in BLU covered by the image
    vein_network: LeafVeinNetwork

    @property
    def size(self) -> tuple[int, int]:
        return self.color.shape[1], self.color.shape[0]


def _smooth_noise(h: int, w: int, cell: float, rng) -> np.ndarray:
    cell = max(1.0, cell)
    gh, gw = int(h / cell) + 3, int(w / cell) + 3
    g = rng.random((gh, gw)).astype(np.float32)
    ys = np.arange(h, dtype=np.float32) / cell
    xs = np.arange(w, dtype=np.float32) / cell
    y0 = ys.astype(int)
    x0 = xs.astype(int)
    fy = ys - y0
    fx = xs - x0
    fy = fy * fy * (3 - 2 * fy)
    fx = fx * fx * (3 - 2 * fx)
    a = g[y0][:, x0]
    b = g[y0][:, x0 + 1]
    c = g[y0 + 1][:, x0]
    d = g[y0 + 1][:, x0 + 1]
    top = a + (b - a) * fx[None, :]
    bot = c + (d - c) * fx[None, :]
    return top + (bot - top) * fy[:, None]


def _box_blur(img: np.ndarray, r: int) -> np.ndarray:
    if r < 1:
        return img
    out = img
    for axis in (0, 1):
        pad = [(0, 0), (0, 0)]
        pad[axis] = (r + 1, r)
        c = np.cumsum(np.pad(out, pad, mode="edge"), axis=axis, dtype=np.float64)
        n = out.shape[axis]
        hi = np.take(c, np.arange(2 * r + 1, 2 * r + 1 + n), axis=axis)
        lo = np.take(c, np.arange(0, n), axis=axis)
        out = ((hi - lo) / (2 * r + 1)).astype(np.float32)
    return out


def _voronoi_edges(h: int, w: int, cell: float, rng, width_px: float = 0.8) -> np.ndarray:
    """Coverage of Voronoi cell borders (F2 - F1 metric) on a jittered grid."""
    gh, gw = int(h / cell) + 3, int(w / cell) + 3
    jitter = rng.random((gh, gw, 2)).astype(np.float32) * 0.85 + 0.075
    ys = (np.arange(h, dtype=np.float32) + 0.5) / cell
    xs = (np.arange(w, dtype=np.float32) + 0.5) / cell
    cy = ys.astype(int)
    cx = xs.astype(int)
    f1 = np.full((h, w), 1e9, dtype=np.float32)
    f2 = np.full((h, w), 1e9, dtype=np.float32)
    for oy in (-1, 0, 1):
        ny = np.clip(cy + oy, 0, gh - 1)
        for ox in (-1, 0, 1):
            nx = np.clip(cx + ox, 0, gw - 1)
            jy = jitter[ny][:, nx, 0]
            jx = jitter[ny][:, nx, 1]
            dy = (ny[:, None] + jy) - ys[:, None]
            dx = (nx[None, :] + jx) - xs[None, :]
            d = np.sqrt(dx * dx + dy * dy)
            f2 = np.where(d < f1, f1, np.minimum(f2, d))
            f1 = np.minimum(f1, d)
    edge_px = (f2 - f1) * cell * 0.5
    return np.clip(1.0 - edge_px / width_px, 0.0, 1.0)


class LeafTextureEngine:
    """Renders colour/alpha and height textures for one foliar unit."""

    def __init__(self, morphology: LeafMorphologyProfile, venation: VenationProfile = None):
        self.morphology = morphology
        self.venation = venation or VenationProfile()

    def render(self, resolution: int = 1024, senescence: float = 0.0, seed: int = 11,
               model: LeafShapeModel = None) -> LeafTexture:
        rng = np.random.default_rng(seed)
        morph = self.morphology
        model = model or LeafShapeBuilder(morph).build()
        x0, y0, x1, y1 = model.bounds
        span_x, span_y = x1 - x0, y1 - y0
        if span_y >= span_x:
            H = int(resolution)
            W = max(32, int(round(resolution * span_x / span_y / 8.0)) * 8)
        else:
            W = int(resolution)
            H = max(32, int(round(resolution * span_y / span_x / 8.0)) * 8)
        px = span_y / H  # BLU per pixel (square pixels up to rounding)
        px_x = span_x / W

        xs = x0 + (np.arange(W, dtype=np.float32) + 0.5) * px_x
        ys = y0 + (np.arange(H, dtype=np.float32) + 0.5) * px

        # ---- 1. Implicit outline (evaluated per blade inside its bounding window)
        F = np.full((H, W), 1e3, dtype=np.float32)
        stalk = np.zeros((H, W), dtype=np.float32)
        stem = np.zeros((H, W), dtype=np.float32)
        for blade in model.blades:
            bx0, by0, bx1, by1 = blade.bbox()
            pad = blade.blend + 2 * px
            i0 = max(0, int((by0 - pad - y0) / px))
            i1 = min(H, int((by1 + pad - y0) / px) + 2)
            j0 = max(0, int((bx0 - pad - x0) / px_x))
            j1 = min(W, int((bx1 + pad - x0) / px_x) + 2)
            if i1 <= i0 or j1 <= j0:
                continue
            X, Y = np.meshgrid(xs[j0:j1], ys[i0:i1])
            f = blade.field(X, Y).astype(np.float32)
            F[i0:i1, j0:j1] = smooth_min(F[i0:i1, j0:j1], f, blade.blend)
            if blade.is_axis:
                target = stem if blade.role == "stem" else stalk
                target[i0:i1, j0:j1] = np.maximum(target[i0:i1, j0:j1], np.clip(0.5 - f / px, 0.0, 1.0))
        alpha = np.clip(0.5 - F / px, 0.0, 1.0)
        inside_dist = np.clip(-F, 0.0, None)  # ~ distance to margin (BLU)

        # ---- 2. Vein network
        blade_mm = morph.blade_length_cm * 10.0
        net = VenationEngine(self.venation).generate(model, blade_mm, seed=seed)
        vein = np.zeros((H, W), dtype=np.float32)
        height = np.zeros((H, W), dtype=np.float32)
        self._stamp_veins(net, vein, height, x0, y0, px_x, px)

        # ---- 3. Minor reticulum (quaternary veins and areoles)
        is_needle = morph.compound_type.value in ("Fascicle", "Spray")
        spacing_px = net.minor_spacing_blu / px
        if not is_needle and self.venation.pattern.value not in ("Parallelodromous",):
            quaternary_cell = max(9.0, spacing_px * 3.2)
            areole_cell = max(4.5, spacing_px)
            q = _voronoi_edges(H, W, quaternary_cell, rng, 0.75) * ORDER_INTENSITY[3] * 0.85
            a = _voronoi_edges(H, W, areole_cell, rng, 0.55) * ORDER_INTENSITY[4] * 0.6
            minor = np.maximum(q, a)
            vein = np.maximum(vein, minor)
            height = np.maximum(height, np.maximum(q * ORDER_HEIGHT[3], a * ORDER_HEIGHT[4]) * 0.6)

        # ---- 4. Colour composition
        ad = np.array(morph.adaxial_color, dtype=np.float32)
        vc = np.array(morph.vein_color, dtype=np.float32)
        au = np.array(morph.autumn_color, dtype=np.float32)

        mottle = _smooth_noise(H, W, max(H, W) / 14.0, rng) - 0.5
        fine = _smooth_noise(H, W, 3.0, rng) - 0.5
        lum = 1.0 + mottle * 0.16 + fine * 0.05
        base = ad[None, None, :] * lum[..., None]

        # Slight chlorophyll gradient: younger tissue toward the margin is lighter
        margin_band = np.clip(1.0 - inside_dist / 0.035, 0.0, 1.0)
        base = base * (1.0 + 0.10 * margin_band[..., None])

        contrast = float(np.clip(self.venation.vein_contrast, 0.0, 1.5))
        vmix = np.clip(vein * contrast, 0.0, 1.0)[..., None]
        color = base * (1.0 - vmix) + vc[None, None, :] * vmix

        # Stalk tissue (petiole, rachis, sheath)
        stalk_col = vc * 0.85 + np.array([0.06, 0.02, 0.0], dtype=np.float32)
        color = color * (1.0 - stalk[..., None]) + stalk_col[None, None, :] * stalk[..., None]
        twig_col = np.array([0.33, 0.26, 0.17], dtype=np.float32)  # Current-year shoot bark
        color = color * (1.0 - stem[..., None]) + twig_col[None, None, :] * stem[..., None]

        # Senescence
        s = float(np.clip(senescence, 0.0, 1.0))
        if s > 0.0:
            near_vein = _box_blur(np.clip(vein * 1.5, 0.0, 1.0), max(1, int(spacing_px * 0.8)))
            blotch = _smooth_noise(H, W, max(H, W) / 9.0, rng)
            m = s * 1.9 - 0.75 + 0.55 * margin_band + 0.45 * (blotch - 0.5) - 0.9 * near_vein * (1.0 - s)
            m = np.clip(m, 0.0, 1.0)[..., None]
            autumn = au[None, None, :] * (0.85 + 0.3 * blotch[..., None])
            color = color * (1.0 - m) + autumn * m

        color = np.clip(color, 0.0, 1.0)
        rgba = np.concatenate([color, alpha[..., None]], axis=-1).astype(np.float32)
        height = np.clip(height * 0.85 + (fine + 0.5) * 0.08, 0.0, 1.0).astype(np.float32)
        return LeafTexture(color=rgba, height=height, bounds=(x0, y0, x1, y1), vein_network=net)

    @staticmethod
    def _stamp_veins(net: LeafVeinNetwork, vein: np.ndarray, height: np.ndarray,
                     x0: float, y0: float, px_x: float, px_y: float):
        """Draws all vein polylines by stamping anti-aliased discs every half pixel (vectorized)."""
        H, W = vein.shape
        samples_x, samples_y, samples_r, samples_i, samples_h = [], [], [], [], []
        for pl in net.polylines:
            pts = pl.points
            if len(pts) < 2:
                continue
            p = np.empty_like(pts)
            p[:, 0] = (pts[:, 0] - x0) / px_x - 0.5
            p[:, 1] = (pts[:, 1] - y0) / px_y - 0.5
            seg = np.diff(p, axis=0)
            seg_len = np.linalg.norm(seg, axis=1)
            n_sub = np.maximum(1, np.ceil(seg_len / 0.5).astype(int))
            idx = np.repeat(np.arange(len(seg)), n_sub)
            offs = np.concatenate([np.arange(k) / k for k in n_sub])
            sx = p[idx, 0] + seg[idx, 0] * offs
            sy = p[idx, 1] + seg[idx, 1] * offs
            r_blu = pl.radii[idx] + (pl.radii[idx + 1] - pl.radii[idx]) * offs
            order = min(4, max(1, pl.order))
            r_px = np.maximum(r_blu / px_y, ORDER_MIN_RADIUS_PX[order])
            samples_x.append(sx)
            samples_y.append(sy)
            samples_r.append(r_px)
            samples_i.append(np.full(len(sx), ORDER_INTENSITY[order], dtype=np.float32))
            samples_h.append(np.full(len(sx), ORDER_HEIGHT[order], dtype=np.float32))
        if not samples_x:
            return
        sx = np.concatenate(samples_x)
        sy = np.concatenate(samples_y)
        sr = np.concatenate(samples_r)
        si = np.concatenate(samples_i)
        sh = np.concatenate(samples_h)
        reach = np.ceil(sr + 1.0).astype(int)
        flat_v = vein.ravel()
        flat_h = height.ravel()
        for R in np.unique(reach):
            sel = reach == R
            gx, gy, gr, gi, gh = sx[sel], sy[sel], sr[sel], si[sel], sh[sel]
            cx = np.round(gx).astype(int)
            cy = np.round(gy).astype(int)
            for oy in range(-R, R + 1):
                for ox in range(-R, R + 1):
                    ix = cx + ox
                    iy = cy + oy
                    ok = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
                    if not np.any(ok):
                        continue
                    d = np.hypot(ix - gx, iy - gy)
                    cov = np.clip(gr + 0.5 - d, 0.0, 1.0)
                    ok &= cov > 0.0
                    if not np.any(ok):
                        continue
                    lin = iy[ok] * W + ix[ok]
                    np.maximum.at(flat_v, lin, (cov * gi)[ok].astype(np.float32))
                    prof = np.sqrt(np.clip(1.0 - (d / np.maximum(gr, 1e-3)) ** 2, 0.0, 1.0)) * gh
                    np.maximum.at(flat_h, lin, prof[ok].astype(np.float32))
