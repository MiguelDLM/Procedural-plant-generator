"""
Vectorized botanical mesh engine.

Produces mesh data directly in Blender's native layout (vertices, per-loop
vertex indices, polygon loop starts/totals, per-loop UVs) so the Blender side
can fill meshes with foreach_set instead of Python lists.

Wood: one ring of `n` vertices per skeleton node, quads between rings, n-gon tip
caps, Gielis-fluted root flare on the trunk, branch collars, and bark UVs whose
U repeats with circumference so bark texel density stays constant.

Foliage: every leaf is an instance of one curved, alpha-mapped card, transformed
with a single batched matrix product.
"""

from dataclasses import dataclass, field
import math
import numpy as np

from .architecture import BranchingGraph
from .foliage import FoliageInstances
from .gielis import GielisEngine, GielisProfile
from .junctions import axis_frames, default_n0, u_repeats


@dataclass
class MeshConfig:
    radial_resolution: int = 12
    twig_resolution: int = 5
    collar_flare_factor: float = 1.0   # Collars are modelled in the skeleton radii
    smooth_caps: bool = True
    buttress_profile: GielisProfile = None
    flare_amplitude: float = 0.45
    flare_decay: float = 12.0
    bark_tile_m: float = 0.6
    flute_azimuth: float | None = None  # World azimuth of one Gielis lobe (aligned with a main root)
    start_caps: bool = False            # Close axis bases too (watertight tubes for volumetric fusion)


@dataclass
class MeshData:
    vertices: np.ndarray                        # (N, 3) float32
    loop_vertex: np.ndarray                     # (L,) int32
    loop_start: np.ndarray                      # (F,) int32
    loop_total: np.ndarray                      # (F,) int32
    loop_uv: np.ndarray                         # (L, 2) float32
    point_attributes: dict = field(default_factory=dict)  # name -> (N,) array

    @property
    def faces(self) -> list[list[int]]:
        return [self.loop_vertex[s:s + t].tolist() for s, t in zip(self.loop_start, self.loop_total)]

    def __len__(self):
        return len(self.vertices)

    @staticmethod
    def empty() -> "MeshData":
        return MeshData(np.zeros((0, 3), np.float32), np.zeros(0, np.int32), np.zeros(0, np.int32),
                        np.zeros(0, np.int32), np.zeros((0, 2), np.float32))

    @staticmethod
    def concatenate(parts: list["MeshData"]) -> "MeshData":
        parts = [p for p in parts if len(p.vertices)]
        if not parts:
            return MeshData.empty()
        v_off = np.cumsum([0] + [len(p.vertices) for p in parts[:-1]])
        l_off = np.cumsum([0] + [len(p.loop_vertex) for p in parts[:-1]])
        attrs = {}
        for name in parts[0].point_attributes:
            attrs[name] = np.concatenate([p.point_attributes[name] for p in parts])
        return MeshData(
            np.concatenate([p.vertices for p in parts]).astype(np.float32),
            np.concatenate([p.loop_vertex + o for p, o in zip(parts, v_off)]).astype(np.int32),
            np.concatenate([p.loop_start + o for p, o in zip(parts, l_off)]).astype(np.int32),
            np.concatenate([p.loop_total for p in parts]).astype(np.int32),
            np.concatenate([p.loop_uv for p in parts]).astype(np.float32),
            attrs,
        )


def _rows_normalize(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)


class BotanicalMeshEngine:
    """Generates wood and foliage meshes from a skeleton."""

    def __init__(self, config: MeshConfig = None):
        self.config = config or MeshConfig()

    def build_wood_mesh(self, skeleton: BranchingGraph, total_height_m: float,
                        gielis_profile: GielisProfile = None, trunk_index: int = 0) -> MeshData:
        """Batches axes that share (sample count, ring resolution) into single array operations."""
        cfg = self.config
        g_prof = gielis_profile or cfg.buttress_profile or GielisProfile(m=0.0)
        groups: dict[tuple, list[int]] = {}
        for i, axis in enumerate(skeleton.axes):
            if len(axis.radii) < 2:
                continue
            n = max(3, int(cfg.radial_resolution if axis.order <= 1 else cfg.twig_resolution))
            is_trunk = (i == trunk_index or getattr(axis, "is_trunk", False)) and not axis.frame_up
            if is_trunk and g_prof.m > 0:
                n = max(n, int(6 * g_prof.m))
            key = (len(axis.radii), n, is_trunk, axis.frame_up, i if is_trunk else -1)
            groups.setdefault(key, []).append(i)
        parts = []
        for (k, n, is_trunk, frame_up, _), ids in groups.items():
            P = np.stack([skeleton.axes[i].positions for i in ids])
            R = np.stack([skeleton.axes[i].radii for i in ids]).astype(float)
            A = np.stack([skeleton.axes[i].aspect if skeleton.axes[i].aspect is not None else np.ones(k)
                          for i in ids]).astype(float)
            orders = np.array([skeleton.axes[i].order for i in ids])
            axes = [skeleton.axes[i] for i in ids]
            parts.append(self._tube_group(P, R, orders, n, is_trunk, total_height_m, g_prof, A, frame_up, axes))
        return MeshData.concatenate(parts)

    def _tube_group(self, P, R, orders, n, is_trunk, total_height_m, g_prof, aspect=None,
                    frame_up=False, axes=None) -> MeshData:
        cfg = self.config
        G, k, _ = P.shape
        # Tangents and a projection frame (stable for axes turning < 90 deg)
        T = np.empty_like(P)
        T[:, 1:-1] = P[:, 2:] - P[:, :-2]
        T[:, 0] = P[:, 1] - P[:, 0]
        T[:, -1] = P[:, -1] - P[:, -2]
        T = _rows_normalize(T)
        n0 = default_n0(T[:, 0])
        if axes is not None:
            for g, a in enumerate(axes):
                if a.frame_n0 is not None:
                    n0[g] = a.frame_n0
        # frame_up: N follows world up so vertically elongated (plank) sections stand upright
        N, B = axis_frames(T, n0, frame_up)

        angles = np.linspace(0.0, 2 * math.pi, n, endpoint=False)
        cos_a, sin_a = np.cos(angles), np.sin(angles)
        ring_r = np.repeat(R[:, :, None], n, axis=2)
        if is_trunk:
            # Radial flare is already in the skeleton radii; here only the angular Gielis fluting
            z_rel = np.clip(P[:, :, 2] / max(1.0, total_height_m), 0.0, 1.0)
            if g_prof.m > 0.0:
                shift = 0.0
                if cfg.flute_azimuth is not None:
                    az_n0 = math.atan2(N[0, 0, 1], N[0, 0, 0])
                    shift = az_n0 - cfg.flute_azimuth
                g = GielisEngine.evaluate(angles + shift, g_prof)
                w = np.exp(-cfg.flare_decay * z_rel)[:, :, None]
                ring_r *= np.clip(1.0 + (g[None, None, :] - 1.0) * w * min(1.5, cfg.flare_amplitude * 1.6),
                                  0.25, 3.0)
        else:
            ring_r[:, 0, :] *= np.where(orders > 0, cfg.collar_flare_factor, 1.0)[:, None]

        if aspect is None:
            aspect = np.ones(R.shape)
        sx = np.sqrt(aspect)[:, :, None, None]
        offs = (cos_a[None, None, :, None] * N[:, :, None, :] * sx
                + sin_a[None, None, :, None] * B[:, :, None, :] / sx)
        radial = ring_r[..., None] * offs
        verts = (P[:, :, None, :] + radial).reshape(-1, 3)
        # 3D bark coordinates (seam-free, world scale): Q = bark_base + k * bark_along in the shader
        origin = P[:, 0, :].copy()
        if axes is not None:
            for g, a in enumerate(axes):
                if a.bark_origin is not None:
                    origin[g] = a.bark_origin
        along = np.broadcast_to((P - origin[:, None, :])[:, :, None, :], radial.shape)
        bark_base = (origin[:, None, None, :] + radial).reshape(-1, 3)
        bark_along = along.reshape(-1, 3)

        # Quads between rings (local indices, then offset per axis)
        ring = np.arange(k - 1)[:, None] * n
        s = np.arange(n)[None, :]
        s1 = (s + 1) % n
        q = np.stack([ring + s, ring + s1, ring + n + s1, ring + n + s], axis=-1).reshape(-1, 4)
        per_axis = k * n
        axis_off = (np.arange(G) * per_axis)[:, None, None]
        quads = (q[None, :, :] + axis_off).reshape(G, -1)

        # Bark UVs: U repeats with circumference, V with arc length
        seg_len = np.linalg.norm(np.diff(P, axis=1), axis=-1)
        v_off = np.array([a.v_offset for a in axes])[:, None] if axes is not None else np.zeros((G, 1))
        v_coord = (np.concatenate([np.zeros((G, 1)), np.cumsum(seg_len, axis=1)], axis=1) + v_off) / cfg.bark_tile_m
        u_rep = u_repeats(R[:, 0], cfg.bark_tile_m)
        if axes is not None:
            u_rep = np.array([a.u_rep if a.u_rep is not None else u for a, u in zip(axes, u_rep)])
        u_lo = (np.arange(n) / n)[None, None, :] * u_rep[:, None, None]
        u_hi = ((np.arange(n) + 1) / n)[None, None, :] * u_rep[:, None, None]
        v0 = np.broadcast_to(v_coord[:, :-1, None], (G, k - 1, n))
        v1 = np.broadcast_to(v_coord[:, 1:, None], (G, k - 1, n))
        u_lo = np.broadcast_to(u_lo, (G, k - 1, n))
        u_hi = np.broadcast_to(u_hi, (G, k - 1, n))
        quad_uv = np.stack([np.stack([u_lo, v0], -1), np.stack([u_hi, v0], -1),
                            np.stack([u_hi, v1], -1), np.stack([u_lo, v1], -1)], axis=3).reshape(G, -1, 2)

        if cfg.smooth_caps:
            caps = ((k - 1) * n + np.arange(n))[None, :] + (np.arange(G) * per_axis)[:, None]
            cap_uv = np.broadcast_to(np.stack([0.5 + 0.5 * cos_a, 0.5 + 0.5 * sin_a], 1), (G, n, 2))
            blocks, uv_blocks, tot = [quads, caps], [quad_uv, cap_uv], [np.full((k - 1) * n, 4), [n]]
            if cfg.start_caps:
                base = np.arange(n)[::-1][None, :] + (np.arange(G) * per_axis)[:, None]  # Reversed: faces outward
                blocks.append(base)
                uv_blocks.append(cap_uv[:, ::-1])
                tot.append([n])
            loops = np.concatenate(blocks, axis=1).reshape(-1)
            uv = np.concatenate(uv_blocks, axis=1).reshape(-1, 2)
            totals = np.tile(np.concatenate(tot), G)
        else:
            loops = quads.reshape(-1)
            uv = quad_uv.reshape(-1, 2)
            totals = np.full(G * (k - 1) * n, 4)
        starts = np.concatenate([[0], np.cumsum(totals[:-1])])
        return MeshData(verts.astype(np.float32), loops.astype(np.int32), starts.astype(np.int32),
                        totals.astype(np.int32), uv.astype(np.float32),
                        {"branch_order": np.repeat(orders, per_axis).astype(np.int32),
                         "bark_base": bark_base.astype(np.float32), "bark_along": bark_along.astype(np.float32),
                         # Local axis radius: proxy for bark age (pipe model), drives the young -> old bark
                         "bark_radius": np.repeat(R[:, :, None], n, axis=2).reshape(-1).astype(np.float32)})

    @staticmethod
    def build_foliage_mesh(instances: FoliageInstances, card: dict, leaf_scale: float = 1.0) -> MeshData:
        n = len(instances)
        V = np.asarray(card["vertices"], dtype=np.float32)
        F = np.asarray(card["faces"], dtype=np.int32)
        UV = np.asarray(card["uvs"], dtype=np.float32)
        if n == 0 or len(V) == 0:
            return MeshData.empty()
        X = instances.axis_x.astype(np.float32)
        Y = instances.axis_y.astype(np.float32)
        Z = instances.axis_z.astype(np.float32)
        s = (instances.scales * leaf_scale).astype(np.float32)[:, None, None]
        world = (instances.positions.astype(np.float32)[:, None, :]
                 + s * (V[None, :, 0:1] * X[:, None, :] + V[None, :, 1:2] * Y[:, None, :]
                        + V[None, :, 2:3] * Z[:, None, :]))
        k = len(V)
        loop_vertex = (F[None, :, :] + (np.arange(n, dtype=np.int32) * k)[:, None, None]).reshape(-1)
        n_faces = n * len(F)
        loop_total = np.full(n_faces, F.shape[1], dtype=np.int32)
        loop_start = (np.arange(n_faces, dtype=np.int32) * F.shape[1])
        loop_uv = np.tile(UV[F.reshape(-1)], (n, 1))
        rnd = np.repeat(instances.randoms.astype(np.float32), k)
        return MeshData(world.reshape(-1, 3), loop_vertex.astype(np.int32), loop_start, loop_total,
                        loop_uv.astype(np.float32), {"leaf_random": rnd})
