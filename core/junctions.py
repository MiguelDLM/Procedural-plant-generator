"""
Continuous junctions between stem, branches and roots.

Tubes meeting at a fork or at the root collar always intersect, so a seam stays
visible however well their girths match. The fix used here is a hybrid:

1. `split_axes` separates every axis into a *thick* part (radius above a
   threshold: trunk, codominant leaders, limb bases, main roots near the
   stem) and a *thin* remainder, overlapping by one segment.
2. The thick parts are fused into a single closed surface by a volumetric
   (voxel) union and smoothed, which turns crotches and root flares into
   continuous fillets (done in Blender with OpenVDB; see blender/junctions.py).
3. `cylindrical_uvs` re-creates bark coordinates on the fused surface with the
   same frames, arc-length offsets and circumferential repeats as the tubes,
   so bark continues across the thick/thin cut. Each face takes the frame of
   its nearest axis; where neighbouring faces belong to different axes the
   bark pattern meets along a line, like the branch-bark ridge of real crotches.

The thin remainders keep their exact tube geometry and start two samples
inside the fused part, whose end is tucked (narrowed) so the tube emerges from
it without a lip or a coincident surface.
"""

import math
import numpy as np

from .architecture import Axis, BranchingGraph

UP = np.array([0.0, 0.0, 1.0])


def _normalize(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)


def axis_tangents(P: np.ndarray) -> np.ndarray:
    """Central-difference tangents for (..., k, 3) polylines."""
    T = np.empty_like(P)
    T[..., 1:-1, :] = P[..., 2:, :] - P[..., :-2, :]
    T[..., 0, :] = P[..., 1, :] - P[..., 0, :]
    T[..., -1, :] = P[..., -1, :] - P[..., -2, :]
    return _normalize(T)


def default_n0(T0: np.ndarray) -> np.ndarray:
    """Reference normal of an axis from its first tangent (same convention for tubes and fused UVs)."""
    T0 = np.atleast_2d(T0)
    ref = np.where(np.abs(T0[:, 2:3]) > 0.9, np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0]))
    return _normalize(np.cross(T0, ref))


def axis_frames(T: np.ndarray, n0: np.ndarray, frame_up: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Projection frames (N, B) along (G, k) axes; `frame_up` aligns N with world up (root planks)."""
    N = n0[:, None, :] - np.sum(n0[:, None, :] * T, axis=-1, keepdims=True) * T
    bad = np.linalg.norm(N, axis=-1) < 1e-4
    if np.any(bad):
        N[bad] = np.cross(T[bad], np.array([1.0, 0.0, 0.0]))
    N = _normalize(N)
    if frame_up:
        Nu = UP - np.sum(UP * T, axis=-1, keepdims=True) * T
        ok = np.linalg.norm(Nu, axis=-1) > 0.2
        N = np.where(ok[..., None], _normalize(Nu), N)
    return N, np.cross(T, N)


def u_repeats(r_base: np.ndarray, tile: float) -> np.ndarray:
    """Bark tiles around the circumference: whole numbers on thick axes (seamless), fractional on thin
    ones so twig bark is not stretched (their wrap seam is too small to notice)."""
    c = 2 * math.pi * np.asarray(r_base, dtype=float) / tile
    return np.where(c < 1.5, np.maximum(c, 0.05), np.round(c))


def _with(axis: Axis, positions, radii, aspect, **kw) -> Axis:
    out = Axis(positions, radii, axis.order, axis.parent_axis, axis.parent_sample, axis.azimuth,
               aspect=aspect, frame_up=axis.frame_up)
    for k, v in kw.items():
        setattr(out, k, v)
    return out


def split_axes(graphs: list[BranchingGraph], r_threshold: float, tile: float,
               tuck: float = 0.8) -> tuple[list[Axis], list[Axis]]:
    """
    Splits axes into thick parts (to be fused) and thin remainders (kept as tubes).

    The thick part runs from the base to the last sample whose radius is >= r_threshold,
    plus one sample of overlap. Both parts carry the original axis' reference frame,
    arc-length offset and bark repeat so bark coordinates stay continuous.
    """
    thick, thin = [], []
    for gi, g in enumerate(graphs):
        if g is None:
            continue
        for ai, a in enumerate(g.axes):
            trunk = gi == 0 and ai == 0 and not a.frame_up
            if len(a.radii) < 2:
                continue
            T0 = axis_tangents(a.positions)[0]
            n0 = a.frame_n0 if a.frame_n0 is not None else default_n0(T0)[0]
            u_rep = a.u_rep if a.u_rep is not None else float(u_repeats(a.radii[0], tile))
            s = a.arc_length + (a.v_offset or 0.0)
            above = np.nonzero(a.radii >= r_threshold)[0]
            if len(above) == 0 or above[0] != 0:
                thin.append(_with(a, a.positions, a.radii, a.aspect, frame_n0=n0, u_rep=u_rep,
                                  v_offset=a.v_offset or 0.0, is_trunk=trunk))
                continue
            # Contiguous thick run from the base
            breaks = np.nonzero(np.diff(above) != 1)[0]
            last = int(above[breaks[0]] if len(breaks) else above[-1])
            end = min(len(a.radii) - 1, last + 1)
            thick_r = a.radii[:end + 1].copy()
            if end < len(a.radii) - 1:
                # Tuck the fused end inside the continuing tube so the tube emerges without a lip
                thick_r[-2:] *= tuck
            thick.append(_with(a, a.positions[:end + 1], thick_r,
                               None if a.aspect is None else a.aspect[:end + 1],
                               frame_n0=n0, u_rep=u_rep, v_offset=a.v_offset or 0.0, is_trunk=trunk))
            if end < len(a.radii) - 1:
                # Overlap: the tube begins two samples inside the fused part (which smoothing shrinks)
                start = max(0, end - 2)
                radii = a.radii[start:].copy()
                thin.append(_with(a, a.positions[start:], radii,
                                  None if a.aspect is None else a.aspect[start:],
                                  frame_n0=n0, u_rep=u_rep, v_offset=float(s[start]), is_trunk=trunk))
    return thick, thin


def graph_of(axes: list[Axis]) -> BranchingGraph:
    """Wraps a list of (split) axes in a graph for meshing; parent links refer to the original graphs."""
    g = BranchingGraph()
    g.axes = list(axes)
    return g


def sample_table(axes: list[Axis]) -> dict:
    """Per-sample frames and bark coordinates of the thick axes (for nearest-sample lookups)."""
    pos, T_all, N_all, B_all, s_all, urep, aid, rad = [], [], [], [], [], [], [], []
    for i, a in enumerate(axes):
        T = axis_tangents(a.positions)
        n0 = a.frame_n0 if a.frame_n0 is not None else default_n0(T[0])[0]
        N, B = axis_frames(T[None], np.asarray(n0)[None], a.frame_up)
        pos.append(a.positions)
        T_all.append(T)
        N_all.append(N[0])
        B_all.append(B[0])
        s_all.append(a.arc_length + (a.v_offset or 0.0))
        urep.append(np.full(len(a.radii), a.u_rep if a.u_rep is not None else 1.0))
        aid.append(np.full(len(a.radii), i))
        rad.append(a.radii * (np.sqrt(a.aspect) if a.aspect is not None else 1.0))
    cat = lambda xs: np.concatenate(xs) if xs else np.zeros((0, 3))
    return {"pos": cat(pos), "T": cat(T_all), "N": cat(N_all), "B": cat(B_all), "s": cat(s_all),
            "u_rep": cat(urep), "axis": cat(aid).astype(int), "r": cat(rad)}


def nearest_samples(points: np.ndarray, sample_pos: np.ndarray, sample_r: np.ndarray = None,
                    chunk: int = 4096) -> np.ndarray:
    """
    Sample whose tube *surface* is nearest (|p - c| - r), so points on a branch running close to
    its parent are not assigned to the parent axis. NumPy fallback; Blender uses a KD-tree.
    """
    r = np.zeros(len(sample_pos)) if sample_r is None else sample_r
    out = np.empty(len(points), dtype=int)
    for i in range(0, len(points), chunk):
        d = np.sqrt(np.sum((points[i:i + chunk, None, :] - sample_pos[None, :, :]) ** 2, axis=-1)) - r[None, :]
        out[i:i + chunk] = np.argmin(d, axis=1)
    return out


def cylindrical_uvs(verts: np.ndarray, loop_vertex: np.ndarray, loop_start: np.ndarray, loop_total: np.ndarray,
                    nearest: np.ndarray, table: dict, tile: float) -> np.ndarray:
    """
    Per-loop bark UVs for a fused surface. Each face uses the frame of the sample nearest to its
    first vertex (one axis per face, so no face is stretched across two axes); U is the angle
    around that axis times the axis' circumferential repeat, V the arc length / tile. U is
    unwrapped per face so faces straddling the 0/2pi seam stay continuous.
    """
    face_of_loop = np.repeat(np.arange(len(loop_start)), loop_total)
    ref_sample = nearest[loop_vertex[loop_start]]          # (F,)
    smp = ref_sample[face_of_loop]                         # (L,)
    d = verts[loop_vertex] - table["pos"][smp]
    along = np.sum(d * table["T"][smp], axis=1)
    ang = np.arctan2(np.sum(d * table["B"][smp], axis=1), np.sum(d * table["N"][smp], axis=1))
    ang = np.mod(ang, 2 * math.pi)
    u = ang / (2 * math.pi) * table["u_rep"][smp]
    v = (table["s"][smp] + along) / tile
    # Unwrap U within each face relative to its first loop
    period = table["u_rep"][smp]
    u0 = u[loop_start][face_of_loop]
    u = u - np.round((u - u0) / period) * period
    return np.stack([u, v], axis=1).astype(np.float32)
