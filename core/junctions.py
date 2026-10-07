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
    out.bark_origin = axis.bark_origin if axis.bark_origin is not None else axis.positions[0].copy()
    for k, v in kw.items():
        setattr(out, k, v)
    return out


def resample_axis(ax: Axis, q: np.ndarray):
    """Positions, radii and aspect of an axis at arc lengths `q` (linear interpolation)."""
    sa = ax.arc_length
    q = np.clip(q, 0.0, sa[-1])
    pos = np.stack([np.interp(q, sa, ax.positions[:, k]) for k in range(3)], axis=1)
    rad = np.interp(q, sa, ax.radii)
    asp = None if ax.aspect is None else np.interp(q, sa, ax.aspect)
    return pos, rad, asp


def handover(ax: Axis, s_cut: float, overlap: float, tuck: float = 0.45, grow_from: float = 0.85):
    """
    Tangential hand-over between a fused part and the tube that continues it.

    The fused part keeps the axis up to `s_cut` and then narrows progressively to `tuck` x radius over
    `overlap`; the tube starts at s_cut - 0.2 x overlap at `grow_from` x radius and grows back to the
    full radius by s_cut + 0.6 x overlap. Both surfaces therefore cross at a shallow angle instead of
    meeting in a step or a crease. Returns (fused_piece_arrays, tube_piece_arrays, tube_start_s).
    """
    sa = ax.arc_length
    overlap = float(min(overlap, max(1e-4, sa[-1] - s_cut)))
    keep = sa < s_cut
    q_tail = s_cut + overlap * np.array([0.0, 0.35, 0.7, 1.0])
    p_t, r_t, a_t = resample_axis(ax, q_tail)
    r_t = r_t * (1.0 - (1.0 - tuck) * (q_tail - s_cut) / overlap)
    f_pos = np.vstack([ax.positions[keep], p_t])
    f_rad = np.concatenate([ax.radii[keep], r_t])
    f_asp = None if ax.aspect is None else np.concatenate([ax.aspect[keep], a_t])

    s0 = max(0.0, s_cut - 0.2 * overlap)
    q_head = s0 + (s_cut + 0.6 * overlap - s0) * np.array([0.0, 0.4, 1.0])
    rest = sa > q_head[-1]
    p_h, r_h, a_h = resample_axis(ax, q_head)
    r_h = r_h * (grow_from + (1.0 - grow_from) * np.array([0.0, 0.6, 1.0]))
    t_pos = np.vstack([p_h, ax.positions[rest]])
    t_rad = np.concatenate([r_h, ax.radii[rest]])
    t_asp = None if ax.aspect is None else np.concatenate([a_h, ax.aspect[rest]])
    return (f_pos, f_rad, f_asp), (t_pos, t_rad, t_asp), s0


def split_axes(graphs: list[BranchingGraph], r_threshold: float, tile: float) -> tuple[list[Axis], list[Axis]]:
    """
    Splits axes into thick parts (to be fused) and thin remainders (kept as tubes).

    The cut is placed where the radius falls below r_threshold (interpolated), with a tangential
    hand-over (see `handover`). Both parts carry the original axis' reference frame, arc-length
    offset, bark repeat and bark origin so bark coordinates stay continuous.
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
            v0 = a.v_offset or 0.0
            if a.radii[0] < r_threshold:
                thin.append(_with(a, a.positions, a.radii, a.aspect, frame_n0=n0, u_rep=u_rep,
                                  v_offset=v0, is_trunk=trunk, src=(gi, ai, 0)))
                continue
            below = np.nonzero(a.radii < r_threshold)[0]
            if len(below) == 0:
                thick.append(_with(a, a.positions, a.radii, a.aspect, frame_n0=n0, u_rep=u_rep,
                                   v_offset=v0, is_trunk=trunk))
                continue
            sa = a.arc_length
            i = int(below[0])
            f = (a.radii[i - 1] - r_threshold) / max(1e-9, a.radii[i - 1] - a.radii[i])
            s_cut = float(sa[i - 1] + f * (sa[i] - sa[i - 1]))
            (fp, fr, fa), (tp, tr, ta), s0 = handover(a, s_cut, overlap=3.0 * r_threshold)
            thick.append(_with(a, fp, fr, fa, frame_n0=n0, u_rep=u_rep, v_offset=v0, is_trunk=trunk))
            thin.append(_with(a, tp, tr, ta, frame_n0=n0, u_rep=u_rep, v_offset=v0 + s0, is_trunk=trunk,
                              src=(gi, ai, 1)))
    return thick, thin


def sector_slab(parent: Axis, lo: int, hi: int, az: float, half_angle: float, offset: float,
                thickness: float, n_ang: int = 13, center_outset: float = 0.0):
    """
    Closed curved slab following the parent's surface over samples [lo, hi) and angles az +- half_angle,
    used as a local stand-in for the parent when fusing a fine branch insertion. Its outer face rises
    `center_outset` above the parent surface around the insertion and sinks smoothly to `offset` below
    it at the borders, so the fused fillet emerges tangentially from the parent instead of creasing.
    """
    from .mesh_engine import MeshData
    P = parent.positions[lo:hi]
    R = parent.radii[lo:hi]
    k = len(P)
    T = axis_tangents(parent.positions)[lo:hi]
    n0 = parent.frame_n0 if parent.frame_n0 is not None else default_n0(axis_tangents(parent.positions)[0])[0]
    N, B = axis_frames(T[None], np.asarray(n0)[None], parent.frame_up)
    N, B = N[0], B[0]
    asp = np.sqrt(parent.aspect[lo:hi]) if parent.aspect is not None else np.ones(k)
    ang = az + np.linspace(-half_angle, half_angle, n_ang)
    off = (np.cos(ang)[None, :, None] * N[:, None, :] * asp[:, None, None]
           + np.sin(ang)[None, :, None] * B[:, None, :] / asp[:, None, None])          # (k, a, 3)
    du = np.abs(np.linspace(-1.0, 1.0, n_ang))[None, :]
    dt = np.abs(np.linspace(-1.0, 1.0, k))[:, None]
    dist = np.sqrt(du ** 2 + dt ** 2)
    w = np.clip((dist - 0.35) / 0.55, 0.0, 1.0)
    w = w * w * (3.0 - 2.0 * w)
    outer_off = center_outset * (1.0 - w) - offset * w                              # >0 outside the parent
    outer = P[:, None, :] + (R[:, None] + outer_off)[..., None] * off
    inner = P[:, None, :] + np.maximum(R[:, None, None] - offset - thickness, 0.15 * R[:, None, None]) * off
    verts = np.concatenate([outer.reshape(-1, 3), inner.reshape(-1, 3)])
    na, m = n_ang, k * n_ang
    idx = lambda i, j: i * na + j
    faces = []
    for i in range(k - 1):
        for j in range(na - 1):
            faces.append([idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)])            # outer
            faces.append([m + idx(i, j), m + idx(i, j + 1), m + idx(i + 1, j + 1), m + idx(i + 1, j)])  # inner
    for i in range(k - 1):                                                                       # angular sides
        faces.append([idx(i, 0), m + idx(i, 0), m + idx(i + 1, 0), idx(i + 1, 0)])
        faces.append([idx(i, na - 1), idx(i + 1, na - 1), m + idx(i + 1, na - 1), m + idx(i, na - 1)])
    for j in range(na - 1):                                                                      # end caps
        faces.append([idx(0, j), idx(0, j + 1), m + idx(0, j + 1), m + idx(0, j)])
        faces.append([idx(k - 1, j), m + idx(k - 1, j), m + idx(k - 1, j + 1), idx(k - 1, j + 1)])
    f = np.ascontiguousarray(np.array(faces, dtype=np.int32)[:, ::-1])  # Outward winding (positive volume)
    zeros = np.zeros((len(verts), 3), np.float32)
    return MeshData(verts.astype(np.float32), f.ravel(), np.arange(len(f), dtype=np.int32) * 4,
                    np.full(len(f), 4, dtype=np.int32), np.zeros((f.size, 2), np.float32),
                    {"branch_order": np.full(len(verts), parent.order, dtype=np.int32),
                     "bark_base": zeros, "bark_along": zeros,
                     "bark_radius": np.zeros(len(verts), np.float32)})


def junction_sleeves(thin: list[Axis], graphs: list[BranchingGraph], r_min: float, tile: float,
                     detail: float = 4.0, max_sleeves: int = 400, tuck: float = 0.8,
                     fused_threshold: float = np.inf, fused_voxel: float = 0.0
                     ) -> tuple[list[dict], list[Axis]]:
    """
    Local fusion "sleeves" for thin axes (hero quality).

    A sleeve unites the base of a child axis (through its collar, to about 2.5 pipe radii) with a
    short segment of its parent around the insertion. The parent segment is shrunk by half a voxel
    so it hides just inside the parent's own surface; fusing the sleeve therefore adds only the
    fillet around the insertion. The child tube restarts inside the sleeve's tucked end.

    Children are ranked by pipe radius (girth after the collar); those >= r_min, up to
    `max_sleeves`, get sleeves. Thick parents are represented only by a curved surface slab around
    the insertion (`sector_slab`), keeping cost proportional to the child's size.
    Returns ([{voxel, tubes, slab, uv_axes, center}, ...], updated_thin_list).
    """
    candidates = []
    for idx, a in enumerate(thin):
        src = getattr(a, "src", None)
        if src is None or src[2] != 0 or a.parent_axis < 0 or len(a.radii) < 4:
            continue
        s = a.arc_length
        r_pipe = float(np.interp(3.0 * a.radii[0], s, a.radii))
        if r_pipe >= r_min:
            candidates.append((r_pipe, idx))
    candidates.sort(reverse=True)
    chosen = {idx: r for r, idx in candidates[:max(0, int(max_sleeves))]}

    def resample(ax: Axis, s0: float, s1: float, n: int):
        sa = ax.arc_length
        q = np.linspace(max(0.0, s0), min(sa[-1], s1), n)
        pos = np.stack([np.interp(q, sa, ax.positions[:, k]) for k in range(3)], axis=1)
        rad = np.interp(q, sa, ax.radii)
        asp = None if ax.aspect is None else np.interp(q, sa, ax.aspect)
        return pos, rad, asp, q

    sleeves, out = [], []
    for idx, a in enumerate(thin):
        if idx not in chosen:
            out.append(a)
            continue
        r_pipe = chosen[idx]
        voxel = max(0.0015, r_pipe / max(2.0, detail))
        gi, ai, _ = a.src
        parent = graphs[gi].axes[a.parent_axis]
        s = a.arc_length
        r0 = float(a.radii[0])
        # Child base through its collar (radius back to ~1.1 x pipe radius), 2-4 pipe radii long
        past = np.nonzero(a.radii <= 1.1 * r_pipe)[0]
        L = float(np.clip(s[past[0]] if len(past) else 3.0 * r_pipe, 2.0 * r_pipe, 4.0 * r_pipe))
        L = min(L, 0.9 * s[-1])
        (pos, rad, asp), (tp, tr, ta), s_start = handover(a, 0.55 * L, overlap=0.45 * L)
        base = _with(a, pos, rad, asp, frame_n0=a.frame_n0, u_rep=a.u_rep, v_offset=a.v_offset)
        # Parent context around the insertion, half a voxel inside the parent's surface
        ps = parent.arc_length
        c = ps[a.parent_sample]
        half = 2.0 * r_pipe + 0.5 * r0
        pT0 = axis_tangents(parent.positions)[0]
        p_n0 = parent.frame_n0 if parent.frame_n0 is not None else default_n0(pT0)[0]
        p_urep = parent.u_rep if parent.u_rep is not None else float(u_repeats(parent.radii[0], tile))
        cpos, crad, casp, cq = resample(parent, c - half, c + half, 5)
        # Stay inside the parent's rendered surface: half a sleeve voxel, plus the shrinkage of
        # voxelising and smoothing when the parent itself belongs to the fused stem/limb surface
        inset = 0.5 * voxel + 0.02 * float(parent.radii[a.parent_sample])
        if parent.radii[a.parent_sample] >= fused_threshold:
            inset += 1.0 * fused_voxel
        # Around the insertion the context rises just above the parent's rendered surface (so the fillet
        # shows), sinking below it toward the ends (so it vanishes tangentially)
        outset = 0.3 * voxel - (inset - 0.5 * voxel - 0.02 * float(parent.radii[a.parent_sample]))
        prof = np.array([-inset, 0.5 * (outset - inset), outset, 0.5 * (outset - inset), -inset])
        ctx = _with(parent, cpos, np.maximum(crad + prof, 0.5 * voxel), casp, frame_n0=p_n0,
                    u_rep=p_urep, v_offset=float(cq[0] + (parent.v_offset or 0.0)))
        r_par = float(parent.radii[a.parent_sample])
        half_angle = (r0 + 1.5 * r_pipe) / max(1e-6, r_par)
        record = {"voxel": voxel, "tubes": [base], "slab": None, "uv_axes": [base, ctx],
                  "center": a.positions[0].copy()}
        # The parent is represented by a curved surface sector around the insertion (also for thin
        # parents: a full ring would rise all around the parent instead of only by the child)
        half_angle = min(half_angle, 0.6 * math.pi)
        ref = _with(parent, cpos, crad, casp, frame_n0=p_n0)
        T = axis_tangents(cpos)
        Np, Bp = axis_frames(T[None], np.asarray(p_n0)[None], parent.frame_up)
        q = base.positions[min(len(base.positions) - 1, int(np.searchsorted(base.arc_length, 0.5 * L)))] - cpos[2]
        sa = np.sqrt(casp[2]) if casp is not None else 1.0
        az = math.atan2(float(q @ Bp[0, 2]) * sa, float(q @ Np[0, 2]) / sa)
        record["slab"] = sector_slab(ref, 0, len(cpos), az, half_angle, inset,
                                     thickness=max(2.5 * r_pipe, 5.0 * voxel), center_outset=outset)
        sleeves.append(record)
        out.append(_with(a, tp, tr, ta, frame_n0=a.frame_n0, u_rep=a.u_rep,
                         v_offset=float(s_start + (a.v_offset or 0.0)), is_trunk=False, src=(gi, ai, 1)))
    return sleeves, out


def graph_of(axes: list[Axis]) -> BranchingGraph:
    """Wraps a list of (split) axes in a graph for meshing; parent links refer to the original graphs."""
    g = BranchingGraph()
    g.axes = list(axes)
    return g


def sample_table(axes: list[Axis]) -> dict:
    """Per-sample frames and bark coordinates of the thick axes (for nearest-sample lookups)."""
    pos, T_all, N_all, B_all, s_all, urep, aid, rad, org = [], [], [], [], [], [], [], [], []
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
        o = a.bark_origin if a.bark_origin is not None else a.positions[0]
        org.append(np.repeat(np.asarray(o)[None, :], len(a.radii), axis=0))
    cat = lambda xs: np.concatenate(xs) if xs else np.zeros((0, 3))
    return {"pos": cat(pos), "T": cat(T_all), "N": cat(N_all), "B": cat(B_all), "s": cat(s_all),
            "u_rep": cat(urep), "axis": cat(aid).astype(int), "r": cat(rad), "origin": cat(org)}


def bark_coordinates(verts: np.ndarray, nearest: np.ndarray, table: dict) -> tuple[np.ndarray, np.ndarray]:
    """(bark_base, bark_along) for points of a fused surface, from their nearest axis sample."""
    c = table["pos"][nearest]
    t = table["T"][nearest]
    d = verts - c
    along = np.sum(d * t, axis=1, keepdims=True)
    centre = c + along * t
    return ((table["origin"][nearest] + (verts - centre)).astype(np.float32),
            (centre - table["origin"][nearest]).astype(np.float32))


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


def grouped_nearest(points: np.ndarray, centers: np.ndarray, groups: list[dict], chunk: int = 8192) -> np.ndarray:
    """
    Nearest-surface sample for points of many small fused pieces: each point is assigned to the
    group (sleeve) with the closest centre, then to that group's sample minimising |p - c| - r.
    `groups[g]` holds 'pos', 'r' and 'offset' (index of its first sample in the global table).
    """
    owner = np.empty(len(points), dtype=int)
    for i in range(0, len(points), chunk):
        d = np.sum((points[i:i + chunk, None, :] - centers[None, :, :]) ** 2, axis=-1)
        owner[i:i + chunk] = np.argmin(d, axis=1)
    out = np.empty(len(points), dtype=int)
    for g, grp in enumerate(groups):
        sel = np.nonzero(owner == g)[0]
        if len(sel) == 0:
            continue
        d = np.sqrt(np.sum((points[sel, None, :] - grp["pos"][None, :, :]) ** 2, axis=-1)) - grp["r"][None, :]
        out[sel] = grp["offset"] + np.argmin(d, axis=1)
    return out
