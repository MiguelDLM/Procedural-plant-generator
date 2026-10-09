"""
Geometry checks shared by the tests: mesh islands (connected components) and the gap between each island
and the surface of the others, to catch parts that float apart (stalks, bunches, stems, roots).

Vectorised with NumPy: connected components by label propagation with pointer jumping, surface samples
drawn for all triangles at once, and the gap test on a uniform spatial hash (cells of the tolerance size), so
only an island that is not touching anything needs an exact distance.
"""

import numpy as np


def _edges(m):
    lv = m.loop_vertex.astype(np.int64)
    starts = m.loop_start.astype(np.int64)
    totals = m.loop_total.astype(np.int64)
    if len(totals) == 0:
        return np.zeros((0, 2), np.int64)
    first = np.repeat(lv[starts], totals)
    return np.stack([first, lv], 1)


def islands(m) -> np.ndarray:
    """Connected-component label of every vertex (vertices sharing a face are connected)."""
    n = len(m.vertices)
    lab = np.arange(n)
    e = _edges(m)
    if len(e) == 0:
        return lab
    a, b = e[:, 0], e[:, 1]
    while True:
        mn = np.minimum(lab[a], lab[b])
        new = lab.copy()
        np.minimum.at(new, a, mn)
        np.minimum.at(new, b, mn)
        new = new[new]                       # Pointer jumping
        new = new[new]
        if np.array_equal(new, lab):
            return lab
        lab = new


def _triangles(m):
    lv = m.loop_vertex.astype(np.int64)
    out = []
    for t in np.unique(m.loop_total):
        sel = np.nonzero(m.loop_total == t)[0]
        base = m.loop_start[sel].astype(np.int64)
        for j in range(1, t - 1):
            out.append(np.stack([lv[base], lv[base + j], lv[base + j + 1]], 1))
    return np.concatenate(out) if out else np.zeros((0, 3), np.int64)


def _surface_samples(m, mask_v, k=8, seed=0):
    rng = np.random.default_rng(seed)
    V = m.vertices.astype(float)
    T = _triangles(m)
    T = T[mask_v[T].all(1)]
    if len(T) == 0:
        return V[mask_v]
    w = rng.dirichlet([1, 1, 1], (len(T), k))                      # (t, k, 3)
    return np.einsum("tkc,tcd->tkd", w, V[T]).reshape(-1, 3)


def _cell_keys(P, cell):
    q = np.floor(P / cell).astype(np.int64)
    return (q[:, 0] * 73856093) ^ (q[:, 1] * 19349663) ^ (q[:, 2] * 83492791), q


def _touches(a, b, tol):
    """Whether any point of a lies within tol of any point of b (spatial hash with cells of size tol)."""
    if len(a) == 0 or len(b) == 0:
        return False
    kb, qb = _cell_keys(b, tol)
    order = np.argsort(kb, kind="stable")
    kb_s = kb[order]
    _, qa = _cell_keys(a, tol)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                q = qa + np.array([dx, dy, dz])
                key = (q[:, 0] * 73856093) ^ (q[:, 1] * 19349663) ^ (q[:, 2] * 83492791)
                lo = np.searchsorted(kb_s, key, "left")
                hi = np.searchsorted(kb_s, key, "right")
                hit = np.nonzero(hi > lo)[0]
                if len(hit) == 0:
                    continue
                # Candidate pairs (a point, b points of its cell)
                counts = hi[hit] - lo[hit]
                ai = np.repeat(hit, counts)
                offs = np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts)
                bi = order[np.repeat(lo[hit], counts) + offs]
                if (((a[ai] - b[bi]) ** 2).sum(1) <= tol * tol).any():
                    return True
    return False


def floating_islands(m, tol: float) -> list:
    """(vertex count, gap in m) of every island farther than `tol` from the surface of all the others."""
    if len(m.vertices) == 0:
        return []
    lab = islands(m)
    ids = np.unique(lab)
    if len(ids) == 1:
        return []
    V = m.vertices.astype(float)
    S = _surface_samples(m, np.ones(len(V), bool))
    T = _triangles(m)
    s_lab = np.repeat(lab[T[:, 0]], 8) if len(T) else lab
    bad = []
    for i in ids:
        a = V[lab == i]
        b = S[s_lab != i]
        lo, hi = a.min(0) - tol, a.max(0) + tol
        b = b[np.all((b >= lo) & (b <= hi), axis=1)]
        if _touches(a, b, tol):
            continue
        g = 1.0
        bb = S[s_lab != i]
        bb = bb[::max(1, len(bb) // 40000)]
        for s0 in range(0, len(a), 256):
            if len(bb):
                g = min(g, float(np.sqrt(((a[s0:s0 + 256, None, :] - bb[None]) ** 2).sum(-1)).min()))
        bad.append((int((lab == i).sum()), round(g, 4)))
    return bad
