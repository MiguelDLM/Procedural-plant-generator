"""
Nearest-neighbour queries on a uniform grid (pure NumPy; no SciPy in Blender's Python).
"""

import numpy as np


def nearest_points(query: np.ndarray, points: np.ndarray, cell: float, chunk: int = 65536) -> tuple[np.ndarray, np.ndarray]:
    """
    For each query point, the distance to and index of the nearest of `points`, searching the 3x3x3
    grid cells around it (cell >= typical spacing of `points`). Points farther than ~1 cell away may be
    missed; their distance is returned as +inf and index -1.
    """
    Q = np.asarray(query, np.float64)
    P = np.asarray(points, np.float64)
    if len(P) == 0 or len(Q) == 0:
        return np.full(len(Q), np.inf), np.full(len(Q), -1, np.int64)
    cell = max(float(cell), 1e-6)
    lo = P.min(0) - cell
    ci = np.floor((P - lo) / cell).astype(np.int64)
    dims = ci.max(0) + 3
    key = (ci[:, 0] * dims[1] + ci[:, 1]) * dims[2] + ci[:, 2]
    order = np.argsort(key, kind="stable")
    ks = key[order]
    uk, start, count = np.unique(ks, return_index=True, return_counts=True)
    per_cell = int(min(256, count.max()))      # Every point of the densest cell is kept
    # Table of up to `per_cell` point indices per occupied cell
    table = np.full((len(uk), per_cell), -1, np.int64)
    rank = np.arange(len(ks)) - np.repeat(start, count)
    ok = rank < per_cell
    table[np.repeat(np.arange(len(uk)), count)[ok], rank[ok]] = order[ok]

    best_d = np.full(len(Q), np.inf)
    best_i = np.full(len(Q), -1, np.int64)
    offs = np.array([(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)])
    for s in range(0, len(Q), chunk):
        q = Q[s:s + chunk]
        qc = np.floor((q - lo) / cell).astype(np.int64)
        bd = np.full(len(q), np.inf)
        bi = np.full(len(q), -1, np.int64)
        for o in offs:
            c = qc + o
            inside = np.all((c >= 0) & (c < dims), axis=1)
            k = (c[:, 0] * dims[1] + c[:, 1]) * dims[2] + c[:, 2]
            pos = np.searchsorted(uk, k)
            pos = np.clip(pos, 0, len(uk) - 1)
            hit = inside & (uk[pos] == k)
            if not np.any(hit):
                continue
            cand = table[pos[hit]]                                    # (h, per_cell)
            valid = cand >= 0
            d = np.linalg.norm(P[np.where(valid, cand, 0)] - q[hit][:, None, :], axis=-1)
            d = np.where(valid, d, np.inf)
            j = np.argmin(d, axis=1)
            dm = d[np.arange(len(j)), j]
            better = dm < bd[hit]
            idx = np.flatnonzero(hit)[better]
            bd[idx] = dm[better]
            bi[idx] = cand[np.arange(len(j)), j][better]
        best_d[s:s + chunk] = bd
        best_i[s:s + chunk] = bi
    return best_d, best_i
