"""
Geometry checks shared by the tests: mesh islands (connected components) and the gap between each island
and the surface of the others, to catch parts that float apart (stalks, bunches, stems, roots).
"""

import numpy as np


def islands(m) -> np.ndarray:
    """Connected-component label of every vertex (vertices sharing a face are connected)."""
    n = len(m.vertices)
    parent = np.arange(n)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for s, t in zip(m.loop_start, m.loop_total):
        f = m.loop_vertex[s:s + t]
        for v in f[1:]:
            ra, rb = find(f[0]), find(v)
            if ra != rb:
                parent[ra] = rb
    return np.array([find(i) for i in range(n)])


def _surface_samples(m, mask_v, k=8, seed=0):
    rng = np.random.default_rng(seed)
    V = m.vertices.astype(float)
    out = []
    for s, t in zip(m.loop_start, m.loop_total):
        f = m.loop_vertex[s:s + t]
        if not mask_v[f].all():
            continue
        for j in range(1, t - 1):
            out.append(rng.dirichlet([1, 1, 1], k) @ V[[f[0], f[j], f[j + 1]]])
    return np.concatenate(out) if out else V[mask_v]


def floating_islands(m, tol: float) -> list:
    """(vertex count, gap in m) of every island farther than `tol` from the surface of all the others."""
    if len(m.vertices) == 0:
        return []
    lab = islands(m)
    ids = np.unique(lab)
    if len(ids) == 1:
        return []
    V = m.vertices.astype(float)
    bad = []
    for i in ids:
        a = V[lab == i]
        b = _surface_samples(m, lab != i)
        lo, hi = a.min(0) - 0.05, a.max(0) + 0.05
        b = b[np.all((b >= lo) & (b <= hi), axis=1)]
        b = b[::max(1, len(b) // 40000)]
        g = 1.0
        for s0 in range(0, len(a), 256):
            if len(b):
                g = min(g, float(np.sqrt(((a[s0:s0 + 256, None, :] - b[None]) ** 2).sum(-1)).min()))
        if g > tol:
            bad.append((int((lab == i).sum()), round(g, 4)))
    return bad
