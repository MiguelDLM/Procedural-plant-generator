"""
Fused wood junctions in Blender.

Quality levels:
- TUBES: plain intersecting tubes (fastest; forests, distant trees).
- FUSED: the thick parts of stem, leaders, limb bases and main roots are meshed
  as watertight tubes, united with a voxel remesh (OpenVDB level set) and
  smoothed with a volume-preserving Laplacian filter, so crotches and root
  flares become continuous fillets. Thin axes stay tubes.
- HERO: FUSED plus a local fusion "sleeve" at every thin-branch insertion whose
  base radius exceeds a limit (core.junctions.junction_sleeves). Each sleeve is
  remeshed with a voxel proportional to the child's radius; sleeves of similar
  size are processed in one batch, so cost grows with the number of junctions,
  not with the volume of the crown.

Bark UVs are rebuilt on every fused surface from the skeleton frames
(core.junctions.cylindrical_uvs) so bark continues into the tubes.
"""

import hashlib
import math
import numpy as np

try:
    import bpy
    from mathutils.kdtree import KDTree
    BLENDER_AVAILABLE = True
except ImportError:
    BLENDER_AVAILABLE = False

try:
    from ..core.junctions import (split_axes, graph_of, sample_table, cylindrical_uvs, junction_sleeves,
                                  grouped_nearest, bark_coordinates)
    from ..core.mesh_engine import MeshData
except (ImportError, ValueError):
    from core.junctions import (split_axes, graph_of, sample_table, cylindrical_uvs, junction_sleeves,
                                grouped_nearest, bark_coordinates)
    from core.mesh_engine import MeshData


def _remesh(mesh_data: MeshData, voxel: float, smooth_iterations: int) -> tuple:
    from .mesh_builder import populate_mesh
    me = bpy.data.meshes.new("PPG_FuseTmp")
    populate_mesh(me, mesh_data, smooth=False)
    obj = bpy.data.objects.new("PPG_FuseTmp", me)
    bpy.context.scene.collection.objects.link(obj)
    try:
        rem = obj.modifiers.new("Remesh", 'REMESH')
        rem.mode = 'VOXEL'
        rem.voxel_size = voxel
        rem.adaptivity = 0.0
        if smooth_iterations > 0:
            sm = obj.modifiers.new("Smooth", 'LAPLACIANSMOOTH')
            sm.iterations = smooth_iterations
            sm.lambda_factor = 0.6
            sm.lambda_border = 0.0
            sm.use_volume_preserve = True
            sm.use_normalized = True
        depsgraph = bpy.context.evaluated_depsgraph_get()
        depsgraph.update()
        ev = obj.evaluated_get(depsgraph)
        em = ev.to_mesh()
        nv, nl, nf = len(em.vertices), len(em.loops), len(em.polygons)
        co = np.empty(nv * 3, dtype=np.float32)
        em.vertices.foreach_get("co", co)
        lv = np.empty(nl, dtype=np.int32)
        em.loops.foreach_get("vertex_index", lv)
        ls = np.empty(nf, dtype=np.int32)
        em.polygons.foreach_get("loop_start", ls)
        lt = np.empty(nf, dtype=np.int32)
        em.polygons.foreach_get("loop_total", lt)
        ev.to_mesh_clear()
    finally:
        bpy.data.objects.remove(obj)
        bpy.data.meshes.remove(me)
    return co.reshape(-1, 3), lv, ls, lt


def _fuse(mesh_engine, axes, voxel: float, smooth_iterations: int, gielis, total_height_m: float,
          extra=None, uv_axes=None, groups=None) -> MeshData:
    """
    Watertight tubes of `axes` (+ `extra` closed meshes) -> voxel union + fillets -> bark UVs from the
    frames of `uv_axes`. With `groups` (sleeves), nearest samples are found per sleeve in NumPy.
    """
    cfg = mesh_engine.config
    cfg.start_caps, cfg.smooth_caps = True, True
    try:
        tubes = mesh_engine.build_wood_mesh(graph_of(axes), total_height_m, gielis, trunk_index=-1)
    finally:
        cfg.start_caps = False
    if extra:
        tubes = MeshData.concatenate([tubes] + list(extra))
    verts, lv, ls, lt = _remesh(tubes, voxel, smooth_iterations)
    if len(verts) == 0:
        return MeshData.empty()
    uv_axes = uv_axes or axes
    table = sample_table(uv_axes)
    if groups is not None:
        nearest = grouped_nearest(verts.astype(float), np.array([g["center"] for g in groups]), groups)
        uvs = cylindrical_uvs(verts, lv, ls, lt, nearest, table, cfg.bark_tile_m)
        orders = np.array([a.order for a in uv_axes])[table["axis"][nearest]].astype(np.int32)
        base, along = bark_coordinates(verts.astype(float), nearest, table)
        return MeshData(verts.astype(np.float32), lv, ls, lt, uvs,
                        {"branch_order": orders, "bark_base": base, "bark_along": along,
                         "bark_radius": table["r"][nearest].astype(np.float32)})
    tree = KDTree(len(table["pos"]))
    for i, p in enumerate(table["pos"]):
        tree.insert(p, i)
    tree.balance()
    r = table["r"]

    def best(v):
        # Nearest tube surface (centre distance minus radius) among the closest axis samples
        hits = tree.find_n(v, 8)
        return min(hits, key=lambda h: h[2] - r[h[1]])[1]

    nearest = np.fromiter((best(v) for v in verts), dtype=np.int64, count=len(verts))
    uvs = cylindrical_uvs(verts, lv, ls, lt, nearest, table, cfg.bark_tile_m)
    orders = np.array([a.order for a in axes])[table["axis"][nearest]].astype(np.int32)
    base, along = bark_coordinates(verts.astype(float), nearest, table)
    return MeshData(verts.astype(np.float32), lv, ls, lt, uvs,
                    {"branch_order": orders, "bark_base": base, "bark_along": along,
                     "bark_radius": table["r"][nearest].astype(np.float32)})


_CACHE: dict = {}


def _key(axes_groups, *extra) -> str:
    h = hashlib.sha1()
    for axes in axes_groups:
        for a in axes:
            h.update(a.positions.astype(np.float32).tobytes())
            h.update(a.radii.astype(np.float32).tobytes())
            if a.aspect is not None:
                h.update(a.aspect.astype(np.float32).tobytes())
    h.update(repr(extra).encode())
    return h.hexdigest()


def build_fused_wood(mesh_engine, graphs, total_height_m: float, gielis, detail: float = 10.0,
                     smooth_iterations: int = 6, quality: str = "FUSED", hero_min_radius: float = 0.025,
                     sleeve_detail: float = 4.0, max_sleeves: int = 400) -> MeshData:
    """Wood mesh with fused junctions; `quality` is 'FUSED' or 'HERO' (adds fine-branch sleeves)."""
    cfg = mesh_engine.config
    trunk = graphs[0].axes[0]
    r_base = float(np.interp(0.0, trunk.positions[:, 2], trunk.radii))
    voxel = max(0.008, r_base / max(2.0, detail))
    # Bound the level-set resolution by the extent of the fused region (small trees, long roots)
    thick, _ = split_axes(graphs, 2.5 * voxel, cfg.bark_tile_m)
    if thick:
        pts = np.vstack([a.positions for a in thick])
        voxel = max(voxel, float(np.max(pts.max(0) - pts.min(0))) / 320.0)
    thick, thin = split_axes(graphs, 2.5 * voxel, cfg.bark_tile_m)

    sleeves = []
    if quality == "HERO":
        sleeves, thin = junction_sleeves(thin, graphs, max(hero_min_radius, 1e-3), cfg.bark_tile_m,
                                         detail=sleeve_detail, max_sleeves=max_sleeves,
                                         fused_threshold=2.5 * voxel, fused_voxel=voxel)

    settings = (round(voxel, 6), smooth_iterations, gielis, cfg.radial_resolution, cfg.twig_resolution,
                cfg.flute_azimuth, quality, round(hero_min_radius, 5), sleeve_detail, max_sleeves)
    key = _key([thick] + [sl["uv_axes"] for sl in sleeves], settings)
    parts = []
    if key in _CACHE:
        parts.append(_CACHE[key])
    else:
        fused = []
        if thick:
            fused.append(_tag(_fuse(mesh_engine, thick, voxel, smooth_iterations, gielis, total_height_m), 1))
        # Sleeves batched by voxel size class (powers of sqrt(2)) so each batch is one remesh
        # Batch sleeves by voxel class AND by 1 m spatial cell: one level set spanning a whole crown
        # with millimetre voxels leaks between the many disjoint pieces, local batches do not
        bins: dict[tuple, list] = {}
        for sl in sleeves:
            cell = tuple(np.floor(sl["center"] / 1.0).astype(int))
            bins.setdefault((int(math.floor(math.log(sl["voxel"], math.sqrt(2.0)))),) + cell, []).append(sl)
        for (b, *_), group in sorted(bins.items()):
            v = math.sqrt(2.0) ** b
            tubes, extra, uv_axes, groups = [], [], [], []
            for sl in group:
                tubes += sl["tubes"]
                if sl["slab"] is not None:
                    extra.append(sl["slab"])
                pos = np.vstack([a.positions for a in sl["uv_axes"]])
                rad = np.concatenate([a.radii * (np.sqrt(a.aspect) if a.aspect is not None else 1.0)
                                      for a in sl["uv_axes"]])
                groups.append({"center": sl["center"], "pos": pos, "r": rad,
                               "offset": sum(len(a.radii) for a in uv_axes)})
                uv_axes += sl["uv_axes"]
            fused.append(_tag(_fuse(mesh_engine, tubes, v, max(4, smooth_iterations + 2), gielis, total_height_m,
                                    extra=extra, uv_axes=uv_axes, groups=groups), 2))
        merged = MeshData.concatenate(fused)
        _CACHE.clear()  # Keep only the latest fused surface
        _CACHE[key] = merged
        parts.append(merged)

    if thin:
        cfg.start_caps = True   # No open tube mouths at fused/thin cuts
        try:
            parts.append(_tag(mesh_engine.build_wood_mesh(graph_of(thin), total_height_m, gielis, trunk_index=-1), 0))
        finally:
            cfg.start_caps = False
    return MeshData.concatenate(parts)


def _tag(mesh: MeshData, stage: int) -> MeshData:
    """Per-vertex 'fuse_stage' (0 tube, 1 fused stem/limbs/roots, 2 junction sleeve), handy for inspection."""
    mesh.point_attributes["fuse_stage"] = np.full(len(mesh.vertices), stage, dtype=np.int32)
    return mesh
