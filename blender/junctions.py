"""
Fused wood junctions in Blender.

The thick parts of stem, leaders, limbs and main roots are meshed as tubes,
united into one closed surface with a voxel remesh (OpenVDB level set), and
smoothed with a volume-preserving Laplacian filter so crotches and root flares
become continuous fillets. Bark UVs are then rebuilt from the skeleton frames
(core.junctions.cylindrical_uvs) so they continue into the thin tubes.
"""

import hashlib
import numpy as np

try:
    import bpy
    from mathutils.kdtree import KDTree
    BLENDER_AVAILABLE = True
except ImportError:
    BLENDER_AVAILABLE = False

try:
    from ..core.junctions import split_axes, graph_of, sample_table, cylindrical_uvs
    from ..core.mesh_engine import MeshData
except (ImportError, ValueError):
    from core.junctions import split_axes, graph_of, sample_table, cylindrical_uvs
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


_CACHE: dict = {}


def _fused_key(thick, voxel, smooth_iterations, gielis, cfg) -> str:
    h = hashlib.sha1()
    for a in thick:
        h.update(a.positions.astype(np.float32).tobytes())
        h.update(a.radii.astype(np.float32).tobytes())
        if a.aspect is not None:
            h.update(a.aspect.astype(np.float32).tobytes())
    h.update(repr((round(voxel, 6), smooth_iterations, gielis, cfg.radial_resolution, cfg.flute_azimuth,
                   cfg.flare_amplitude, cfg.flare_decay)).encode())
    return h.hexdigest()


def build_fused_wood(mesh_engine, graphs, total_height_m: float, gielis, detail: float = 10.0,
                     smooth_iterations: int = 6) -> MeshData:
    """Wood mesh with fused thick junctions (voxel union + smoothing) and tube twigs."""
    cfg = mesh_engine.config
    trunk = graphs[0].axes[0]
    r_base = float(np.interp(0.0, trunk.positions[:, 2], trunk.radii))
    voxel = max(0.008, r_base / max(2.0, detail))
    # Bound the level-set resolution by the extent of the fused region (small trees, long roots)
    thick, _ = split_axes(graphs, 2.5 * voxel, cfg.bark_tile_m)
    if thick:
        pts = np.vstack([a.positions for a in thick])
        voxel = max(voxel, float(np.max(pts.max(0) - pts.min(0))) / 320.0)
    threshold = 2.5 * voxel
    thick, thin = split_axes(graphs, threshold, cfg.bark_tile_m)

    parts = []
    key = _fused_key(thick, voxel, smooth_iterations, gielis, cfg) if thick else None
    if thick and key in _CACHE:
        parts.append(_CACHE[key])
        thick = []
    if thick:
        cfg.start_caps, cfg.smooth_caps = True, True   # Watertight input for the level set
        try:
            tubes = mesh_engine.build_wood_mesh(graph_of(thick), total_height_m, gielis, trunk_index=-1)
        finally:
            cfg.start_caps = False
        verts, lv, ls, lt = _remesh(tubes, voxel, smooth_iterations)
        table = sample_table(thick)
        tree = KDTree(len(table["pos"]))
        for i, p in enumerate(table["pos"]):
            tree.insert(p, i)
        tree.balance()
        # Nearest tube surface (centre distance minus radius) among the closest axis samples
        r = table["r"]

        def best(v):
            hits = tree.find_n(v, 8)
            return min(hits, key=lambda h: h[2] - r[h[1]])[1]
        nearest = np.fromiter((best(v) for v in verts), dtype=np.int64, count=len(verts))
        uvs = cylindrical_uvs(verts, lv, ls, lt, nearest, table, cfg.bark_tile_m)
        orders = np.array([a.order for a in thick])[table["axis"][nearest]].astype(np.int32)
        fused = MeshData(verts.astype(np.float32), lv, ls, lt, uvs, {"branch_order": orders})
        _CACHE.clear()  # Keep only the latest fused surface
        _CACHE[key] = fused
        parts.append(fused)
    if thin:
        cfg.start_caps = True   # No open tube mouths at the fused/thin cut
        try:
            parts.append(mesh_engine.build_wood_mesh(graph_of(thin), total_height_m, gielis, trunk_index=-1))
        finally:
            cfg.start_caps = False
    return MeshData.concatenate(parts)
