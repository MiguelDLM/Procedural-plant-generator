"""
Blender side of grasses: properties generated from core.grass, one attribute-driven material, a single plant
or a lawn / meadow patch (several tuft variants instanced over an area with Geometry Nodes).
"""

import copy
import hashlib
import math
from dataclasses import fields

import numpy as np

try:
    import bpy
    from bpy.props import BoolProperty, FloatProperty, IntProperty, PointerProperty
    BLENDER_AVAILABLE = True
except ImportError:
    BLENDER_AVAILABLE = False

try:
    from ..core.grass import GrassEngine, GrassProfile, patch_points, surface_points
    from ..core.grass_db import GRASS_CATALOG, GRASS_RANGES
    from ..core.mesh_engine import MeshData
    from .flowers import _props_for, _instancer_nodes, _euler_xyz
    from .mesh_builder import populate_mesh
except (ImportError, ValueError):
    from core.grass import GrassEngine, GrassProfile, patch_points, surface_points
    from core.grass_db import GRASS_CATALOG, GRASS_RANGES
    from core.mesh_engine import MeshData
    from blender.flowers import _props_for, _instancer_nodes, _euler_xyz
    from blender.mesh_builder import populate_mesh

GRA = "gra_"
SUFFIXES = ("_GrassLeaves", "_GrassCulms", "_GrassHeads", "_GrassEars", "_GrassRoots")
LAYOUT = [
    ("Tillers & Culm", ["tillers", "tiller_spread_deg", "tiller_variation", "flowering_tillers", "clump_radius_cm",
                        "culm_height_m", "culm_radius_mm", "nodes", "internode_gradient", "basal_leaves",
                        "culm_color"]),
    ("Nodes & Internodes", ["node_swell", "growth_ring", "internode_barrel", "bud_groove", "bud_size_mm",
                            "zigzag_deg", "wax_band", "root_primordia", "node_color"]),
    ("Leaves", ["leaf_length_cm", "leaf_width_cm", "leaf_peak", "leaf_angle_deg", "leaf_droop", "leaf_twist_deg",
                "leaf_fold", "margin_wave", "sheath_fraction", "leaf_loss", "ligule", "ligule_mm", "auricles",
                "auricle_mm", "leaf_color", "midrib_color",
                "tip_dryness"]),
    ("Inflorescence", ["head", "head_length_cm", "head_width_cm", "peduncle_cm", "spikelets", "spikelet_mm",
                       "awn_cm", "branches", "branch_angle_deg", "nod", "head_color", "awn_color"]),
    ("Maize Ears", ["ears", "ear_node", "ear_length_cm", "ear_diameter_cm", "kernel_rows", "husk", "silk_cm",
                    "kernel_color", "silk_color"]),
    ("Roots & Ripeness", ["crown_roots", "root_length_cm", "brace_roots", "fine_roots", "ripeness"]),
]


def grass_properties(update) -> dict:
    if not BLENDER_AVAILABLE:
        return {}
    out = _props_for(GrassProfile, GRA, GRASS_RANGES, update)
    out.update({
        "grass_patch": BoolProperty(name="Lawn / Meadow Patch", default=False, update=update,
                                    description="Scatter several tuft variants over an area instead of one plant"),
        "grass_patch_size": FloatProperty(name="Patch Size (m)", default=2.0, min=0.2, max=50.0, update=update),
        "grass_patch_density": FloatProperty(name="Tufts per m²", default=110.0, min=0.5, max=2000.0, update=update),
        "grass_patch_variants": IntProperty(name="Variants", default=4, min=1, max=8, update=update),
        "grass_patch_spacing": FloatProperty(name="Min Spacing (cm)", default=3.0, min=0.0, max=200.0, update=update),
        "grass_patch_scale_var": FloatProperty(name="Size Variation", default=0.25, min=0.0, max=0.9, update=update),
        "grass_patch_surface": PointerProperty(
            name="Terrain", type=bpy.types.Object, update=update, poll=lambda self, o: o.type == 'MESH',
            description="Mesh to grow the patch on (empty = flat square of Patch Size)"),
        "grass_patch_align": FloatProperty(name="Follow Slope", default=0.15, min=0.0, max=1.0, update=update,
                                           description="0 tufts stay vertical (gravitropism) .. 1 perpendicular "
                                                       "to the terrain"),
        "grass_patch_max_slope": FloatProperty(name="Max Slope", default=math.radians(60), min=0.0,
                                               max=math.radians(90), subtype='ANGLE', update=update,
                                               description="No tufts on steeper faces (rock, cliffs)"),
    })
    return out


def write_grass_to_props(props, key):
    sp = GRASS_CATALOG.get(key)
    if sp is None:
        return
    for f in fields(sp.profile):
        name = GRA + f.name
        if hasattr(props, name):
            v = getattr(sp.profile, f.name)
            setattr(props, name, v.value if hasattr(v, "value") else v)


def grass_from_props(props) -> GrassProfile:
    obj = copy.deepcopy(GRASS_CATALOG[props.grass_species].profile)
    for f in fields(obj):
        name = GRA + f.name
        if not hasattr(props, name):
            continue
        cur, val = getattr(obj, f.name), getattr(props, name)
        if hasattr(cur, "value"):
            val = type(cur)(val)
        elif isinstance(cur, tuple):
            val = tuple(float(x) for x in val)
        elif isinstance(cur, bool):
            val = bool(val)
        elif isinstance(cur, int):
            val = int(val)
        else:
            val = float(val)
        setattr(obj, f.name, val)
    return obj


def current_grass_preset(props):
    sp = copy.deepcopy(GRASS_CATALOG[props.grass_species])
    sp.profile = grass_from_props(props)
    return sp


def grass_values(props) -> dict:
    return {f.name: getattr(props, GRA + f.name) for f in fields(GrassProfile) if hasattr(props, GRA + f.name)}


# -----------------------------------------------------------------------------
# Material
# -----------------------------------------------------------------------------
def _key(*parts) -> str:
    return hashlib.sha1(repr(parts).encode()).hexdigest()[:12]


STRAW = (0.80, 0.68, 0.38)


def grass_material(name, p: GrassProfile):
    """One material for every organ, chosen by the 'grass_part' attribute: blades with a pale midrib, fine
    parallel veins and dry tips; culms and sheaths; spikelets; awns and hairs; kernels; husk; silks; roots.
    Ripeness turns the green organs to straw-gold."""
    from .succulents import _new_mat, _attr, _mix, _noise, _math
    from .materials import _srgb_to_linear, _set
    mat, nt, b = _new_mat(name)

    def rgb(c, loc):
        n = nt.nodes.new('ShaderNodeRGB')
        n.location = loc
        n.outputs[0].default_value = _srgb_to_linear(c)
        return n.outputs[0]

    def mix_c(c):                      # Ripeness: green organs toward straw
        return tuple(x * (1 - p.ripeness) + y * p.ripeness for x, y in zip(c, STRAW))
    part = _attr(nt, "grass_part", (-1800, -400)).outputs['Fac']
    u = _attr(nt, "grass_u", (-1800, 300)).outputs['Fac']
    v = _attr(nt, "grass_v", (-1800, 100)).outputs['Fac']

    def is_part(k, loc):
        lo = _math(nt, 'GREATER_THAN', part, k - 0.5, loc)
        hi = _math(nt, 'LESS_THAN', part, k + 0.5, (loc[0], loc[1] - 60))
        return _math(nt, 'MULTIPLY', lo, hi, (loc[0] + 150, loc[1]))
    # Blade: midrib, veins, dry tip
    col = rgb(mix_c(p.leaf_color), (-1300, 600))
    mid = _math(nt, 'LESS_THAN', _math(nt, 'ABSOLUTE', v, None, (-1550, 150)), 0.12, (-1400, 150))
    col = _mix(nt, _math(nt, 'MULTIPLY', mid, 0.7, (-1250, 150)), col, rgb(mix_c(p.midrib_color), (-1250, 350)),
               (-1050, 500))
    veins = _math(nt, 'COSINE', _math(nt, 'MULTIPLY', v, math.pi * 9, (-1550, 0)), None, (-1400, 0))
    veins = _math(nt, 'MULTIPLY', _math(nt, 'GREATER_THAN', veins, 0.85, (-1250, 0)), 0.18, (-1100, 0))
    col = _mix(nt, veins, col, (0.02, 0.05, 0.01, 1.0), (-900, 450))
    dry = _math(nt, 'SUBTRACT', u, 1.0 - p.tip_dryness, (-1550, 450))
    dry = _math(nt, 'MINIMUM', _math(nt, 'MAXIMUM', _math(nt, 'MULTIPLY', dry, 1.0 / max(p.tip_dryness, 0.02),
                                                             (-1400, 450)), 0.0, (-1250, 450)), 1.0, (-1100, 450))
    col = _mix(nt, _math(nt, 'MULTIPLY', dry, min(1.0, p.tip_dryness * 4), (-950, 600)), col, rgb(STRAW, (-950, 750)),
               (-750, 450))
    # Other organs
    for k, c, loc in ((1, mix_c(p.culm_color), -600), (2, mix_c(p.head_color), -450), (3, p.awn_color, -300),
                      (4, p.kernel_color, -150), (5, mix_c((0.55, 0.68, 0.40)), 0), (6, p.silk_color, 150),
                      (7, (0.85, 0.80, 0.68), 300), (8, (0.88, 0.90, 0.78), 450), (9, mix_c(p.culm_color), 600)):
        col = _mix(nt, is_part(k, (loc - 500, -700 - k * 40)), col, rgb(c, (loc, 650)), (loc, 400 - k * 20))
    # Culm nodes (grass_node: signed distance to the nearest node in culm radii; far elsewhere)
    nd = _attr(nt, "grass_node", (-1800, -1300)).outputs['Fac']

    from .succulents import _smooth_s

    def band(lo, hi, w, amount, loc):
        """Soft band lo..hi of grass_node (edges `w` wide), times `amount`."""
        rise = _smooth_s(nt, nd, lo - w, lo + w, loc)
        fall = _math(nt, 'SUBTRACT', 1.0, _smooth_s(nt, nd, hi - w, hi + w, (loc[0], loc[1] - 80)),
                     (loc[0] + 150, loc[1] - 80))
        m = _math(nt, 'MULTIPLY', rise, fall, (loc[0] + 300, loc[1]))
        return _math(nt, 'MULTIPLY', m, amount, (loc[0] + 450, loc[1]))
    nid = _attr(nt, "grass_nid", (-1800, -1900)).outputs['Fac']
    vary = _math(nt, 'ADD', 0.55, _math(nt, 'MULTIPLY', nid, 0.9, (-1650, -1900)), (-1500, -1900))
    wax = band(-1.8, -0.2, 0.35, 0.45 * p.wax_band, (-1500, -1300))              # Waxy bloom below the node
    wax = _math(nt, 'MULTIPLY', wax, vary, (-1300, -1300))                      # ...varying from node to node
    col = _mix(nt, wax, col, rgb((0.74, 0.76, 0.70), (-1100, -1150)), (450, 250))
    ring = band(-0.15, 1.2, 0.15, 0.45, (-1500, -1500))                           # Node ring and root band
    ring = _math(nt, 'MULTIPLY', ring, _math(nt, 'ADD', 0.7, _math(nt, 'MULTIPLY', nid, 0.6, (-1650, -2050)),
                                             (-1500, -2050)), (-1300, -1500))
    col = _mix(nt, ring, col, rgb(mix_c(p.node_color), (-1100, -1350)), (500, 220))
    gr = band(0.8, 1.0, 0.05, 0.55 * p.growth_ring, (-1500, -1700))               # Growth ring line
    col = _mix(nt, gr, col, (0.22, 0.20, 0.10, 1.0), (550, 200))
    # Root primordia (sugarcane): two staggered rows of dots in the root band, upper row irregular
    # (Artschwager & Brandes 1958); drawn as bump and slight darkening
    if p.root_primordia > 0 and p.growth_ring > 0:
        culm = is_part(1, (-1500, -2300))
        dots = None
        for row, (dr, off) in enumerate(((0.32, 0.0), (0.58, 0.5))):
            band_r = _math(nt, 'EXPONENT', _math(nt, 'MULTIPLY', _math(nt, 'POWER', _math(
                nt, 'SUBTRACT', nd, dr, (-1500, -2450 - 200 * row)), 2.0, (-1350, -2450 - 200 * row)),
                -1.0 / 0.0035, (-1200, -2450 - 200 * row)), None, (-1050, -2450 - 200 * row))
            ang = _math(nt, 'MULTIPLY', _math(nt, 'ADD', v, off / 28.0, (-1500, -2550 - 200 * row)),
                        2 * math.pi * 28, (-1350, -2550 - 200 * row))
            dot = _math(nt, 'POWER', _math(nt, 'ADD', 0.5, _math(nt, 'MULTIPLY', _math(
                nt, 'COSINE', ang, None, (-1200, -2550 - 200 * row)), 0.5, (-1050, -2550 - 200 * row)),
                (-900, -2550 - 200 * row)), 3.0, (-750, -2550 - 200 * row))
            if row == 1:                                   # Upper row incomplete, varying per node
                gate = _math(nt, 'GREATER_THAN', _math(nt, 'SINE', _math(nt, 'ADD', _math(
                    nt, 'MULTIPLY', v, 18.85, (-1500, -2900)), _math(nt, 'MULTIPLY', nid, 17.0, (-1500, -3000)),
                    (-1350, -2950)), None, (-1200, -2950)), -0.1, (-1050, -2950))
                dot = _math(nt, 'MULTIPLY', dot, gate, (-600, -2750))
            d1 = _math(nt, 'MULTIPLY', dot, band_r, (-600, -2450 - 200 * row))
            dots = d1 if dots is None else _math(nt, 'ADD', dots, d1, (-450, -2500))
        dots = _math(nt, 'MULTIPLY', dots, culm, (-300, -2500))
        col = _mix(nt, _math(nt, 'MULTIPLY', dots, 0.35 * p.root_primordia, (-150, -2500)), col,
                   rgb((0.45, 0.40, 0.25), (-150, -2350)), (580, 180))
        bump = nt.nodes.new('ShaderNodeBump')
        bump.location = (300, -2500)
        bump.inputs['Strength'].default_value = 0.6 * p.root_primordia
        bump.inputs['Distance'].default_value = 0.0006
        nt.links.new(dots, bump.inputs['Height'])
        nt.links.new(bump.outputs['Normal'], b.inputs['Normal'])
    # Kernels: per-kernel brightness variation
    tc = nt.nodes.new('ShaderNodeTexCoord')
    tc.location = (-800, -1000)
    nz = _noise(nt, tc.outputs['Object'], 90.0, 2.0, (-600, -1000))
    col = _mix(nt, 0.15, col, nz.outputs['Color'], (500, 300), blend='OVERLAY')
    nt.links.new(col, b.inputs['Base Color'])
    _set(b, 0.55, "Roughness")
    _set(b, 0.12, "Subsurface Weight", "Subsurface")
    return mat


# -----------------------------------------------------------------------------
# Geometry
# -----------------------------------------------------------------------------
def hide_grass_parts(root):
    for c in root.children:
        if c.name.endswith(SUFFIXES) or "_GrassPatch" in c.name or "_GrassProto" in c.name:
            c.hide_viewport = c.hide_render = True


def _terrain_points(context, terrain, root, props):
    """Tuft positions on a terrain mesh, in the plant root's space."""
    dg = context.evaluated_depsgraph_get()
    ev = terrain.evaluated_get(dg)
    me = ev.to_mesh()
    try:
        me.calc_loop_triangles()
        nv = len(me.vertices)
        co = np.empty(nv * 3, np.float32)
        me.vertices.foreach_get("co", co)
        co = co.reshape(-1, 3)
        tri = np.empty(len(me.loop_triangles) * 3, np.int32)
        me.loop_triangles.foreach_get("vertices", tri)
    finally:
        ev.to_mesh_clear()
    M = np.array(root.matrix_world.inverted() @ terrain.matrix_world)
    W = co @ M[:3, :3].T + M[:3, 3]
    pts, nrm = surface_points(W[tri.reshape(-1, 3)], props.grass_patch_density, props.seed,
                              props.grass_patch_spacing * 0.01, math.degrees(props.grass_patch_max_slope))
    return pts, nrm


def _tilts(normals, align):
    """Rotations taking +Z toward the blend of vertical and the terrain normal."""
    out = []
    for nv in normals:
        t = np.array([0.0, 0.0, 1.0]) * (1 - align) + nv * align
        t = t / max(np.linalg.norm(t), 1e-9)
        ax = np.cross([0.0, 0.0, 1.0], t)
        s, c = np.linalg.norm(ax), t[2]
        if s < 1e-8:
            out.append(np.eye(3))
            continue
        k = ax / s
        K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        out.append(np.eye(3) + s * K + (1 - c) * K @ K)
    return np.array(out).reshape(-1, 3, 3)


def update_grass_geometry(context, props, find_root):
    from .succulents import _child
    from .flowers import _child as _child_plain, _assign
    sp = GRASS_CATALOG.get(props.grass_species)
    if sp is None:
        print("[PPG] No valid grass preset selected")
        return None
    root = find_root(context, "PPG_" + sp.scientific_name.split(" (")[0].replace(" ", "_"))
    for c in root.children:
        c.hide_viewport = c.hide_render = True
    prof = grass_from_props(props)
    mat = None
    if props.assign_materials:
        k = _key(prof)
        mat = bpy.data.materials.get(root.name + "_GrassMat")
        if mat is None or mat.get("ppg_key") != k:
            mat = grass_material(root.name + "_GrassMat", prof)
            mat["ppg_key"] = k
    if not props.grass_patch:
        res = GrassEngine(prof).generate(seed=props.seed, detail=props.succ_detail, with_roots=props.show_roots)
        for suffix, m in zip(SUFFIXES, (res.leaves, res.culms, res.heads, res.ears,
                                        res.roots if props.show_roots else None)):
            _child(context, root, suffix, m, mat)
        root["leaves"] = res.stats["leaves"]
    else:
        # Lawn / meadow: tuft variants (own seeds) instanced over the patch
        nvar = max(1, props.grass_patch_variants)
        terrain = getattr(props, "grass_patch_surface", None)
        nrm = None
        if terrain is not None and terrain.type == 'MESH':
            pts, nrm = _terrain_points(context, terrain, root, props)
        else:
            pts = patch_points(props.grass_patch_size, props.grass_patch_density, props.seed,
                               props.grass_patch_spacing * 0.01)
        rng = np.random.default_rng(props.seed + 31)
        which = rng.integers(nvar, size=len(pts))
        for i in range(nvar):
            res = GrassEngine(prof).generate(seed=props.seed * 17 + i, detail=props.succ_detail * 0.8,
                                             with_roots=props.show_roots)
            proto = _child_plain(context, root, f"_GrassProto{i}")
            populate_mesh(proto.data, MeshData.concatenate([m for m in (res.leaves, res.culms, res.heads, res.ears,
                                                                        res.roots) if len(m.vertices)]))
            _assign(proto, mat)
            proto.hide_viewport = proto.hide_render = True
            sel = pts[which == i]
            obj = _child_plain(context, root, f"_GrassPatch{i}")
            n = len(sel)
            yaw = rng.uniform(0, 2 * math.pi, n)
            R = np.array([[[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]] for a in yaw])
            if nrm is not None and n:
                R = np.einsum("nij,njk->nik", _tilts(nrm[which == i], props.grass_patch_align), R)
            scale = 1.0 + props.grass_patch_scale_var * rng.uniform(-1, 1, n)
            populate_mesh(obj.data, MeshData(
                sel.astype(np.float32), np.zeros(0, np.int32), np.zeros(0, np.int32), np.zeros(0, np.int32),
                np.zeros((0, 2), np.float32),
                {"frot": (_euler_xyz(R) if n else np.zeros((0, 3))).astype(np.float32),
                 "fscale": scale.astype(np.float32)}))
            _instancer_nodes(obj, proto)
            obj.hide_viewport = obj.hide_render = False
        root["tufts"] = len(pts)
    root["scientific_name"] = sp.scientific_name
    root["common_name"] = sp.common_name
    root["family"] = sp.family
    root["growth_form"] = "Grass"
    return root
