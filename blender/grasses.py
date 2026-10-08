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
    from bpy.props import BoolProperty, FloatProperty, IntProperty
    BLENDER_AVAILABLE = True
except ImportError:
    BLENDER_AVAILABLE = False

try:
    from ..core.grass import GrassEngine, GrassProfile, patch_points
    from ..core.grass_db import GRASS_CATALOG, GRASS_RANGES
    from ..core.mesh_engine import MeshData
    from .flowers import _props_for, _instancer_nodes, _euler_xyz
    from .mesh_builder import populate_mesh
except (ImportError, ValueError):
    from core.grass import GrassEngine, GrassProfile, patch_points
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
    ("Leaves", ["leaf_length_cm", "leaf_width_cm", "leaf_peak", "leaf_angle_deg", "leaf_droop", "leaf_twist_deg",
                "leaf_fold", "margin_wave", "sheath_fraction", "leaf_color", "midrib_color", "tip_dryness"]),
    ("Inflorescence", ["head", "head_length_cm", "head_width_cm", "peduncle_cm", "spikelets", "spikelet_mm",
                       "awn_cm", "branches", "branch_angle_deg", "nod", "head_color", "awn_color"]),
    ("Maize Ears", ["ears", "ear_node", "ear_length_cm", "ear_diameter_cm", "kernel_rows", "husk", "silk_cm",
                    "kernel_color", "silk_color"]),
    ("Roots & Ripeness", ["crown_roots", "root_length_cm", "brace_roots", "ripeness"]),
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
                      (7, (0.85, 0.80, 0.68), 300)):
        col = _mix(nt, is_part(k, (loc - 500, -700 - k * 40)), col, rgb(c, (loc, 650)), (loc, 400 - k * 20))
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
        pts = patch_points(props.grass_patch_size, props.grass_patch_density, props.seed,
                           props.grass_patch_spacing * 0.01)
        rng = np.random.default_rng(props.seed + 31)
        which = rng.integers(nvar, size=len(pts))
        for i in range(nvar):
            res = GrassEngine(prof).generate(seed=props.seed * 17 + i, detail=props.succ_detail * 0.8,
                                             with_roots=False)
            proto = _child_plain(context, root, f"_GrassProto{i}")
            populate_mesh(proto.data, MeshData.concatenate([m for m in (res.leaves, res.culms, res.heads, res.ears)
                                                            if len(m.vertices)]))
            _assign(proto, mat)
            proto.hide_viewport = proto.hide_render = True
            sel = pts[which == i]
            obj = _child_plain(context, root, f"_GrassPatch{i}")
            n = len(sel)
            yaw = rng.uniform(0, 2 * math.pi, n)
            R = np.array([[[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]] for a in yaw])
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
