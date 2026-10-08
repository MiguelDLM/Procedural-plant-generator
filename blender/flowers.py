"""
Blender side of flowers: properties generated from the core profiles, one organ-aware
material, a standalone flower / inflorescence form, and flowers instanced on trees,
cacti and rosettes through a Geometry Nodes "Instance on Points" modifier (one
prototype inflorescence, any number of sites).
"""

import copy
import hashlib
import math
from dataclasses import fields
from enum import Enum

import numpy as np

try:
    import bpy
    from bpy.props import EnumProperty, FloatProperty, IntProperty, BoolProperty, FloatVectorProperty
    BLENDER_AVAILABLE = True
except ImportError:
    BLENDER_AVAILABLE = False

try:
    from ..core.flower import FlowerProfile, SEPAL, PETAL, FILAMENT, ANTHER, PISTIL, RECEPTACLE, FLORET, STEM
    from ..core.inflorescence import (InflorescenceProfile, InflorescenceEngine, FlowerSites, tree_flower_sites,
                                      surface_flower_sites, FlowerSiteMode)
    from ..core.flower_db import (FLOWER_CATALOG, FLOWER_RANGES, INFL_RANGES, TREE_FLOWERS, CACTUS_FLOWERS,
                                  ROSETTE_FLOWERS)
    from ..core.mesh_engine import MeshData
    from .mesh_builder import populate_mesh
    from .materials import _srgb_to_linear, _set
except (ImportError, ValueError):
    from core.flower import FlowerProfile, SEPAL, PETAL, FILAMENT, ANTHER, PISTIL, RECEPTACLE, FLORET, STEM
    from core.inflorescence import (InflorescenceProfile, InflorescenceEngine, FlowerSites, tree_flower_sites,
                                    surface_flower_sites, FlowerSiteMode)
    from core.flower_db import (FLOWER_CATALOG, FLOWER_RANGES, INFL_RANGES, TREE_FLOWERS, CACTUS_FLOWERS,
                                ROSETTE_FLOWERS)
    from core.mesh_engine import MeshData
    from blender.mesh_builder import populate_mesh
    from blender.materials import _srgb_to_linear, _set

FLW, INF = "flw_", "inf_"

LAYOUT = [
    ("Floral Diagram", FLW, ["arrangement", "merosity", "petal_whorls", "spiral_tepals", "zygomorphy", "lip_bias",
                             "petals_visible"]),
    ("Petal Shape", FLW, ["petal_length_cm", "petal_aspect", "widest_position", "base_angle_deg", "apex_angle_deg",
                          "apex_curvature", "claw", "truncation", "notch", "fringe", "fringe_count"]),
    ("Petal Posture", FLW, ["opening_deg", "reflex_deg", "cup", "twist_deg", "undulation", "inner_scale",
                            "inner_opening_deg"]),
    ("Tube & Calyx", FLW, ["tube_length_cm", "tube_radius_cm", "tube_flare", "limb_fusion", "sepal_length_ratio", "sepal_aspect",
                           "sepal_opening_deg", "hypanthium_cm", "hypanthium_scales"]),
    ("Stamens & Pistil", FLW, ["stamen_count", "stamen_length_ratio", "stamen_spread_deg", "anther_size_mm",
                               "staminal_column", "stamen_declination", "carpels", "style_length_ratio",
                               "stigma_size_mm", "receptacle_radius_cm", "receptacle_height_cm"]),
    ("Capitulum", FLW, ["capitulum", "disc_florets", "disc_radius_cm", "disc_dome", "ray_count", "ray_fill",
                        "involucre_bracts"]),
    ("Flower Colour", FLW, ["petal_color", "tip_color", "tip_start", "eye_color", "eye_size", "guide_lines",
                            "guide_contrast", "spots", "spot_color", "outer_color", "outer_tint", "sepal_color",
                            "stamen_color", "anther_color", "pistil_color", "disc_color", "disc_center_color",
                            "sheen", "translucency"]),
    ("Inflorescence", INF, ["kind", "flower_count", "peduncle_cm", "rachis_cm", "pedicel_cm", "pedicel_angle_deg",
                            "divergence_deg", "branches", "branch_start", "branch_length_ratio", "branch_angle_deg",
                            "branch_umbels", "maturation", "nodding", "droop", "stem_radius_mm", "stem_color",
                            "site_mode", "orientation"]),
]


def _label(name: str) -> str:
    return (name.replace("_deg", " (deg)").replace("_cm", " (cm)").replace("_mm", " (mm)")
            .replace("_", " ").strip().title())


def _props_for(cls, prefix, ranges, update) -> dict:
    out = {}
    default = cls()
    for f in fields(cls):
        v = getattr(default, f.name)
        name = prefix + f.name
        if isinstance(v, Enum):
            out[name] = EnumProperty(name=_label(f.name), items=[(e.value, e.value, "") for e in type(v)],
                                     default=v.value, update=update)
        elif isinstance(v, bool):
            out[name] = BoolProperty(name=_label(f.name), default=v, update=update)
        elif isinstance(v, int):
            lo, hi = ranges.get(f.name, (0, 100))
            out[name] = IntProperty(name=_label(f.name), default=v, min=int(lo), max=int(hi), update=update)
        elif isinstance(v, float):
            lo, hi = ranges.get(f.name, (0.0, 1.0))
            out[name] = FloatProperty(name=_label(f.name), default=v, min=float(lo), max=float(hi), update=update)
        elif isinstance(v, tuple) and len(v) == 3:
            out[name] = FloatVectorProperty(name=_label(f.name), subtype='COLOR_GAMMA', size=3, min=0.0, max=1.0,
                                            default=v, update=update)
    return out


def flower_properties(update) -> dict:
    if not BLENDER_AVAILABLE:
        return {}
    out = _props_for(FlowerProfile, FLW, FLOWER_RANGES, update)
    out.update(_props_for(InflorescenceProfile, INF, INFL_RANGES, update))
    return out


def _write(props, obj, prefix):
    for f in fields(obj):
        name = prefix + f.name
        if hasattr(props, name):
            v = getattr(obj, f.name)
            setattr(props, name, v.value if isinstance(v, Enum) else v)


def flower_key(props):
    """The selected flower preset id, or None if the selection is stale (removed preset, older file)."""
    key = getattr(props, "flower_species", "")
    return key if key in FLOWER_CATALOG else None


def write_flower_to_props(props, key):
    if key not in FLOWER_CATALOG:
        return
    sp = FLOWER_CATALOG[key]
    _write(props, sp.flower, FLW)
    _write(props, sp.infl, INF)


def _read(props, base, prefix):
    for f in fields(base):
        name = prefix + f.name
        if not hasattr(props, name):
            continue
        cur = getattr(base, f.name)
        val = getattr(props, name)
        if isinstance(cur, Enum):
            val = type(cur)(val)
        elif isinstance(cur, tuple):
            val = tuple(float(x) for x in val)
        elif isinstance(cur, bool):
            val = bool(val)
        elif isinstance(cur, int):
            val = int(val)
        else:
            val = float(val)
        setattr(base, f.name, val)
    return base


def flower_from_props(props):
    sp = FLOWER_CATALOG[flower_key(props) or "rosa_canina"]
    return _read(props, copy.deepcopy(sp.flower), FLW), _read(props, copy.deepcopy(sp.infl), INF)


def default_flower_for(props):
    """Flower preset of the current plant preset (None when it has no showy flowers)."""
    form = props.growth_form
    if form == 'Tree':
        return TREE_FLOWERS.get(props.species_enum)
    if form == 'Cactus':
        return CACTUS_FLOWERS.get(props.cactus_species)
    if form == 'Rosette':
        return ROSETTE_FLOWERS.get(props.rosette_species)
    return None


def sync_flowers_to_plant(props):
    """Load the flower of the selected plant preset into the flower sliders (no update)."""
    key = default_flower_for(props)
    if key and key in FLOWER_CATALOG:
        props.flower_species = key
        write_flower_to_props(props, key)


def flower_variant(props):
    """Blend with a second flower and/or mutate within the trait ranges (trait-space arithmetic)."""
    try:
        from ..core.trait_space import blend_profile, mutate_profile
        from .runtime import set_updating, schedule_update
    except (ImportError, ValueError):
        from core.trait_space import blend_profile, mutate_profile
        from blender.runtime import set_updating, schedule_update
    f, ip = flower_from_props(props)
    other = FLOWER_CATALOG[props.flower_blend]
    if props.blend_factor > 0 and props.flower_blend != props.flower_species:
        f = blend_profile(f, other.flower, props.blend_factor, FLOWER_RANGES)
        ip = blend_profile(ip, other.infl, props.blend_factor, INFL_RANGES)
    if props.variation_amount > 0:
        f = mutate_profile(f, props.variation_amount, props.variation_seed, FLOWER_RANGES)
        ip = mutate_profile(ip, props.variation_amount, props.variation_seed + 1, INFL_RANGES)
    set_updating(True)
    try:
        _write(props, f, FLW)
        _write(props, ip, INF)
    finally:
        set_updating(False)
    if getattr(props, "auto_update", True):
        schedule_update()


# -----------------------------------------------------------------------------
# Material
# -----------------------------------------------------------------------------
def _key(*parts) -> str:
    return hashlib.sha1(repr(parts).encode()).hexdigest()[:12]


def _node(nt, kind, loc, **inputs):
    n = nt.nodes.new(kind)
    n.location = loc
    return n


def _math(nt, op, a, b=None, loc=(0, 0), c=None, clamp=False):
    m = nt.nodes.new('ShaderNodeMath')
    m.operation = op
    m.location = loc
    m.use_clamp = clamp
    for i, v in enumerate((a, b, c)):
        if v is None:
            continue
        if isinstance(v, (int, float)):
            m.inputs[i].default_value = v
        else:
            nt.links.new(v, m.inputs[i])
    return m.outputs[0]


def _mix(nt, fac, a, b, loc):
    m = nt.nodes.new('ShaderNodeMix')
    m.data_type = 'RGBA'
    m.location = loc
    for sock, val in ((m.inputs[0], fac), (m.inputs[6], a), (m.inputs[7], b)):
        if isinstance(val, (int, float, tuple)):
            sock.default_value = val
        else:
            nt.links.new(val, sock)
    return m.outputs[2]


def _smooth(nt, x, lo, hi, loc):
    m = nt.nodes.new('ShaderNodeMapRange')
    m.interpolation_type = 'SMOOTHSTEP'
    m.location = loc
    m.inputs['From Min'].default_value = lo
    m.inputs['From Max'].default_value = max(hi, lo + 1e-4)
    nt.links.new(x, m.inputs['Value'])
    return m.outputs['Result']


def flower_material(name, f: FlowerProfile, ip: InflorescenceProfile):
    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = _node(nt, 'ShaderNodeOutputMaterial', (1600, 0))
    bsdf = _node(nt, 'ShaderNodeBsdfPrincipled', (1300, 0))
    nt.links.new(bsdf.outputs['BSDF'], out.inputs['Surface'])
    A = {}
    for i, k in enumerate(("organ", "pu", "pv", "whorl", "orand")):
        a = _node(nt, 'ShaderNodeAttribute', (-1600, 300 - 160 * i))
        a.attribute_name = k
        A[k] = a.outputs['Fac']
    C = lambda c: _srgb_to_linear(tuple(c))  # noqa: E731
    pu, pv = A["pu"], A["pv"]
    # Petal: base -> tip gradient, eye zone, nectar guides, spots, outer-whorl tint
    col = _mix(nt, _smooth(nt, pu, f.tip_start, 1.0, (-1200, 400)), C(f.petal_color), C(f.tip_color), (-1000, 400))
    if f.eye_size > 0:
        eye = _math(nt, 'SUBTRACT', 1.0, _smooth(nt, pu, f.eye_size * 0.7, f.eye_size, (-1200, 200)), (-1000, 200))
        col = _mix(nt, eye, col, C(f.eye_color), (-800, 300))
    if f.guide_lines > 0 and f.guide_contrast > 0:
        x = _math(nt, 'MULTIPLY_ADD', pv, 0.5 * f.guide_lines, (-1300, 0), 0.5 * f.guide_lines)
        d = _math(nt, 'ABSOLUTE', _math(nt, 'SUBTRACT', _math(nt, 'FRACT', x, None, (-1150, 0)), 0.5, (-1000, 0)),
                  None, (-850, 0))
        line = _math(nt, 'SUBTRACT', 1.0, _smooth(nt, d, 0.02, 0.12, (-700, 0)), (-550, 0))
        reach = _math(nt, 'SUBTRACT', 1.0, _smooth(nt, pu, 0.3, 0.8, (-700, -150)), (-550, -150))
        g = _math(nt, 'MULTIPLY', _math(nt, 'MULTIPLY', line, reach, (-400, -50)), f.guide_contrast, (-250, -50))
        col = _mix(nt, g, col, C(f.eye_color), (-100, 200))
    if f.spots > 0:
        comb = _node(nt, 'ShaderNodeCombineXYZ', (-1150, -350))
        nt.links.new(_math(nt, 'MULTIPLY', pu, 9.0, (-1300, -300)), comb.inputs[0])
        nt.links.new(_math(nt, 'MULTIPLY', pv, 3.0, (-1300, -400)), comb.inputs[1])
        nt.links.new(_math(nt, 'MULTIPLY', A["orand"], 17.0, (-1300, -500)), comb.inputs[2])
        vor = _node(nt, 'ShaderNodeTexVoronoi', (-950, -350))
        vor.inputs['Scale'].default_value = 1.0
        nt.links.new(comb.outputs[0], vor.inputs['Vector'])
        dot = _math(nt, 'SUBTRACT', 1.0, _smooth(nt, vor.outputs['Distance'], 0.12, 0.2, (-750, -350)), (-600, -350))
        zone = _math(nt, 'MULTIPLY', _smooth(nt, pu, 0.08, 0.2, (-750, -500)),
                     _math(nt, 'SUBTRACT', 1.0, _smooth(nt, pu, 0.5, 0.7, (-750, -600)), (-600, -600)), (-450, -500))
        sp = _math(nt, 'MULTIPLY', _math(nt, 'MULTIPLY', dot, zone, (-300, -400)), f.spots, (-150, -400))
        col = _mix(nt, sp, col, C(f.spot_color), (50, 100))
    if f.outer_tint > 0:
        ow = _math(nt, 'MULTIPLY', _math(nt, 'POWER', _math(nt, 'SUBTRACT', 1.0, A["whorl"], (-500, -700)), 2.0,
                                         (-350, -700)), f.outer_tint, (-200, -700))
        ow = _math(nt, 'MULTIPLY', ow, _math(nt, 'SUBTRACT', 1.0, _smooth(nt, pu, 0.4, 1.0, (-350, -850)),
                                             (-200, -850)), (-50, -750))
        ow = _math(nt, 'MAXIMUM', ow, _math(nt, 'MULTIPLY', _math(nt, 'SUBTRACT', 1.0, A["whorl"], (-350, -950)),
                                            0.6 * f.outer_tint, (-200, -950)), (100, -800))
        col = _mix(nt, ow, col, C(f.outer_color), (250, 0))
    # Per-flower value jitter
    hsv = _node(nt, 'ShaderNodeHueSaturation', (400, 0))
    nt.links.new(col, hsv.inputs['Color'])
    nt.links.new(_math(nt, 'MULTIPLY_ADD', A["orand"], 0.12, (250, -200), 0.94), hsv.inputs['Value'])
    col = hsv.outputs['Color']
    # Disc florets: centre colour -> disc colour, outer mature florets slightly paler
    disc = _mix(nt, _smooth(nt, pu, 0.15, 0.55, (100, -1100)), C(f.disc_center_color), C(f.disc_color), (300, -1100))
    organ_cols = [(SEPAL, C(f.sepal_color)), (FILAMENT, C(f.stamen_color)), (ANTHER, C(f.anther_color)),
                  (PISTIL, C(f.pistil_color)), (RECEPTACLE, C(f.pistil_color)), (FLORET, disc),
                  (STEM, C(ip.stem_color))]
    y = -300
    for code, c in organ_cols:
        m = _math(nt, 'COMPARE', A["organ"], float(code), (500, y), 0.5)
        col = _mix(nt, m, col, c, (700, y))
        y -= 160
    nt.links.new(col, bsdf.inputs['Base Color'])
    is_petal = _math(nt, 'COMPARE', A["organ"], float(PETAL), (900, -600), 0.5)
    _set(bsdf, 0.45, "Roughness")
    nt.links.new(_math(nt, 'MULTIPLY', is_petal, f.sheen, (1050, -600)), bsdf.inputs['Sheen Weight'])
    nt.links.new(_math(nt, 'MULTIPLY', is_petal, 0.6 * f.translucency, (1050, -750)), bsdf.inputs['Subsurface Weight'])
    return mat


# -----------------------------------------------------------------------------
# Objects
# -----------------------------------------------------------------------------
def _child(context, root, suffix):
    obj = next((c for c in root.children if c.name.endswith(suffix)), None)
    if obj is None:
        name = f"{root.name}{suffix}"
        obj = bpy.data.objects.new(name, bpy.data.meshes.new(name))
        context.scene.collection.objects.link(obj)
        obj.parent = root
    return obj


def _assign(obj, mat):
    if mat is None:
        return
    if len(obj.data.materials) == 0:
        obj.data.materials.append(mat)
    else:
        obj.data.materials[0] = mat


def hide_flowers(root):
    for c in root.children:
        if c.name.endswith(("_Flowers", "_FlowerProto", "_Bloom")):
            c.hide_viewport = c.hide_render = True


def _instancer_nodes(obj, proto):
    mod = obj.modifiers.get("PPG_Flowers")
    if mod is None:
        mod = obj.modifiers.new("PPG_Flowers", 'NODES')
    group = mod.node_group
    if group is None or group.get("ppg_version") != 1:
        group = bpy.data.node_groups.new(f"{obj.name}_Instancer", 'GeometryNodeTree')
        group["ppg_version"] = 1
        group.interface.new_socket('Geometry', in_out='INPUT', socket_type='NodeSocketGeometry')
        group.interface.new_socket('Geometry', in_out='OUTPUT', socket_type='NodeSocketGeometry')
        n, l = group.nodes, group.links
        gi = n.new('NodeGroupInput')
        go = n.new('NodeGroupOutput')
        go.location = (800, 0)
        info = n.new('GeometryNodeObjectInfo')
        info.name = "Proto"
        info.location = (0, -200)
        inst = n.new('GeometryNodeInstanceOnPoints')
        inst.location = (500, 0)
        rot = n.new('GeometryNodeInputNamedAttribute')
        rot.data_type = 'FLOAT_VECTOR'
        rot.inputs['Name'].default_value = "frot"
        rot.location = (0, -400)
        e2r = n.new('FunctionNodeEulerToRotation')
        e2r.location = (250, -400)
        scl = n.new('GeometryNodeInputNamedAttribute')
        scl.data_type = 'FLOAT'
        scl.inputs['Name'].default_value = "fscale"
        scl.location = (0, -600)
        l.new(gi.outputs[0], inst.inputs['Points'])
        l.new(info.outputs['Geometry'], inst.inputs['Instance'])
        l.new(rot.outputs['Attribute'], e2r.inputs[0])
        l.new(e2r.outputs[0], inst.inputs['Rotation'])
        l.new(scl.outputs['Attribute'], inst.inputs['Scale'])
        l.new(inst.outputs['Instances'], go.inputs[0])
        mod.node_group = group
    group.nodes["Proto"].inputs['Object'].default_value = proto
    return mod


def _euler_xyz(R):
    """Blender XYZ Euler angles of rotation matrices R (N, 3, 3)."""
    try:
        from .instancing import euler_xyz
    except (ImportError, ValueError):
        from blender.instancing import euler_xyz
    return euler_xyz(R)


def _site_rotations(D, rng):
    z = D / np.maximum(np.linalg.norm(D, axis=1, keepdims=True), 1e-9)
    ref = np.where(np.abs(z[:, 2:3]) > 0.95, np.array([[1.0, 0.0, 0.0]]), np.array([[0.0, 0.0, 1.0]]))
    x = np.cross(ref, z)
    x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-9)
    y = np.cross(z, x)
    roll = rng.uniform(0, 2 * math.pi, len(z))[:, None]
    x, y = np.cos(roll) * x + np.sin(roll) * y, -np.sin(roll) * x + np.cos(roll) * y
    return np.stack([x, y, z], axis=2)          # Columns: local axes in plant space


def _flower_mat(root, props, f, ip):
    if not props.assign_materials:
        return None
    mkey = _key(f, ip)
    mat = bpy.data.materials.get(root.name + "_Flower")
    if mat is None or mat.get("ppg_key") != mkey:
        mat = flower_material(root.name + "_Flower", f, ip)
        mat["ppg_key"] = mkey
    return mat


def build_inflorescence(props, f, ip, single=False):
    if single:
        ip = copy.deepcopy(ip)
        ip.kind = type(ip.kind)("Solitary")
    return InflorescenceEngine(f, ip).generate(seed=props.seed, detail=props.flower_detail, bloom=props.flower_bloom)


def update_flowers_on_plant(context, root, props, sites: FlowerSites):
    """Instances the current inflorescence at `sites` (plant-local coordinates)."""
    if not props.show_flowers or sites is None or len(sites) == 0 or flower_key(props) is None:
        hide_flowers(root)
        return
    f, ip = flower_from_props(props)
    res = build_inflorescence(props, f, ip)
    proto = _child(context, root, "_FlowerProto")
    populate_mesh(proto.data, res.mesh)
    _assign(proto, _flower_mat(root, props, f, ip))
    proto.hide_viewport = proto.hide_render = True
    pts = _child(context, root, "_Flowers")
    rng = np.random.default_rng(props.seed + 7)
    R = _site_rotations(sites.directions, rng)
    data = MeshData(sites.positions.astype(np.float32), np.zeros(0, np.int32), np.zeros(0, np.int32),
                    np.zeros(0, np.int32), np.zeros((0, 2), np.float32),
                    {"frot": _euler_xyz(R).astype(np.float32),
                     "fscale": (sites.scales * props.flower_scale).astype(np.float32)})
    populate_mesh(pts.data, data)
    _instancer_nodes(pts, proto)
    pts.hide_viewport = pts.hide_render = False
    for c in root.children:
        if c.name.endswith("_Bloom"):
            c.hide_viewport = c.hide_render = True
    root["flower_sites"] = len(sites)
    root["flowers_per_site"] = res.flower_count


def flower_site_count(props, available: int) -> int:
    """Inflorescences to place: a fraction of the available sites, at least one when flowering is on."""
    if available <= 0 or props.flower_density <= 0:
        return 0
    return int(min(props.flower_max, max(1, round(available * props.flower_density))))


def update_flower_geometry(context, props, find_root):
    """Standalone flower / inflorescence (growth form 'Flower')."""
    if flower_key(props) is None:
        print("[PPG] No valid flower preset selected")
        return None
    sp = FLOWER_CATALOG[props.flower_species]
    root = find_root(context, f"PPG_{sp.scientific_name.replace(' ', '_').replace(chr(39), '').replace('×', 'x')}")
    for c in root.children:
        if not c.name.endswith("_Bloom"):
            c.hide_viewport = c.hide_render = True
    f, ip = flower_from_props(props)
    res = build_inflorescence(props, f, ip, single=props.flower_single)
    obj = _child(context, root, "_Bloom")
    populate_mesh(obj.data, res.mesh)
    _assign(obj, _flower_mat(root, props, f, ip))
    obj.scale = (props.flower_scale,) * 3
    obj.hide_viewport = obj.hide_render = False
    root["scientific_name"] = sp.scientific_name
    root["common_name"] = sp.common_name
    root["family"] = sp.family
    root["floral_formula"] = sp.formula
    root["growth_form"] = "Flower"
    root["flower_count"] = res.flower_count
    return root
