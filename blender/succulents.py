"""
Blender side of the succulent growth forms (Cactaceae and rosette leaf succulents):
auto-generated properties from the core profiles, materials and geometry updates.
"""

import copy
import hashlib
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
    from ..core.succulent_db import GrowthForm, CATALOGS, RANGES, PROFILE_CLASSES
    from ..core.cactus import CactusEngine
    from ..core.rosette import RosetteEngine
    from .mesh_builder import populate_mesh
    from .materials import _srgb_to_linear, _set
except (ImportError, ValueError):
    from core.succulent_db import GrowthForm, CATALOGS, RANGES, PROFILE_CLASSES
    from core.cactus import CactusEngine
    from core.rosette import RosetteEngine
    from blender.mesh_builder import populate_mesh
    from blender.materials import _srgb_to_linear, _set

PREFIX = {GrowthForm.CACTUS: "cac_", GrowthForm.ROSETTE: "ros_"}

# Panel layout: (title, [field names]) per form
LAYOUT = {
    GrowthForm.CACTUS: [
        ("Stem", ["habit", "height_m", "diameter_m", "base_taper", "apex_dome", "apex_roundness",
                  "apex_depression"]),
        ("Ribs & Tubercles", ["arrangement", "rib_count", "rib_depth", "rib_sharpness", "rib_twist_deg_per_m",
                              "tubercle_height", "areole_spacing_cm"]),
        ("Spines & Wool", ["radial_spines", "radial_length_cm", "central_spines", "central_length_cm",
                           "spine_thickness_mm", "spine_curvature", "central_hook", "radial_lift_deg",
                           "spine_jitter", "wool", "apical_wool"]),
        ("Branching", ["arm_count", "arm_height_min", "arm_height_max", "arm_radius_ratio", "arm_reach_m",
                       "arm_length_ratio", "offsets", "offset_scale"]),
        ("Cladodes (Opuntia)", ["pad_length_cm", "pad_width_ratio", "pad_thickness_ratio", "pad_levels",
                                "pad_branching"]),
        ("Colour", ["stem_color", "groove_color", "spine_color", "spine_tip_color", "wool_color", "glaucous",
                    "flecks"]),
    ],
    GrowthForm.ROSETTE: [
        ("Rosette", ["phyllotaxis", "leaf_count", "stem_height_m", "stem_radius_m", "rosette_height_m",
                     "elevation_outer_deg", "elevation_inner_deg", "elevation_power", "size_gradient",
                     "offsets", "offset_scale"]),
        ("Leaf", ["leaf_length_cm", "leaf_aspect", "leaf_thickness", "thickness_taper", "curvature_deg",
                  "widest_position", "base_angle_deg", "apex_angle_deg", "base_curvature", "apex_curvature"]),
        ("Section & Armature", ["channel", "keel", "section_exponent", "terminal_spine_cm", "teeth_count",
                                "teeth_size_cm", "teeth_hook"]),
        ("Colour", ["leaf_color", "blush_color", "blush_amount", "glaucous", "spots", "bands",
                    "armature_color"]),
    ],
}


def _label(name: str) -> str:
    return (name.replace("_deg_per_m", " (deg/m)").replace("_deg", " (deg)").replace("_cm", " (cm)")
            .replace("_mm", " (mm)").replace("_m", " (m)").replace("_", " ").strip().title())


def profile_properties(form: GrowthForm, update) -> dict:
    """Blender properties for every field of a succulent profile (prefixed by form)."""
    out = {}
    if not BLENDER_AVAILABLE:
        return out
    cls = PROFILE_CLASSES[form]
    ranges = RANGES[form]
    default = cls()
    for f in fields(cls):
        v = getattr(default, f.name)
        name = PREFIX[form] + f.name
        if isinstance(v, Enum):
            items = [(e.value, e.value, "") for e in type(v)]
            out[name] = EnumProperty(name=_label(f.name), items=items, default=v.value, update=update)
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


def write_profile_to_props(props, form: GrowthForm, profile):
    for f in fields(profile):
        name = PREFIX[form] + f.name
        if hasattr(props, name):
            v = getattr(profile, f.name)
            setattr(props, name, v.value if isinstance(v, Enum) else v)


def profile_from_props(props, form: GrowthForm, species_key: str):
    base = copy.deepcopy(CATALOGS[form][species_key].profile)
    for f in fields(base):
        name = PREFIX[form] + f.name
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


# -----------------------------------------------------------------------------
# Materials
# -----------------------------------------------------------------------------
def _new_mat(name):
    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new('ShaderNodeOutputMaterial')
    out.location = (900, 0)
    bsdf = nt.nodes.new('ShaderNodeBsdfPrincipled')
    bsdf.location = (600, 0)
    nt.links.new(bsdf.outputs['BSDF'], out.inputs['Surface'])
    return mat, nt, bsdf


def _attr(nt, name, loc):
    a = nt.nodes.new('ShaderNodeAttribute')
    a.attribute_name = name
    a.location = loc
    return a


def _mix(nt, fac, a, b, loc, blend='MIX'):
    m = nt.nodes.new('ShaderNodeMix')
    m.data_type = 'RGBA'
    m.blend_type = blend
    m.location = loc
    for sock, val in ((m.inputs[0], fac), (m.inputs[6], a), (m.inputs[7], b)):
        if isinstance(val, (int, float, tuple)):
            sock.default_value = val
        else:
            nt.links.new(val, sock)
    return m.outputs[2]


def _glaucous(nt, colour_socket, amount, bsdf, loc):
    bloom = (0.70, 0.78, 0.80, 1.0)
    out = _mix(nt, 0.5 * amount, colour_socket, _srgb_to_linear(bloom[:3]), loc)
    _set(bsdf, 0.4 + 0.45 * amount, "Roughness")
    _set(bsdf, 0.6 * amount, "Sheen Weight", "Sheen")
    return out


def cactus_materials(name, p):
    skin, nt, bsdf = _new_mat(name + "_Skin")
    rib = _attr(nt, "rib", (-900, 200))
    tub = _attr(nt, "tubercle", (-900, -50))
    pw = nt.nodes.new('ShaderNodeMath')
    pw.operation = 'POWER'
    pw.location = (-700, 200)
    pw.inputs[1].default_value = 0.7
    nt.links.new(rib.outputs['Fac'], pw.inputs[0])
    col = _mix(nt, pw.outputs[0], _srgb_to_linear(p.groove_color), _srgb_to_linear(p.stem_color), (-450, 200))
    # Tubercle tips slightly paler (younger epidermis, less chlorophyll at the podarium apex)
    tub_f = nt.nodes.new('ShaderNodeMath')
    tub_f.operation = 'MULTIPLY'
    tub_f.location = (-450, -50)
    tub_f.inputs[1].default_value = 0.25
    nt.links.new(tub.outputs['Fac'], tub_f.inputs[0])
    col = _mix(nt, tub_f.outputs[0], col, (0.5, 0.6, 0.45, 1.0), (-250, 100), 'SCREEN')
    coord = nt.nodes.new('ShaderNodeTexCoord')
    coord.location = (-900, -300)
    noise = nt.nodes.new('ShaderNodeTexNoise')
    noise.location = (-650, -300)
    _set(noise, 30.0, "Scale")
    nt.links.new(coord.outputs['Object'], noise.inputs['Vector'])
    col = _mix(nt, 0.15, col, noise.outputs['Color'], (-250, 0), 'OVERLAY')
    if p.flecks > 0:
        vor = nt.nodes.new('ShaderNodeTexVoronoi')
        vor.location = (-650, -550)
        _set(vor, 220.0, "Scale")
        nt.links.new(coord.outputs['Object'], vor.inputs['Vector'])
        ramp = nt.nodes.new('ShaderNodeValToRGB')
        ramp.location = (-450, -550)
        ramp.color_ramp.elements[0].position = 0.18
        ramp.color_ramp.elements[0].color = (1, 1, 1, 1)
        ramp.color_ramp.elements[1].position = 0.30
        ramp.color_ramp.elements[1].color = (0, 0, 0, 1)
        nt.links.new(vor.outputs['Distance'], ramp.inputs['Fac'])
        mul = nt.nodes.new('ShaderNodeMath')
        mul.operation = 'MULTIPLY'
        mul.location = (-250, -550)
        mul.inputs[1].default_value = p.flecks
        nt.links.new(ramp.outputs['Color'], mul.inputs[0])
        col = _mix(nt, mul.outputs[0], col, (0.92, 0.92, 0.88, 1.0), (-50, -200))
    col = _glaucous(nt, col, p.glaucous, bsdf, (150, 0))
    nt.links.new(col, bsdf.inputs['Base Color'])
    _set(bsdf, 0.08, "Subsurface Weight")
    bump = nt.nodes.new('ShaderNodeBump')
    bump.location = (350, -300)
    _set(bump, 0.15, "Strength")
    nt.links.new(noise.outputs['Fac'], bump.inputs['Height'])
    nt.links.new(bump.outputs['Normal'], bsdf.inputs['Normal'])

    spines, nt2, b2 = _new_mat(name + "_Spines")
    st = _attr(nt2, "spine_t", (-700, 200))
    wl = _attr(nt2, "wool", (-700, -100))
    c = _mix(nt2, st.outputs['Fac'], _srgb_to_linear(p.spine_color), _srgb_to_linear(p.spine_tip_color), (-400, 200))
    c = _mix(nt2, wl.outputs['Fac'], c, _srgb_to_linear(p.wool_color), (-150, 100))
    nt2.links.new(c, b2.inputs['Base Color'])
    rough = nt2.nodes.new('ShaderNodeMapRange')
    rough.location = (-150, -150)
    _set(rough, 0.35, "To Min")
    _set(rough, 0.95, "To Max")
    nt2.links.new(wl.outputs['Fac'], rough.inputs['Value'])
    nt2.links.new(rough.outputs['Result'], b2.inputs['Roughness'])
    _set(b2, 0.15, "Subsurface Weight")
    return skin, spines


def rosette_materials(name, p):
    leaf, nt, bsdf = _new_mat(name + "_Leaf")
    u = _attr(nt, "leaf_u", (-1100, 300))
    edge = _attr(nt, "leaf_edge", (-1100, 100))
    top = _attr(nt, "leaf_top", (-1100, -100))
    rnd = _attr(nt, "leaf_rand", (-1100, -300))
    # Blush on margins and tip
    pe = nt.nodes.new('ShaderNodeMath')
    pe.operation = 'POWER'
    pe.location = (-850, 100)
    pe.inputs[1].default_value = 6.0
    nt.links.new(edge.outputs['Fac'], pe.inputs[0])
    pu = nt.nodes.new('ShaderNodeMath')
    pu.operation = 'POWER'
    pu.location = (-850, 300)
    pu.inputs[1].default_value = 5.0
    nt.links.new(u.outputs['Fac'], pu.inputs[0])
    mx = nt.nodes.new('ShaderNodeMath')
    mx.operation = 'MAXIMUM'
    mx.location = (-650, 200)
    nt.links.new(pe.outputs[0], mx.inputs[0])
    nt.links.new(pu.outputs[0], mx.inputs[1])
    amt = nt.nodes.new('ShaderNodeMath')
    amt.operation = 'MULTIPLY'
    amt.location = (-450, 200)
    amt.inputs[1].default_value = p.blush_amount
    nt.links.new(mx.outputs[0], amt.inputs[0])
    base = (_srgb_to_linear(p.leaf_color))
    hsv = nt.nodes.new('ShaderNodeHueSaturation')
    hsv.location = (-650, -250)
    hsv.inputs['Color'].default_value = base
    vmap = nt.nodes.new('ShaderNodeMapRange')
    vmap.location = (-850, -300)
    _set(vmap, 0.9, "To Min")
    _set(vmap, 1.1, "To Max")
    nt.links.new(rnd.outputs['Fac'], vmap.inputs['Value'])
    nt.links.new(vmap.outputs['Result'], hsv.inputs['Value'])
    col = _mix(nt, amt.outputs[0], hsv.outputs['Color'], _srgb_to_linear(p.blush_color), (-250, 100))
    coord = nt.nodes.new('ShaderNodeTexCoord')
    coord.location = (-1100, -550)
    if p.spots > 0:
        vor = nt.nodes.new('ShaderNodeTexVoronoi')
        vor.location = (-850, -550)
        _set(vor, 45.0, "Scale")
        nt.links.new(coord.outputs['Object'], vor.inputs['Vector'])
        ramp = nt.nodes.new('ShaderNodeValToRGB')
        ramp.location = (-650, -550)
        ramp.color_ramp.elements[0].position = 0.12
        ramp.color_ramp.elements[0].color = (1, 1, 1, 1)
        ramp.color_ramp.elements[1].position = 0.2
        ramp.color_ramp.elements[1].color = (0, 0, 0, 1)
        nt.links.new(vor.outputs['Distance'], ramp.inputs['Fac'])
        sm = nt.nodes.new('ShaderNodeMath')
        sm.operation = 'MULTIPLY'
        sm.location = (-450, -550)
        sm.inputs[1].default_value = p.spots
        nt.links.new(ramp.outputs['Color'], sm.inputs[0])
        col = _mix(nt, sm.outputs[0], col, (0.85, 0.88, 0.75, 1.0), (-50, -100))
    if p.bands > 0:
        # Transverse bands of white tubercles on the abaxial face (Haworthiopsis)
        freq = nt.nodes.new('ShaderNodeMath')
        freq.operation = 'MULTIPLY'
        freq.location = (-850, -800)
        freq.inputs[1].default_value = 2 * 3.14159 * 14
        nt.links.new(u.outputs['Fac'], freq.inputs[0])
        sn = nt.nodes.new('ShaderNodeMath')
        sn.operation = 'SINE'
        sn.location = (-650, -800)
        nt.links.new(freq.outputs[0], sn.inputs[0])
        band = nt.nodes.new('ShaderNodeMapRange')
        band.location = (-450, -800)
        _set(band, 0.55, "From Min")
        _set(band, 0.9, "From Max")
        nt.links.new(sn.outputs[0], band.inputs['Value'])
        inv = nt.nodes.new('ShaderNodeMath')
        inv.operation = 'SUBTRACT'
        inv.location = (-450, -1000)
        inv.inputs[0].default_value = 1.0
        nt.links.new(top.outputs['Fac'], inv.inputs[1])
        m1 = nt.nodes.new('ShaderNodeMath')
        m1.operation = 'MULTIPLY'
        m1.location = (-250, -850)
        nt.links.new(band.outputs['Result'], m1.inputs[0])
        nt.links.new(inv.outputs[0], m1.inputs[1])
        m2 = nt.nodes.new('ShaderNodeMath')
        m2.operation = 'MULTIPLY'
        m2.location = (-100, -850)
        m2.inputs[1].default_value = p.bands
        nt.links.new(m1.outputs[0], m2.inputs[0])
        col = _mix(nt, m2.outputs[0], col, (0.92, 0.93, 0.9, 1.0), (50, -300))
    col = _glaucous(nt, col, p.glaucous, bsdf, (300, 0))
    nt.links.new(col, bsdf.inputs['Base Color'])
    _set(bsdf, 0.25, "Subsurface Weight")
    _set(bsdf, 0.4, "Coat Weight")
    _set(bsdf, 0.3, "Coat Roughness")

    arm, nt2, b2 = _new_mat(name + "_Armature")
    b2.inputs['Base Color'].default_value = _srgb_to_linear(p.armature_color)
    _set(b2, 0.4, "Roughness")
    stem, nt3, b3 = _new_mat(name + "_RosetteStem")
    b3.inputs['Base Color'].default_value = _srgb_to_linear(tuple(0.55 * c + 0.45 * d for c, d in
                                                                   zip(p.leaf_color, (0.45, 0.38, 0.28))))
    _set(b3, 0.7, "Roughness")
    return leaf, arm


# -----------------------------------------------------------------------------
# Geometry update
# -----------------------------------------------------------------------------
def _key(*parts) -> str:
    return hashlib.sha1(repr(parts).encode()).hexdigest()[:12]


def _child(context, root, suffix, mesh_data, mat):
    name = f"{root.name}{suffix}"
    obj = next((c for c in root.children if c.name.endswith(suffix)), None)
    if obj is None:
        obj = bpy.data.objects.new(name, bpy.data.meshes.new(name))
        context.scene.collection.objects.link(obj)
        obj.parent = root
    if mesh_data is None or len(mesh_data.vertices) == 0:
        obj.data.clear_geometry()
        obj.hide_viewport = obj.hide_render = True
        return obj
    populate_mesh(obj.data, mesh_data)
    obj.hide_viewport = obj.hide_render = False
    if mat is not None:
        if len(obj.data.materials) == 0:
            obj.data.materials.append(mat)
        else:
            obj.data.materials[0] = mat
    return obj


def update_succulent_geometry(context, props, find_root):
    form = GrowthForm(props.growth_form)
    key = props.cactus_species if form == GrowthForm.CACTUS else props.rosette_species
    preset = CATALOGS[form][key]
    profile = profile_from_props(props, form, key)
    root = find_root(context, f"PPG_{preset.scientific_name.replace(' ', '_').replace(chr(39), '')}")
    # Objects of other growth forms (if this root was a tree before) are hidden
    for c in root.children:
        if c.name.endswith(("_Wood", "_Foliage", "_Roots")):
            c.hide_viewport = c.hide_render = True
    mkey = _key(profile)
    if form == GrowthForm.CACTUS:
        res = CactusEngine(profile).generate(seed=props.seed, detail=props.succ_detail,
                                             spine_budget=props.spine_budget)
        skin = spines = None
        if props.assign_materials:
            skin = bpy.data.materials.get(root.name + "_Skin")
            if skin is None or skin.get("ppg_key") != mkey:
                skin, spines = cactus_materials(root.name, profile)
                skin["ppg_key"] = mkey
            spines = bpy.data.materials.get(root.name + "_Spines")
        _child(context, root, "_Stem", res.stem, skin)
        _child(context, root, "_Spines", res.spines, spines)
        for suffix in ("_Leaves", "_Armature", "_RosetteBase"):
            _child(context, root, suffix, None, None)
        root["areoles"] = res.areole_count
        root["spines"] = res.spine_count
    else:
        res = RosetteEngine(profile).generate(seed=props.seed, detail=props.succ_detail)
        leaf = arm = None
        if props.assign_materials:
            leaf = bpy.data.materials.get(root.name + "_Leaf")
            if leaf is None or leaf.get("ppg_key") != mkey:
                leaf, arm = rosette_materials(root.name, profile)
                leaf["ppg_key"] = mkey
            arm = bpy.data.materials.get(root.name + "_Armature")
        _child(context, root, "_Leaves", res.leaves, leaf)
        _child(context, root, "_Armature", res.armature, arm)
        _child(context, root, "_RosetteBase", res.stem, bpy.data.materials.get(root.name + "_RosetteStem"))
        for suffix in ("_Stem", "_Spines"):
            _child(context, root, suffix, None, None)
        root["leaf_count"] = res.leaf_count
    root["scientific_name"] = preset.scientific_name
    root["common_name"] = preset.common_name
    root["family"] = preset.family
    root["growth_form"] = form.value
    return root
