"""
Blender side of vegetables (root crops, potato, brassica heads): properties generated from the core profiles,
materials and the update of the plant objects.
"""

import copy
import hashlib
from dataclasses import fields
from types import SimpleNamespace

try:
    import bpy
    BLENDER_AVAILABLE = True
except ImportError:
    BLENDER_AVAILABLE = False

try:
    from ..core.vegetable import VegetableEngine, VegetableProfile
    from ..core.vegetable_db import VEGETABLE_CATALOG, VEGETABLE_RANGES
    from ..core.vine_db import leaf_ranges
    from ..core.leaf_morphology import LeafMorphologyProfile
    from ..core.leaf_venation import VenationProfile
    from .flowers import _props_for
except (ImportError, ValueError):
    from core.vegetable import VegetableEngine, VegetableProfile
    from core.vegetable_db import VEGETABLE_CATALOG, VEGETABLE_RANGES
    from core.vine_db import leaf_ranges
    from core.leaf_morphology import LeafMorphologyProfile
    from core.leaf_venation import VenationProfile
    from blender.flowers import _props_for

VEG, VGL, VGV = "veg_", "vgl_", "vgv_"
SUFFIXES = ("_VegRoot", "_VegStems", "_VegHead", "_VegLeaves")
SECTION = {VEG: "profile", VGL: "leaf", VGV: "venation"}

LAYOUT = [
    ("Shoot & Leaves", VEG, ["organ", "stem_count", "stem_height_cm", "stem_radius_mm", "leaf_count", "leaf_size",
                             "elevation_outer_deg", "elevation_inner_deg", "head_wrap", "stem_color"]),
    ("Storage Root", VEG, ["root_length_cm", "root_diameter_cm", "widest_position", "shoulder", "taper", "exposure",
                           "tail_cm", "rings", "rootlets", "rootlet_ranks", "root_color", "shoulder_color",
                           "shoulder_tint", "tip_color", "tip_tint"]),
    ("Tubers", VEG, ["tuber_count", "tuber_length_cm", "tuber_diameter_cm", "tuber_depth_cm", "stolon_length_cm",
                     "eyes", "eye_depth", "tuber_color", "tuber_dots"]),
    ("Head", VEG, ["head_type", "head_diameter_cm", "head_height_ratio", "head_levels", "florets", "floret_scale",
                   "bud_size_mm", "head_color", "branch_color"]),
    ("Leaf Shape", VGL, ["archetype", "blade_length_cm", "aspect_ratio", "widest_position", "base_angle_deg",
                         "apex_angle_deg", "cordate_depth", "petiole_length_ratio", "lobe_type", "lobe_count",
                         "lobe_depth", "margin_type", "teeth_count", "tooth_height_ratio", "compound_type",
                         "leaflet_count", "leaflet_angle_deg", "rachis_length_ratio", "terminal_leaflet"]),
    ("Leaf Surface & Colour", VGL, ["transverse_curl", "longitudinal_droop", "undulation_amplitude", "adaxial_color",
                                    "abaxial_color", "vein_color", "gloss"]),
    ("Leaf Venation", VGV, ["pattern", "secondary_vein_pairs", "divergence_angle_deg", "vein_contrast"]),
]


def _ranges(section):
    if section == "profile":
        return VEGETABLE_RANGES
    return {k: v for (sec, k), v in leaf_ranges().items() if sec == section}


def vegetable_properties(update) -> dict:
    if not BLENDER_AVAILABLE:
        return {}
    out = {}
    out.update(_props_for(VegetableProfile, VEG, _ranges("profile"), update))
    out.update(_props_for(LeafMorphologyProfile, VGL, _ranges("leaf"), update))
    out.update(_props_for(VenationProfile, VGV, _ranges("venation"), update))
    return out


def _write(props, obj, prefix):
    for f in fields(obj):
        name = prefix + f.name
        if hasattr(props, name):
            v = getattr(obj, f.name)
            setattr(props, name, v.value if hasattr(v, "value") else v)


def _read(props, obj, prefix):
    for f in fields(obj):
        name = prefix + f.name
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


def write_vegetable_to_props(props, key):
    sp = VEGETABLE_CATALOG.get(key)
    if sp is None:
        return
    _write(props, sp.profile, VEG)
    _write(props, sp.leaf, VGL)
    _write(props, sp.venation, VGV)


def vegetable_from_props(props):
    sp = VEGETABLE_CATALOG[props.vegetable_species]
    return (_read(props, copy.deepcopy(sp.profile), VEG), _read(props, copy.deepcopy(sp.leaf), VGL),
            _read(props, copy.deepcopy(sp.venation), VGV))


def current_vegetable_preset(props):
    sp = copy.deepcopy(VEGETABLE_CATALOG[props.vegetable_species])
    sp.profile, sp.leaf, sp.venation = vegetable_from_props(props)
    return sp


def section_values(props, prefix) -> dict:
    cls = {VEG: VegetableProfile, VGL: LeafMorphologyProfile, VGV: VenationProfile}[prefix]
    return {f.name: getattr(props, prefix + f.name) for f in fields(cls) if hasattr(props, prefix + f.name)}


# -----------------------------------------------------------------------------
# Materials
# -----------------------------------------------------------------------------
def _key(*parts) -> str:
    return hashlib.sha1(repr(parts).encode()).hexdigest()[:12]


def root_material(name, p: VegetableProfile):
    """Storage root or tubers: body colour, above-soil shoulder tint, tip tint, darker rings, pale rootlets,
    lenticel dots on tubers."""
    from .succulents import _new_mat, _attr, _mix, _noise, _math
    from .materials import _srgb_to_linear, _set
    mat, nt, b = _new_mat(name)

    def rgb(c, loc):
        n = nt.nodes.new('ShaderNodeRGB')
        n.location = loc
        n.outputs[0].default_value = _srgb_to_linear(c)
        return n.outputs[0]
    tubers = p.organ.value == "Tubers"
    t = _attr(nt, "veg_t", (-1400, 300)).outputs['Fac']
    above = _attr(nt, "veg_above", (-1400, 100)).outputs['Fac']
    ring = _attr(nt, "veg_ring", (-1400, -100)).outputs['Fac']
    part = _attr(nt, "veg_part", (-1400, -300)).outputs['Fac']
    tc = nt.nodes.new('ShaderNodeTexCoord')
    tc.location = (-1600, -500)
    col = rgb(p.tuber_color if tubers else p.root_color, (-1000, 500))
    if not tubers:
        col = _mix(nt, _math(nt, 'MULTIPLY', above, p.shoulder_tint, (-1100, 200)), col,
                   rgb(p.shoulder_color, (-1000, 300)), (-800, 400))
        tip = _math(nt, 'MULTIPLY', _math(nt, 'SUBTRACT', t, 0.55, (-1200, 0)), 2.5 * p.tip_tint, (-1050, 0))
        tip = _math(nt, 'MINIMUM', _math(nt, 'MAXIMUM', tip, 0.0, (-900, 0)), 1.0, (-750, 0))
        col = _mix(nt, tip, col, rgb(p.tip_color, (-800, 200)), (-600, 350))
    nz = _noise(nt, tc.outputs['Object'], 60.0, 4.0, (-1200, -500))
    col = _mix(nt, 0.25, col, nz.outputs['Color'], (-450, 300), blend='OVERLAY')
    dark = _math(nt, 'MULTIPLY', ring, 3.0, (-1000, -150))
    col = _mix(nt, _math(nt, 'MINIMUM', dark, 0.6, (-850, -150)), col, (0.08, 0.05, 0.03, 1.0), (-300, 250))
    if tubers:
        vor = nt.nodes.new('ShaderNodeTexVoronoi')
        vor.location = (-1000, -600)
        nt.links.new(tc.outputs['Object'], vor.inputs['Vector'])
        _set(vor, 90.0, "Scale")
        d = _math(nt, 'LESS_THAN', vor.outputs['Distance'], 0.12, (-800, -600))
        col = _mix(nt, _math(nt, 'MULTIPLY', d, p.tuber_dots, (-650, -600)), col, (0.35, 0.22, 0.12, 1.0),
                   (-150, 200))
    col = _mix(nt, part, col, rgb((0.85, 0.80, 0.68), (-300, 450)), (0, 200))      # Rootlets and stolons
    nt.links.new(col, b.inputs['Base Color'])
    _set(b, 0.6, "Roughness")
    _set(b, 0.08, "Subsurface Weight", "Subsurface")
    return mat


def head_material(name, p: VegetableProfile):
    from .succulents import _new_mat, _attr, _mix, _noise, _math
    from .materials import _srgb_to_linear, _set
    mat, nt, b = _new_mat(name)
    head = nt.nodes.new('ShaderNodeRGB')
    head.location = (-900, 300)
    head.outputs[0].default_value = _srgb_to_linear(p.head_color)
    br = nt.nodes.new('ShaderNodeRGB')
    br.location = (-900, 100)
    br.outputs[0].default_value = _srgb_to_linear(p.branch_color)
    branch = _attr(nt, "veg_branch", (-900, -100)).outputs['Fac']
    hgt = _attr(nt, "veg_h", (-1100, 450)).outputs['Fac']
    crev = _math(nt, 'POWER', _math(nt, 'SUBTRACT', 1.0, hgt, (-900, 500)), 2.0, (-750, 500))
    shade = _mix(nt, _math(nt, 'MULTIPLY', crev, 0.55, (-600, 500)), head.outputs[0], (0.05, 0.05, 0.03, 1.0),
                 (-500, 350))
    col = _mix(nt, branch, shade, br.outputs[0], (-350, 200))
    tc = nt.nodes.new('ShaderNodeTexCoord')
    tc.location = (-1100, -300)
    nz = _noise(nt, tc.outputs['Object'], 140.0 if p.head_type.value == "Buds" else 40.0, 4.0, (-900, -300))
    col = _mix(nt, 0.3, col, nz.outputs['Color'], (-350, 200), blend='OVERLAY')
    nt.links.new(col, b.inputs['Base Color'])
    _set(b, 0.65, "Roughness")
    _set(b, 0.15, "Subsurface Weight", "Subsurface")
    if p.head_type.value == "Buds":            # Bud texture as fine bump
        bump = nt.nodes.new('ShaderNodeBump')
        bump.location = (300, -300)
        _set(bump, 0.4, "Strength")
        nt.links.new(nz.outputs['Fac'], bump.inputs['Height'])
        nt.links.new(bump.outputs['Normal'], b.inputs['Normal'])
    return mat


# -----------------------------------------------------------------------------
# Geometry
# -----------------------------------------------------------------------------
def hide_vegetable_parts(root):
    for c in root.children:
        if c.name.endswith(SUFFIXES):
            c.hide_viewport = c.hide_render = True


def update_vegetable_geometry(context, props, find_root):
    from .succulents import _child
    from .runtime import _ensure_leaf_material
    from .vines import stem_material, _transparency_depth
    sp = VEGETABLE_CATALOG.get(props.vegetable_species)
    if sp is None:
        print("[PPG] No valid vegetable preset selected")
        return None
    root = find_root(context, f"PPG_{sp.scientific_name.split(' ')[0]}_{sp.scientific_name.split(' ')[1]}"
                     .replace("'", "").replace(".", ""))
    for c in root.children:
        if not c.name.endswith(SUFFIXES):
            c.hide_viewport = c.hide_render = True
    prof, leaf, ven = vegetable_from_props(props)
    res = VegetableEngine(prof, leaf, ven).generate(
        seed=props.seed, detail=props.succ_detail, leaf_density=1.0 if props.show_leaves else 0.0,
        with_roots=props.show_roots, lift=props.veg_lift * prof.root_length_cm * 0.01
        if prof.organ.value == "Taproot" else props.veg_lift * (prof.tuber_depth_cm + prof.tuber_length_cm) * 0.01)
    mats = {}
    if props.assign_materials:
        def cached(suffix, k, make):
            mat = bpy.data.materials.get(root.name + suffix)
            if mat is None or mat.get("ppg_key") != k:
                mat = make(root.name + suffix)
                mat["ppg_key"] = k
            return mat
        mats["root"] = cached("_RootCropMat", _key(prof, "root"), lambda n: root_material(n, prof))
        mats["head"] = cached("_HeadMat", _key(prof, "head"), lambda n: head_material(n, prof))
        stem_prof = SimpleNamespace(stem_color=prof.stem_color, stem_color_old=prof.stem_color, hairiness=0.1)
        mats["stem"] = cached("_VegStemMat", _key(prof.stem_color), lambda n: stem_material(n, stem_prof))
        if len(res.foliage):
            mats["leaf"] = _ensure_leaf_material(root, SimpleNamespace(leaf_morphology=leaf, venation=ven), props,
                                                 res.leaf_engine.shape_model, 0)
    _transparency_depth(context.scene, len(res.foliage))
    _child(context, root, "_VegRoot", res.root if props.show_roots else None, mats.get("root"))
    _child(context, root, "_VegStems", res.stems, mats.get("stem"))
    _child(context, root, "_VegHead", res.head, mats.get("head"))
    _child(context, root, "_VegLeaves", res.leaves if props.show_leaves else None, mats.get("leaf"))
    root["scientific_name"] = sp.scientific_name
    root["common_name"] = sp.common_name
    root["family"] = sp.family
    root["growth_form"] = "Vegetable"
    root["storage_organ"] = prof.organ.value
    root["leaf_count"] = len(res.foliage)
    return root
