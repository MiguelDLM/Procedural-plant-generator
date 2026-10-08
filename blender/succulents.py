"""
Blender side of the succulent growth forms (Cactaceae and rosette leaf succulents):
auto-generated properties from the core profiles, materials and geometry updates.
"""

import copy
import math
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
                       "arm_length_ratio", "arm_lean_deg", "arm_branching", "crown_fill", "offsets", "offset_scale"]),
        ("Cladodes (Opuntia)", ["pad_length_cm", "pad_width_ratio", "pad_thickness_ratio", "pad_levels",
                                "pad_branching"]),
        ("Colour", ["stem_color", "groove_color", "spine_color", "spine_tip_color", "wool_color", "glaucous",
                    "flecks"]),
        ("Weathering", ["browning_height_m", "equator_bias", "equator_azimuth_deg", "scaling_color",
                        "barking_color", "scars", "scar_color", "areole_stain", "crest_light", "groove_dust",
                        "streaks"]),
        ("Roots", ["root_system", "root_count", "root_spread_ratio", "root_depth_m", "taproot_share",
                   "taproot_depth_m", "root_core_ratio", "tuber_length_cm", "tuber_radius_ratio"]),
    ],
    GrowthForm.ROSETTE: [
        ("Rosette", ["phyllotaxis", "leaf_count", "stem_height_m", "stem_radius_m", "rosette_height_m",
                     "elevation_outer_deg", "elevation_inner_deg", "elevation_power", "size_gradient",
                     "offsets", "offset_scale"]),
        ("Leaf", ["leaf_length_cm", "leaf_aspect", "leaf_thickness", "thickness_taper", "curvature_deg",
                  "widest_position", "base_angle_deg", "apex_angle_deg", "base_curvature", "apex_curvature"]),
        ("Section & Armature", ["channel", "keel", "section_exponent", "base_width", "clasp", "base_swell", "furl",
                                "dead_leaves", "stem_scars", "scar_spacing_mm",
                                "terminal_spine_cm", "teeth_count", "teeth_size_cm", "teeth_hook"]),
        ("Colour", ["leaf_color", "blush_color", "blush_amount", "blush_tip", "glaucous", "spots", "bands",
                    "armature_color", "margin_band", "striation", "imprints", "dead_color", "stem_color"]),
        ("Roots", ["root_system", "root_count", "root_spread_ratio", "root_depth_m", "root_radius_mm"]),
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


def _sep_r(nt, colour, loc):
    sp = nt.nodes.new('ShaderNodeSeparateColor')
    sp.location = loc
    nt.links.new(colour, sp.inputs[0])
    return sp.outputs[0]


def _smooth_s(nt, x, lo, hi, loc):
    m = nt.nodes.new('ShaderNodeMapRange')
    m.interpolation_type = 'SMOOTHSTEP'
    m.location = loc
    for name, v in (("From Min", lo), ("From Max", hi)):
        if isinstance(v, (int, float)):
            m.inputs[name].default_value = v
        else:
            nt.links.new(v, m.inputs[name])
    nt.links.new(x, m.inputs['Value'])
    return m.outputs['Result']


def _noise(nt, vec, scale, detail, loc, rough=0.55):
    n = nt.nodes.new('ShaderNodeTexNoise')
    n.location = loc
    _set(n, scale, "Scale")
    _set(n, detail, "Detail")
    _set(n, rough, "Roughness")
    nt.links.new(vec, n.inputs['Vector'])
    return n


def cactus_materials(name, p):
    """
    Cactus epidermis with weathering layers (Evans et al. 1994; Kiesling): rib crest/groove tones with
    paler crests and dust in the grooves, vertical streaks, soil splash near the ground, dark halos and
    drip streaks below the areoles, corky scars, and epidermal browning ("scaling" tan to red-orange,
    "barking" dark brown) rising from the base, higher on the equator-facing side.
    """
    skin, nt, bsdf = _new_mat(name + "_Skin")
    g = lambda k, d: getattr(p, k, d)  # noqa: E731
    rib = _attr(nt, "rib", (-1500, 300))
    tub = _attr(nt, "tubercle", (-1500, 100))
    halo = _attr(nt, "areole_halo", (-1500, -100))
    drip = _attr(nt, "areole_drip", (-1500, -250))
    tc = nt.nodes.new('ShaderNodeTexCoord')
    tc.location = (-1700, -500)
    sep = nt.nodes.new('ShaderNodeSeparateXYZ')
    sep.location = (-1500, -500)
    nt.links.new(tc.outputs['Object'], sep.inputs[0])
    Z = sep.outputs['Z']                                         # Height above the ground (plant space)

    # Base tones: groove -> crest, crests paler and yellower
    crest = _math(nt, 'POWER', rib.outputs['Fac'], 0.7, (-1300, 300))
    col = _mix(nt, crest, _srgb_to_linear(p.groove_color), _srgb_to_linear(p.stem_color), (-1100, 300))
    cl = _math(nt, 'MULTIPLY', _math(nt, 'POWER', rib.outputs['Fac'], 4.0, (-1300, 450)), g("crest_light", 0.25),
               (-1100, 450))
    col = _mix(nt, cl, col, _srgb_to_linear((0.80, 0.82, 0.55)), (-900, 350))
    tub_f = _math(nt, 'MULTIPLY', tub.outputs['Fac'], 0.25, (-1100, 150))
    col = _mix(nt, tub_f, col, (0.5, 0.6, 0.45, 1.0), (-750, 250), 'SCREEN')

    # Vertical streaks (axially stretched noise) and fine mottling
    stretch = nt.nodes.new('ShaderNodeMapping')
    stretch.location = (-1500, -750)
    stretch.inputs['Scale'].default_value = (14.0, 14.0, 1.2)
    nt.links.new(tc.outputs['Object'], stretch.inputs['Vector'])
    st = _noise(nt, stretch.outputs['Vector'], 3.0, 4.0, (-1300, -750))
    sv = _math(nt, 'MULTIPLY', _math(nt, 'SUBTRACT', st.outputs['Fac'], 0.5, (-1100, -750)),
               2.0 * g("streaks", 0.3), (-950, -750))
    hsv = nt.nodes.new('ShaderNodeHueSaturation')
    hsv.location = (-600, 250)
    nt.links.new(col, hsv.inputs['Color'])
    nt.links.new(_math(nt, 'ADD', sv, 1.0, (-800, -700)), hsv.inputs['Value'])
    col = hsv.outputs['Color']
    fine = _noise(nt, tc.outputs['Object'], 30.0, 6.0, (-1300, -1000))
    col = _mix(nt, 0.12, col, fine.outputs['Color'], (-450, 200), 'OVERLAY')
    col = _glaucous(nt, col, p.glaucous, bsdf, (-350, 350))          # Wax only on living epidermis

    # Dust in the grooves (more near the ground) and soil splash on the lowest 25 cm
    groove = _math(nt, 'SUBTRACT', 1.0, crest, (-900, 0))
    low = _math(nt, 'SUBTRACT', 1.0, _smooth_s(nt, Z, 0.0, 0.6, (-1100, -100)), (-900, -100))
    dust_amt = _math(nt, 'MULTIPLY', _math(nt, 'MULTIPLY_ADD', low, 0.7, (-750, -50), 0.3), groove, (-600, -50))
    dust_amt = _math(nt, 'MULTIPLY', dust_amt, g("groove_dust", 0.3), (-450, -50))
    splash = _math(nt, 'MULTIPLY', _math(nt, 'SUBTRACT', 1.0, _smooth_s(nt, Z, 0.02, 0.25, (-750, -200)),
                                         (-600, -200)), 0.8 * g("groove_dust", 0.3), (-450, -200))
    dust_c = _srgb_to_linear((0.60, 0.52, 0.40))
    col = _mix(nt, _math(nt, 'MAXIMUM', dust_amt, splash, (-300, -100)), col, dust_c, (-250, 150))

    # Areole halos and drip streaks
    stain = _math(nt, 'MAXIMUM', _math(nt, 'MULTIPLY', halo.outputs['Fac'], 0.55, (-900, -300)),
                  drip.outputs['Fac'], (-750, -300))
    stain = _math(nt, 'MULTIPLY', stain, g("areole_stain", 0.4), (-600, -300))
    col = _mix(nt, stain, col, _srgb_to_linear((0.20, 0.20, 0.13)), (-100, 150))

    # Corky scars: irregular, domain-warped blobs selected per Voronoi cell (density = scars)
    warp_n = _noise(nt, tc.outputs['Object'], 9.0, 3.0, (-1300, -1250))
    warp = nt.nodes.new('ShaderNodeMix')
    warp.data_type = 'VECTOR'
    warp.location = (-1100, -1250)
    warp.inputs[0].default_value = 0.08
    nt.links.new(tc.outputs['Object'], warp.inputs[4])
    nt.links.new(warp_n.outputs['Color'], warp.inputs[5])
    vor = nt.nodes.new('ShaderNodeTexVoronoi')
    vor.location = (-900, -1250)
    vor.voronoi_dimensions = '3D'
    _set(vor, 12.0, "Scale")                                       # ~8 cm cells
    nt.links.new(warp.outputs[1], vor.inputs['Vector'])
    pick = _math(nt, 'LESS_THAN', _sep_r(nt, vor.outputs['Color'], (-700, -1350)), g("scars", 0.15), (-550, -1350))
    blob = _math(nt, 'SUBTRACT', 1.0, _smooth_s(nt, vor.outputs['Distance'], 0.22, 0.30, (-700, -1200)),
                 (-550, -1200))
    scar = _math(nt, 'MULTIPLY', blob, pick, (-400, -1250))
    rim = _math(nt, 'MULTIPLY', _smooth_s(nt, vor.outputs['Distance'], 0.16, 0.26, (-550, -1050)), scar,
                (-400, -1050))
    col = _mix(nt, scar, col, _srgb_to_linear(g("scar_color", (0.64, 0.55, 0.41))), (50, 100))

    # Epidermal browning from the base: higher on the equator-facing side, patchy, starting on crests
    nrm = nt.nodes.new('ShaderNodeVectorTransform')
    nrm.location = (-1500, -1550)
    nrm.vector_type = 'NORMAL'
    nrm.convert_from = 'WORLD'
    nrm.convert_to = 'OBJECT'
    geo = nt.nodes.new('ShaderNodeNewGeometry')
    geo.location = (-1700, -1550)
    nt.links.new(geo.outputs['Normal'], nrm.inputs[0])
    import math as _m
    az = _m.radians(g("equator_azimuth_deg", 270.0))
    eq = nt.nodes.new('ShaderNodeVectorMath')
    eq.operation = 'DOT_PRODUCT'
    eq.location = (-1300, -1550)
    eq.inputs[1].default_value = (_m.cos(az), _m.sin(az), 0.0)
    nt.links.new(nrm.outputs[0], eq.inputs[0])
    facing = _math(nt, 'MULTIPLY_ADD', eq.outputs['Value'], 0.5, (-1100, -1550), 0.5)
    H = g("browning_height_m", 0.0)
    if H > 0:
        front = _math(nt, 'MULTIPLY', _math(nt, 'MULTIPLY_ADD', facing, 2.0 * g("equator_bias", 0.6), (-950, -1550),
                                            1.0 - g("equator_bias", 0.6)), H, (-800, -1550))
        patch = _noise(nt, tc.outputs['Object'], 4.0, 8.0, (-1100, -1750), 0.65)
        zj = _math(nt, 'ADD', Z, _math(nt, 'MULTIPLY', _math(nt, 'SUBTRACT', patch.outputs['Fac'], 0.5, (-950, -1750)),
                                       1.2 * H, (-800, -1750)), (-650, -1650))
        brown = _math(nt, 'SUBTRACT', 1.0, _smooth_s(nt, zj, _math(nt, 'MULTIPLY', front, 0.96, (-650, -1500)),
                                                     front, (-500, -1550)), (-350, -1550))
        # Isolated crusts ahead of the front (scaling starts in patches on exposed crests)
        isl = _noise(nt, tc.outputs['Object'], 9.0, 6.0, (-1100, -1950), 0.6)
        above = _math(nt, 'SUBTRACT', 1.0, _smooth_s(nt, Z, front, _math(nt, 'MULTIPLY', front, 1.8, (-650, -2050)),
                                                     (-500, -2000)), (-350, -2000))
        islands = _math(nt, 'MULTIPLY', _smooth_s(nt, isl.outputs['Fac'], 0.68, 0.71, (-500, -1900)), above,
                        (-200, -1950))
        brown = _math(nt, 'MAXIMUM', brown, islands, (-100, -1750))
        brown = _math(nt, 'MULTIPLY', brown, _math(nt, 'MULTIPLY_ADD', crest, 0.4, (-350, -1700), 0.6), (-200, -1600))
        deep = _smooth_s(nt, _math(nt, 'SUBTRACT', front, zj, (-350, -1850)), 0.0, 0.5 * H, (-200, -1850))
        bcol = _mix(nt, deep, _srgb_to_linear(g("scaling_color", (0.62, 0.44, 0.30))),
                    _srgb_to_linear(g("barking_color", (0.27, 0.22, 0.18))), (0, -1700))
        col = _mix(nt, brown, col, bcol, (200, 50))
    else:
        brown = None

    if p.flecks > 0:
        vf = nt.nodes.new('ShaderNodeTexVoronoi')
        vf.location = (-650, -550)
        _set(vf, 220.0, "Scale")
        nt.links.new(tc.outputs['Object'], vf.inputs['Vector'])
        fl = _math(nt, 'MULTIPLY', _math(nt, 'SUBTRACT', 1.0, _smooth_s(nt, vf.outputs['Distance'], 0.18, 0.30,
                                                                        (-450, -550)), (-300, -550)),
                   p.flecks, (-150, -550))
        col = _mix(nt, fl, col, (0.92, 0.92, 0.88, 1.0), (300, -200))
    nt.links.new(col, bsdf.inputs['Base Color'])
    _set(bsdf, 0.08, "Subsurface Weight")
    # Relief: fine cuticle noise, raised scar rims, rough bark crust
    height = _math(nt, 'ADD', _math(nt, 'MULTIPLY', fine.outputs['Fac'], 0.3, (400, -400)),
                   _math(nt, 'MULTIPLY', rim, 0.8, (400, -550)), (550, -450))
    if brown is not None:
        crust = _noise(nt, tc.outputs['Object'], 60.0, 8.0, (250, -700), 0.7)
        height = _math(nt, 'ADD', height, _math(nt, 'MULTIPLY', brown, crust.outputs['Fac'], (550, -650)), (700, -500))
        rough = _math(nt, 'MULTIPLY_ADD', brown, 0.35, (700, -250), 0.5)
        nt.links.new(rough, bsdf.inputs['Roughness'])
    bump = nt.nodes.new('ShaderNodeBump')
    bump.location = (850, -400)
    _set(bump, 0.25, "Strength")
    _set(bump, 0.002, "Distance")
    nt.links.new(height, bump.inputs['Height'])
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
    # Wool: matted-hair look (fine noise in colour and bump, soft sheen) only where wool = 1
    wn = nt2.nodes.new('ShaderNodeTexNoise')
    wn.location = (-700, -400)
    _set(wn, 900.0, "Scale")
    _set(wn, 8.0, "Detail")
    shade = _math(nt2, 'MULTIPLY', _math(nt2, 'MULTIPLY', wn.outputs['Fac'], 0.35, (-450, -400)), wl.outputs['Fac'],
                  (-300, -400))
    c2 = _mix(nt2, shade, c, (0.35, 0.33, 0.30, 1.0), (50, 0), 'MULTIPLY')
    nt2.links.new(c2, b2.inputs['Base Color'])
    wb = nt2.nodes.new('ShaderNodeBump')
    wb.location = (100, -400)
    _set(wb, 0.6, "Strength")
    nt2.links.new(wn.outputs['Fac'], wb.inputs['Height'])
    nt2.links.new(wl.outputs['Fac'], wb.inputs['Strength'])
    nt2.links.new(wb.outputs['Normal'], b2.inputs['Normal'])
    nt2.links.new(wl.outputs['Fac'], b2.inputs['Sheen Weight'])
    return skin, spines


def _math(nt, op, a, b=None, loc=(0, 0), c=None):
    m = nt.nodes.new('ShaderNodeMath')
    m.operation = op
    m.location = loc
    for i, v in enumerate((a, b, c)):
        if v is None:
            continue
        if isinstance(v, (int, float)):
            m.inputs[i].default_value = v
        else:
            nt.links.new(v, m.inputs[i])
    return m.outputs[0]


def _agave_details(nt, p, u, edge, top, col):
    """Horny margin, longitudinal striations and bud imprints (teeth and outline of the neighbouring leaf)."""
    e = edge.outputs['Fac']
    uu = u.outputs['Fac']
    tt = top.outputs['Fac']
    if getattr(p, "striation", 0.0) > 0:
        s = _math(nt, 'SINE', _math(nt, 'MULTIPLY', e, 2 * 3.14159 * 22, (-650, -1200)), None, (-450, -1200))
        s = _math(nt, 'MULTIPLY', s, 0.05 * p.striation, (-250, -1200))
        col = _mix(nt, s, col, (1.0, 1.0, 1.0, 1.0), (-50, -1200), 'ADD')
    if getattr(p, "imprints", 0.0) > 0:
        # Tooth imprints: a row of pale marks inside the margin, one per tooth of the neighbouring leaf
        per = _math(nt, 'COSINE', _math(nt, 'MULTIPLY', uu, 2 * 3.14159 * max(1, p.teeth_count), (-650, -1400)),
                    None, (-450, -1400))
        per = _math(nt, 'POWER', _math(nt, 'MULTIPLY_ADD', per, 0.5, (-300, -1400), 0.5), 6.0, (-150, -1400))
        # Soft band just inside the margin: exp(-((edge - 0.78) / 0.07)^2)
        dz = _math(nt, 'DIVIDE', _math(nt, 'SUBTRACT', e, 0.78, (-600, -1550)), 0.07, (-450, -1550))
        band = _math(nt, 'EXPONENT', _math(nt, 'MULTIPLY', _math(nt, 'MULTIPLY', dz, dz, (-350, -1600)), -1.0,
                                           (-250, -1600)), None, (-150, -1600))
        teeth = _math(nt, 'MULTIPLY', per, band, (0, -1500))
        # Outline imprint: a pale line converging toward the tip
        line_pos = _math(nt, 'MULTIPLY_ADD', uu, -0.7, (-650, -1800), 0.78)
        d = _math(nt, 'ABSOLUTE', _math(nt, 'SUBTRACT', e, line_pos, (-450, -1800)), None, (-300, -1800))
        line = _math(nt, 'EXPONENT', _math(nt, 'MULTIPLY', _math(nt, 'MULTIPLY', d, d, (-150, -1800)), -900.0,
                                           (0, -1800)), None, (100, -1800))
        imp = _math(nt, 'MULTIPLY', _math(nt, 'MAXIMUM', teeth, line, (200, -1650)), tt, (300, -1650))
        imp = _math(nt, 'MULTIPLY', imp, 0.55 * p.imprints, (400, -1650))
        col = _mix(nt, imp, col, (0.80, 0.85, 0.86, 1.0), (500, -1500))
    if getattr(p, "margin_band", 0.0) > 0:
        mb = _math(nt, 'MULTIPLY', _math(nt, 'GREATER_THAN', e, 0.93, (-450, -2000)), p.margin_band, (-300, -2000))
        col = _mix(nt, mb, col, _srgb_to_linear(p.armature_color), (600, -2000))
    return col


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
    pu.inputs[0].default_value = 0.0
    tipw = nt.nodes.new('ShaderNodeMath')
    tipw.operation = 'MULTIPLY'
    tipw.location = (-850, 450)
    tipw.inputs[1].default_value = getattr(p, "blush_tip", 1.0)
    nt.links.new(u.outputs['Fac'], tipw.inputs[0])
    nt.links.new(tipw.outputs[0], pu.inputs[0])
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
    col = _agave_details(nt, p, u, edge, top, col)
    col = _glaucous(nt, col, p.glaucous, bsdf, (300, 0))
    if getattr(p, "dead_leaves", 0) > 0:
        dead = _attr(nt, "leaf_dead", (300, -400))
        # Withered leaves: straw/brown, darker toward the base where they rot
        dry = _mix(nt, u.outputs['Fac'], tuple(0.6 * c for c in _srgb_to_linear(p.dead_color)[:3]) + (1.0,),
                   _srgb_to_linear(p.dead_color), (450, -500))
        col = _mix(nt, dead.outputs['Fac'], col, dry, (600, -300))
    nt.links.new(col, bsdf.inputs['Base Color'])
    _set(bsdf, 0.25, "Subsurface Weight")
    _set(bsdf, 0.4, "Coat Weight")
    _set(bsdf, 0.3, "Coat Roughness")

    arm, nt2, b2 = _new_mat(name + "_Armature")
    b2.inputs['Base Color'].default_value = _srgb_to_linear(p.armature_color)
    _set(b2, 0.4, "Roughness")
    stem, nt3, b3 = _new_mat(name + "_RosetteStem")
    # Pale leaf-base tissue in the insertion zone, corky bark below, darker crescent leaf scars
    zone = _attr(nt3, "stem_zone", (-900, 200))
    scar = _attr(nt3, "scar", (-900, -100))
    pale = tuple(0.6 * c + 0.4 * d for c, d in zip(p.leaf_color, (0.85, 0.82, 0.62)))
    zs = _math(nt3, 'POWER', zone.outputs['Fac'], 0.5, (-700, 200))
    col = _mix(nt3, zs, _srgb_to_linear(getattr(p, "stem_color", (0.5, 0.45, 0.36))), _srgb_to_linear(pale),
               (-450, 200))
    noise = nt3.nodes.new('ShaderNodeTexNoise')
    noise.location = (-700, -350)
    _set(noise, 60.0, "Scale")
    col = _mix(nt3, _math(nt3, 'MULTIPLY', noise.outputs['Fac'], 0.25, (-500, -350)), col, (0.1, 0.08, 0.06, 1.0),
               (-250, 0))
    sc = _math(nt3, 'MULTIPLY', scar.outputs['Fac'], getattr(p, "stem_scars", 0.0), (-700, -100))
    col = _mix(nt3, sc, col, (0.12, 0.09, 0.06, 1.0), (-50, 0))
    nt3.links.new(col, b3.inputs['Base Color'])
    _set(b3, 0.75, "Roughness")
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


def _root_material(name):
    mat = bpy.data.materials.get(name + "_RootMat")
    if mat is None:
        mat, nt, b = _new_mat(name + "_RootMat")
        b.inputs['Base Color'].default_value = _srgb_to_linear((0.42, 0.33, 0.24))
        _set(b, 0.85, "Roughness")
    return mat


def _succulent_flowers(context, root, props, form, res, profile):
    from .flowers import update_flowers_on_plant, hide_flowers, flower_from_props, flower_site_count
    from ..core.inflorescence import surface_flower_sites, FlowerSites, FlowerSiteMode
    for c in root.children:
        if c.name.endswith("_Bloom"):
            c.hide_viewport = c.hide_render = True
    if not props.show_flowers:
        hide_flowers(root)
        return
    _, ip = flower_from_props(props)
    if form == GrowthForm.CACTUS:
        w = res.flower_weight
        # Crowns of 10-20 flowers around each apex (barrels, Mammillaria) at the default density
        n = flower_site_count(props, int(np.count_nonzero(w > 0.05)) // 4 + 1)
        clad = getattr(getattr(profile, "habit", None), "value", "") == "Cladode"
        if clad:     # Margin sites already carry their direction (in the pad plane)
            n = flower_site_count(props, len(w))
        f, _ = flower_from_props(props)
        span = 0.01 * (f.petal_length_cm * 1.6 + f.receptacle_radius_cm) * props.flower_scale
        sites = surface_flower_sites(res.flower_pos, res.flower_normal, w, n, seed=props.seed,
                                     lean=0.0 if clad else 0.3 + 0.5 * ip.orientation, min_dist=span,
                                     sink=0.0 if clad else 0.006 * f.receptacle_radius_cm * props.flower_scale)
    else:
        pos, dirs = res.terminal_sites if ip.site_mode == FlowerSiteMode.TERMINAL else res.axillary_sites
        rng = np.random.default_rng(props.seed + 3)
        if ip.site_mode == FlowerSiteMode.TERMINAL:
            n = max(1 if len(pos) else 0, flower_site_count(props, len(pos)))
        else:     # Only a few axils bolt per season (1-3 lateral inflorescences per rosette)
            n = max(1 if len(pos) else 0, flower_site_count(props, int(math.ceil(len(pos) / 12))))
        idx = rng.choice(len(pos), min(n, len(pos)), replace=False) if len(pos) else np.zeros(0, int)
        up = np.array([0.0, 0.0, 1.0])
        d = dirs[idx] * (1 - ip.orientation) + up * ip.orientation
        sites = FlowerSites(pos[idx], d, np.ones(len(idx)))
    update_flowers_on_plant(context, root, props, sites)


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
                                             spine_budget=props.spine_budget, with_roots=props.show_roots,
                                             root_display_depth=props.root_display_depth)
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
        res = RosetteEngine(profile).generate(seed=props.seed, detail=props.succ_detail, with_roots=props.show_roots,
                                              root_display_depth=props.root_display_depth)
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
    _child(context, root, "_SuccRoots", res.roots if props.show_roots else None, _root_material(root.name))
    _succulent_flowers(context, root, props, form, res, profile)
    root["scientific_name"] = preset.scientific_name
    root["common_name"] = preset.common_name
    root["family"] = preset.family
    root["growth_form"] = form.value
    return root
