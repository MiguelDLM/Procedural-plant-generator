"""
Blender side of orchids: properties generated from core.orchid, one attribute-driven material (organ identity,
pigmentation patterns: full colour, spots and venation as in Phalaenopsis PeMYB2 / 11 / 12) and the plant
objects (leaves, stems and pseudobulbs, roots, flowers, inflorescence axes, support).
"""

import copy
import hashlib
import math
from dataclasses import fields

try:
    import bpy
    BLENDER_AVAILABLE = True
except ImportError:
    BLENDER_AVAILABLE = False

try:
    from ..core.orchid import (OrchidEngine, OrchidProfile, STEM, ROOT, SEPAL, PETAL, LIP, COLUMN, ANTHER,
                               CALLUS, BRACT, SPIKE, BUD, SUPPORT, CAPSULE, STIGMA)
    from ..core.orchid_db import ORCHID_CATALOG, ORCHID_RANGES
    from .flowers import _props_for
except (ImportError, ValueError):
    from core.orchid import (OrchidEngine, OrchidProfile, STEM, ROOT, SEPAL, PETAL, LIP, COLUMN, ANTHER,
                             CALLUS, BRACT, SPIKE, BUD, SUPPORT, CAPSULE, STIGMA)
    from core.orchid_db import ORCHID_CATALOG, ORCHID_RANGES
    from blender.flowers import _props_for

ORC = "orc_"
SUFFIXES = ("_OrchidLeaves", "_OrchidStems", "_OrchidRoots", "_OrchidFlowers", "_OrchidSpikes", "_OrchidSupport")
LAYOUT = [
    ("Habit & Stems", ["habit", "growths", "rhizome_cm", "growth_angle_deg", "leafless_growths", "stem_length_cm",
                       "stem_radius_mm", "internode_cm", "support_height_m", "support_radius_cm", "stem_color",
                       "sheath_color"]),
    ("Pseudobulbs", ["pseudobulb_cm", "pseudobulb_diameter_cm", "bulb_widest", "bulb_fullness", "bulb_flatten",
                     "bulb_ridges", "bulb_lean_deg", "sheath_cover"]),
    ("Leaves", ["leaf_placement", "leaves", "leaf_length_cm", "leaf_width_cm", "leaf_thickness_mm", "leaf_widest",
                "leaf_apex_deg", "leaf_angle_deg", "leaf_droop", "leaf_fold", "leaf_twist_deg", "leaf_size_gradient",
                "leaf_color", "leaf_mottle", "mottle_color"]),
    ("Aerial Roots", ["roots", "root_diameter_mm", "root_length_cm", "root_wander", "aerial_share", "root_color",
                      "root_tip_color"]),
    ("Inflorescence", ["infl_origin", "inflorescences", "flowers", "peduncle_cm", "rachis_cm", "spike_radius_mm",
                       "spike_angle_deg", "spike_flex", "branches", "branch_ratio", "divergence_deg", "pedicel_cm",
                       "flower_facing", "maturation", "bract_mm", "spike_color", "fruit_set", "capsule_cm",
                       "capsule_diameter_cm", "capsule_color"]),
    ("Sepals & Petals", ["resupination", "flower_tilt_deg", "sepal_length_cm", "sepal_width_cm", "sepal_widest",
                         "sepal_apex_deg", "lateral_sepal_deg", "lateral_sepal_scale", "synsepal", "sepal_cup",
                         "sepal_reflex_deg", "sepal_twist_deg", "sepal_wave", "petal_length_cm", "petal_width_cm",
                         "petal_widest", "petal_apex_deg", "petal_claw", "petal_angle_deg", "petal_cup",
                         "petal_reflex_deg", "petal_twist_deg", "petal_wave", "perianth_forward_deg"]),
    ("Labellum (Lip)", ["lip_length_cm", "lip_width_cm", "lip_angle_deg", "lip_deflex_deg", "lip_claw", "side_lobes",
                        "side_lobe_pos", "side_lobe_length", "side_lobe_erect_deg", "midlobe_width", "isthmus",
                        "lip_widest", "lip_apex_deg", "lip_roll", "lip_roll_extent", "lip_sac", "lip_wave",
                        "lip_waves", "callus_mm", "callus_ridges", "callus_pos", "cirrhi_mm"]),
    ("Column", ["column_length_cm", "column_width_mm", "column_arch", "staminode_mm", "mentum_mm"]),
    ("Flower Colours", ["sepal_color", "petal_color", "lip_color", "lip_throat_color", "callus_color",
                        "column_color", "anther_color", "pattern_color", "tip_color", "tip_amount", "spots",
                        "spot_size_mm", "spot_stretch",
                        "veins", "lip_spots", "lip_veins", "sheen"]),
]


def orchid_properties(update) -> dict:
    if not BLENDER_AVAILABLE:
        return {}
    return _props_for(OrchidProfile, ORC, ORCHID_RANGES, update)


def write_orchid_to_props(props, key):
    sp = ORCHID_CATALOG.get(key)
    if sp is None:
        return
    for f in fields(sp.profile):
        name = ORC + f.name
        if hasattr(props, name):
            v = getattr(sp.profile, f.name)
            setattr(props, name, v.value if hasattr(v, "value") else v)


def orchid_from_props(props) -> OrchidProfile:
    obj = copy.deepcopy(ORCHID_CATALOG[props.orchid_species].profile)
    for f in fields(obj):
        name = ORC + f.name
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


def current_orchid_preset(props):
    sp = copy.deepcopy(ORCHID_CATALOG[props.orchid_species])
    sp.profile = orchid_from_props(props)
    return sp


def orchid_values(props) -> dict:
    return {f.name: getattr(props, ORC + f.name) for f in fields(OrchidProfile) if hasattr(props, ORC + f.name)}


# -----------------------------------------------------------------------------
# Material
# -----------------------------------------------------------------------------
def _key(*parts) -> str:
    return hashlib.sha1(repr(parts).encode()).hexdigest()[:12]


def orchid_material(name, p: OrchidProfile):
    """One material for every organ, chosen by the 'orc_part' attribute. Tepals and lip carry the three
    pigmentation patterns of Phalaenopsis (full colour, spots, venation; Hsu et al. 2015) drawn in organ space
    (orc_s along, orc_t across, in metres) so spots keep their size; the lip has a contrasting throat; roots
    a green growing tip; pseudobulbs papery sheaths; leaves optional tessellated mottling."""
    from .succulents import _new_mat, _attr, _mix, _noise, _math, _smooth_s
    from .materials import _srgb_to_linear, _set
    mat, nt, b = _new_mat(name)

    def rgb(c, loc):
        n = nt.nodes.new('ShaderNodeRGB')
        n.location = loc
        n.outputs[0].default_value = _srgb_to_linear(c)
        return n.outputs[0]
    part = _attr(nt, "orc_part", (-2200, -400)).outputs['Fac']
    u = _attr(nt, "orc_u", (-2200, 300)).outputs['Fac']
    v = _attr(nt, "orc_v", (-2200, 100)).outputs['Fac']
    s = _attr(nt, "orc_s", (-2200, -100)).outputs['Fac']
    t = _attr(nt, "orc_t", (-2200, -250)).outputs['Fac']
    rnd = _attr(nt, "orc_rand", (-2200, -600)).outputs['Fac']

    def is_part(k, loc):
        lo = _math(nt, 'GREATER_THAN', part, k - 0.5, loc)
        hi = _math(nt, 'LESS_THAN', part, k + 0.5, (loc[0], loc[1] - 60))
        return _math(nt, 'MULTIPLY', lo, hi, (loc[0] + 150, loc[1]))
    tc = nt.nodes.new('ShaderNodeTexCoord')
    tc.location = (-2200, -900)
    # Leaves (base), with tessellated mottling
    col = rgb(p.leaf_color, (-1500, 700))
    if p.leaf_mottle > 0:
        nz = _noise(nt, tc.outputs['Object'], 60.0, 2.0, (-1700, 850))
        m = _smooth_s(nt, nz.outputs['Fac'], 0.48, 0.56, (-1550, 850))
        col = _mix(nt, _math(nt, 'MULTIPLY', m, p.leaf_mottle, (-1400, 850)), col, rgb(p.mottle_color, (-1400, 1000)),
                   (-1250, 750))
    # Organ colours
    bud_c = tuple(0.65 * a + 0.35 * b_ for a, b_ in zip(p.sepal_color, p.spike_color))
    table = [(STEM, p.stem_color), (ROOT, p.root_color), (SEPAL, p.sepal_color), (PETAL, p.petal_color),
             (LIP, p.lip_color), (COLUMN, p.column_color), (ANTHER, p.anther_color), (CALLUS, p.callus_color),
             (BRACT, p.sheath_color), (SPIKE, p.spike_color), (BUD, bud_c), (SUPPORT, (0.33, 0.25, 0.17)),
             (CAPSULE, p.capsule_color), (STIGMA, (0.30, 0.26, 0.12))]
    for i, (k, c) in enumerate(table):
        col = _mix(nt, is_part(k, (-1300, -800 - 120 * i)), col, rgb(c, (-1000, 600 - 60 * i)),
                   (-800 + 40 * i, 500 - 30 * i))
    # Stems: papery sheaths from the base and dry node rings
    stem = is_part(STEM, (-1700, -2600))
    sheath = _math(nt, 'SUBTRACT', 1.0, _smooth_s(nt, u, p.sheath_cover - 0.04, p.sheath_cover + 0.04, (-1700, -2750)),
                   (-1550, -2750))
    node = _math(nt, 'SUBTRACT', 1.0, _smooth_s(nt, t, 0.1, 0.35, (-1700, -2900)), (-1550, -2900))
    sh = _math(nt, 'MULTIPLY', stem, _math(nt, 'MAXIMUM', sheath, _math(nt, 'MULTIPLY', node, 0.6, (-1400, -2900)),
                                           (-1250, -2800)), (-1100, -2700))
    col = _mix(nt, sh, col, rgb(p.sheath_color, (-1100, -2500)), (0, 200))
    # Roots: green growing tip
    tip = _math(nt, 'MULTIPLY', is_part(ROOT, (-1700, -3100)), _smooth_s(nt, u, 0.86, 0.97, (-1700, -3250)),
                (-1400, -3150))
    col = _mix(nt, tip, col, rgb(p.root_tip_color, (-1100, -3100)), (60, 180))
    # Lip throat
    lip = is_part(LIP, (-1700, -3400))
    throat = _math(nt, 'SUBTRACT', 1.0, _smooth_s(nt, u, 0.12, 0.45, (-1700, -3550)), (-1550, -3550))
    col = _mix(nt, _math(nt, 'MULTIPLY', lip, throat, (-1400, -3450)), col, rgb(p.lip_throat_color, (-1100, -3400)),
               (120, 160))
    # Coloured tips of the tepals and lip (anthocyanin toward the apex)
    if p.tip_amount > 0:
        tl = _math(nt, 'ADD', _math(nt, 'ADD', is_part(SEPAL, (-1700, -3650)), is_part(PETAL, (-1700, -3700)),
                                    (-1550, -3650)), lip, (-1400, -3650))
        tipz = _smooth_s(nt, u, 1.0 - p.tip_amount, 1.0, (-1550, -3750))
        col = _mix(nt, _math(nt, 'MULTIPLY', _math(nt, 'MINIMUM', tl, 1.0, (-1250, -3650)), tipz, (-1100, -3700)), col,
                   rgb(p.tip_color, (-1100, -3550)), (140, 150))
    # Pigmentation patterns: spots (Voronoi cells in organ space) and venation (lines along the organ)
    tep = _math(nt, 'ADD', is_part(SEPAL, (-1700, -3800)), is_part(PETAL, (-1700, -3950)), (-1400, -3850))
    comb = nt.nodes.new('ShaderNodeCombineXYZ')
    comb.location = (-1700, -4200)
    nt.links.new(_math(nt, 'MULTIPLY', s, p.spot_stretch / max(p.spot_size_mm * 0.001, 1e-5), (-1900, -4150)),
                 comb.inputs[0])
    nt.links.new(_math(nt, 'MULTIPLY', t, 1.0 / max(p.spot_size_mm * 0.001, 1e-5), (-1900, -4300)), comb.inputs[1])
    nt.links.new(_math(nt, 'MULTIPLY', rnd, 37.0, (-1900, -4450)), comb.inputs[2])
    vor = nt.nodes.new('ShaderNodeTexVoronoi')
    vor.location = (-1500, -4200)
    _set(vor, 1.0, "Scale")
    _set(vor, 1.0, "Randomness")
    nt.links.new(comb.outputs[0], vor.inputs['Vector'])
    dots = _math(nt, 'SUBTRACT', 1.0, _smooth_s(nt, vor.outputs['Distance'], 0.22, 0.32, (-1300, -4200)),
                 (-1150, -4200))
    toward_base = _math(nt, 'SUBTRACT', 1.0, _math(nt, 'MULTIPLY', u, 0.55, (-1300, -4400)), (-1150, -4400))
    vein = _math(nt, 'ABSOLUTE', _math(nt, 'COSINE', _math(nt, 'MULTIPLY', v, math.pi * 6.5, (-1500, -4600)),
                                         None, (-1350, -4600)), None, (-1200, -4600))
    vein = _smooth_s(nt, vein, 0.82, 0.97, (-1050, -4600))
    vein = _math(nt, 'MULTIPLY', vein, toward_base, (-900, -4600))
    spots_t = _math(nt, 'MULTIPLY', _math(nt, 'MULTIPLY', dots, toward_base, (-1000, -4200)), p.spots, (-850, -4200))
    pat_t = _math(nt, 'MAXIMUM', spots_t, _math(nt, 'MULTIPLY', vein, p.veins, (-750, -4600)), (-650, -4400))
    pat_t = _math(nt, 'MULTIPLY', pat_t, _math(nt, 'MINIMUM', tep, 1.0, (-800, -3850)), (-500, -4300))
    lipcal = _math(nt, 'ADD', lip, is_part(CALLUS, (-1700, -4900)), (-1400, -4900))
    pat_l = _math(nt, 'MAXIMUM', _math(nt, 'MULTIPLY', dots, p.lip_spots, (-900, -4800)),
                  _math(nt, 'MULTIPLY', vein, p.lip_veins, (-900, -4950)), (-700, -4850))
    pat_l = _math(nt, 'MULTIPLY', pat_l, _math(nt, 'MINIMUM', lipcal, 1.0, (-1200, -4900)), (-500, -4850))
    pat = _math(nt, 'MINIMUM', _math(nt, 'ADD', pat_t, pat_l, (-350, -4500)), 1.0, (-200, -4500))
    col = _mix(nt, pat, col, rgb(p.pattern_color, (-200, -4300)), (200, 140))
    # Fine variation
    nz2 = _noise(nt, tc.outputs['Object'], 140.0, 3.0, (-200, -1100))
    col = _mix(nt, 0.08, col, nz2.outputs['Color'], (300, 100), blend='OVERLAY')
    nt.links.new(col, b.inputs['Base Color'])
    # Tepals: smoother, sheen and translucency; vegetative organs: rougher
    floral = _math(nt, 'MINIMUM', _math(nt, 'ADD', _math(nt, 'ADD', tep, lipcal, (-600, -1500)),
                                          is_part(BUD, (-800, -1600)), (-450, -1500)), 1.0, (-300, -1500))
    rough = _math(nt, 'SUBTRACT', 0.55, _math(nt, 'MULTIPLY', floral, 0.25 * p.sheen + 0.05, (-150, -1500)), (0, -1500))
    nt.links.new(rough, b.inputs['Roughness'])
    sub = _math(nt, 'MULTIPLY', floral, 0.25, (0, -1700))
    sock = b.inputs.get('Subsurface Weight') or b.inputs.get('Subsurface')
    if sock is not None:
        nt.links.new(sub, sock)
    _set(b, 0.003, "Subsurface Scale")
    sheen = b.inputs.get('Sheen Weight')
    if sheen is not None:
        nt.links.new(_math(nt, 'MULTIPLY', floral, 0.6 * p.sheen, (150, -1800)), sheen)
    return mat


# -----------------------------------------------------------------------------
# Geometry
# -----------------------------------------------------------------------------
def hide_orchid_parts(root):
    for c in root.children:
        if c.name.endswith(SUFFIXES):
            c.hide_viewport = c.hide_render = True


def update_orchid_geometry(context, props, find_root):
    from .succulents import _child
    sp = ORCHID_CATALOG.get(props.orchid_species)
    if sp is None:
        print("[PPG] No valid orchid preset selected")
        return None
    root = find_root(context, "PPG_" + sp.scientific_name.split(" (")[0].replace(" ", "_"))
    for c in root.children:
        c.hide_viewport = c.hide_render = True
    prof = orchid_from_props(props)
    mat = None
    if props.assign_materials:
        k = _key(prof)
        mat = bpy.data.materials.get(root.name + "_OrchidMat")
        if mat is None or mat.get("ppg_key") != k:
            mat = orchid_material(root.name + "_OrchidMat", prof)
            mat["ppg_key"] = k
    res = OrchidEngine(prof).generate(seed=props.seed, detail=props.succ_detail, with_roots=props.show_roots)
    for suffix, m in zip(SUFFIXES, (res.leaves, res.stems, res.roots if props.show_roots else None, res.flowers,
                                    res.spikes, res.support)):
        _child(context, root, suffix, m, mat)
    root["leaves"] = res.stats["leaves"]
    root["flowers"] = res.stats["flowers"]
    root["scientific_name"] = sp.scientific_name
    root["common_name"] = sp.common_name
    root["family"] = sp.family
    root["growth_form"] = "Orchid"
    return root
