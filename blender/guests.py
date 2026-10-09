"""
Blender side of the guests of a tree: mistletoe clumps (parasites) and epiphytic orchids on its branches.
"""

import hashlib

try:
    import bpy
    from bpy.props import EnumProperty, FloatProperty, IntProperty
    BLENDER_AVAILABLE = True
except ImportError:
    BLENDER_AVAILABLE = False

try:
    from ..core.guests import epiphytic_orchids, mistletoes
    from ..core.mistletoe import MISTLETOE_CATALOG, MistletoeProfile
    from ..core.orchid_db import ORCHID_CATALOG, orchid_items
except (ImportError, ValueError):
    from core.guests import epiphytic_orchids, mistletoes
    from core.mistletoe import MISTLETOE_CATALOG, MistletoeProfile
    from core.orchid_db import ORCHID_CATALOG, orchid_items

SUFFIXES = ("_Mistletoe", "_Epiphytes")


def guest_properties(update) -> dict:
    if not BLENDER_AVAILABLE:
        return {}
    return {
        "guest_mistletoe": IntProperty(name="Mistletoe Clumps", default=0, min=0, max=60, update=update,
                                       description="Hemiparasitic mistletoes on young sunlit branches (zones 4-5)"),
        "mistletoe_species": EnumProperty(name="Mistletoe", update=update, items=[
            (k, f"{v.scientific_name} ({v.common_name})", v.notes) for k, v in MISTLETOE_CATALOG.items()]),
        "mistletoe_age": IntProperty(name="Clump Age (years)", default=6, min=2, max=12, update=update,
                                     description="One forked generation per year"),
        "guest_epiphytes": IntProperty(name="Epiphytic Orchids", default=0, min=0, max=40, update=update,
                                       description="Orchids on the main branches (Johansson zones 3-4)"),
        "epiphyte_species": EnumProperty(name="Epiphyte", update=update,
                                         items=[(k, n, d) for k, n, d in orchid_items()
                                                if ORCHID_CATALOG[k].profile.habit.value != "Climbing"]),
        "epiphyte_scale": FloatProperty(name="Epiphyte Scale", default=1.0, min=0.2, max=3.0, update=update),
    }


def _key(*a):
    return hashlib.sha1(repr(a).encode()).hexdigest()[:12]


def mistletoe_material(name, p: MistletoeProfile):
    from .succulents import _new_mat, _attr, _mix, _math, _noise
    from .materials import _srgb_to_linear, _set
    mat, nt, b = _new_mat(name)

    def rgb(c, loc):
        n = nt.nodes.new('ShaderNodeRGB')
        n.location = loc
        n.outputs[0].default_value = _srgb_to_linear(c)
        return n.outputs[0]
    part = _attr(nt, "mst_part", (-1200, 0)).outputs['Fac']

    def is_part(k, loc):
        lo = _math(nt, 'GREATER_THAN', part, k - 0.5, loc)
        hi = _math(nt, 'LESS_THAN', part, k + 0.5, (loc[0], loc[1] - 60))
        return _math(nt, 'MULTIPLY', lo, hi, (loc[0] + 150, loc[1]))
    col = rgb(p.stem_color, (-900, 400))
    for k, c in ((1, p.leaf_color), (2, p.berry_color), (3, (0.36, 0.30, 0.24))):
        col = _mix(nt, is_part(k, (-900, -200 * k)), col, rgb(c, (-700, 300 - 100 * k)), (-400 + 100 * k, 300))
    tc = nt.nodes.new('ShaderNodeTexCoord')
    tc.location = (-900, -900)
    nz = _noise(nt, tc.outputs['Object'], 80.0, 3.0, (-600, -900))
    col = _mix(nt, 0.12, col, nz.outputs['Color'], (100, 300), blend='OVERLAY')
    nt.links.new(col, b.inputs['Base Color'])
    _set(b, 0.45, "Roughness")
    _set(b, 0.15, "Subsurface Weight", "Subsurface")
    return mat


def hide_guests(root):
    for c in root.children:
        if c.name.endswith(SUFFIXES):
            c.hide_viewport = c.hide_render = True


def update_tree_guests(context, root, props, result):
    """Rebuilds the mistletoes and epiphytes of a tree (hidden when their counts are 0)."""
    from .flowers import _child, _assign
    from .mesh_builder import populate_mesh
    import copy
    skel = result.skeleton_graph
    n_m = int(getattr(props, "guest_mistletoe", 0))
    n_e = int(getattr(props, "guest_epiphytes", 0))
    obj = _child(context, root, "_Mistletoe")
    if n_m > 0 and props.mistletoe_species in MISTLETOE_CATALOG:
        prof = copy.deepcopy(MISTLETOE_CATALOG[props.mistletoe_species].profile)
        prof.generations = props.mistletoe_age
        populate_mesh(obj.data, mistletoes(prof, skel, n_m, seed=props.seed, detail=0.8))
        if props.assign_materials:
            name = root.name + "_MistletoeMat"
            mat = bpy.data.materials.get(name)
            k = _key(prof)
            if mat is None or mat.get("ppg_key") != k:
                mat = mistletoe_material(name, prof)
                mat["ppg_key"] = k
            _assign(obj, mat)
        obj.hide_viewport = obj.hide_render = False
    else:
        obj.data.clear_geometry()
        obj.hide_viewport = obj.hide_render = True
    obj = _child(context, root, "_Epiphytes")
    if n_e > 0 and props.epiphyte_species in ORCHID_CATALOG:
        prof = ORCHID_CATALOG[props.epiphyte_species].profile
        populate_mesh(obj.data, epiphytic_orchids(prof, skel, n_e, seed=props.seed, detail=0.6,
                                                  scale=props.epiphyte_scale))
        if props.assign_materials:
            from .orchids import orchid_material
            name = root.name + "_EpiphyteMat"
            mat = bpy.data.materials.get(name)
            k = _key(prof)
            if mat is None or mat.get("ppg_key") != k:
                mat = orchid_material(name, prof)
                mat["ppg_key"] = k
            _assign(obj, mat)
        obj.hide_viewport = obj.hide_render = False
    else:
        obj.data.clear_geometry()
        obj.hide_viewport = obj.hide_render = True
