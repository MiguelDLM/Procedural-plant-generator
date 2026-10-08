"""
Blender side of vines: properties generated from the core profiles, guide paths (a parametric base shape or
any curve object of the scene, one stem per spline), materials, and live regeneration while the guide curve
is edited.
"""

import copy
import hashlib
import math
from dataclasses import fields
from types import SimpleNamespace

import numpy as np

try:
    import bpy
    from bpy.app.handlers import persistent
    from bpy.props import EnumProperty, FloatProperty, BoolProperty, PointerProperty
    from bpy.types import Operator
    BLENDER_AVAILABLE = True
except ImportError:
    BLENDER_AVAILABLE = False
    Operator = object

    def persistent(f):
        return f

try:
    from ..core.vine import VineEngine, VineProfile, GuideShape, guide_shape, ClimbingMode
    from ..core.vine_db import VINE_CATALOG, VINE_RANGES, leaf_ranges
    from ..core.leaf_morphology import LeafMorphologyProfile
    from ..core.leaf_venation import VenationProfile
    from ..core.inflorescence import FlowerSites
    from .flowers import _props_for
except (ImportError, ValueError):
    from core.vine import VineEngine, VineProfile, GuideShape, guide_shape, ClimbingMode
    from core.vine_db import VINE_CATALOG, VINE_RANGES, leaf_ranges
    from core.leaf_morphology import LeafMorphologyProfile
    from core.leaf_venation import VenationProfile
    from core.inflorescence import FlowerSites
    from blender.flowers import _props_for

VIN, VLF, VVN = "vin_", "vlf_", "vvn_"
SUFFIXES = ("_VineStem", "_Tendrils", "_VineLeaves", "_Fruits", "_VineRoots")
GUIDE_NAME = "VineGuide"          # Not "PPG_...": PPG_ objects are taken for plant roots when selected

LAYOUT = [
    ("Climbing Habit", VIN, ["mode", "chirality", "coil_radius_cm", "coil_pitch_cm", "wander_cm", "tip_length_cm",
                             "tip_hook"]),
    ("Stem & Nodes", VIN, ["stem_radius_mm", "stem_taper", "internode_cm", "leaf_arrangement", "leaf_size",
                           "young_leaf_size", "expansion_zone_cm", "leaf_facing", "basal_leaf_loss"]),
    ("Lateral Shoots & Roots", VIN, ["branch_probability", "branch_length_cm", "branch_angle_deg", "branch_droop",
                                     "aerial_roots", "rootlet_length_cm"]),
    ("Tendrils", VIN, ["tendril_mode", "tendril_length_cm", "tendril_branches", "tendril_coils", "tendril_coil_mm",
                       "tendril_radius_mm", "tendril_reach"]),
    ("Fruits", VIN, ["fruit_count", "fruit_length_cm", "fruit_diameter_cm", "fruit_widest_position", "fruit_neck",
                     "fruit_ribs", "fruit_rib_depth", "fruit_end_depression", "fruit_stalk_cm", "fruit_color",
                     "fruit_stripe_color", "fruit_stripes", "fruit_mottle", "fruit_gloss"]),
    ("Stem Colour", VIN, ["stem_color", "stem_color_old", "woodiness", "hairiness"]),
    ("Leaf Shape", VLF, ["archetype", "blade_length_cm", "aspect_ratio", "widest_position", "base_angle_deg",
                         "apex_angle_deg", "base_curvature", "apex_curvature", "cordate_depth", "petiole_length_ratio",
                         "petiole_angle_deg", "mean_leaf_angle_deg"]),
    ("Lobes, Teeth & Leaflets", VLF, ["lobe_type", "lobe_count", "lobe_depth", "lobe_angle_deg", "lobe_spread_deg",
                                      "lobe_width", "lobe_roundness", "lobe_apex_angle_deg", "margin_type",
                                      "teeth_count", "tooth_height_ratio", "tooth_skew", "compound_type",
                                      "leaflet_count", "leaflet_angle_deg", "rachis_length_ratio",
                                      "terminal_leaflet", "leaflet_size_gradient"]),
    ("Leaf Surface & Colour", VLF, ["transverse_curl", "longitudinal_droop", "undulation_amplitude", "adaxial_color",
                                    "abaxial_color", "vein_color", "gloss"]),
    ("Leaf Venation", VVN, ["pattern", "secondary_vein_pairs", "divergence_angle_deg", "secondary_curvature",
                            "vein_contrast"]),
]
SECTION = {VIN: "profile", VLF: "leaf", VVN: "venation"}


def _ranges(section):
    if section == "profile":
        return VINE_RANGES
    return {k: v for (sec, k), v in leaf_ranges().items() if sec == section}


def vine_properties(update) -> dict:
    """Blender properties: species traits (vin_, vlf_, vvn_) and the guide controls."""
    if not BLENDER_AVAILABLE:
        return {}
    out = {}
    out.update(_props_for(VineProfile, VIN, _ranges("profile"), update))
    out.update(_props_for(LeafMorphologyProfile, VLF, _ranges("leaf"), update))
    out.update(_props_for(VenationProfile, VVN, _ranges("venation"), update))
    out.update({
        "vine_guide_source": EnumProperty(name="Guide", default='SHAPE', update=update, items=[
            ('SHAPE', "Base Shape", "A parametric guide (pole, arch, spiral, fence, wall, ground)", 'MESH_CONE', 0),
            ('CURVE', "Scene Curve", "Any curve object of the scene; each spline grows one stem", 'CURVE_BEZCURVE', 1)]),
        "vine_guide_shape": EnumProperty(name="Shape", default='Pole', update=update,
                                         items=[(g.value, g.value, "") for g in GuideShape]),
        "vine_guide_height": FloatProperty(name="Height (m)", default=2.0, min=0.05, max=50.0, update=update),
        "vine_guide_width": FloatProperty(name="Width / Run (m)", default=1.5, min=0.05, max=50.0, update=update),
        "vine_guide_turns": FloatProperty(name="Turns", default=3.0, min=0.25, max=30.0, update=update),
        "vine_curve": PointerProperty(name="Curve", type=bpy.types.Object, update=update,
                                      poll=lambda self, obj: obj.type == 'CURVE',
                                      description="Curve that guides the stems (edit it to reshape the vine)"),
        "vine_growth": FloatProperty(name="Growth", default=1.0, min=0.0, max=1.0, update=update, subtype='FACTOR',
                                     description="How much of the guide the stems have covered (animatable)"),
        "vine_flip_side": BoolProperty(name="Flip Support Side", default=False, update=update,
                                       description="Clinging plants and planar guides: put the support on the "
                                                   "other side of the stem"),
        "vine_leaf_density": FloatProperty(name="Leaf Density", default=1.0, min=0.0, max=1.0, update=update,
                                           subtype='FACTOR', description="Fraction of the nodes' leaves shown"),
    })
    return out


# -----------------------------------------------------------------------------
# Presets <-> properties
# -----------------------------------------------------------------------------
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


DEFAULT_SHAPE = {"Twining": ("Pole", 2.0, 1.5), "Tendril": ("Arch", 2.0, 1.5), "Clinging": ("Wall", 2.2, 1.5),
                 "Trailing": ("Ground", 0.3, 3.0)}


def write_vine_to_props(props, key):
    sp = VINE_CATALOG.get(key)
    if sp is None:
        return
    _write(props, sp.profile, VIN)
    _write(props, sp.leaf, VLF)
    _write(props, sp.venation, VVN)
    # With a base shape, switch to the one that suits the climbing mode (a runner on the soil, a twiner on a
    # pole...); a curve chosen by the user is never replaced
    if getattr(props, "vine_guide_source", 'SHAPE') == 'SHAPE':
        shape, h, w = DEFAULT_SHAPE[sp.profile.mode.value]
        if props.vine_guide_shape != shape:
            props.vine_guide_shape = shape
            props.vine_guide_height = h
            props.vine_guide_width = w


def vine_from_props(props):
    """(profile, leaf, venation) of the current sliders, on top of the selected preset."""
    sp = VINE_CATALOG[props.vine_species]
    return (_read(props, copy.deepcopy(sp.profile), VIN), _read(props, copy.deepcopy(sp.leaf), VLF),
            _read(props, copy.deepcopy(sp.venation), VVN))


def current_vine_preset(props):
    sp = copy.deepcopy(VINE_CATALOG[props.vine_species])
    sp.profile, sp.leaf, sp.venation = vine_from_props(props)
    return sp


def section_values(props, prefix) -> dict:
    cls = {VIN: VineProfile, VLF: LeafMorphologyProfile, VVN: VenationProfile}[prefix]
    return {f.name: getattr(props, prefix + f.name) for f in fields(cls) if hasattr(props, prefix + f.name)}


# -----------------------------------------------------------------------------
# Guides
# -----------------------------------------------------------------------------
def curve_polylines(obj, depsgraph=None) -> list:
    """World-space polylines of a curve object, one per spline, in spline order. Uses the evaluated curve
    (reflects edit-mode changes, modifiers and resolution); falls back to the spline points when the curve
    has a bevel or extrusion (its evaluated mesh is a surface)."""
    out = []
    mw = np.array(obj.matrix_world)
    data = obj.data
    surface = data.bevel_depth > 0 or data.extrude > 0 or data.bevel_mode == 'OBJECT' and data.bevel_object
    if depsgraph is not None and not surface:
        ev = obj.evaluated_get(depsgraph)
        try:
            me = ev.to_mesh()
        except RuntimeError:
            me = None
        if me is not None:
            n = len(me.vertices)
            co = np.zeros(n * 3)
            me.vertices.foreach_get("co", co)
            co = co.reshape(-1, 3)
            ed = np.zeros(len(me.edges) * 2, dtype=np.int64)
            me.edges.foreach_get("vertices", ed)
            ev.to_mesh_clear()
            out = _chains(co, ed.reshape(-1, 2))
    if not out:
        for sp in data.splines:
            if sp.type == 'BEZIER':
                pts = _bezier_spline(sp, max(4, data.resolution_u))
            else:
                pts = np.array([p.co[:3] for p in sp.points])
            if len(pts) >= 2:
                out.append(pts)
    return [(np.c_[P, np.ones(len(P))] @ mw.T)[:, :3] for P in out]


def _bezier_spline(sp, res):
    from mathutils.geometry import interpolate_bezier
    bp = list(sp.bezier_points)
    segs = list(zip(bp[:-1], bp[1:])) + ([(bp[-1], bp[0])] if sp.use_cyclic_u and len(bp) > 1 else [])
    pts = []
    for a, b in segs:
        seg = interpolate_bezier(a.co, a.handle_right, b.handle_left, b.co, res + 1)
        pts.extend([tuple(v) for v in (seg if not pts else seg[1:])])
    return np.array(pts) if pts else np.array([tuple(bp[0].co)]) if bp else np.zeros((0, 3))


def _chains(co, edges) -> list:
    """Ordered vertex chains of a polyline mesh (each connected component), starting at the lowest-index
    end so that the spline direction (start = base of the stem) is kept."""
    n = len(co)
    adj = [[] for _ in range(n)]
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    seen = np.zeros(n, bool)
    out = []
    ends = [i for i in range(n) if len(adj[i]) == 1]
    for start in ends + list(range(n)):
        if seen[start] or not adj[start]:
            continue
        chain, prev, cur = [start], -1, start
        seen[start] = True
        while True:
            nxt = [v for v in adj[cur] if v != prev and not seen[v]]
            if not nxt:
                break
            prev, cur = cur, nxt[0]
            seen[cur] = True
            chain.append(cur)
        if len(chain) >= 2:
            out.append(co[chain])
    return out


def guides_for(context, props, root) -> list:
    """Guide polylines in the root's local space."""
    if props.vine_guide_source == 'CURVE':
        obj = props.vine_curve
        if obj is None or obj.type != 'CURVE':
            return []
        world = curve_polylines(obj, context.evaluated_depsgraph_get())
        inv = np.array(root.matrix_world.inverted())
        return [(np.c_[P, np.ones(len(P))] @ inv.T)[:, :3] for P in world]
    return guide_shape(props.vine_guide_shape, props.vine_guide_height, props.vine_guide_width,
                       props.vine_guide_turns)


CONTROL_POINTS = {"Pole": 3, "Arch": 7, "Fence": 9, "Wall": 9, "Ground": 8}


def make_guide_curve(context, props, root):
    """Bakes the base shape into an editable Bezier curve placed like the plant, and selects it as guide."""
    P = guide_shape(props.vine_guide_shape, props.vine_guide_height, props.vine_guide_width,
                    props.vine_guide_turns)[0]
    n = CONTROL_POINTS.get(props.vine_guide_shape, int(props.vine_guide_turns * 6) + 1)
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    t = np.linspace(0.0, s[-1], max(2, n))
    C = np.stack([np.interp(t, s, P[:, k]) for k in range(3)], 1)
    cu = bpy.data.curves.new(GUIDE_NAME, 'CURVE')
    cu.dimensions = '3D'
    cu.resolution_u = 16
    sp = cu.splines.new('BEZIER')
    sp.bezier_points.add(len(C) - 1)
    for bp, c in zip(sp.bezier_points, C):
        bp.co = c
        bp.handle_left_type = bp.handle_right_type = 'AUTO'
    obj = bpy.data.objects.new(GUIDE_NAME, cu)
    context.scene.collection.objects.link(obj)
    obj.matrix_world = root.matrix_world.copy()
    return obj


# -----------------------------------------------------------------------------
# Materials
# -----------------------------------------------------------------------------
def _key(*parts) -> str:
    return hashlib.sha1(repr(parts).encode()).hexdigest()[:12]


def stem_material(name, p: VineProfile):
    from .succulents import _new_mat, _attr, _mix, _noise
    from .materials import _srgb_to_linear, _set
    mat, nt, b = _new_mat(name)
    young = nt.nodes.new('ShaderNodeRGB')
    young.location = (-700, 200)
    young.outputs[0].default_value = _srgb_to_linear(p.stem_color)
    old = nt.nodes.new('ShaderNodeRGB')
    old.location = (-700, 0)
    old.outputs[0].default_value = _srgb_to_linear(p.stem_color_old)
    woody = _attr(nt, "woody", (-700, -200))
    col = _mix(nt, woody.outputs['Fac'], young.outputs[0], old.outputs[0], (-450, 100))
    tc = nt.nodes.new('ShaderNodeTexCoord')
    tc.location = (-900, -350)
    nz = _noise(nt, tc.outputs['UV'], 18.0, 4.0, (-650, -350))
    var = _mix(nt, 0.25, col, nz.outputs['Color'], (-250, 0), blend='OVERLAY')
    nt.links.new(var, b.inputs['Base Color'])
    _set(b, 0.55 + 0.3 * p.hairiness, "Roughness")
    if "Sheen Weight" in b.inputs:
        _set(b, 0.8 * p.hairiness, "Sheen Weight")
    return mat


def fruit_material(name, p: VineProfile):
    from .succulents import _new_mat, _attr, _mix, _noise, _math
    from .materials import _srgb_to_linear, _set
    mat, nt, b = _new_mat(name)
    base = nt.nodes.new('ShaderNodeRGB')
    base.location = (-900, 300)
    base.outputs[0].default_value = _srgb_to_linear(p.fruit_color)
    stripe = nt.nodes.new('ShaderNodeRGB')
    stripe.location = (-900, 100)
    stripe.outputs[0].default_value = _srgb_to_linear(p.fruit_stripe_color)
    u = _attr(nt, "fruit_u", (-1300, -100))
    rib = _attr(nt, "fruit_rib", (-1300, -300))
    tc = nt.nodes.new('ShaderNodeTexCoord')
    tc.location = (-1500, -500)
    warp = _noise(nt, tc.outputs['Object'], 6.0, 3.0, (-1300, -500))
    # Longitudinal stripes: cos(2 pi n u) with a noisy, irregular edge (watermelon, squash)
    n = max(p.fruit_ribs, 14)
    a = _math(nt, 'MULTIPLY', u.outputs['Fac'], 2 * math.pi * n, (-1100, -100))
    a = _math(nt, 'MULTIPLY_ADD', warp.outputs['Fac'], 2.5, (-950, -150), c=a)
    s = _math(nt, 'COSINE', a, None, (-800, -150))
    s = _math(nt, 'MULTIPLY_ADD', s, 0.5, (-650, -150), c=0.5)
    s = _math(nt, 'GREATER_THAN', s, 0.55, (-500, -150))
    s = _math(nt, 'MULTIPLY', s, p.fruit_stripes, (-350, -150))
    col = _mix(nt, s, base.outputs[0], stripe.outputs[0], (-250, 200))
    mot = _noise(nt, tc.outputs['Object'], 14.0, 6.0, (-650, -450))
    m = _math(nt, 'MULTIPLY', mot.outputs['Fac'], p.fruit_mottle, (-450, -450))
    col = _mix(nt, m, col, stripe.outputs[0], (-100, 150))
    # Grooves between ribs slightly darker
    g = _math(nt, 'SUBTRACT', 1.0, rib.outputs['Fac'], (-650, -650))
    g = _math(nt, 'MULTIPLY', g, 0.35, (-450, -650))
    dark = nt.nodes.new('ShaderNodeRGB')
    dark.location = (-300, -650)
    dark.outputs[0].default_value = (0.0, 0.0, 0.0, 1.0)
    col = _mix(nt, g, col, dark.outputs[0], (50, 100))
    nt.links.new(col, b.inputs['Base Color'])
    _set(b, 0.75 - 0.6 * p.fruit_gloss, "Roughness")
    return mat


# -----------------------------------------------------------------------------
# Geometry
# -----------------------------------------------------------------------------
def hide_vine_parts(root):
    for c in root.children:
        if c.name.endswith(SUFFIXES):
            c.hide_viewport = c.hide_render = True


def _transparency_depth(scene, leaves):
    """Overlapping alpha-mapped leaf cards exceed Cycles' default 8 transparent bounces where the foliage is
    dense (e.g. at the crown of a pumpkin runner) and render black there. Raise the limit, never lower it."""
    cy = getattr(scene, "cycles", None)
    if leaves and cy is not None and cy.transparent_max_bounces < 32:
        cy.transparent_max_bounces = 32
        print("[PPG] Cycles transparent bounces raised to 32 for the leaf cards")


def update_vine_geometry(context, props, find_root):
    from .succulents import _child, _root_material
    from .runtime import _ensure_leaf_material
    from .flowers import update_flowers_on_plant, hide_flowers, flower_site_count
    sp = VINE_CATALOG.get(props.vine_species)
    if sp is None:
        print("[PPG] No valid vine preset selected")
        return None
    root = find_root(context, f"PPG_{sp.scientific_name.replace(' ', '_').replace(chr(39), '')}")
    keep = SUFFIXES + ("_Flowers", "_FlowerProto")
    for c in root.children:
        if not c.name.endswith(keep):
            c.hide_viewport = c.hide_render = True
    prof, leaf, ven = vine_from_props(props)
    guides = guides_for(context, props, root)
    res = VineEngine(prof, leaf, ven).generate(
        guides, seed=props.seed, growth=props.vine_growth, detail=props.succ_detail,
        leaf_density=props.vine_leaf_density if props.show_leaves else 0.0, with_roots=props.show_roots,
        side_flip=props.vine_flip_side)
    stem = fruit = leaf_mat = None
    if props.assign_materials:
        k = _key(prof.stem_color, prof.stem_color_old, prof.hairiness)
        stem = bpy.data.materials.get(root.name + "_VineStemMat")
        if stem is None or stem.get("ppg_key") != k:
            stem = stem_material(root.name + "_VineStemMat", prof)
            stem["ppg_key"] = k
        if prof.fruit_count > 0:
            k = _key(*(getattr(prof, f.name) for f in fields(prof) if f.name.startswith("fruit_")))
            fruit = bpy.data.materials.get(root.name + "_FruitMat")
            if fruit is None or fruit.get("ppg_key") != k:
                fruit = fruit_material(root.name + "_FruitMat", prof)
                fruit["ppg_key"] = k
        if res.leaf_count:
            leaf_mat = _ensure_leaf_material(root, SimpleNamespace(leaf_morphology=leaf, venation=ven), props,
                                             res.leaf_engine.shape_model, 0)
    _transparency_depth(context.scene, res.leaf_count)
    _child(context, root, "_VineStem", res.stem, stem)
    _child(context, root, "_Tendrils", res.tendrils, stem)
    _child(context, root, "_VineLeaves", res.leaves if props.show_leaves else None, leaf_mat)
    _child(context, root, "_Fruits", res.fruits, fruit)
    _child(context, root, "_VineRoots", res.roots if props.show_roots else None, _root_material(root.name))
    # Flowers in the leaf axils
    if props.show_flowers and len(res.flower_pos):
        rng = np.random.default_rng(props.seed + 5)
        n = flower_site_count(props, int(math.ceil(len(res.flower_pos) / 4)))
        idx = np.sort(rng.choice(len(res.flower_pos), min(n, len(res.flower_pos)), replace=False))
        update_flowers_on_plant(context, root, props, FlowerSites(res.flower_pos[idx], res.flower_dir[idx],
                                                                  np.ones(len(idx))))
    else:
        hide_flowers(root)
    root["scientific_name"] = sp.scientific_name
    root["common_name"] = sp.common_name
    root["family"] = sp.family
    root["growth_form"] = "Vine"
    root["stems"] = res.stem_count
    root["nodes"] = res.node_count
    root["leaf_count"] = res.leaf_count
    root["tendrils"] = res.tendril_count
    root["fruits"] = res.fruit_count
    root["stem_length_m"] = round(res.stats["stem_length_m"], 2)
    if not guides:
        print("[PPG] Vine: no guide (pick a curve object or switch to a base shape)")
    return root


# -----------------------------------------------------------------------------
# Live update while the guide curve is edited
# -----------------------------------------------------------------------------
@persistent
def _on_depsgraph(scene, depsgraph):
    props = getattr(scene, "ppg_properties", None)
    if props is None or props.growth_form != 'Vine' or props.vine_guide_source != 'CURVE' or not props.auto_update:
        return
    curve = props.vine_curve
    if curve is None:
        return
    for upd in depsgraph.updates:
        idv = upd.id.original
        if idv == curve and (upd.is_updated_geometry or upd.is_updated_transform) or idv == curve.data:
            from .runtime import schedule_update, is_updating
            if not is_updating():
                schedule_update(0.15)
            return


# -----------------------------------------------------------------------------
# Operators
# -----------------------------------------------------------------------------
class PPG_OT_VineMakeCurve(Operator):
    """Turn the base shape into an editable Bezier curve and use it as the guide"""
    bl_idname = "ppg.vine_make_curve"
    bl_label = "Convert to Editable Curve"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        from .mesh_builder import BlenderMeshBuilder
        props = context.scene.ppg_properties
        sp = VINE_CATALOG.get(props.vine_species)
        if sp is None:
            self.report({'ERROR'}, "No valid vine preset selected")
            return {'CANCELLED'}
        root = BlenderMeshBuilder._find_root(None, context,
                                             f"PPG_{sp.scientific_name.replace(' ', '_').replace(chr(39), '')}")
        obj = make_guide_curve(context, props, root)
        props.vine_curve = obj
        props.vine_guide_source = 'CURVE'
        self.report({'INFO'}, f"Guide curve '{obj.name}' created: edit it to reshape the vine")
        return {'FINISHED'}


class PPG_OT_VineUseSelected(Operator):
    """Use the active curve object as the vine guide (each spline grows one stem)"""
    bl_idname = "ppg.vine_use_selected"
    bl_label = "Use Active Curve"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj is not None and obj.type == 'CURVE'

    def execute(self, context):
        props = context.scene.ppg_properties
        props.vine_curve = context.active_object
        props.vine_guide_source = 'CURVE'
        return {'FINISHED'}


class PPG_OT_VineEditGuide(Operator):
    """Select the guide curve and enter Edit Mode (the vine follows your edits)"""
    bl_idname = "ppg.vine_edit_guide"
    bl_label = "Edit Guide Curve"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        p = getattr(context.scene, "ppg_properties", None)
        return p is not None and p.vine_curve is not None and context.mode == 'OBJECT'

    def execute(self, context):
        obj = context.scene.ppg_properties.vine_curve
        if obj.name not in context.view_layer.objects:
            self.report({'ERROR'}, "The guide curve is not in the current view layer")
            return {'CANCELLED'}
        for o in context.selected_objects:
            o.select_set(False)
        obj.hide_set(False)
        obj.select_set(True)
        context.view_layer.objects.active = obj
        bpy.ops.object.mode_set(mode='EDIT')
        return {'FINISHED'}


VINE_CLASSES = (PPG_OT_VineMakeCurve, PPG_OT_VineUseSelected, PPG_OT_VineEditGuide) if BLENDER_AVAILABLE else ()


def register_handlers():
    if _on_depsgraph not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(_on_depsgraph)


def unregister_handlers():
    for h in list(bpy.app.handlers.depsgraph_update_post):
        if getattr(h, "__name__", "") == "_on_depsgraph" and getattr(h, "__module__", "") == __name__:
            bpy.app.handlers.depsgraph_update_post.remove(h)
