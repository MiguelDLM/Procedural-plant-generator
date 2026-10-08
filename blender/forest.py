"""
Forest pipeline: a library of unique tree variants per species, scattered with native Geometry Nodes.

1. Build Library: for every species in the forest list, N variants are generated with their own seed,
   intraspecific trait variation (trait-space mutation) and size (DBH drawn around the species value);
   each variant is a collection (wood + instanced foliage) under "PPG Forest Library", excluded from
   the view layer so it is only seen through instances. Variants of one species share materials.
2. Scatter: a "PPG Forest" object carries the "PPG Forest Scatter" Geometry Nodes group: Poisson-disk
   points on a surface (min spacing, density in trees/ha), slope filter, random variant, Z rotation and
   scale, and a viewport display fraction. Everything stays editable in the modifier panel, and the
   library collection can be used in any other Geometry Nodes setup.
"""

import copy
import math
import time

import numpy as np

try:
    import bpy
    from bpy.types import Operator, PropertyGroup, UIList
    from bpy.props import StringProperty, IntProperty, FloatProperty, EnumProperty, BoolProperty
    BLENDER_AVAILABLE = True
except ImportError:
    BLENDER_AVAILABLE = False
    Operator = PropertyGroup = UIList = object

try:
    from ..core.species_db import get_species_preset, get_preset_names
    from ..core.plant_pipeline import BotanicalPlantPipeline
    from ..core.gielis import GielisProfile
    from ..core.trait_space import mutate
    from .mesh_builder import BlenderMeshBuilder
except (ImportError, ValueError):
    from core.species_db import get_species_preset, get_preset_names
    from core.plant_pipeline import BotanicalPlantPipeline
    from core.gielis import GielisProfile
    from core.trait_space import mutate
    from blender.mesh_builder import BlenderMeshBuilder

LIBRARY = "PPG Forest Library"
SCATTER_GROUP = "PPG Forest Scatter"
SCATTER_VERSION = 2


# -----------------------------------------------------------------------------
# Properties
# -----------------------------------------------------------------------------
if BLENDER_AVAILABLE:
    class PPG_ForestSpecies(PropertyGroup):
        species: EnumProperty(name="Species", items=get_preset_names())
        variants: IntProperty(name="Variants", default=4, min=1, max=32,
                              description="Unique trees generated for this species (also its share of the mix)")
        use_sliders: BoolProperty(name="Use Current Sliders", default=False,
                                  description="Build from the current panel settings instead of the catalogue preset")
else:
    PPG_ForestSpecies = None


def forest_properties() -> dict:
    if not BLENDER_AVAILABLE:
        return {}
    return {
        "forest_species": bpy.props.CollectionProperty(type=PPG_ForestSpecies),
        "forest_index": IntProperty(default=0),
        "forest_variation": FloatProperty(name="Trait Variation", default=0.6, min=0.0, max=3.0,
                                          description="Intraspecific variability of each variant (trait CVs)"),
        "forest_size_jitter": FloatProperty(name="Size Variation", default=0.3, min=0.0, max=0.8,
                                            description="Relative spread of stem diameter (and so height) "
                                                        "among variants"),
        "forest_quality": EnumProperty(name="Variant Quality", default='TUBES', items=[
            ('TUBES', "Fast", "Tube junctions: for large forests"),
            ('FUSED', "Fused", "Fused stem, limbs and roots: for mid-ground trees")]),
        "forest_leaf_budget": IntProperty(name="Leaf Cards per Tree", default=60000, min=500, max=200000),
        "forest_roots": BoolProperty(name="Root Flare", default=False,
                                     description="Include the visible root collar (slower, mostly buried)"),
        "forest_density": FloatProperty(name="Density (trees/ha)", default=250.0, min=1.0, max=5000.0),
        "forest_spacing": FloatProperty(name="Min Spacing (m)", default=5.0, min=0.2, max=50.0,
                                        description="Poisson-disk minimum distance between trunks"),
        "forest_area": FloatProperty(name="Area Size (m)", default=100.0, min=5.0, max=5000.0,
                                     description="Side of the ground plane created when no surface is selected"),
        "forest_max_slope": FloatProperty(name="Max Slope (deg)", default=35.0, min=0.0, max=90.0),
        "forest_scale_min": FloatProperty(name="Scale Min", default=0.85, min=0.1, max=3.0),
        "forest_scale_max": FloatProperty(name="Scale Max", default=1.15, min=0.1, max=3.0),
        "forest_viewport": FloatProperty(name="Viewport Display", default=0.35, min=0.0, max=1.0,
                                         description="Fraction of trees shown in the viewport (all render)"),
        "forest_seed": IntProperty(name="Forest Seed", default=1, min=0, max=999999),
        "forest_assets": BoolProperty(name="Mark Variants as Assets", default=False,
                                      description="Make every variant collection available in the Asset Browser"),
    }


# -----------------------------------------------------------------------------
# Library
# -----------------------------------------------------------------------------
def _library(context):
    lib = bpy.data.collections.get(LIBRARY)
    if lib is None:
        lib = bpy.data.collections.new(LIBRARY)
        context.scene.collection.children.link(lib)
    return lib


def _exclude(context, coll, exclude=True):
    def walk(lc):
        if lc.collection == coll:
            lc.exclude = exclude
            return True
        return any(walk(c) for c in lc.children)
    walk(context.view_layer.layer_collection)


def _clear_species(lib, species_name: str):
    for c in list(lib.children):
        if c.get("ppg_species") == species_name:
            for o in list(c.all_objects):
                bpy.data.objects.remove(o, do_unlink=True)
            bpy.data.collections.remove(c)


def _move_tree(root, coll, context):
    objs = [root] + list(root.children_recursive)
    for o in objs:
        for c in list(o.users_collection):
            c.objects.unlink(o)
        coll.objects.link(o)


def build_variant(context, props, key: str, variant: int, base_preset, gielis, rng):
    """One unique tree: own seed, mutated traits, size drawn around the species DBH."""
    from .runtime import _ensure_bark_material, _ensure_leaf_material, _assign
    preset = copy.deepcopy(base_preset)
    vseed = int(rng.integers(0, 10 ** 6))
    if props.forest_variation > 0:
        preset = mutate(preset, props.forest_variation, seed=vseed)
    dbh = float(preset.allometry.dbh_default_m * rng.uniform(1.0 - props.forest_size_jitter,
                                                              1.0 + props.forest_size_jitter))
    shoot = props.shoot_leaves if props.foliage_unit == 'SHOOT' else 0
    result = BotanicalPlantPipeline(preset).generate(
        dbh_m=dbh, leaf_density=props.leaf_density, seed=vseed, leaf_budget=props.forest_leaf_budget,
        shoot_leaves=shoot, roots=props.forest_roots, root_display_depth_m=0.6)
    root = bpy.data.objects.new(f"PPG_{preset.scientific_name.replace(' ', '_')}_v{variant + 1}", None)
    context.scene.collection.objects.link(root)
    builder = BlenderMeshBuilder(result, radial_resolution=min(props.radial_resolution, 10),
                                 buttress_profile=gielis, twig_resolution=min(props.twig_resolution, 4))
    built = builder.build_or_update_plant(
        context=context, existing_root=root, leaf_density=props.leaf_density, leaf_scale=props.leaf_scale,
        show_leaves=True, use_subsurf=False, show_roots=props.forest_roots,
        junction_quality=props.forest_quality, fuse_detail=min(props.fuse_detail, 8.0),
        fuse_smoothing=props.fuse_smoothing, foliage_instanced=True)
    root.location = (0.0, 0.0, 0.0)
    if props.assign_materials:
        # Variants of a species share its bark and leaf materials (and leaf textures): the variation lives
        # in the geometry, so a forest needs one texture set per species, not per tree
        bark = _ensure_bark_material(root, base_preset, False)
        _assign(built["wood"], bark)
        _assign(built.get("roots"), bark)
        leaf = _ensure_leaf_material(root, base_preset, props, result.leaf_engine.shape_model,
                                     result.leaf_engine.shoot_leaves)
        _assign(built["foliage"], leaf)
        _assign(built.get("leaf_card"), leaf)
    root["ppg_variant_seed"] = vseed
    root["dbh_m"] = dbh
    return root, result


def build_library(context, props, report=print):
    lib = _library(context)
    _exclude(context, lib, False)
    t0 = time.time()
    rng_all = np.random.default_rng(props.forest_seed)
    n = 0
    for item in props.forest_species:
        if item.use_sliders and item.species == props.species_enum:
            from .runtime import build_custom_preset_from_props
            base, gielis = build_custom_preset_from_props(props)
        else:
            base = copy.deepcopy(get_species_preset(item.species))
            gielis = GielisProfile(m=float(base.allometry.buttress_lobes), n1=0.5, n2=1.8, n3=1.8)
        _clear_species(lib, base.scientific_name)
        rng = np.random.default_rng(rng_all.integers(0, 2 ** 31))
        for v in range(item.variants):
            coll = bpy.data.collections.new(f"{base.scientific_name} v{v + 1}")
            coll["ppg_species"] = base.scientific_name
            lib.children.link(coll)
            root, _ = build_variant(context, props, item.species, v, base, gielis, rng)
            _move_tree(root, coll, context)
            if props.forest_assets and hasattr(coll, "asset_mark"):
                coll.asset_mark()
                coll.asset_data.tags.new(base.scientific_name)
            n += 1
    _exclude(context, lib, True)
    if "ppg_active_root_name" in context.scene:   # Library trees must not be edited by the sliders
        del context.scene["ppg_active_root_name"]
    report(f"Forest library: {n} variants in {time.time() - t0:.1f} s")
    return lib


# -----------------------------------------------------------------------------
# Geometry Nodes scatter
# -----------------------------------------------------------------------------
def _socket(group, name, kind, default=None, lo=None, hi=None, in_out='INPUT'):
    s = group.interface.new_socket(name, in_out=in_out, socket_type=kind)
    if default is not None and hasattr(s, "default_value"):
        s.default_value = default
    if lo is not None and hasattr(s, "min_value"):
        s.min_value = lo
    if hi is not None and hasattr(s, "max_value"):
        s.max_value = hi
    return s


def scatter_group():
    """Native GN group: surface -> Poisson points -> slope filter -> random variant instances."""
    g = bpy.data.node_groups.get(SCATTER_GROUP)
    if g is not None and g.get("ppg_version") == SCATTER_VERSION:
        return g
    if g is not None:
        g.name = SCATTER_GROUP + " (old)"
    g = bpy.data.node_groups.new(SCATTER_GROUP, 'GeometryNodeTree')
    g["ppg_version"] = SCATTER_VERSION
    _socket(g, "Geometry", 'NodeSocketGeometry')
    _socket(g, "Surface", 'NodeSocketObject')
    _socket(g, "Library", 'NodeSocketCollection')
    _socket(g, "Density (trees/ha)", 'NodeSocketFloat', 250.0, 0.0, 10000.0)
    _socket(g, "Min Spacing", 'NodeSocketFloat', 5.0, 0.0, 100.0)
    _socket(g, "Max Slope", 'NodeSocketFloat', 35.0, 0.0, 90.0)
    _socket(g, "Scale Min", 'NodeSocketFloat', 0.85, 0.01, 10.0)
    _socket(g, "Scale Max", 'NodeSocketFloat', 1.15, 0.01, 10.0)
    _socket(g, "Viewport Display", 'NodeSocketFloat', 0.35, 0.0, 1.0)
    _socket(g, "Seed", 'NodeSocketInt', 1, 0, 1000000)
    _socket(g, "Geometry", 'NodeSocketGeometry', in_out='OUTPUT')
    N, L = g.nodes, g.links
    gi = N.new('NodeGroupInput')
    gi.location = (-1400, 0)
    go = N.new('NodeGroupOutput')
    go.location = (1400, 0)
    I = {s.name: s for s in gi.outputs}

    def math(op, a, b=None, loc=(0, 0)):
        m = N.new('ShaderNodeMath')
        m.operation = op
        m.location = loc
        for i, v in enumerate((a, b)):
            if v is None:
                continue
            if isinstance(v, (int, float)):
                m.inputs[i].default_value = v
            else:
                L.new(v, m.inputs[i])
        return m.outputs[0]

    info = N.new('GeometryNodeObjectInfo')
    info.location = (-1150, 300)
    info.transform_space = 'RELATIVE'
    L.new(I["Surface"], info.inputs['Object'])
    dist = N.new('GeometryNodeDistributePointsOnFaces')
    dist.location = (-900, 300)
    dist.distribute_method = 'POISSON'
    L.new(info.outputs['Geometry'], dist.inputs['Mesh'])
    L.new(I["Min Spacing"], dist.inputs['Distance Min'])
    L.new(math('DIVIDE', I["Density (trees/ha)"], 10000.0, (-1150, 100)), dist.inputs['Density Max'])
    L.new(I["Seed"], dist.inputs['Seed'])
    # Slope filter: delete points whose surface normal is steeper than Max Slope
    sep = N.new('ShaderNodeSeparateXYZ')
    sep.location = (-900, 0)
    L.new(dist.outputs['Normal'], sep.inputs[0])
    cos_max = math('COSINE', math('RADIANS', I["Max Slope"], None, (-1150, -150)), None, (-1000, -150))
    steep = math('LESS_THAN', sep.outputs['Z'], cos_max, (-750, 0))
    # Viewport thinning: only a fraction of the trees in the viewport, all of them in renders
    isv = N.new('GeometryNodeIsViewport')
    isv.location = (-900, -300)
    rnd_b = N.new('FunctionNodeRandomValue')
    rnd_b.data_type = 'BOOLEAN'
    rnd_b.location = (-900, -450)
    L.new(math('SUBTRACT', 1.0, I["Viewport Display"], (-1150, -450)), rnd_b.inputs['Probability'])
    L.new(I["Seed"], rnd_b.inputs['Seed'])
    thin = N.new('FunctionNodeBooleanMath')
    thin.operation = 'AND'
    thin.location = (-700, -350)
    L.new(isv.outputs[0], thin.inputs[0])
    L.new(rnd_b.outputs[3], thin.inputs[1])
    drop = N.new('FunctionNodeBooleanMath')
    drop.operation = 'OR'
    drop.location = (-550, -150)
    L.new(steep, drop.inputs[0])
    L.new(thin.outputs[0], drop.inputs[1])
    dele = N.new('GeometryNodeDeleteGeometry')
    dele.location = (-400, 250)
    L.new(dist.outputs['Points'], dele.inputs['Geometry'])
    L.new(drop.outputs[0], dele.inputs['Selection'])
    # Instances: random variant, Z rotation and scale (trees stay upright on slopes)
    coll = N.new('GeometryNodeCollectionInfo')
    coll.location = (-400, -100)
    coll.inputs['Separate Children'].default_value = True
    coll.inputs['Reset Children'].default_value = True
    L.new(I["Library"], coll.inputs['Collection'])
    rnd_i = N.new('FunctionNodeRandomValue')
    rnd_i.data_type = 'INT'
    rnd_i.location = (-400, -350)
    rnd_i.inputs[4].default_value = 0              # Integer Min / Max (Random Value has one pair per type)
    rnd_i.inputs[5].default_value = 100000
    L.new(math('ADD', I["Seed"], 17.0, (-600, -500)), rnd_i.inputs['Seed'])
    rnd_r = N.new('FunctionNodeRandomValue')
    rnd_r.data_type = 'FLOAT'
    rnd_r.location = (-200, -500)
    rnd_r.inputs[2].default_value = 0.0
    rnd_r.inputs[3].default_value = 2.0 * math_pi()
    L.new(math('ADD', I["Seed"], 31.0, (-400, -650)), rnd_r.inputs['Seed'])
    comb = N.new('ShaderNodeCombineXYZ')
    comb.location = (0, -500)
    L.new(rnd_r.outputs[1], comb.inputs['Z'])
    e2r = N.new('FunctionNodeEulerToRotation')
    e2r.location = (200, -500)
    L.new(comb.outputs[0], e2r.inputs[0])
    rnd_s = N.new('FunctionNodeRandomValue')
    rnd_s.data_type = 'FLOAT'
    rnd_s.location = (0, -750)
    L.new(I["Scale Min"], rnd_s.inputs[2])
    L.new(I["Scale Max"], rnd_s.inputs[3])
    L.new(math('ADD', I["Seed"], 53.0, (-200, -850)), rnd_s.inputs['Seed'])
    inst = N.new('GeometryNodeInstanceOnPoints')
    inst.location = (500, 100)
    L.new(dele.outputs['Geometry'], inst.inputs['Points'])
    L.new(coll.outputs['Instances'], inst.inputs['Instance'])
    inst.inputs['Pick Instance'].default_value = True
    L.new(rnd_i.outputs[2], inst.inputs['Instance Index'])
    L.new(e2r.outputs[0], inst.inputs['Rotation'])
    L.new(rnd_s.outputs[1], inst.inputs['Scale'])
    L.new(inst.outputs['Instances'], go.inputs[0])
    return g


def math_pi():
    return math.pi


def _set_input(mod, name, value):
    for it in mod.node_group.interface.items_tree:
        if getattr(it, "in_out", "") == 'INPUT' and it.name == name:
            mod[it.identifier] = value
            return


def scatter(context, props, surface=None):
    lib = bpy.data.collections.get(LIBRARY)
    if lib is None or not lib.children:
        raise RuntimeError("Build the forest library first")
    if surface is None:
        surface = bpy.data.objects.get("PPG_Forest_Ground")
        if surface is None:
            s = props.forest_area * 0.5
            me = bpy.data.meshes.new("PPG_Forest_Ground")
            me.from_pydata([(-s, -s, 0), (s, -s, 0), (s, s, 0), (-s, s, 0)], [], [(0, 1, 2, 3)])
            surface = bpy.data.objects.new("PPG_Forest_Ground", me)
            context.scene.collection.objects.link(surface)
        else:
            s = props.forest_area * 0.5
            surface.data.vertices.foreach_set("co", [-s, -s, 0, s, -s, 0, s, s, 0, -s, s, 0])
            surface.data.update()
    forest = bpy.data.objects.get("PPG_Forest")
    if forest is None:
        forest = bpy.data.objects.new("PPG_Forest", bpy.data.meshes.new("PPG_Forest"))
        context.scene.collection.objects.link(forest)
    forest.matrix_world = surface.matrix_world.copy()
    mod = forest.modifiers.get("PPG Forest") or forest.modifiers.new("PPG Forest", 'NODES')
    mod.node_group = scatter_group()
    for name, value in (("Surface", surface), ("Library", lib), ("Density (trees/ha)", props.forest_density),
                        ("Min Spacing", props.forest_spacing), ("Max Slope", props.forest_max_slope),
                        ("Scale Min", props.forest_scale_min), ("Scale Max", props.forest_scale_max),
                        ("Viewport Display", props.forest_viewport), ("Seed", props.forest_seed)):
        _set_input(mod, name, value)
    forest.update_tag()
    return forest


# -----------------------------------------------------------------------------
# Operators and UI
# -----------------------------------------------------------------------------
class PPG_OT_ForestAddSpecies(Operator):
    """Add the current species to the forest mix"""
    bl_idname = "ppg.forest_add_species"
    bl_label = "Add Species"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        p = context.scene.ppg_properties
        it = p.forest_species.add()
        it.species = p.species_enum
        p.forest_index = len(p.forest_species) - 1
        return {'FINISHED'}


class PPG_OT_ForestRemoveSpecies(Operator):
    """Remove the selected species from the forest mix"""
    bl_idname = "ppg.forest_remove_species"
    bl_label = "Remove Species"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        p = context.scene.ppg_properties
        if 0 <= p.forest_index < len(p.forest_species):
            p.forest_species.remove(p.forest_index)
            p.forest_index = max(0, p.forest_index - 1)
        return {'FINISHED'}


class PPG_OT_ForestBuildLibrary(Operator):
    """Generate the unique tree variants of every species in the forest mix"""
    bl_idname = "ppg.forest_build_library"
    bl_label = "Build Variants"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        p = context.scene.ppg_properties
        if not p.forest_species:
            self.report({'WARNING'}, "Add at least one species to the forest mix")
            return {'CANCELLED'}
        build_library(context, p, report=lambda m: self.report({'INFO'}, m))
        return {'FINISHED'}


class PPG_OT_ForestScatter(Operator):
    """Scatter the variant library on the selected surface (or a new ground plane) with Geometry Nodes"""
    bl_idname = "ppg.forest_scatter"
    bl_label = "Scatter Forest"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        p = context.scene.ppg_properties
        obj = context.active_object
        surface = obj if (obj is not None and obj.type == 'MESH' and not obj.name.startswith("PPG_")) else None
        try:
            scatter(context, p, surface)
        except RuntimeError as e:
            self.report({'WARNING'}, str(e))
            return {'CANCELLED'}
        return {'FINISHED'}


class PPG_OT_ForestCreate(Operator):
    """Build the variant library and scatter the forest in one step"""
    bl_idname = "ppg.forest_create"
    bl_label = "Create Forest"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        p = context.scene.ppg_properties
        if not p.forest_species:
            it = p.forest_species.add()
            it.species = p.species_enum
        obj = context.active_object
        surface = obj if (obj is not None and obj.type == 'MESH' and not obj.name.startswith("PPG_")) else None
        build_library(context, p, report=lambda m: self.report({'INFO'}, m))
        scatter(context, p, surface)
        return {'FINISHED'}


class PPG_UL_ForestSpecies(UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index=0):
        row = layout.row(align=True)
        row.prop(item, "species", text="")
        row.prop(item, "variants", text="")
        row.prop(item, "use_sliders", text="", icon='OPTIONS')


FOREST_CLASSES = (PPG_ForestSpecies, PPG_UL_ForestSpecies, PPG_OT_ForestAddSpecies, PPG_OT_ForestRemoveSpecies,
                  PPG_OT_ForestBuildLibrary, PPG_OT_ForestScatter, PPG_OT_ForestCreate) if BLENDER_AVAILABLE else ()


def draw_forest_panel(layout, p):
    row = layout.row()
    row.template_list("PPG_UL_ForestSpecies", "", p, "forest_species", p, "forest_index", rows=3)
    col = row.column(align=True)
    col.operator("ppg.forest_add_species", icon='ADD', text="")
    col.operator("ppg.forest_remove_species", icon='REMOVE', text="")
    box = layout.box()
    box.label(text="Variants", icon='OUTLINER_COLLECTION')
    c = box.column(align=True)
    for name in ("forest_variation", "forest_size_jitter", "forest_quality", "forest_leaf_budget", "forest_roots",
                 "forest_assets"):
        c.prop(p, name)
    box.operator("ppg.forest_build_library", icon='OUTLINER_OB_GROUP_INSTANCE')
    box = layout.box()
    box.label(text="Scatter (Geometry Nodes)", icon='GEOMETRY_NODES')
    c = box.column(align=True)
    for name in ("forest_density", "forest_spacing", "forest_area", "forest_max_slope", "forest_scale_min",
                 "forest_scale_max", "forest_viewport", "forest_seed"):
        c.prop(p, name)
    box.operator("ppg.forest_scatter", icon='PARTICLES')
    layout.operator("ppg.forest_create", icon='OUTLINER_OB_FORCE_FIELD')
