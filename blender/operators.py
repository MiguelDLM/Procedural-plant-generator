"""
Blender Operators for Procedural Plant Generator.
Supports live real-time procedural updating, continuous quad meshing, and species preset loading.
"""

import json

try:
    import bpy
    from bpy.types import Operator
    from bpy.props import StringProperty
    BLENDER_AVAILABLE = True
except ImportError:
    BLENDER_AVAILABLE = False
    Operator = object

try:
    from ..core.species_db import get_species_preset, SPECIES_CATALOG
    from ..core.allometry import AllometricProfile
    from ..core.architecture import ArchitectureProfile, HalleOldemanModel, PhyllotaxisType
    from ..core.leaf_morphology import LeafMorphologyProfile, LeafArchetype, MarginType
    from ..core.leaf_venation import VenationProfile, VenationPattern
    from ..core.biomechanics import BiomechanicalProfile
    from ..core.species_preset import BotanicalSpeciesPreset
    from ..core.gielis import GielisProfile
    from ..core.plant_pipeline import BotanicalPlantPipeline
    from ..core.ontology import get_po_term
    from .mesh_builder import BlenderMeshBuilder
    from .materials import create_bark_material, create_foliage_material
except (ImportError, ValueError):
    from core.species_db import get_species_preset, SPECIES_CATALOG
    from core.allometry import AllometricProfile
    from core.architecture import ArchitectureProfile, HalleOldemanModel, PhyllotaxisType
    from core.leaf_morphology import LeafMorphologyProfile, LeafArchetype, MarginType
    from core.leaf_venation import VenationProfile, VenationPattern
    from core.biomechanics import BiomechanicalProfile
    from core.species_preset import BotanicalSpeciesPreset
    from core.gielis import GielisProfile
    from core.plant_pipeline import BotanicalPlantPipeline
    from core.ontology import get_po_term
    from blender.mesh_builder import BlenderMeshBuilder
    from blender.materials import create_bark_material, create_foliage_material


def build_custom_preset_from_props(props) -> tuple[BotanicalSpeciesPreset, GielisProfile]:
    """Constructs dynamic botanical profiles directly from the user's interactive sliders."""
    base_preset = get_species_preset(props.species_enum)

    # User-adjusted allometry
    allometry = AllometricProfile(
        dbh_min_m=0.02,
        dbh_max_m=props.dbh_m * 2.0,
        dbh_default_m=props.dbh_m,
        height_max_m=max(props.tree_height_m, base_preset.allometry.height_max_m),
        pipe_exponent_delta=props.pipe_delta,
        crown_depth_ratio=max(0.1, min(0.9, (props.tree_height_m - props.crown_base_height_m) / max(0.5, props.tree_height_m))),
        buttress_amplitude=props.buttress_amplitude,
        buttress_decay=props.buttress_decay,
        wood_density_g_cm3=props.wood_density
    )

    # User-adjusted architecture
    architecture = ArchitectureProfile(
        model=base_preset.architecture.model,
        phyllotaxis=PhyllotaxisType.SPIRAL if abs(props.phyllotaxis_angle_deg - 137.5) < 10.0 else (
            PhyllotaxisType.DECUSSATE if abs(props.phyllotaxis_angle_deg - 90.0) < 10.0 else PhyllotaxisType.DISTICHOUS
        ),
        max_order=props.branch_levels,
        branch_angle_mean_deg=props.branch_angle_deg,
        branch_angle_std_deg=5.0,
        divergence_angle_deg=props.phyllotaxis_angle_deg,
        apical_dominance=props.apical_dominance,
        gravitropism=props.branch_gravity,
        internode_decay_per_order=props.branch_length_decay,
        crookedness=props.crookedness
    )

    # User-adjusted leaf morphology
    leaf_morph = LeafMorphologyProfile(
        archetype=LeafArchetype(props.leaf_archetype),
        margin_type=MarginType(props.margin_type),
        blade_length_cm=props.leaf_length_cm,
        aspect_ratio=props.leaf_aspect_ratio,
        teeth_count=props.teeth_count,
        transverse_curl=0.20,
        longitudinal_droop=props.leaf_droop
    )

    # Gielis buttress cross section profile
    gielis = GielisProfile(
        m=float(props.buttress_m),
        n1=props.gielis_n1,
        n2=props.gielis_n2,
        n3=props.gielis_n3,
        a=1.0,
        b=1.0
    )

    preset = BotanicalSpeciesPreset(
        scientific_name=base_preset.scientific_name,
        common_name=base_preset.common_name,
        family=base_preset.family,
        biome=base_preset.biome,
        growth_habit=base_preset.growth_habit,
        allometry=allometry,
        architecture=architecture,
        leaf_morphology=leaf_morph,
        venation=base_preset.venation,
        biomechanics=base_preset.biomechanics,
        po_growth_form=base_preset.po_growth_form
    )

    return preset, gielis


try:
    from .runtime import (
        update_tree_geometry,
        apply_species_preset_to_props,
        build_custom_preset_from_props,
        is_updating,
        set_updating
    )
except (ImportError, ValueError):
    from blender.runtime import (
        update_tree_geometry,
        apply_species_preset_to_props,
        build_custom_preset_from_props,
        is_updating,
        set_updating
    )


class PPG_OT_LiveUpdate(Operator):
    """Internal operator to update tree geometry live in the 3D viewport"""
    bl_idname = "ppg.live_update"
    bl_label = "Live Update Tree"
    bl_options = {'INTERNAL'}

    def execute(self, context):
        if not is_updating():
            set_updating(True)
            try:
                update_tree_geometry(context)
            finally:
                set_updating(False)
        return {'FINISHED'}


class PPG_OT_GeneratePlant(Operator):
    """Generate or refresh procedural plant mesh in active collection"""
    bl_idname = "ppg.generate_plant"
    bl_label = "Update Botanical Tree"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        if not is_updating():
            set_updating(True)
            try:
                update_tree_geometry(context)
            finally:
                set_updating(False)
        self.report({'INFO'}, "Updated botanical tree.")
        return {'FINISHED'}


class PPG_OT_NewPlant(Operator):
    """Create a brand new procedural plant without overwriting the current one"""
    bl_idname = "ppg.new_plant"
    bl_label = "New Plant"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        if "ppg_active_root_name" in context.scene:
            del context.scene["ppg_active_root_name"]

        for obj in context.selected_objects:
            obj.select_set(False)
        context.view_layer.objects.active = None

        if not is_updating():
            set_updating(True)
            try:
                update_tree_geometry(context)
            finally:
                set_updating(False)

        self.report({'INFO'}, "Created new separate botanical plant.")
        return {'FINISHED'}


class PPG_OT_ApplySpeciesPreset(Operator):
    """Load default empirical allometric values for the selected species into sliders"""
    bl_idname = "ppg.apply_species_preset"
    bl_label = "Load Species Empirical Data"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        props = context.scene.ppg_properties
        apply_species_preset_to_props(props, props.species_enum, context)
        preset = get_species_preset(props.species_enum)
        self.report({'INFO'}, f"Loaded empirical parameters for {preset.scientific_name}")
        return {'FINISHED'}


class PPG_OT_ExportTraits(Operator):
    """Export current active plant parameters to a structured JSON file"""
    bl_idname = "ppg.export_traits"
    bl_label = "Export Trait Report"
    bl_options = {'REGISTER'}

    filepath: StringProperty(
        name="File Path",
        description="Path to save botanical trait report",
        default="botanical_traits.json",
        subtype='FILE_PATH'
    )

    def execute(self, context):
        props = context.scene.ppg_properties
        preset, gielis = build_custom_preset_from_props(props)

        data = {
            "species": preset.scientific_name,
            "family": preset.family,
            "biome": preset.biome,
            "plant_ontology": {
                "trunk": "PO:0004712",
                "branch": "PO:0025073",
                "leaf_lamina": "PO:0020039",
                "leaf_vein": "PO:0005022"
            },
            "parameters": {
                "dbh_m": props.dbh_m,
                "tree_height_m": props.tree_height_m,
                "crown_base_height_m": props.crown_base_height_m,
                "pipe_delta": props.pipe_delta,
                "gielis_buttress_m": props.buttress_m,
                "branch_levels": props.branch_levels,
                "branch_angle_deg": props.branch_angle_deg,
                "branch_gravity": props.branch_gravity,
                "leaf_archetype": props.leaf_archetype,
                "leaf_length_cm": props.leaf_length_cm
            }
        }

        target = bpy.path.abspath(self.filepath)
        with open(target, 'w') as f:
            json.dump(data, f, indent=2)

        self.report({'INFO'}, f"Saved botanical report to {target}")
        return {'FINISHED'}

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}
