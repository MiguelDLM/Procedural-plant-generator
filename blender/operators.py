"""
Blender Operators for Procedural Plant Generator.
"""

import json
import os

try:
    import bpy
    from bpy.types import Operator
    from bpy.props import StringProperty, FloatProperty, IntProperty, BoolProperty, EnumProperty
    BLENDER_AVAILABLE = True
except ImportError:
    BLENDER_AVAILABLE = False
    Operator = object

from data.species_db import get_species_preset, SPECIES_CATALOG
from core.plant_pipeline import BotanicalPlantPipeline
from blender.mesh_builder import BlenderMeshBuilder
from blender.materials import create_bark_material, create_foliage_material


class PPG_OT_GeneratePlant(Operator):
    """Generate a 3D plant or tree grounded in empirical botanical datasets"""
    bl_idname = "ppg.generate_plant"
    bl_label = "Generate Plant"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        props = context.scene.ppg_properties

        preset = get_species_preset(props.species_enum)
        pipeline = BotanicalPlantPipeline(preset)

        # Execute procedural generation
        result = pipeline.generate(
            dbh_m=props.dbh_m,
            leaf_density=props.leaf_density,
            seed=props.seed
        )

        builder = BlenderMeshBuilder(result)
        built = builder.build_full_plant(context)

        # Assign materials
        if props.assign_materials:
            bark_mat = create_bark_material()
            foliage_mat = create_foliage_material()

            if built["wood"]:
                built["wood"].data.materials.append(bark_mat)
            if built["leaf_master"]:
                built["leaf_master"].data.materials.append(foliage_mat)
            if built["foliage"]:
                built["foliage"].data.materials.append(foliage_mat)

        # Select the newly created plant empty
        bpy.ops.object.select_all(action='DESELECT')
        built["root"].select_set(True)
        context.view_layer.objects.active = built["root"]

        self.report(
            {'INFO'},
            f"Generated {preset.scientific_name}: H={result.total_height_m:.2f}m, CR={result.crown_radius_m:.2f}m, DBH={result.dbh_m*100:.1f}cm"
        )
        return {'FINISHED'}


class PPG_OT_GenerateLeaf(Operator):
    """Generate a high-detail standalone leaf with 3D venation network"""
    bl_idname = "ppg.generate_leaf"
    bl_label = "Generate Macro Leaf"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        props = context.scene.ppg_properties
        preset = get_species_preset(props.species_enum)

        pipeline = BotanicalPlantPipeline(preset)
        result = pipeline.generate(dbh_m=props.dbh_m, seed=props.seed)

        builder = BlenderMeshBuilder(result)

        # Build blade
        leaf_obj = builder.build_leaf_mesh_object(f"Leaf_{preset.scientific_name.replace(' ', '_')}")
        context.scene.collection.objects.link(leaf_obj)

        # Build 3D veins
        vein_obj = builder.build_vein_curve_object(f"Veins_{preset.scientific_name.replace(' ', '_')}")
        context.scene.collection.objects.link(vein_obj)
        vein_obj.parent = leaf_obj

        # Assign materials
        if props.assign_materials:
            foliage_mat = create_foliage_material()
            leaf_obj.data.materials.append(foliage_mat)
            vein_mat = create_foliage_material("Botanical_Vein_PBR")
            vein_mat.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.20, 0.45, 0.10, 1.0)
            vein_obj.data.materials.append(vein_mat)

        bpy.ops.object.select_all(action='DESELECT')
        leaf_obj.select_set(True)
        context.view_layer.objects.active = leaf_obj

        self.report(
            {'INFO'},
            f"Generated Leaf ({preset.scientific_name}): VLA={preset.venation.vla_mm_per_mm2} mm/mm², Pattern={preset.venation.pattern.value}"
        )
        return {'FINISHED'}


class PPG_OT_ExportTraits(Operator):
    """Export the active species empirical traits to a JSON report"""
    bl_idname = "ppg.export_traits"
    bl_label = "Export Botanical Report"
    bl_options = {'REGISTER'}

    filepath: StringProperty(
        name="File Path",
        description="Path to save botanical trait report",
        default="botanical_traits.json",
        subtype='FILE_PATH'
    )

    def execute(self, context):
        props = context.scene.ppg_properties
        preset = get_species_preset(props.species_enum)

        data = {
            "scientific_name": preset.scientific_name,
            "common_name": preset.common_name,
            "family": preset.family,
            "biome": preset.biome,
            "growth_habit": preset.growth_habit,
            "allometry": {
                "height_power_law": f"H = {preset.allometry.height_a} * (DBH_cm)^{preset.allometry.height_b}",
                "crown_radius_law": f"CR = {preset.allometry.crown_radius_c} * (DBH_cm)^{preset.allometry.crown_radius_d}",
                "max_height_m": preset.allometry.height_max_m,
                "pipe_delta": preset.allometry.pipe_exponent_delta,
                "wood_density_g_cm3": preset.allometry.wood_density_g_cm3
            },
            "architecture": {
                "model": preset.architecture.model.value,
                "branch_angle_deg": preset.architecture.branch_angle_mean_deg,
                "phyllotaxis": preset.architecture.phyllotaxis.value,
                "apical_dominance": preset.architecture.apical_dominance,
                "gravitropism": preset.architecture.gravitropism
            },
            "leaf_morphometrics": {
                "archetype": preset.leaf_morphology.archetype.value,
                "margin_type": preset.leaf_morphology.margin_type.value,
                "blade_length_cm": preset.leaf_morphology.blade_length_cm,
                "aspect_ratio": preset.leaf_morphology.aspect_ratio,
                "teeth_count": preset.leaf_morphology.teeth_count
            },
            "venation_network": {
                "pattern": preset.venation.pattern.value,
                "vein_length_per_area_vla": preset.venation.vla_mm_per_mm2,
                "secondary_vein_pairs": preset.venation.secondary_vein_pairs,
                "divergence_angle_deg": preset.venation.divergence_angle_deg
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
