"""
Blender operators for Procedural Plant Generator.
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

    def StringProperty(**kw):
        return None

try:
    from ..core.species_db import get_species_preset, SPECIES_CATALOG
    from ..core.trait_space import to_vector, nearest_species, TRAITS, COLOR_TRAITS, get_path
    from .runtime import (update_tree_geometry, apply_species_preset_to_props, apply_variant_to_props,
                          build_custom_preset_from_props, is_updating, set_updating)
except (ImportError, ValueError):
    from core.species_db import get_species_preset, SPECIES_CATALOG
    from core.trait_space import to_vector, nearest_species, TRAITS, COLOR_TRAITS, get_path
    from blender.runtime import (update_tree_geometry, apply_species_preset_to_props, apply_variant_to_props,
                                 build_custom_preset_from_props, is_updating, set_updating)


def _run_update(context):
    if is_updating():
        return
    set_updating(True)
    try:
        update_tree_geometry(context)
    finally:
        set_updating(False)


class PPG_OT_LiveUpdate(Operator):
    """Internal operator to update tree geometry live in the 3D viewport"""
    bl_idname = "ppg.live_update"
    bl_label = "Live Update Tree"
    bl_options = {'INTERNAL'}

    def execute(self, context):
        _run_update(context)
        return {'FINISHED'}


class PPG_OT_GeneratePlant(Operator):
    """Generate or refresh the active procedural plant"""
    bl_idname = "ppg.generate_plant"
    bl_label = "Update Plant"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        _run_update(context)
        return {'FINISHED'}


class PPG_OT_NewPlant(Operator):
    """Create a new plant at the 3D cursor without replacing the current one"""
    bl_idname = "ppg.new_plant"
    bl_label = "New Plant"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        if "ppg_active_root_name" in context.scene:
            del context.scene["ppg_active_root_name"]
        for obj in context.selected_objects:
            obj.select_set(False)
        context.view_layer.objects.active = None
        _run_update(context)
        return {'FINISHED'}


class PPG_OT_ApplySpeciesPreset(Operator):
    """Reload the selected species' traits into all sliders"""
    bl_idname = "ppg.apply_species_preset"
    bl_label = "Reload Species"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        props = context.scene.ppg_properties
        if props.growth_form == 'Flower':
            from .flowers import write_flower_to_props
            from .runtime import schedule_update
            set_updating(True)
            try:
                write_flower_to_props(props, props.flower_species)
            finally:
                set_updating(False)
            schedule_update()
            self.report({'INFO'}, "Loaded flower preset")
            return {'FINISHED'}
        if props.growth_form != 'Tree':
            from .runtime import apply_succulent_preset
            apply_succulent_preset(props, context)
            self.report({'INFO'}, "Loaded succulent preset")
            return {'FINISHED'}
        apply_species_preset_to_props(props, props.species_enum, context)
        self.report({'INFO'}, f"Loaded {get_species_preset(props.species_enum).scientific_name}")
        return {'FINISHED'}


class PPG_OT_ApplyVariant(Operator):
    """Blend with a second species and/or add intraspecific variation; results are written to the sliders"""
    bl_idname = "ppg.apply_variant"
    bl_label = "Create Variant"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        props = context.scene.ppg_properties
        if props.growth_form == 'Flower':
            from .flowers import flower_variant
            flower_variant(props)
            self.report({'INFO'}, "Flower variant written to the sliders")
            return {'FINISHED'}
        apply_variant_to_props(props, context)
        if props.growth_form != 'Tree':
            self.report({'INFO'}, "Variant written to the sliders")
            return {'FINISHED'}
        preset, _ = build_custom_preset_from_props(props)
        near = nearest_species(preset, SPECIES_CATALOG)[:3]
        self.report({'INFO'}, "Closest species: " + ", ".join(f"{SPECIES_CATALOG[k].scientific_name} ({d:.3f})"
                                                              for k, d in near))
        return {'FINISHED'}


class PPG_OT_ExportTraits(Operator):
    """Export the current plant's traits (and normalised trait vector) to JSON"""
    bl_idname = "ppg.export_traits"
    bl_label = "Export Trait Report"
    bl_options = {'REGISTER'}

    filepath: StringProperty(name="File Path", default="botanical_traits.json", subtype='FILE_PATH')

    def execute(self, context):
        props = context.scene.ppg_properties

        def plain(v):
            return v.value if hasattr(v, "value") else (list(v) if isinstance(v, tuple) else v)

        if props.growth_form == 'Flower':
            from dataclasses import fields as dc_fields
            from .flowers import flower_from_props
            from ..core.flower_db import FLOWER_CATALOG
            f, ip = flower_from_props(props)
            if props.flower_species not in FLOWER_CATALOG:
                self.report({'ERROR'}, "No valid flower preset selected")
                return {'CANCELLED'}
            sp = FLOWER_CATALOG[props.flower_species]
            data = {"growth_form": "Flower", "species": sp.scientific_name, "common_name": sp.common_name,
                    "family": sp.family, "floral_formula": sp.formula,
                    "flower": {x.name: plain(getattr(f, x.name)) for x in dc_fields(f)},
                    "inflorescence": {x.name: plain(getattr(ip, x.name)) for x in dc_fields(ip)}}
            target = bpy.path.abspath(self.filepath)
            with open(target, "w") as fh:
                json.dump(data, fh, indent=2)
            self.report({'INFO'}, f"Saved trait report to {target}")
            return {'FINISHED'}
        if props.growth_form != 'Tree':
            from dataclasses import fields as dc_fields
            from .succulents import profile_from_props
            from .runtime import _succulent_key
            from ..core.succulent_db import CATALOGS
            form, key = _succulent_key(props)
            prof = profile_from_props(props, form, key)
            sp = CATALOGS[form][key]
            data = {"growth_form": form.value, "species": sp.scientific_name, "common_name": sp.common_name,
                    "family": sp.family, "traits": {f.name: plain(getattr(prof, f.name)) for f in dc_fields(prof)}}
            target = bpy.path.abspath(self.filepath)
            with open(target, "w") as f:
                json.dump(data, f, indent=2)
            self.report({'INFO'}, f"Saved trait report to {target}")
            return {'FINISHED'}
        preset, _ = build_custom_preset_from_props(props)

        def plain(v):
            return v.value if hasattr(v, "value") else (list(v) if isinstance(v, tuple) else v)

        data = {
            "species": preset.scientific_name,
            "common_name": preset.common_name,
            "family": preset.family,
            "dimensions": {"dbh_m": props.dbh_m, "height_m": props.tree_height_m,
                           "crown_radius_m": props.crown_radius_m, "clear_bole_m": props.crown_base_height_m},
            "traits": {s.path: plain(get_path(preset, s.path)) for s in TRAITS},
            "colors_srgb": {c: plain(get_path(preset, c)) for c in COLOR_TRAITS},
            "trait_vector": [round(float(x), 5) for x in to_vector(preset)],
            "nearest_species": [{"id": k, "distance": round(d, 4)}
                                for k, d in nearest_species(preset, SPECIES_CATALOG)[:5]],
            "plant_ontology": {"trunk": "PO:0004712", "branch": "PO:0025073", "leaf_lamina": "PO:0020039",
                               "leaf_vein": "PO:0005022"},
        }
        target = bpy.path.abspath(self.filepath)
        with open(target, "w") as f:
            json.dump(data, f, indent=2)
        self.report({'INFO'}, f"Saved trait report to {target}")
        return {'FINISHED'}

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}


OPERATOR_CLASSES = (PPG_OT_LiveUpdate, PPG_OT_GeneratePlant, PPG_OT_NewPlant, PPG_OT_ApplySpeciesPreset,
                    PPG_OT_ApplyVariant, PPG_OT_ExportTraits)
