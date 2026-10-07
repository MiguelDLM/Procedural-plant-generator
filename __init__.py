"""
Procedural Plant Generator - Blender Extension
A scientifically grounded botanical procedural plant and tree generator
backed by empirical datasets (TALLO, Dryad, LeavesBank, PlantCLEF, Plant Ontology, Gielis 2003, Runions et al. 2005).
"""

bl_info = {
    "name": "Procedural Plant Generator",
    "author": "Miguel Diaz de Leon-Munoz",
    "version": (0, 1, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Sidebar > Plant Gen",
    "description": "Data-driven botanical 3D plant and tree generator backed by empirical traits and Plant Ontology",
    "category": "Add Mesh",
}

try:
    import bpy
    from bpy.props import PointerProperty
    BLENDER_AVAILABLE = True
except ImportError:
    BLENDER_AVAILABLE = False

if BLENDER_AVAILABLE:
    from .blender.panel import PPG_Properties, PPG_PT_MainPanel
    from .blender.operators import (
        PPG_OT_GeneratePlant,
        PPG_OT_GenerateLeaf,
        PPG_OT_ExportTraits,
    )

    classes = (
        PPG_Properties,
        PPG_OT_GeneratePlant,
        PPG_OT_GenerateLeaf,
        PPG_OT_ExportTraits,
        PPG_PT_MainPanel,
    )

    def register():
        for cls in classes:
            bpy.utils.register_class(cls)
        bpy.types.Scene.ppg_properties = PointerProperty(type=PPG_Properties)

    def unregister():
        if hasattr(bpy.types.Scene, "ppg_properties"):
            del bpy.types.Scene.ppg_properties
        for cls in reversed(classes):
            bpy.utils.unregister_class(cls)

else:
    def register():
        pass

    def unregister():
        pass


if __name__ == "__main__":
    register()
