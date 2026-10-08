"""
Procedural Plant Generator - Blender Extension
A scientifically grounded botanical procedural plant and tree generator
built on measurable botanical traits: allometry, Halle-Oldeman architecture, leaf-architecture
descriptors (Ellis et al. 2009), hierarchical venation and a normalised trait space.
"""

bl_info = {
    "name": "Procedural Plant Generator",
    "author": "Miguel Diaz de Leon-Munoz",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Sidebar > Plant Gen",
    "description": "Botanical trait-driven trees, succulents and flowers backed by empirical traits and Plant Ontology",
    "category": "Add Mesh",
}

try:
    import bpy
    from bpy.props import PointerProperty
    BLENDER_AVAILABLE = True
except ImportError:
    BLENDER_AVAILABLE = False

if BLENDER_AVAILABLE:
    from .blender.forest import FOREST_CLASSES
    from .blender.panel import PPG_Properties, PANEL_CLASSES
    from .blender.operators import OPERATOR_CLASSES

    # The forest species item type must be registered before the properties that hold a collection of it
    classes = FOREST_CLASSES[:1] + (PPG_Properties,) + OPERATOR_CLASSES + FOREST_CLASSES[1:] + PANEL_CLASSES

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
