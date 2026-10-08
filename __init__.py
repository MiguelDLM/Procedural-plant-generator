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
    "description": "Botanical trait-driven trees, succulents, flowers and vines backed by empirical traits and Plant Ontology",
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
    from .blender.preset_io import PRESET_CLASSES, load_library
    from .blender.vines import VINE_CLASSES, register_handlers, unregister_handlers

    # The forest species item type must be registered before the properties that hold a collection of it
    classes = (FOREST_CLASSES[:1] + (PPG_Properties,) + OPERATOR_CLASSES + FOREST_CLASSES[1:] + PRESET_CLASSES
               + VINE_CLASSES + PANEL_CLASSES)

    def register():
        for cls in classes:
            bpy.utils.register_class(cls)
        bpy.types.Scene.ppg_properties = PointerProperty(type=PPG_Properties)
        register_handlers()                       # Vines regenerate while their guide curve is edited
        try:
            n = load_library()                    # User presets join the species menus
            if n:
                print(f"[PPG] {n} user presets loaded")
        except Exception as e:                    # Never block registration on a bad preset folder
            print(f"[PPG] preset library not loaded: {e}")

    def unregister():
        unregister_handlers()
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
