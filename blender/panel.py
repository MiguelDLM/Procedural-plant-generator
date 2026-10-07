"""
Blender 3D Viewport UI Panel and Property Groups.
Features an interactive Empirical Botanical Trait Inspector with Plant Ontology grounding.
"""

try:
    import bpy
    from bpy.types import Panel, PropertyGroup
    from bpy.props import EnumProperty, FloatProperty, IntProperty, BoolProperty
    BLENDER_AVAILABLE = True
except ImportError:
    BLENDER_AVAILABLE = False
    Panel = object
    PropertyGroup = object

try:
    from ..core.species_db import get_preset_names, get_species_preset, SPECIES_CATALOG
    from ..core.ontology import get_po_term
except (ImportError, ValueError):
    from core.species_db import get_preset_names, get_species_preset, SPECIES_CATALOG
    from core.ontology import get_po_term


class PPG_Properties(PropertyGroup):
    """Container for active plant generation parameters in the Blender scene."""

    def update_species(self, context):
        preset = get_species_preset(self.species_enum)
        self.dbh_m = preset.allometry.dbh_default_m

    species_enum: EnumProperty(
        name="Species",
        description="Botanical species preset grounded in empirical allometric and trait data",
        items=get_preset_names() if BLENDER_AVAILABLE else [],
        default="quercus_robur",
        update=update_species if BLENDER_AVAILABLE else None
    )

    dbh_m: FloatProperty(
        name="DBH (Diameter)",
        description="Stem Diameter at Breast Height (1.3m) in meters [PO:0004712]",
        default=0.45,
        min=0.02,
        max=4.50,
        step=1.0,
        precision=2,
        unit='LENGTH'
    )

    leaf_density: FloatProperty(
        name="Foliage Density",
        description="Relative density of leaves on terminal branches [PO:0025004]",
        default=1.0,
        min=0.0,
        max=2.0,
        step=10.0
    )

    seed: IntProperty(
        name="Seed",
        description="Random seed for procedural branching variation",
        default=42,
        min=0,
        max=999999
    )

    assign_materials: BoolProperty(
        name="Assign PBR Materials",
        description="Generate and attach botanical shaders with subsurface scattering",
        default=True
    )

    use_geometry_nodes: BoolProperty(
        name="Use Geometry Nodes",
        description="Attach procedural Geometry Nodes modifier for foliage instancing",
        default=True
    )

    show_traits: BoolProperty(
        name="Empirical Trait Inspector",
        description="Display scientific trait values from empirical datasets and Plant Ontology",
        default=True
    )


class PPG_PT_MainPanel(Panel):
    """Main Sidebar Panel in 3D Viewport."""
    bl_label = "Procedural Plant Generator"
    bl_idname = "PPG_PT_main_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Plant Gen'

    def draw(self, context):
        layout = self.layout
        props = context.scene.ppg_properties
        preset = get_species_preset(props.species_enum)

        # -------------------------------------------------------------
        # 1. Species Selection
        # -------------------------------------------------------------
        box = layout.box()
        box.label(text="Botanical Taxon (Plant Ontology)", icon='OUTLINER_OB_ARMATURE')
        box.prop(props, "species_enum", text="")

        col = box.column(align=True)
        col.label(text=f"Family: {preset.family} | Biome: {preset.biome}")
        col.label(text=f"Habit: {preset.growth_habit} [{preset.po_growth_form}]")
        if preset.notes:
            col.label(text=f"Note: {preset.notes[:45]}...")

        # -------------------------------------------------------------
        # 2. Empirical Trait Inspector
        # -------------------------------------------------------------
        trait_box = layout.box()
        row = trait_box.row()
        row.prop(
            props, "show_traits",
            icon="TRIA_DOWN" if props.show_traits else "TRIA_RIGHT",
            icon_only=True,
            emboss=False
        )
        row.label(text="Empirical Trait Inspector", icon='SOLO_ON')

        if props.show_traits:
            col = trait_box.column(align=True)
            col.scale_y = 0.9

            # Allometry box
            col.label(text="Allometry (TALLO & Forests) [PO:0004712]:", icon='DRIVER_DISTANCE')
            col.label(text=f" • H Power Law: H = {preset.allometry.height_a}·D^{preset.allometry.height_b}")
            col.label(text=f" • Crown Radius: CR = {preset.allometry.crown_radius_c}·D^{preset.allometry.crown_radius_d}")
            col.label(text=f" • Leonardo Exponent: Δ = {preset.allometry.pipe_exponent_delta}")

            col.separator()
            # Architecture box
            col.label(text="Architecture (Hallé-Oldeman) [PO:0025029]:", icon='CON_KINEMATIC')
            col.label(text=f" • Model: {preset.architecture.model.value}")
            col.label(text=f" • Branch Angle: {preset.architecture.branch_angle_mean_deg:.1f}°")
            col.label(text=f" • Phyllotaxis: {preset.architecture.phyllotaxis.value}")

            col.separator()
            # Venation and Lamina box
            col.label(text="Leaf & Venation (Dryad & Runions) [PO:0020039]:", icon='MESH_GRID')
            col.label(text=f" • Archetype: {preset.leaf_morphology.archetype.value}")
            col.label(text=f" • Margin [PO:0020042]: {preset.leaf_morphology.margin_type.value}")
            col.label(text=f" • Venation [PO:0005022]: {preset.venation.pattern.value}")
            col.label(text=f" • Vein Density (VLA): {preset.venation.vla_mm_per_mm2} mm/mm²")
            col.label(text=f" • 2nd Vein Pairs: {preset.venation.secondary_vein_pairs}")

            col.separator()
            # Biomechanics box
            col.label(text="Biomechanics & Wood Density [PO:0005352]:", icon='PHYSICS')
            col.label(text=f" • Wood Density: {preset.biomechanics.wood_density_g_cm3} g/cm³")
            col.label(text=f" • Leaf Mass per Area: {preset.biomechanics.leaf_mass_per_area_g_m2} g/m²")

        # -------------------------------------------------------------
        # 3. Morphological & Scaling Sliders
        # -------------------------------------------------------------
        gen_box = layout.box()
        gen_box.label(text="Growth & Allometry Controls", icon='MOD_LENGTH')
        gen_box.prop(props, "dbh_m")
        gen_box.prop(props, "leaf_density")
        gen_box.prop(props, "seed")

        gen_box.separator()
        gen_box.prop(props, "assign_materials")
        gen_box.prop(props, "use_geometry_nodes")

        # -------------------------------------------------------------
        # 4. Action Buttons
        # -------------------------------------------------------------
        layout.separator()
        col = layout.column(align=True)
        col.scale_y = 1.4
        col.operator("ppg.generate_plant", text="Generate Botanical Plant", icon='OUTLINER_OB_SURFACE')

        row = layout.row(align=True)
        row.operator("ppg.generate_leaf", text="Generate Macro Leaf (3D Veins)", icon='CURVE_DATA')
        row.operator("ppg.export_traits", text="Export Botanical Report", icon='TEXT')
