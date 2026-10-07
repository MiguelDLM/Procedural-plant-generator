"""
Interactive Blender 3D Viewport UI Panel and Property Groups.
Provides real-time interactive procedural updates with adjustable botanical sliders.
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
except (ImportError, ValueError):
    from core.species_db import get_preset_names, get_species_preset, SPECIES_CATALOG


try:
    from .runtime import is_updating, set_updating, update_tree_geometry, apply_species_preset_to_props
except (ImportError, ValueError):
    from blender.runtime import is_updating, set_updating, update_tree_geometry, apply_species_preset_to_props


def on_param_update(self, context):
    """Callback when any slider changes in the UI - updates tree in real time without operator deadlock."""
    if is_updating():
        return
    if not getattr(self, "auto_update", True):
        return
    set_updating(True)
    try:
        update_tree_geometry(context)
    finally:
        set_updating(False)


def on_species_change(self, context):
    """When species dropdown changes, safely auto-load empirical parameters into sliders."""
    if is_updating():
        return
    apply_species_preset_to_props(self, self.species_enum, context)


class PPG_Properties(PropertyGroup):
    """Interactive parameters for real-time procedural tree generation."""

    auto_update: BoolProperty(
        name="Real-time Live Update",
        description="Update 3D tree in viewport immediately when any slider is adjusted",
        default=True
    )

    species_enum: EnumProperty(
        name="Species Preset",
        description="Empirical botanical species archetype",
        items=get_preset_names() if BLENDER_AVAILABLE else [],
        default="quercus_robur",
        update=on_species_change if BLENDER_AVAILABLE else None
    )

    # -------------------------------------------------------------
    # 1. Trunk & Allometry Controls [PO:0004712]
    # -------------------------------------------------------------
    dbh_m: FloatProperty(
        name="DBH (Trunk Diameter)",
        description="Diameter at Breast Height (1.3m) in meters [PO:0004712]",
        default=0.45,
        min=0.04,
        max=4.50,
        step=1.0,
        precision=2,
        unit='LENGTH',
        update=on_param_update if BLENDER_AVAILABLE else None
    )

    tree_height_m: FloatProperty(
        name="Total Height",
        description="Total tree height in meters",
        default=14.0,
        min=1.0,
        max=95.0,
        step=10.0,
        precision=1,
        unit='LENGTH',
        update=on_param_update if BLENDER_AVAILABLE else None
    )

    crown_base_height_m: FloatProperty(
        name="Trunk Clear Height",
        description="Height from ground to first primary branch",
        default=3.5,
        min=0.2,
        max=45.0,
        step=5.0,
        precision=1,
        unit='LENGTH',
        update=on_param_update if BLENDER_AVAILABLE else None
    )

    pipe_delta: FloatProperty(
        name="Taper Exponent (Δ)",
        description="Leonardo da Vinci / pipe model tapering exponent (2.1 - 2.5)",
        default=2.25,
        min=1.8,
        max=3.0,
        precision=2,
        update=on_param_update if BLENDER_AVAILABLE else None
    )

    crookedness: FloatProperty(
        name="Trunk Gnarliness",
        description="Natural winding / tortuosity of trunk and branches",
        default=0.14,
        min=0.0,
        max=0.50,
        precision=2,
        update=on_param_update if BLENDER_AVAILABLE else None
    )

    wood_density: FloatProperty(
        name="Wood Density (g/cm³)",
        description="Basic wood density affecting gravitational droop [PO:0005352]",
        default=0.68,
        min=0.20,
        max=1.25,
        precision=2,
        update=on_param_update if BLENDER_AVAILABLE else None
    )

    # -------------------------------------------------------------
    # 2. Gielis Buttress & Fluting Controls
    # -------------------------------------------------------------
    buttress_m: IntProperty(
        name="Buttress Ribs (m)",
        description="Gielis symmetry parameter (0=round, 3=triangular, 4=4-fluted, 6=6-buttress)",
        default=4,
        min=0,
        max=12,
        update=on_param_update if BLENDER_AVAILABLE else None
    )

    buttress_amplitude: FloatProperty(
        name="Flare Amplitude",
        description="Root buttress expansion factor at soil level",
        default=0.65,
        min=0.0,
        max=3.0,
        precision=2,
        update=on_param_update if BLENDER_AVAILABLE else None
    )

    buttress_decay: FloatProperty(
        name="Flare Decay Rate",
        description="How rapidly buttressing decays upwards (higher = stays near ground)",
        default=10.0,
        min=3.0,
        max=25.0,
        precision=1,
        update=on_param_update if BLENDER_AVAILABLE else None
    )

    gielis_n1: FloatProperty(
        name="Rib Sharpness (n1)",
        description="Gielis curvature exponent: smaller values produce sharper buttress ribs",
        default=0.50,
        min=0.20,
        max=3.0,
        precision=2,
        update=on_param_update if BLENDER_AVAILABLE else None
    )

    gielis_n2: FloatProperty(
        name="Rib Curvature (n2)",
        description="Gielis cosine exponent",
        default=1.8,
        min=0.20,
        max=4.0,
        precision=2,
        update=on_param_update if BLENDER_AVAILABLE else None
    )

    gielis_n3: FloatProperty(
        name="Rib Curvature (n3)",
        description="Gielis sine exponent",
        default=1.8,
        min=0.20,
        max=4.0,
        precision=2,
        update=on_param_update if BLENDER_AVAILABLE else None
    )

    # -------------------------------------------------------------
    # 3. Branching Architecture Controls [PO:0025073]
    # -------------------------------------------------------------
    branch_levels: IntProperty(
        name="Branch Levels",
        description="Maximum branching hierarchy order (1=trunk only, 2=scaffolds, 3=twigs)",
        default=3,
        min=1,
        max=3,
        update=on_param_update if BLENDER_AVAILABLE else None
    )

    branch_angle_deg: FloatProperty(
        name="Branch Angle (°)",
        description="Insertion angle relative to parent shoot axis",
        default=48.0,
        min=20.0,
        max=85.0,
        precision=1,
        unit='ROTATION',
        update=on_param_update if BLENDER_AVAILABLE else None
    )

    branch_length_decay: FloatProperty(
        name="Length Decay",
        description="Ratio of child branch length relative to parent branch",
        default=0.65,
        min=0.30,
        max=0.90,
        precision=2,
        update=on_param_update if BLENDER_AVAILABLE else None
    )

    branch_gravity: FloatProperty(
        name="Gravity Droop",
        description="Negative = upright (orthotropic), 0 = neutral, Positive = weeping (pendulous)",
        default=-0.15,
        min=-0.50,
        max=0.50,
        precision=2,
        update=on_param_update if BLENDER_AVAILABLE else None
    )

    phyllotaxis_angle_deg: FloatProperty(
        name="Phyllotaxis Divergence (°)",
        description="Divergence angle (137.5° = golden angle spiral, 90° = decussate, 180° = distichous)",
        default=137.5,
        min=45.0,
        max=240.0,
        precision=1,
        unit='ROTATION',
        update=on_param_update if BLENDER_AVAILABLE else None
    )

    apical_dominance: FloatProperty(
        name="Apical Dominance",
        description="Strength of central leader shoot suppression over lateral branches",
        default=0.60,
        min=0.10,
        max=1.00,
        precision=2,
        update=on_param_update if BLENDER_AVAILABLE else None
    )

    seed: IntProperty(
        name="Random Seed",
        description="Stochastic seed for branch generation",
        default=42,
        min=0,
        max=999999,
        update=on_param_update if BLENDER_AVAILABLE else None
    )

    # -------------------------------------------------------------
    # 4. Foliage & Leaf Controls [PO:0020039]
    # -------------------------------------------------------------
    show_leaves: BoolProperty(
        name="Show Leaves",
        description="Generate anchored foliage leaves on twigs",
        default=True,
        update=on_param_update if BLENDER_AVAILABLE else None
    )

    leaf_archetype: EnumProperty(
        name="Leaf Archetype",
        description="Morphological leaf silhouette",
        items=[
            ('Pinnate_Lobed', 'Pinnate Lobed (Oak)', 'Quercus robur'),
            ('Palmate_5', 'Palmate 5-Lobed (Maple)', 'Acer palmatum'),
            ('Ovate', 'Ovate (Birch, Apple)', 'Betula, Malus'),
            ('Elliptic', 'Elliptic (Beech, Fig)', 'Fagus, Ficus'),
            ('Lanceolate', 'Lanceolate (Eucalyptus)', 'Eucalyptus, Salix'),
            ('Cordate', 'Cordate (Linden)', 'Tilia'),
            ('Flabellate', 'Flabellate (Ginkgo)', 'Ginkgo biloba'),
            ('Acicular', 'Acicular (Needle)', 'Pinus conifer needle')
        ],
        default='Pinnate_Lobed',
        update=on_param_update if BLENDER_AVAILABLE else None
    )

    margin_type: EnumProperty(
        name="Leaf Margin",
        description="Botanical margin type [PO:0020042]",
        items=[
            ('Entire', 'Entire (Smooth)', 'Smooth unbroken margin'),
            ('Serrate', 'Serrate (Teeth forward)', 'Forward-pointing teeth'),
            ('Dentate', 'Dentate (Teeth outward)', 'Outward-pointing teeth'),
            ('Crenate', 'Crenate (Scalloped)', 'Rounded teeth')
        ],
        default='Entire',
        update=on_param_update if BLENDER_AVAILABLE else None
    )

    leaf_density: FloatProperty(
        name="Foliage Density",
        description="Relative quantity of leaves anchored along twigs",
        default=1.0,
        min=0.0,
        max=2.0,
        precision=2,
        update=on_param_update if BLENDER_AVAILABLE else None
    )

    leaf_scale: FloatProperty(
        name="Leaf Scale",
        description="Uniform scaling multiplier for leaves",
        default=1.0,
        min=0.2,
        max=3.0,
        precision=2,
        update=on_param_update if BLENDER_AVAILABLE else None
    )

    leaf_length_cm: FloatProperty(
        name="Blade Length (cm)",
        description="Length of leaf lamina from petiole base to apex in centimeters",
        default=11.0,
        min=1.0,
        max=35.0,
        precision=1,
        update=on_param_update if BLENDER_AVAILABLE else None
    )

    leaf_aspect_ratio: FloatProperty(
        name="Aspect Ratio (L/W)",
        description="Blade length to width ratio",
        default=1.75,
        min=0.8,
        max=8.0,
        precision=2,
        update=on_param_update if BLENDER_AVAILABLE else None
    )

    teeth_count: IntProperty(
        name="Teeth Count",
        description="Number of serration teeth per margin side",
        default=24,
        min=0,
        max=60,
        update=on_param_update if BLENDER_AVAILABLE else None
    )

    leaf_droop: FloatProperty(
        name="Leaf Droop",
        description="Longitudinal cantilever curvature bending downward",
        default=0.30,
        min=0.0,
        max=1.0,
        precision=2,
        update=on_param_update if BLENDER_AVAILABLE else None
    )

    # -------------------------------------------------------------
    # 5. Topology & Material Controls
    # -------------------------------------------------------------
    radial_resolution: IntProperty(
        name="Radial Sides",
        description="Circumferential quad polygon resolution (8, 12, 16)",
        default=12,
        min=6,
        max=24,
        update=on_param_update if BLENDER_AVAILABLE else None
    )

    use_subsurf: BoolProperty(
        name="Subdivision Surface",
        description="Add a smooth Subdivision Surface modifier for sculpted organic bark",
        default=False,
        update=on_param_update if BLENDER_AVAILABLE else None
    )

    assign_materials: BoolProperty(
        name="Assign PBR Materials",
        description="Attach botanical procedural shaders with subsurface scattering",
        default=True,
        update=on_param_update if BLENDER_AVAILABLE else None
    )


class PPG_PT_MainPanel(Panel):
    """Main Interactive Sidebar Panel in 3D Viewport."""
    bl_label = "Procedural Plant Generator"
    bl_idname = "PPG_PT_main_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Plant Gen'

    def draw(self, context):
        layout = self.layout
        props = context.scene.ppg_properties

        # Top Bar: Live update toggle, update active tree, or spawn new separate tree
        row = layout.row(align=True)
        row.prop(props, "auto_update", icon='PLAY', toggle=True)
        row.operator("ppg.generate_plant", text="Update", icon='FILE_REFRESH')
        row.operator("ppg.new_plant", text="New Tree", icon='ADD')

        # -------------------------------------------------------------
        # Section 1: Species Preset
        # -------------------------------------------------------------
        box = layout.box()
        box.label(text="Species Preset", icon='BOOKMARKS')
        box.prop(props, "species_enum", text="")
        box.operator("ppg.apply_species_preset", text="Load Species Defaults", icon='IMPORT')

        # -------------------------------------------------------------
        # Section 2: Trunk & Allometry [PO:0004712]
        # -------------------------------------------------------------
        box = layout.box()
        box.label(text="Trunk & Allometry [PO:0004712]", icon='MOD_LENGTH')
        box.prop(props, "dbh_m")
        box.prop(props, "tree_height_m")
        box.prop(props, "crown_base_height_m")
        box.prop(props, "pipe_delta")
        box.prop(props, "crookedness")
        box.prop(props, "wood_density")

        # -------------------------------------------------------------
        # Section 3: Gielis Buttress & Fluting (Modular Tree)
        # -------------------------------------------------------------
        box = layout.box()
        box.label(text="Gielis Buttress & Fluting", icon='MESH_CONE')
        box.prop(props, "buttress_m")
        if props.buttress_m > 0:
            box.prop(props, "buttress_amplitude")
            box.prop(props, "buttress_decay")
            box.prop(props, "gielis_n1")
            row = box.row(align=True)
            row.prop(props, "gielis_n2")
            row.prop(props, "gielis_n3")

        # -------------------------------------------------------------
        # Section 4: Branch Architecture [PO:0025073]
        # -------------------------------------------------------------
        box = layout.box()
        box.label(text="Branch Architecture [PO:0025073]", icon='OUTLINER_OB_ARMATURE')
        box.prop(props, "branch_levels")
        box.prop(props, "branch_angle_deg")
        box.prop(props, "branch_length_decay")
        box.prop(props, "branch_gravity")
        box.prop(props, "phyllotaxis_angle_deg")
        box.prop(props, "apical_dominance")
        box.prop(props, "seed")

        # -------------------------------------------------------------
        # Section 5: Foliage & Leaves [PO:0020039]
        # -------------------------------------------------------------
        box = layout.box()
        box.label(text="Foliage & Leaves [PO:0020039]", icon='SNAP_FACE')
        box.prop(props, "show_leaves")
        if props.show_leaves:
            box.prop(props, "leaf_archetype")
            box.prop(props, "margin_type")
            box.prop(props, "leaf_density")
            box.prop(props, "leaf_scale")
            box.prop(props, "leaf_length_cm")
            box.prop(props, "leaf_aspect_ratio")
            if props.margin_type != 'Entire':
                box.prop(props, "teeth_count")
            box.prop(props, "leaf_droop")

        # -------------------------------------------------------------
        # Section 6: Topology & Shading
        # -------------------------------------------------------------
        box = layout.box()
        box.label(text="Topology & Shading", icon='SHADING_SOLID')
        box.prop(props, "radial_resolution")
        box.prop(props, "use_subsurf")
        box.prop(props, "assign_materials")

        # Export Report
        layout.separator()
        layout.operator("ppg.export_traits", text="Export Trait Report (JSON)", icon='TEXT')
