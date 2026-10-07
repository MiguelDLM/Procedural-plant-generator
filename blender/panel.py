"""
Interactive 3D Viewport UI: properties and sidebar panels.
"""

try:
    import bpy
    from bpy.types import Panel, PropertyGroup
    from bpy.props import EnumProperty, FloatProperty, IntProperty, BoolProperty, FloatVectorProperty
    BLENDER_AVAILABLE = True
except ImportError:
    BLENDER_AVAILABLE = False
    Panel = object
    PropertyGroup = object

try:
    from ..core.species_db import get_preset_names, get_species_preset
    from ..core.architecture import HalleOldemanModel, PhyllotaxisType, CrownShape
    from ..core.leaf_morphology import LeafArchetype, MarginType, LobeType, CompoundType
    from ..core.leaf_venation import VenationPattern
    from ..core.bark import BarkPattern
    from ..core.roots import RootSystemType
    from .runtime import (is_updating, schedule_update, apply_species_preset_to_props, apply_allometry,
                          apply_leaf_template, apply_crown_shape, apply_succulent_preset)
    from ..core.succulent_db import GrowthForm, CATALOGS, preset_items
    from .succulents import profile_properties, LAYOUT, PREFIX
    from ..core.flower_db import FLOWER_CATALOG, flower_items
    from .flowers import flower_properties, LAYOUT as FLOWER_LAYOUT, write_flower_to_props, sync_flowers_to_plant
    from .runtime import set_updating
except (ImportError, ValueError):
    from core.species_db import get_preset_names, get_species_preset
    from core.architecture import HalleOldemanModel, PhyllotaxisType, CrownShape
    from core.leaf_morphology import LeafArchetype, MarginType, LobeType, CompoundType
    from core.leaf_venation import VenationPattern
    from core.bark import BarkPattern
    from core.roots import RootSystemType
    from blender.runtime import (is_updating, schedule_update, apply_species_preset_to_props, apply_allometry,
                                 apply_leaf_template, apply_crown_shape, apply_succulent_preset)
    from core.succulent_db import GrowthForm, CATALOGS, preset_items
    from blender.succulents import profile_properties, LAYOUT, PREFIX
    from core.flower_db import FLOWER_CATALOG, flower_items
    from blender.flowers import (flower_properties, LAYOUT as FLOWER_LAYOUT, write_flower_to_props,
                                 sync_flowers_to_plant)
    from blender.runtime import set_updating


def on_param_update(self, context):
    if is_updating() or not getattr(self, "auto_update", True):
        return
    schedule_update()


def _sync_flowers(props):
    set_updating(True)
    try:
        sync_flowers_to_plant(props)
    finally:
        set_updating(False)


def on_species_change(self, context):
    if is_updating():
        return
    _sync_flowers(self)
    apply_species_preset_to_props(self, self.species_enum, context)


def on_form_change(self, context):
    if is_updating():
        return
    if self.growth_form == 'Tree':
        _sync_flowers(self)
        apply_species_preset_to_props(self, self.species_enum, context)
    elif self.growth_form == 'Flower':
        on_flower_species(self, context)
    else:
        _sync_flowers(self)
        apply_succulent_preset(self, context)


def on_flower_species(self, context):
    if is_updating():
        return
    set_updating(True)
    try:
        write_flower_to_props(self, self.flower_species)
    finally:
        set_updating(False)
    if getattr(self, "auto_update", True):
        schedule_update()


def on_succulent_species(self, context):
    if is_updating():
        return
    _sync_flowers(self)
    apply_succulent_preset(self, context)


def on_dbh_change(self, context):
    if is_updating():
        return
    if self.allometric_lock:
        from .runtime import set_updating
        set_updating(True)
        try:
            apply_allometry(self)
        finally:
            set_updating(False)
    on_param_update(self, context)


def on_leaf_archetype(self, context):
    if is_updating():
        return
    apply_leaf_template(self)
    on_param_update(self, context)


def on_crown_shape(self, context):
    if is_updating():
        return
    apply_crown_shape(self)
    on_param_update(self, context)


def _enum_items(enum_cls):
    return [(e.value, e.value.replace("_", " "), "") for e in enum_cls]


U = on_param_update if BLENDER_AVAILABLE else None


def F(name, default, lo, hi, desc="", **kw):
    return FloatProperty(name=name, default=default, min=lo, max=hi, description=desc, update=U, **kw)


def I(name, default, lo, hi, desc=""):
    return IntProperty(name=name, default=default, min=lo, max=hi, description=desc, update=U)


def E(name, enum_cls, default, desc="", update=None):
    return EnumProperty(name=name, items=_enum_items(enum_cls), default=default, description=desc,
                        update=update or U)


def COL(name, default, desc=""):
    return FloatVectorProperty(name=name, subtype='COLOR_GAMMA', size=3, min=0.0, max=1.0,
                               default=default, description=desc, update=U)


if BLENDER_AVAILABLE:
    SPECIES_ITEMS = get_preset_names()

    class PPG_Properties(PropertyGroup):
        """Interactive parameters for procedural plant generation."""
        auto_update: BoolProperty(name="Live Update", default=True,
                                  description="Regenerate the plant when any parameter changes")
        species_enum: EnumProperty(name="Species", items=SPECIES_ITEMS, default="quercus_robur",
                                   update=on_species_change, description="Species preset")

        # Trait-space variation
        blend_species: EnumProperty(name="Blend With", items=SPECIES_ITEMS, default="fagus_sylvatica",
                                    description="Second species for morphological interpolation")
        blend_factor: FloatProperty(name="Blend", default=0.0, min=0.0, max=1.0,
                                    description="0 = current sliders, 1 = second species")
        variation_amount: FloatProperty(name="Variation", default=0.0, min=0.0, max=3.0,
                                        description="Intraspecific variability (in units of each trait's typical CV)")
        variation_seed: IntProperty(name="Variant Seed", default=1, min=0, max=999999)

        # Trunk & allometry
        allometric_lock: BoolProperty(name="Allometric Scaling", default=True,
                                      description="Derive height and crown from DBH using the species allometry")
        dbh_m: FloatProperty(name="DBH", default=0.8, min=0.02, max=5.0, precision=2, unit='LENGTH',
                             description="Stem diameter at breast height (1.3 m)", update=on_dbh_change)
        tree_height_m: F("Height", 24.0, 0.5, 100.0, "Total height", unit='LENGTH')
        crown_radius_m: F("Crown Radius", 9.0, 0.2, 25.0, "Mean horizontal crown radius", unit='LENGTH')
        crown_base_height_m: F("Clear Bole", 6.0, 0.0, 80.0, "Height of the lowest live branch", unit='LENGTH')
        pipe_delta: F("Pipe Exponent", 2.3, 1.8, 3.0, "Leonardo/pipe model exponent: r^D = sum r_i^D")
        crookedness: F("Tortuosity", 0.3, 0.0, 0.6, "Natural winding of axes")
        wood_density: F("Wood Density", 0.7, 0.15, 1.25, "g/cm3; heavier wood bends branches more")

        # Buttress
        buttress_m: I("Buttress Lobes (m)", 4, 0, 12, "Gielis symmetry of the fluted base (0 = round)")
        buttress_amplitude: F("Flare", 0.5, 0.0, 3.0, "Root flare at soil level")
        buttress_decay: F("Flare Decay", 10.0, 2.0, 25.0, "Higher = flare confined near the ground")
        gielis_n1: F("Rib Sharpness n1", 0.5, 0.2, 3.0)
        gielis_n2: F("n2", 1.8, 0.2, 4.0)
        gielis_n3: F("n3", 1.8, 0.2, 4.0)

        # Crown & architecture
        arch_model: E("Architecture", HalleOldemanModel, "Rauh", "Halle-Oldeman architectural model")
        crown_shape: E("Crown Template", CrownShape, "Spherical", "Sets widest position and fullness",
                       update=on_crown_shape if BLENDER_AVAILABLE else None)
        crown_widest: F("Widest Point", 0.45, 0.02, 0.98, "Relative height of the widest crown section")
        crown_fullness: F("Fullness", 1.5, 0.3, 3.0, "Low = columnar/flat sides, high = peaked")
        apical_dominance: F("Apical Dominance", 0.3, 0.0, 1.0, "High = single excurrent leader")
        leader_count: I("Leaders", 0, 0, 6, "Codominant stems (0 = from apical dominance)")
        branch_levels: I("Branch Orders", 3, 0, 4, "0 = unbranched, 3 = twigs")
        branch_angle_deg: F("Branch Angle", 55.0, 5.0, 100.0, "Scaffold insertion angle")
        twig_angle_deg: F("Twig Angle", 55.0, 5.0, 95.0, "Higher-order insertion angle")
        branch_frequency: F("Branch Frequency", 2.0, 0.3, 8.0, "Lateral nodes per meter")
        branch_length_decay: F("Length Ratio", 0.68, 0.3, 0.95, "Child/parent axis length")
        internode_length: F("Internode", 0.45, 0.05, 1.5, "Base segment length (m)")
        branch_gravity: F("Gravitropism", -0.05, -0.6, 0.9, "< 0 upright, > 0 weeping")
        phototropism: F("Phototropism", 0.35, 0.0, 1.0, "Outward light-seeking bias")
        plagiotropy: F("Plagiotropy", 0.2, 0.0, 1.0, "Flatten laterals into horizontal sprays")
        phyllotaxis_type: E("Phyllotaxis", PhyllotaxisType, "Spiral")
        phyllotaxis_angle_deg: F("Divergence", 137.5, 30.0, 180.0, "Spiral divergence angle (degrees)")
        whorl_size: I("Whorl Size", 1, 1, 8)
        seed: I("Seed", 42, 0, 999999)

        # Leaf shape
        leaf_archetype: E("Template", LeafArchetype, "Pinnate_Lobed", "Applies a leaf-shape template",
                          update=on_leaf_archetype if BLENDER_AVAILABLE else None)
        leaf_length_cm: F("Blade Length (cm)", 10.0, 0.2, 150.0)
        leaf_aspect_ratio: F("Length/Width", 1.9, 0.5, 120.0)
        leaf_widest: F("Widest Point", 0.6, 0.05, 0.95, "0 = base (ovate) .. 1 = apex (obovate)")
        leaf_base_angle: F("Base Angle", 90.0, 5.0, 179.0, "Angle enclosed by the base")
        leaf_apex_angle: F("Apex Angle", 70.0, 5.0, 179.0, "Angle enclosed by the apex")
        leaf_base_curvature: F("Base Curvature", 0.3, -1.0, 1.0, "-1 concave (attenuate) .. +1 rounded")
        leaf_apex_curvature: F("Apex Curvature", 0.0, -1.0, 1.0, "-1 acuminate .. +1 obtuse")
        leaf_cordate: F("Cordate Base", 0.0, 0.0, 1.0)
        leaf_asymmetry: F("Base Asymmetry", 0.0, 0.0, 1.0)
        leaf_notch: F("Apical Notch", 0.0, 0.0, 0.5)
        leaf_falcate: F("Falcate Bend", 0.0, 0.0, 0.3)
        lobe_type: E("Lobation", LobeType, "None")
        lobe_count: I("Lobes", 5, 1, 11)
        lobe_depth: F("Sinus Depth", 0.5, 0.0, 0.95)
        lobe_angle: F("Lobe Angle", 50.0, 15.0, 85.0)
        lobe_spread: F("Lobe Spread", 200.0, 10.0, 330.0)
        lobe_width: F("Lobe Width", 0.55, 0.2, 1.2)
        lobe_roundness: F("Sinus Roundness", 0.5, 0.0, 1.0)
        lobe_apex_angle: F("Lobe Apex Angle", 80.0, 10.0, 179.0)
        margin_type: E("Margin", MarginType, "Entire")
        teeth_count: I("Teeth", 24, 0, 80)
        tooth_height: F("Tooth Depth", 0.035, 0.0, 0.2)
        tooth_skew: F("Tooth Skew", 0.75, 0.3, 0.92, "0.5 symmetric .. 0.9 apically pointing")
        compound_type: E("Organisation", CompoundType, "Simple")
        leaflet_count: I("Leaflets", 7, 1, 80)
        leaflet_angle: F("Leaflet Angle", 60.0, 5.0, 90.0)
        rachis_ratio: F("Rachis Length", 2.5, 0.2, 20.0)
        terminal_leaflet: BoolProperty(name="Terminal Leaflet", default=True, update=U)
        leaflet_gradient: F("Leaflet Gradient", 0.3, 0.0, 0.8)
        petiole_ratio: F("Petiole Length", 0.35, 0.0, 1.5)
        petiole_angle: F("Petiole Angle", 50.0, 5.0, 120.0)
        leaf_curl: F("Transverse Curl", 0.25, 0.0, 1.0)
        leaf_droop: F("Droop", 0.35, 0.0, 1.0)
        leaf_undulation: F("Undulation", 0.05, 0.0, 0.5)
        leaf_thickness: F("Thickness (mm)", 0.22, 0.05, 1.5)

        # Venation & texture
        vein_pattern: E("Venation", VenationPattern, "Craspedodromous")
        vein_vla: F("VLA (mm/mm2)", 7.0, 1.0, 20.0, "Vein length per area; sets areole size")
        vein_pairs: I("Secondary Pairs", 8, 0, 30)
        vein_angle: F("Secondary Angle", 46.0, 10.0, 88.0)
        vein_curvature: F("Arcuation", 0.45, 0.0, 1.0)
        vein_reticulation: F("Reticulation", 0.6, 0.0, 1.0)
        vein_contrast: F("Vein Contrast", 0.6, 0.0, 1.5)
        texture_resolution: EnumProperty(name="Texture", items=[("256", "256", ""), ("512", "512", ""),
                                                               ("1024", "1024", ""), ("2048", "2048", "")],
                                         default="1024", update=U)
        senescence: F("Season (Senescence)", 0.0, 0.0, 1.0, "0 summer green .. 1 full autumn colour")
        color_adaxial: COL("Upper Surface", (0.19, 0.31, 0.09))
        color_abaxial: COL("Lower Surface", (0.36, 0.46, 0.25))
        color_vein: COL("Veins", (0.48, 0.56, 0.25))
        color_autumn: COL("Autumn", (0.55, 0.38, 0.15))
        leaf_gloss: F("Gloss", 0.3, 0.0, 1.0)

        # Foliage
        show_leaves: BoolProperty(name="Show Leaves", default=True, update=U)
        foliage_unit: EnumProperty(name="Card", default='SHOOT', update=U, items=[
            ('SHOOT', "Leafy Shoot", "Each card is a shoot carrying several leaves: dense crowns, few polygons"),
            ('LEAF', "Single Leaf", "One leaf per card: maximum texture detail for close-ups")])
        shoot_leaves: I("Leaves per Shoot", 6, 2, 16)
        leaf_density: F("Density", 1.0, 0.0, 3.0, "Multiplier on the leaf-area-index target")
        leaf_area_index: F("Leaf Area Index", 4.5, 0.5, 12.0, "Leaf area per unit crown projection area")
        leaf_angle: F("Mean Leaf Angle", 40.0, 5.0, 85.0,
                      "Lamina inclination: planophile ~25, spherical ~57, erectophile ~70 degrees")
        leaf_budget: IntProperty(name="Max Cards", default=60000, min=100, max=500000, update=U,
                                 description="Upper bound on leaf cards (performance)")
        leaf_scale: F("Leaf Scale", 1.0, 0.2, 4.0)

        # Bark
        bark_pattern: E("Bark", BarkPattern, "Fissured")
        bark_color: COL("Bark Colour", (0.36, 0.33, 0.29))
        bark_color2: COL("Furrow Colour", (0.14, 0.12, 0.10))
        bark_scale: F("Feature Size (m)", 0.06, 0.005, 0.5)
        bark_relief: F("Relief", 0.8, 0.0, 1.0)
        bark_color_young: COL("Young Bark Colour", (0.42, 0.40, 0.34), "Smooth periderm of thin, young axes")
        bark_onset_cm: F("Fissuring Onset (cm)", 4.0, 0.1, 30.0,
                         "Axis radius at which the mature bark pattern appears; fissures widen beyond it")
        bark_weathering: F("Weathering", 0.35, 0.0, 1.0, "Grey, bleached ridge tops on old bark")

        # Roots
        show_roots: BoolProperty(name="Show Roots", default=True, update=U)
        root_display_depth: F("Display Depth (m)", 2.5, 0.2, 20.0, "Roots below this depth are not meshed")
        root_system: E("Root System", RootSystemType, "Heart",
                       "Koestler et al. (1968) type: taproot, heart, plate; plus buttress and fibrous")
        root_laterals: I("Main Laterals", 6, 1, 16, "Structural laterals (= stem flutes when buttressed)")
        root_spread: F("Spread / Crown", 1.4, 0.3, 4.0, "Lateral reach relative to crown radius")
        root_max_depth: F("Max Depth (m)", 3.0, 0.3, 60.0, "Maximum rooting depth (Canadell et al. 1996)")
        root_beta: F("Depth Coefficient (beta)", 0.966, 0.90, 0.99,
                     "Jackson et al. (1996): cumulative root fraction Y = 1 - beta^d (d in cm)", precision=3)
        root_taproot_share: F("Taproot Share", 0.3, 0.0, 0.8, "Collar flow taken by taproot / oblique roots")
        root_zrt: F("Rapid Taper Zone (xDBH)", 2.2, 0.5, 5.0, "Radius of the zone of rapid taper")
        root_sinker_spacing: F("Sinker Spacing (m)", 1.2, 0.2, 5.0)
        root_exposure: F("Surface Exposure", 0.15, 0.0, 1.0, "How much laterals ride above the soil near the stem")
        root_plank: F("Plank / Buttress", 0.5, 0.0, 5.0, "Vertical elongation of root sections near the stem")
        root_buttress_height: F("Collar Height (xDBH)", 0.6, 0.0, 4.0, "Where laterals merge into the stem")
        root_knees: I("Knees", 0, 0, 20, "Pneumatophores per lateral (Taxodium distichum)")

        # Topology
        radial_resolution: I("Trunk Sides", 12, 4, 32)
        twig_resolution: I("Twig Sides", 5, 3, 16)
        use_subsurf: BoolProperty(name="Subdivision Surface", default=False, update=U)
        junction_quality: EnumProperty(name="Junctions", default='FUSED', update=U, items=[
            ('TUBES', "Tubes", "Intersecting tubes: fastest, for forests and distant trees"),
            ('FUSED', "Fused", "Stem, limb bases and roots fused into one surface: for many trees"),
            ('HERO', "Hero", "Fused plus local fillets at every fine-branch insertion: few close-up trees")])
        fuse_detail: F("Fusion Detail", 10.0, 3.0, 30.0, "Voxels per stem radius (higher = finer, slower)")
        fuse_smoothing: I("Fillet Smoothing", 6, 0, 30, "Laplacian iterations rounding crotches and flares")
        hero_min_radius_cm: F("Finest Fused Branch (cm)", 2.5, 0.3, 20.0,
                              "Branches thicker than this (past their collar) get a fused fillet (Hero)")
        sleeve_detail: F("Fillet Detail", 4.0, 2.0, 12.0, "Voxels per branch radius at fine insertions (Hero)")
        hero_max_junctions: IntProperty(name="Max Fused Junctions", default=400, min=10, max=20000, update=U,
                                        description="Thickest insertions fused first; bounds Hero cost")
        assign_materials: BoolProperty(name="Materials", default=True, update=U)
    # Growth form selector and succulent properties (generated from the core profiles)
    PPG_Properties.__annotations__.update({
        "growth_form": EnumProperty(name="Growth Form", default='Tree', update=on_form_change, items=[
            ('Tree', "Tree / Shrub", "Woody plants: trunk, branches, leaves, roots", 'OUTLINER_OB_FORCE_FIELD', 0),
            ('Cactus', "Cactus / Stem Succulent", "Ribbed or tuberculate succulent stems with areoles and spines",
             'MESH_CYLINDER', 1),
            ('Rosette', "Rosette Succulent", "Agave, Aloe, Echeveria: thick leaves in a Fibonacci rosette",
             'MESH_CIRCLE', 2),
            ('Flower', "Flower / Inflorescence", "An isolated flower or inflorescence built from its floral diagram",
             'FREEZE', 3)]),
        "flower_species": EnumProperty(name="Flower", items=flower_items(), default="rosa_canina",
                                       update=on_flower_species),
        "show_flowers": BoolProperty(name="Show Flowers", default=False, update=U,
                                     description="Place the selected inflorescence on the plant"),
        "flower_blend": EnumProperty(name="Blend With", items=flower_items(), default="hibiscus_rosa_sinensis"),
        "flower_single": BoolProperty(name="Single Flower", default=False, update=U,
                                      description="Show one flower instead of the whole inflorescence"),
        "flower_density": FloatProperty(name="Flowering Density", default=0.35, min=0.0, max=1.0, update=U,
                                        description="Fraction of the possible sites (shoot tips, axils, areoles) "
                                                    "that bear an inflorescence"),
        "flower_max": IntProperty(name="Max Inflorescences", default=2000, min=1, max=20000, update=U),
        "flower_scale": FloatProperty(name="Flower Scale", default=1.0, min=0.1, max=10.0, update=U),
        "flower_bloom": FloatProperty(name="Bloom Stage", default=0.5, min=0.0, max=1.0, update=U,
                                      description="0 buds .. 0.5 peak (acropetal gradient) .. 1 all open"),
        "flower_detail": FloatProperty(name="Flower Detail", default=1.0, min=0.3, max=3.0, update=U),
        "cactus_species": EnumProperty(name="Cactus", items=preset_items(GrowthForm.CACTUS),
                                       default="carnegiea_gigantea", update=on_succulent_species),
        "rosette_species": EnumProperty(name="Rosette", items=preset_items(GrowthForm.ROSETTE),
                                        default="echeveria_elegans", update=on_succulent_species),
        "cactus_blend": EnumProperty(name="Blend With", items=preset_items(GrowthForm.CACTUS),
                                     default="echinocactus_grusonii"),
        "rosette_blend": EnumProperty(name="Blend With", items=preset_items(GrowthForm.ROSETTE),
                                      default="agave_americana"),
        "succ_detail": FloatProperty(name="Mesh Detail", default=1.0, min=0.3, max=3.0, update=U,
                                     description="Surface sampling density of stems and leaves"),
        "spine_budget": IntProperty(name="Max Spines", default=60000, min=0, max=1000000, update=U,
                                    description="Radial spines are thinned evenly above this count"),
    })
    for _form in (GrowthForm.CACTUS, GrowthForm.ROSETTE):
        PPG_Properties.__annotations__.update(profile_properties(_form, U))
    PPG_Properties.__annotations__.update(flower_properties(U))
else:
    PPG_Properties = None


class _PPGSub:
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Plant Gen'
    bl_parent_id = "PPG_PT_main_panel"
    forms = ('Tree',)

    @classmethod
    def poll(cls, context):
        p = getattr(context.scene, "ppg_properties", None)
        return p is not None and p.growth_form in cls.forms


class PPG_PT_MainPanel(Panel):
    bl_label = "Procedural Plant Generator"
    bl_idname = "PPG_PT_main_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Plant Gen'

    def draw(self, context):
        layout = self.layout
        props = context.scene.ppg_properties
        row = layout.row(align=True)
        row.prop(props, "auto_update", icon='PLAY', toggle=True)
        row.operator("ppg.generate_plant", text="Update", icon='FILE_REFRESH')
        row.operator("ppg.new_plant", text="New", icon='ADD')

        layout.prop(props, "growth_form", text="")
        box = layout.box()
        if props.growth_form == 'Flower':
            box.prop(props, "flower_species", text="")
            spec = FLOWER_CATALOG[props.flower_species]
            col = box.column(align=True)
            col.scale_y = 0.8
            col.label(text=f"{spec.family}", icon='FREEZE')
            col.label(text=spec.formula)
            box.operator("ppg.apply_species_preset", text="Reload Flower", icon='IMPORT')
            layout.operator("ppg.export_traits", text="Export Traits (JSON)", icon='TEXT')
            return
        if props.growth_form == 'Tree':
            box.prop(props, "species_enum", text="")
            spec = get_species_preset(props.species_enum)
            family, habit, biome = spec.family, spec.growth_habit, spec.biome
        else:
            form = GrowthForm(props.growth_form)
            key = props.cactus_species if form == GrowthForm.CACTUS else props.rosette_species
            box.prop(props, "cactus_species" if form == GrowthForm.CACTUS else "rosette_species", text="")
            spec = CATALOGS[form][key]
            family, habit, biome = spec.family, form.value, spec.biome
        col = box.column(align=True)
        col.scale_y = 0.8
        col.label(text=f"{family} · {habit}", icon='OUTLINER_OB_FORCE_FIELD')
        col.label(text=biome)
        box.operator("ppg.apply_species_preset", text="Reload Species", icon='IMPORT')
        layout.operator("ppg.export_traits", text="Export Traits (JSON)", icon='TEXT')


class PPG_PT_Variation(_PPGSub, Panel):
    bl_label = "Variants (Trait Space)"
    bl_idname = "PPG_PT_variation"
    bl_options = {'DEFAULT_CLOSED'}
    forms = ('Tree', 'Cactus', 'Rosette', 'Flower')

    def draw(self, context):
        p = context.scene.ppg_properties
        col = self.layout.column(align=True)
        col.prop(p, {"Tree": "blend_species", "Cactus": "cactus_blend", "Rosette": "rosette_blend",
                     "Flower": "flower_blend"}[p.growth_form], text="")
        col.prop(p, "blend_factor", slider=True)
        col.separator()
        col.prop(p, "variation_amount", slider=True)
        col.prop(p, "variation_seed")
        self.layout.operator("ppg.apply_variant", icon='SHADERFX')


class PPG_PT_Trunk(_PPGSub, Panel):
    bl_label = "Trunk & Allometry"
    bl_idname = "PPG_PT_trunk"

    def draw(self, context):
        p = context.scene.ppg_properties
        col = self.layout.column(align=True)
        col.prop(p, "allometric_lock")
        col.prop(p, "dbh_m")
        sub = col.column(align=True)
        sub.prop(p, "tree_height_m")
        sub.prop(p, "crown_radius_m")
        sub.prop(p, "crown_base_height_m")
        col.separator()
        col.prop(p, "pipe_delta")
        col.prop(p, "crookedness")
        col.prop(p, "wood_density")
        col.separator()
        col.prop(p, "buttress_m")
        if p.buttress_m > 0:
            col.prop(p, "buttress_amplitude")
            col.prop(p, "buttress_decay")
            col.prop(p, "gielis_n1")
            row = col.row(align=True)
            row.prop(p, "gielis_n2")
            row.prop(p, "gielis_n3")


class PPG_PT_Crown(_PPGSub, Panel):
    bl_label = "Crown & Branching"
    bl_idname = "PPG_PT_crown"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        p = context.scene.ppg_properties
        col = self.layout.column(align=True)
        col.prop(p, "arch_model")
        col.prop(p, "crown_shape")
        col.prop(p, "crown_widest", slider=True)
        col.prop(p, "crown_fullness")
        col.separator()
        col.prop(p, "apical_dominance", slider=True)
        col.prop(p, "leader_count")
        col.prop(p, "branch_levels")
        col.prop(p, "branch_angle_deg")
        col.prop(p, "twig_angle_deg")
        col.prop(p, "branch_frequency")
        col.prop(p, "branch_length_decay")
        col.prop(p, "internode_length")
        col.separator()
        col.prop(p, "branch_gravity")
        col.prop(p, "phototropism")
        col.prop(p, "plagiotropy")
        col.separator()
        col.prop(p, "phyllotaxis_type")
        if p.phyllotaxis_type == "Spiral":
            col.prop(p, "phyllotaxis_angle_deg")
        col.prop(p, "whorl_size")
        col.prop(p, "seed")


class PPG_PT_Leaf(_PPGSub, Panel):
    bl_label = "Leaf Shape"
    bl_idname = "PPG_PT_leaf"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        p = context.scene.ppg_properties
        layout = self.layout
        layout.prop(p, "leaf_archetype")
        col = layout.column(align=True)
        col.label(text="Lamina")
        for name in ("leaf_length_cm", "leaf_aspect_ratio", "leaf_widest", "leaf_base_angle", "leaf_apex_angle",
                     "leaf_base_curvature", "leaf_apex_curvature", "leaf_cordate", "leaf_asymmetry",
                     "leaf_notch", "leaf_falcate"):
            col.prop(p, name)
        col = layout.column(align=True)
        col.prop(p, "lobe_type")
        if p.lobe_type != "None":
            for name in ("lobe_count", "lobe_depth", "lobe_angle" if p.lobe_type == "Pinnate" else "lobe_spread",
                         "lobe_width", "lobe_roundness", "lobe_apex_angle"):
                col.prop(p, name)
        col = layout.column(align=True)
        col.prop(p, "margin_type")
        if p.margin_type != "Entire":
            col.prop(p, "teeth_count")
            col.prop(p, "tooth_height")
            col.prop(p, "tooth_skew")
        col = layout.column(align=True)
        col.prop(p, "compound_type")
        if p.compound_type != "Simple":
            col.prop(p, "leaflet_count")
            col.prop(p, "leaflet_angle" if p.compound_type in ("Pinnate", "Spray") else "lobe_spread")
            if p.compound_type in ("Pinnate", "Spray"):
                col.prop(p, "rachis_ratio")
                col.prop(p, "leaflet_gradient")
            if p.compound_type == "Pinnate":
                col.prop(p, "terminal_leaflet")
        col = layout.column(align=True)
        col.label(text="Petiole & 3D form")
        for name in ("petiole_ratio", "petiole_angle", "leaf_curl", "leaf_droop", "leaf_undulation",
                     "leaf_thickness"):
            col.prop(p, name)


class PPG_PT_Venation(_PPGSub, Panel):
    bl_label = "Venation & Colour"
    bl_idname = "PPG_PT_venation"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        p = context.scene.ppg_properties
        col = self.layout.column(align=True)
        for name in ("vein_pattern", "vein_vla", "vein_pairs", "vein_angle", "vein_curvature",
                     "vein_reticulation", "vein_contrast"):
            col.prop(p, name)
        col.separator()
        col.prop(p, "senescence", slider=True)
        col.prop(p, "texture_resolution")
        col.separator()
        for name in ("color_adaxial", "color_abaxial", "color_vein", "color_autumn", "leaf_gloss"):
            col.prop(p, name)


class PPG_PT_Foliage(_PPGSub, Panel):
    bl_label = "Foliage"
    bl_idname = "PPG_PT_foliage"

    def draw(self, context):
        p = context.scene.ppg_properties
        col = self.layout.column(align=True)
        col.prop(p, "show_leaves")
        if p.show_leaves:
            col.prop(p, "foliage_unit")
            if p.foliage_unit == 'SHOOT':
                col.prop(p, "shoot_leaves")
            col.prop(p, "leaf_area_index")
            col.prop(p, "leaf_angle")
            col.prop(p, "leaf_density")
            col.prop(p, "leaf_scale")
            col.prop(p, "leaf_budget")


class PPG_PT_Roots(_PPGSub, Panel):
    bl_label = "Roots"
    bl_idname = "PPG_PT_roots"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        p = context.scene.ppg_properties
        col = self.layout.column(align=True)
        col.prop(p, "show_roots")
        if not p.show_roots:
            return
        col.prop(p, "root_system")
        col.prop(p, "root_display_depth")
        col.separator()
        for name in ("root_laterals", "root_spread", "root_max_depth", "root_beta", "root_taproot_share",
                     "root_zrt", "root_sinker_spacing", "root_exposure", "root_plank", "root_buttress_height",
                     "root_knees"):
            col.prop(p, name)


class PPG_PT_Bark(_PPGSub, Panel):
    bl_label = "Bark"
    bl_idname = "PPG_PT_bark"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        p = context.scene.ppg_properties
        col = self.layout.column(align=True)
        for name in ("bark_pattern", "bark_color", "bark_color2", "bark_scale", "bark_relief", "bark_color_young",
                     "bark_onset_cm", "bark_weathering"):
            col.prop(p, name)


class PPG_PT_Topology(_PPGSub, Panel):
    bl_label = "Topology & Shading"
    bl_idname = "PPG_PT_topology"
    bl_options = {'DEFAULT_CLOSED'}
    forms = ('Tree', 'Cactus', 'Rosette')

    def draw(self, context):
        p = context.scene.ppg_properties
        col = self.layout.column(align=True)
        if p.growth_form == 'Flower':
            col.prop(p, "seed")
            col.prop(p, "assign_materials")
            return
        if p.growth_form != 'Tree':
            col.prop(p, "seed")
            col.prop(p, "succ_detail")
            col.prop(p, "show_roots")
            if p.show_roots:
                col.prop(p, "root_display_depth")
            if p.growth_form == 'Cactus':
                col.prop(p, "spine_budget")
            col.prop(p, "assign_materials")
            return
        col.prop(p, "junction_quality")
        if p.junction_quality != 'TUBES':
            col.prop(p, "fuse_detail")
            col.prop(p, "fuse_smoothing")
        if p.junction_quality == 'HERO':
            col.prop(p, "hero_min_radius_cm")
            col.prop(p, "sleeve_detail")
            col.prop(p, "hero_max_junctions")
        col.separator()
        for name in ("radial_resolution", "twig_resolution", "use_subsurf", "assign_materials"):
            col.prop(p, name)


def _succulent_panel(form, index, title, names):
    def draw(self, context):
        p = context.scene.ppg_properties
        col = self.layout.column(align=True)
        for n in names:
            col.prop(p, PREFIX[form] + n)
    return type(f"PPG_PT_{form.value}_{index}", (_PPGSub, Panel), {
        "bl_label": title, "bl_idname": f"PPG_PT_{form.value.lower()}_{index}",
        "bl_options": {'DEFAULT_CLOSED'} if index else set(), "forms": (form.value,), "draw": draw})


class PPG_PT_Flowers(_PPGSub, Panel):
    bl_label = "Flowers"
    bl_idname = "PPG_PT_flowers"
    bl_options = {'DEFAULT_CLOSED'}
    forms = ('Tree', 'Cactus', 'Rosette', 'Flower')

    def draw_header(self, context):
        p = context.scene.ppg_properties
        if p.growth_form != 'Flower':
            self.layout.prop(p, "show_flowers", text="")

    def draw(self, context):
        p = context.scene.ppg_properties
        col = self.layout.column(align=True)
        if p.growth_form != 'Flower':
            col.prop(p, "flower_species", text="")
            spec = FLOWER_CATALOG[p.flower_species]
            col.label(text=spec.formula)
            col.separator()
            col.prop(p, "flower_density", slider=True)
            col.prop(p, "flower_max")
        else:
            col.prop(p, "flower_single")
        col.prop(p, "flower_bloom", slider=True)
        col.prop(p, "flower_scale")
        col.prop(p, "flower_detail")


def _flower_panel(index, title, prefix, names):
    def draw(self, context):
        p = context.scene.ppg_properties
        col = self.layout.column(align=True)
        for n in names:
            col.prop(p, prefix + n)

    @classmethod
    def poll(cls, context):
        p = getattr(context.scene, "ppg_properties", None)
        return p is not None and (p.growth_form == 'Flower' or p.show_flowers)
    return type(f"PPG_PT_Flower_{index}", (_PPGSub, Panel), {
        "bl_label": title, "bl_idname": f"PPG_PT_flower_{index}", "bl_parent_id": "PPG_PT_flowers",
        "bl_options": {'DEFAULT_CLOSED'}, "draw": draw, "poll": poll})


FLOWER_PANELS = tuple(_flower_panel(i, t, pre, n) for i, (t, pre, n) in enumerate(FLOWER_LAYOUT))

SUCCULENT_PANELS = tuple(_succulent_panel(f, i, t, n) for f in (GrowthForm.CACTUS, GrowthForm.ROSETTE)
                         for i, (t, n) in enumerate(LAYOUT[f]))

PANEL_CLASSES = (PPG_PT_MainPanel, PPG_PT_Variation, PPG_PT_Trunk, PPG_PT_Crown, PPG_PT_Leaf,
                 PPG_PT_Venation, PPG_PT_Foliage, PPG_PT_Roots, PPG_PT_Bark) + SUCCULENT_PANELS + \
                (PPG_PT_Flowers,) + FLOWER_PANELS + (PPG_PT_Topology,)
