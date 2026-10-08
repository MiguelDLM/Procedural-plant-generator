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
    from .forest import forest_properties, draw_forest_panel
    from .preset_io import enum_items, item_number, draw_presets
    from ..core.relevance import applies, condition
    from .runtime import PROP_MAP
    from ..core.vine_db import VINE_CATALOG
    from .vines import vine_properties, LAYOUT as VINE_LAYOUT, SECTION as VINE_SECTION, section_values, \
        write_vine_to_props
    from ..core.fruit_db import FRUIT_CATALOG
    from .fruits import fruit_properties, LAYOUT as FRUIT_LAYOUT, FRT, fruit_values, write_fruit_to_props, \
        sync_fruits_to_plant
    from ..core.vegetable_db import VEGETABLE_CATALOG
    from . import vegetables as VG
    from ..core.grass_db import GRASS_CATALOG
    from . import grasses as GR
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
    from blender.forest import forest_properties, draw_forest_panel
    from blender.preset_io import enum_items, item_number, draw_presets
    from core.relevance import applies, condition
    from blender.runtime import PROP_MAP
    from core.vine_db import VINE_CATALOG
    from blender.vines import vine_properties, LAYOUT as VINE_LAYOUT, SECTION as VINE_SECTION, section_values, \
        write_vine_to_props
    from core.fruit_db import FRUIT_CATALOG
    from blender.fruits import fruit_properties, LAYOUT as FRUIT_LAYOUT, FRT, fruit_values, write_fruit_to_props, \
        sync_fruits_to_plant
    from core.vegetable_db import VEGETABLE_CATALOG
    import blender.vegetables as VG
    from core.grass_db import GRASS_CATALOG
    import blender.grasses as GR


# -----------------------------------------------------------------------------
# Context-dependent fields: inactive ones are greyed out with a short note (core.relevance)
# -----------------------------------------------------------------------------
_PROP_PATH = {prop: path for prop, path in PROP_MAP}


def _section_values(props, prefix: str, names) -> dict:
    return {n: getattr(props, prefix + n) for n in names if hasattr(props, prefix + n)}


def _tree_values(props, section: str) -> dict:
    out = {}
    for prop, path in PROP_MAP:
        sec, _, field = path.partition(".")
        if sec == section and hasattr(props, prop):
            out[field] = getattr(props, prop)
    return out


def draw_fields(col, props, form: str, section: str, items, values: dict, notes=None):
    """items: (property name, field name) pairs. Inactive fields stay visible but greyed, with a note
    (pass a shared `notes` list to collect the notes and draw them once at the end of a group)."""
    own = notes is None
    notes = [] if own else notes
    for prop, field in items:
        row = col.row(align=True)
        ok = applies(form, section, field, values)
        row.active = ok
        row.prop(props, prop)
        if not ok:
            note = condition(form, section, field)[1]
            if note not in notes:
                notes.append(note)
    if own:
        _draw_notes(col, notes)


def _draw_notes(col, notes):
    for note in notes[:3]:
        r = col.row()
        r.active = False
        r.label(text=note, icon='INFO')


def draw_tree_fields(col, props, names):
    """Tree properties (PROP_MAP names), greyed out when their trait does not apply."""
    by_section: dict = {}
    notes: list = []
    for name in names:
        path = _PROP_PATH.get(name)
        if path is None:
            col.prop(props, name)
            continue
        sec, _, field = path.partition(".")
        draw_fields(col, props, "Tree", sec, [(name, field)], by_section.setdefault(sec, _tree_values(props, sec)),
                    notes)
    _draw_notes(col, notes)


def on_param_update(self, context):
    if is_updating() or not getattr(self, "auto_update", True):
        return
    schedule_update()


def _sync_flowers(props):
    set_updating(True)
    try:
        sync_flowers_to_plant(props)
        sync_fruits_to_plant(props)
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
    elif self.growth_form == 'Vine':
        on_vine_species(self, context)
    elif self.growth_form == 'Fruit':
        on_fruit_species(self, context)
    elif self.growth_form == 'Vegetable':
        on_vegetable_species(self, context)
    elif self.growth_form == 'Grass':
        on_grass_species(self, context)
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


def on_grass_species(self, context):
    if is_updating():
        return
    set_updating(True)
    try:
        GR.write_grass_to_props(self, self.grass_species)
    finally:
        set_updating(False)
    if getattr(self, "auto_update", True):
        schedule_update()


def on_vegetable_species(self, context):
    if is_updating():
        return
    set_updating(True)
    try:
        VG.write_vegetable_to_props(self, self.vegetable_species)
    finally:
        set_updating(False)
    if getattr(self, "auto_update", True):
        schedule_update()


def on_fruit_species(self, context):
    if is_updating():
        return
    set_updating(True)
    try:
        write_fruit_to_props(self, self.fruit_species)
    finally:
        set_updating(False)
    if getattr(self, "auto_update", True):
        schedule_update()


def on_vine_species(self, context):
    if is_updating():
        return
    _sync_flowers(self)
    set_updating(True)
    try:
        write_vine_to_props(self, self.vine_species)
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
        # Dynamic items: presets imported or saved by the user appear without restarting Blender
        species_enum: EnumProperty(name="Species", items=enum_items("Tree"), default=item_number("quercus_robur"),
                                   update=on_species_change, description="Species preset")

        # Trait-space variation
        blend_species: EnumProperty(name="Blend With", items=enum_items("Tree"), default=item_number("fagus_sylvatica"),
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
        leaflet_count: I("Leaflets", 7, 1, 150)
        leaflet_angle: F("Leaflet Angle", 60.0, 5.0, 90.0)
        rachis_ratio: F("Rachis Length", 2.5, 0.2, 30.0)
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
        leaf_budget: IntProperty(name="Max Cards", default=150000, min=100, max=1000000, update=U,
                                 description="Upper bound on leaf cards (performance)")
        leaf_scale: F("Leaf Scale", 1.0, 0.2, 4.0)
        foliage_instancing: BoolProperty(name="Instanced Leaves", default=True, update=U,
                                         description="Instance one leaf card on every foliage point (Geometry "
                                                     "Nodes): far less memory and faster updates. Apply the "
                                                     "modifier to get real geometry for export")

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
        bark_blockiness: F("Blockiness", 0.4, 0.0, 1.0, "Irregular rounded cells .. rectangular blocks")
        bark_segments: F("Transverse Splits", 0.3, 0.0, 1.0, "Ridges cut into segments along the axis")
        bark_plate_tilt: F("Plate Tilt", 0.3, 0.0, 1.0, "Per-plate tilt and offset")
        bark_warp: F("Interlacing", 0.4, 0.0, 1.0, "Organic distortion; high = anastomosing ridges")
        bark_moss: F("Moss", 0.15, 0.0, 1.0, "Moss on upper, shaded and basal surfaces")
        bark_lichen: F("Lichen", 0.15, 0.0, 1.0, "Crustose lichen patches")
        bark_displacement: BoolProperty(name="Bark Displacement", default=False, update=U,
                                        description="True displacement (Cycles) with adaptive subdivision: "
                                                    "real relief on close-ups, slower renders")

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
             'FREEZE', 3),
            ('Vine', "Vine / Climber", "Twining, tendril, clinging or trailing plants grown along a guide path or "
             "curve", 'CURVE_BEZCURVE', 4),
            ('Fruit', "Fruit / Bunch", "A single fruit or bunch (apple, pear, pomegranate, grapes...) on its own",
             'SHADING_SOLID', 5),
            ('Vegetable', "Vegetable (root / tuber / head)", "Root crops (carrot, radish, beet), potato tubers and "
             "brassica heads (cauliflower, broccoli, Romanesco)", 'OUTLINER_OB_POINTCLOUD', 6),
            ('Grass', "Grass / Cereal", "Grasses: maize, wheat, barley, oats, rice, sorghum, sugarcane, lawn and "
             "ornamental grasses; single clump or lawn / meadow patch", 'STRANDS', 7)]),
        "grass_species": EnumProperty(name="Grass", items=enum_items("Grass"), default=item_number("zea_mays"),
                                      update=on_grass_species),
        "vegetable_species": EnumProperty(name="Vegetable", items=enum_items("Vegetable"),
                                          default=item_number("daucus_carota"), update=on_vegetable_species),
        "veg_lift": FloatProperty(name="Lift (show underground)", default=0.0, min=0.0, max=1.5, update=U,
                                  subtype='FACTOR', description="Raise the plant so its root or tubers stand above "
                                  "the ground plane (as when harvested); 1 = the whole storage organ"),
        "fruit_species": EnumProperty(name="Fruit", items=enum_items("Fruit"), default=item_number("malus_domestica"),
                                      update=on_fruit_species),
        "show_fruits": BoolProperty(name="Show Fruits", default=False, update=U,
                                    description="Hang the selected fruit (or bunch) on the plant"),
        "fruit_density": FloatProperty(name="Fruiting Density", default=0.3, min=0.0, max=1.0, update=U,
                                       description="Fraction of the flowering sites that bear a fruit (trees)"),
        "fruit_max": IntProperty(name="Max Fruits", default=1500, min=1, max=20000, update=U),
        "fruit_scale": FloatProperty(name="Fruit Scale", default=1.0, min=0.1, max=10.0, update=U),
        "fruit_detail": FloatProperty(name="Fruit Detail", default=1.0, min=0.3, max=3.0, update=U),
        "vine_species": EnumProperty(name="Vine", items=enum_items("Vine"), default=item_number("ipomoea_purpurea"),
                                     update=on_vine_species),
        "flower_species": EnumProperty(name="Flower", items=enum_items("Flower"), default=item_number("rosa_canina"),
                                       update=on_flower_species),
        "show_flowers": BoolProperty(name="Show Flowers", default=False, update=U,
                                     description="Place the selected inflorescence on the plant"),
        "flower_blend": EnumProperty(name="Blend With", items=enum_items("Flower"),
                                     default=item_number("hibiscus_rosa_sinensis")),
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
        "cactus_species": EnumProperty(name="Cactus", items=enum_items("Cactus"),
                                       default=item_number("carnegiea_gigantea"), update=on_succulent_species),
        "rosette_species": EnumProperty(name="Rosette", items=enum_items("Rosette"),
                                        default=item_number("echeveria_elegans"), update=on_succulent_species),
        "cactus_blend": EnumProperty(name="Blend With", items=enum_items("Cactus"),
                                     default=item_number("echinocactus_grusonii")),
        "rosette_blend": EnumProperty(name="Blend With", items=enum_items("Rosette"),
                                      default=item_number("agave_americana")),
        "succ_detail": FloatProperty(name="Mesh Detail", default=1.0, min=0.3, max=3.0, update=U,
                                     description="Surface sampling density of stems and leaves"),
        "spine_budget": IntProperty(name="Max Spines", default=60000, min=0, max=1000000, update=U,
                                    description="Radial spines are thinned evenly above this count"),
    })
    for _form in (GrowthForm.CACTUS, GrowthForm.ROSETTE):
        PPG_Properties.__annotations__.update(profile_properties(_form, U))
    PPG_Properties.__annotations__.update(flower_properties(U))
    PPG_Properties.__annotations__.update(vine_properties(U))
    PPG_Properties.__annotations__.update(fruit_properties(U))
    PPG_Properties.__annotations__.update(VG.vegetable_properties(U))
    PPG_Properties.__annotations__.update(GR.grass_properties(U))
    PPG_Properties.__annotations__.update(forest_properties())
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
        if props.growth_form == 'Fruit':
            box.prop(props, "fruit_species", text="")
            spec = FRUIT_CATALOG.get(props.fruit_species)
            if spec is None:
                box.label(text="Preset not found: choose one from the list", icon='ERROR')
                return
            col = box.column(align=True)
            col.scale_y = 0.8
            col.label(text=f"{spec.family} · {props.frt_kind}", icon='SHADING_SOLID')
            box.operator("ppg.apply_species_preset", text="Reload Fruit", icon='IMPORT')
            layout.operator("ppg.export_traits", text="Export Traits (JSON)", icon='TEXT')
            return
        if props.growth_form == 'Flower':
            box.prop(props, "flower_species", text="")
            spec = FLOWER_CATALOG.get(props.flower_species)
            if spec is None:
                box.label(text="Preset not found: choose one from the list", icon='ERROR')
                return
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
        elif props.growth_form == 'Grass':
            box.prop(props, "grass_species", text="")
            spec = GRASS_CATALOG.get(props.grass_species)
            if spec is None:
                box.label(text="Preset not found: choose one from the list", icon='ERROR')
                return
            family, habit, biome = spec.family, props.gra_head, spec.common_name
        elif props.growth_form == 'Vegetable':
            box.prop(props, "vegetable_species", text="")
            spec = VEGETABLE_CATALOG.get(props.vegetable_species)
            if spec is None:
                box.label(text="Preset not found: choose one from the list", icon='ERROR')
                return
            family, habit, biome = spec.family, props.veg_organ, spec.common_name
        elif props.growth_form == 'Vine':
            box.prop(props, "vine_species", text="")
            spec = VINE_CATALOG.get(props.vine_species)
            if spec is None:
                box.label(text="Preset not found: choose one from the list", icon='ERROR')
                return
            family, habit, biome = spec.family, props.vin_mode, spec.biome
        else:
            form = GrowthForm(props.growth_form)
            key = props.cactus_species if form == GrowthForm.CACTUS else props.rosette_species
            box.prop(props, "cactus_species" if form == GrowthForm.CACTUS else "rosette_species", text="")
            spec = CATALOGS[form].get(key)
            if spec is None:
                box.label(text="Preset not found: choose one from the list", icon='ERROR')
                return
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
        draw_tree_fields(col, p, ("crown_widest", "crown_fullness"))
        col.separator()
        draw_tree_fields(col, p, ("apical_dominance", "leader_count", "branch_levels", "branch_angle_deg",
                                  "twig_angle_deg", "branch_frequency", "branch_length_decay", "internode_length"))
        col.separator()
        draw_tree_fields(col, p, ("branch_gravity", "phototropism", "plagiotropy"))
        col.separator()
        draw_tree_fields(col, p, ("phyllotaxis_type", "phyllotaxis_angle_deg", "whorl_size"))
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
            col.prop(p, "foliage_instancing")
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
        draw_tree_fields(col, p, ("root_laterals", "root_spread", "root_max_depth", "root_beta",
                                  "root_taproot_share", "root_zrt", "root_sinker_spacing", "root_exposure",
                                  "root_plank", "root_buttress_height", "root_knees"))


class PPG_PT_Bark(_PPGSub, Panel):
    bl_label = "Bark"
    bl_idname = "PPG_PT_bark"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        p = context.scene.ppg_properties
        col = self.layout.column(align=True)
        draw_tree_fields(col, p, ("bark_pattern", "bark_color", "bark_color2", "bark_scale", "bark_relief",
                                  "bark_color_young", "bark_onset_cm", "bark_weathering", "bark_blockiness",
                                  "bark_segments", "bark_plate_tilt", "bark_warp", "bark_moss", "bark_lichen"))
        for name in ("bark_displacement",):
            col.prop(p, name)


class PPG_PT_Topology(_PPGSub, Panel):
    bl_label = "Topology & Shading"
    bl_idname = "PPG_PT_topology"
    bl_options = {'DEFAULT_CLOSED'}
    forms = ('Tree', 'Cactus', 'Rosette', 'Vine', 'Vegetable', 'Grass')

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
            if p.growth_form in ('Vine', 'Vegetable'):
                col.prop(p, "show_leaves")
            if p.growth_form == 'Vegetable':
                col.prop(p, "veg_lift", slider=True)
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


def _succulent_values(p, form):
    from dataclasses import fields as _fields
    from ..core.succulent_db import PROFILE_CLASSES
    return _section_values(p, PREFIX[form], [f.name for f in _fields(PROFILE_CLASSES[form])])


def _succulent_panel(form, index, title, names):
    def draw(self, context):
        p = context.scene.ppg_properties
        col = self.layout.column(align=True)
        draw_fields(col, p, form.value, "profile", [(PREFIX[form] + n, n) for n in names], _succulent_values(p, form))

    @classmethod
    def poll(cls, context):
        p = getattr(context.scene, "ppg_properties", None)
        if p is None or p.growth_form != form.value:
            return False
        values = _succulent_values(p, form)       # Hide sections where nothing applies (e.g. Cladodes)
        return any(applies(form.value, "profile", n, values) for n in names)
    return type(f"PPG_PT_{form.value}_{index}", (_PPGSub, Panel), {
        "bl_label": title, "bl_idname": f"PPG_PT_{form.value.lower()}_{index}",
        "bl_options": {'DEFAULT_CLOSED'} if index else set(), "forms": (form.value,), "draw": draw,
        "poll": poll})


class PPG_PT_Presets(_PPGSub, Panel):
    bl_label = "Presets (Import / Export)"
    bl_idname = "PPG_PT_presets"
    bl_options = {'DEFAULT_CLOSED'}
    forms = ('Tree', 'Cactus', 'Rosette', 'Flower', 'Vine', 'Fruit', 'Vegetable', 'Grass')

    def draw(self, context):
        draw_presets(self.layout, context.scene.ppg_properties)


class PPG_PT_Forest(_PPGSub, Panel):
    bl_label = "Forest"
    bl_idname = "PPG_PT_forest"
    bl_options = {'DEFAULT_CLOSED'}
    forms = ('Tree',)

    def draw(self, context):
        draw_forest_panel(self.layout, context.scene.ppg_properties)


class PPG_PT_Flowers(_PPGSub, Panel):
    bl_label = "Flowers"
    bl_idname = "PPG_PT_flowers"
    bl_options = {'DEFAULT_CLOSED'}
    forms = ('Tree', 'Cactus', 'Rosette', 'Flower', 'Vine')

    def draw_header(self, context):
        p = context.scene.ppg_properties
        if p.growth_form != 'Flower':
            self.layout.prop(p, "show_flowers", text="")

    def draw(self, context):
        p = context.scene.ppg_properties
        col = self.layout.column(align=True)
        if p.growth_form != 'Flower':
            col.prop(p, "flower_species", text="")
            spec = FLOWER_CATALOG.get(p.flower_species)
            col.label(text=spec.formula if spec else "Preset not found: choose one from the list")
            col.separator()
            col.prop(p, "flower_density", slider=True)
            col.prop(p, "flower_max")
        else:
            col.prop(p, "flower_single")
        col.prop(p, "flower_bloom", slider=True)
        col.prop(p, "flower_scale")
        col.prop(p, "flower_detail")


def _flower_values(p, prefix):
    from dataclasses import fields as _fields
    from ..core.flower import FlowerProfile
    from ..core.inflorescence import InflorescenceProfile
    cls = FlowerProfile if prefix == "flw_" else InflorescenceProfile
    return _section_values(p, prefix, [f.name for f in _fields(cls)])


def _flower_panel(index, title, prefix, names):
    section = "flower" if prefix == "flw_" else "infl"

    def draw(self, context):
        p = context.scene.ppg_properties
        col = self.layout.column(align=True)
        draw_fields(col, p, "Flower", section, [(prefix + n, n) for n in names], _flower_values(p, prefix))

    @classmethod
    def poll(cls, context):
        p = getattr(context.scene, "ppg_properties", None)
        if p is None or not (p.growth_form == 'Flower' or p.show_flowers):
            return False
        values = _flower_values(p, prefix)        # Hide sections where nothing applies (e.g. Capitulum)
        return any(applies("Flower", section, n, values) for n in names)
    return type(f"PPG_PT_Flower_{index}", (_PPGSub, Panel), {
        "bl_label": title, "bl_idname": f"PPG_PT_flower_{index}", "bl_parent_id": "PPG_PT_flowers",
        "bl_options": {'DEFAULT_CLOSED'}, "draw": draw, "poll": poll})


FLOWER_PANELS = tuple(_flower_panel(i, t, pre, n) for i, (t, pre, n) in enumerate(FLOWER_LAYOUT))


class PPG_PT_VineGuide(_PPGSub, Panel):
    bl_label = "Guide Path"
    bl_idname = "PPG_PT_vine_guide"
    forms = ('Vine',)

    def draw(self, context):
        p = context.scene.ppg_properties
        layout = self.layout
        layout.row().prop(p, "vine_guide_source", expand=True)
        col = layout.column(align=True)
        if p.vine_guide_source == 'SHAPE':
            col.prop(p, "vine_guide_shape")
            col.prop(p, "vine_guide_height")
            if p.vine_guide_shape != 'Pole':
                col.prop(p, "vine_guide_width")
            if p.vine_guide_shape == 'Spiral':
                col.prop(p, "vine_guide_turns")
            layout.operator("ppg.vine_make_curve", icon='CURVE_BEZCURVE')
        else:
            col.prop(p, "vine_curve", text="")
            row = col.row(align=True)
            row.operator("ppg.vine_use_selected", icon='EYEDROPPER')
            row.operator("ppg.vine_edit_guide", icon='EDITMODE_HLT', text="Edit")
            if p.vine_curve is None:
                r = col.row()
                r.active = False
                r.label(text="Pick a curve: each spline grows one stem", icon='INFO')
            else:
                r = col.row()
                r.active = False
                r.label(text=f"{len(p.vine_curve.data.splines)} spline(s): stems start at the first point",
                        icon='INFO')
        col = layout.column(align=True)
        col.prop(p, "vine_growth", slider=True)
        col.prop(p, "vine_leaf_density", slider=True)
        col.prop(p, "vine_flip_side")
        col.prop(p, "seed")


def _vine_panel(index, title, prefix, names):
    section = VINE_SECTION[prefix]

    def draw(self, context):
        p = context.scene.ppg_properties
        col = self.layout.column(align=True)
        draw_fields(col, p, "Vine", section, [(prefix + n, n) for n in names], section_values(p, prefix))

    @classmethod
    def poll(cls, context):
        p = getattr(context.scene, "ppg_properties", None)
        if p is None or p.growth_form != 'Vine':
            return False
        values = section_values(p, prefix)          # Hide sections where nothing applies (e.g. no fruits)
        return any(applies("Vine", section, n, values) for n in names)
    return type(f"PPG_PT_Vine_{index}", (_PPGSub, Panel), {
        "bl_label": title, "bl_idname": f"PPG_PT_vine_{index}", "bl_options": {'DEFAULT_CLOSED'},
        "forms": ('Vine',), "draw": draw, "poll": poll})


class PPG_PT_Fruits(_PPGSub, Panel):
    bl_label = "Fruits"
    bl_idname = "PPG_PT_fruits"
    bl_options = {'DEFAULT_CLOSED'}
    forms = ('Tree', 'Vine', 'Fruit')

    def draw_header(self, context):
        p = context.scene.ppg_properties
        if p.growth_form != 'Fruit':
            self.layout.prop(p, "show_fruits", text="")

    def draw(self, context):
        p = context.scene.ppg_properties
        col = self.layout.column(align=True)
        if p.growth_form != 'Fruit':
            col.prop(p, "fruit_species", text="")
            if p.growth_form == 'Tree':
                col.prop(p, "fruit_density", slider=True)
                col.prop(p, "fruit_max")
            else:
                r = col.row()
                r.active = False
                r.label(text="Fruits per stem: Stem & Nodes panel", icon='INFO')
        col.prop(p, "fruit_scale")
        col.prop(p, "fruit_detail")
        if p.growth_form == 'Fruit':
            col.prop(p, "seed")
            col.prop(p, "assign_materials")


def _fruit_panel(index, title, names):
    def draw(self, context):
        p = context.scene.ppg_properties
        col = self.layout.column(align=True)
        draw_fields(col, p, "Fruit", "fruit", [(FRT + n, n) for n in names], fruit_values(p))

    @classmethod
    def poll(cls, context):
        p = getattr(context.scene, "ppg_properties", None)
        if p is None or not (p.growth_form == 'Fruit' or (p.show_fruits and p.growth_form in ('Tree', 'Vine'))):
            return False
        values = fruit_values(p)
        return any(applies("Fruit", "fruit", n, values) for n in names)
    return type(f"PPG_PT_Fruit_{index}", (_PPGSub, Panel), {
        "bl_label": title, "bl_idname": f"PPG_PT_fruit_{index}", "bl_parent_id": "PPG_PT_fruits",
        "bl_options": {'DEFAULT_CLOSED'}, "draw": draw, "poll": poll})


FRUIT_PANELS = (PPG_PT_Fruits,) + tuple(_fruit_panel(i, t, n) for i, (t, n) in enumerate(FRUIT_LAYOUT))

def _veg_panel(index, title, prefix, names):
    section = VG.SECTION[prefix]

    def draw(self, context):
        p = context.scene.ppg_properties
        col = self.layout.column(align=True)
        draw_fields(col, p, "Vegetable", section, [(prefix + n, n) for n in names], VG.section_values(p, prefix))

    @classmethod
    def poll(cls, context):
        p = getattr(context.scene, "ppg_properties", None)
        if p is None or p.growth_form != 'Vegetable':
            return False
        values = VG.section_values(p, prefix)       # Hide what the storage organ does not use
        return any(applies("Vegetable", section, n, values) for n in names)
    return type(f"PPG_PT_Veg_{index}", (_PPGSub, Panel), {
        "bl_label": title, "bl_idname": f"PPG_PT_veg_{index}", "bl_options": {'DEFAULT_CLOSED'} if index else set(),
        "forms": ('Vegetable',), "draw": draw, "poll": poll})


class PPG_PT_GrassPatch(_PPGSub, Panel):
    bl_label = "Lawn / Meadow"
    bl_idname = "PPG_PT_grass_patch"
    forms = ('Grass',)

    def draw_header(self, context):
        self.layout.prop(context.scene.ppg_properties, "grass_patch", text="")

    def draw(self, context):
        p = context.scene.ppg_properties
        col = self.layout.column(align=True)
        col.active = p.grass_patch
        for name in ("grass_patch_size", "grass_patch_density", "grass_patch_spacing", "grass_patch_variants",
                     "grass_patch_scale_var"):
            col.prop(p, name)


def _grass_panel(index, title, names):
    def draw(self, context):
        p = context.scene.ppg_properties
        col = self.layout.column(align=True)
        draw_fields(col, p, "Grass", "profile", [(GR.GRA + n, n) for n in names], GR.grass_values(p))

    @classmethod
    def poll(cls, context):
        p = getattr(context.scene, "ppg_properties", None)
        if p is None or p.growth_form != 'Grass':
            return False
        values = GR.grass_values(p)
        return any(applies("Grass", "profile", n, values) for n in names)
    return type(f"PPG_PT_Grass_{index}", (_PPGSub, Panel), {
        "bl_label": title, "bl_idname": f"PPG_PT_grass_{index}", "bl_options": {'DEFAULT_CLOSED'} if index else set(),
        "forms": ('Grass',), "draw": draw, "poll": poll})


GRASS_PANELS = (PPG_PT_GrassPatch,) + tuple(_grass_panel(i, t, n) for i, (t, n) in enumerate(GR.LAYOUT))

VEG_PANELS = tuple(_veg_panel(i, t, pre, n) for i, (t, pre, n) in enumerate(VG.LAYOUT))

VINE_PANELS = (PPG_PT_VineGuide,) + tuple(_vine_panel(i, t, pre, n) for i, (t, pre, n) in enumerate(VINE_LAYOUT))

SUCCULENT_PANELS = tuple(_succulent_panel(f, i, t, n) for f in (GrowthForm.CACTUS, GrowthForm.ROSETTE)
                         for i, (t, n) in enumerate(LAYOUT[f]))

PANEL_CLASSES = (PPG_PT_MainPanel, PPG_PT_Variation, PPG_PT_Trunk, PPG_PT_Crown, PPG_PT_Leaf,
                 PPG_PT_Venation, PPG_PT_Foliage, PPG_PT_Roots, PPG_PT_Bark) + SUCCULENT_PANELS + VINE_PANELS + VEG_PANELS + GRASS_PANELS + \
                (PPG_PT_Flowers,) + FLOWER_PANELS + FRUIT_PANELS + (PPG_PT_Forest, PPG_PT_Presets, PPG_PT_Topology)
