"""
Procedural Plant Generator runtime.

- PROP_MAP: single source of truth linking UI properties to preset trait paths
  (used both to load species into sliders and to build a preset from sliders).
- Debounced live updates: slider drags schedule one regeneration via a timer
  instead of rebuilding on every intermediate value.
- Leaf textures and materials are cached per plant and only regenerated when
  the parameters that affect them change.
"""

import copy
import hashlib
from enum import Enum

try:
    import bpy
    BLENDER_AVAILABLE = True
except ImportError:
    BLENDER_AVAILABLE = False

try:
    from ..core.species_db import get_species_preset
    from ..core.architecture import CROWN_SHAPE_PARAMS, CrownShape
    from ..core.leaf_morphology import ARCHETYPE_TEMPLATES, LeafArchetype
    from ..core.leaf_texture import LeafTextureEngine
    from ..core.gielis import GielisProfile
    from ..core.plant_pipeline import BotanicalPlantPipeline
    from ..core.trait_space import get_path, set_path, blend, mutate, blend_profile, mutate_profile
    from ..core.succulent_db import GrowthForm, CATALOGS, RANGES
    from .mesh_builder import BlenderMeshBuilder
    from .materials import create_bark_material, create_leaf_material, leaf_images_from_texture
except (ImportError, ValueError):
    from core.species_db import get_species_preset
    from core.architecture import CROWN_SHAPE_PARAMS, CrownShape
    from core.leaf_morphology import ARCHETYPE_TEMPLATES, LeafArchetype
    from core.leaf_texture import LeafTextureEngine
    from core.gielis import GielisProfile
    from core.plant_pipeline import BotanicalPlantPipeline
    from core.trait_space import get_path, set_path, blend, mutate, blend_profile, mutate_profile
    from core.succulent_db import GrowthForm, CATALOGS, RANGES
    from blender.mesh_builder import BlenderMeshBuilder
    from blender.materials import create_bark_material, create_leaf_material, leaf_images_from_texture


# (UI property, preset attribute path)
PROP_MAP: list[tuple[str, str]] = [
    ("pipe_delta", "allometry.pipe_exponent_delta"),
    ("buttress_m", "allometry.buttress_lobes"),
    ("buttress_amplitude", "allometry.buttress_amplitude"),
    ("buttress_decay", "allometry.buttress_decay"),
    ("wood_density", "biomechanics.wood_density_g_cm3"),
    ("arch_model", "architecture.model"),
    ("phyllotaxis_type", "architecture.phyllotaxis"),
    ("phyllotaxis_angle_deg", "architecture.divergence_angle_deg"),
    ("whorl_size", "architecture.whorl_size"),
    ("apical_dominance", "architecture.apical_dominance"),
    ("leader_count", "architecture.leader_count"),
    ("branch_levels", "architecture.max_order"),
    ("branch_angle_deg", "architecture.branch_angle_mean_deg"),
    ("twig_angle_deg", "architecture.twig_angle_mean_deg"),
    ("branch_frequency", "architecture.branch_frequency_per_meter"),
    ("branch_length_decay", "architecture.internode_decay_per_order"),
    ("internode_length", "architecture.internode_length_base_m"),
    ("branch_gravity", "architecture.gravitropism"),
    ("phototropism", "architecture.phototropism"),
    ("plagiotropy", "architecture.plagiotropy"),
    ("crookedness", "architecture.crookedness"),
    ("crown_widest", "architecture.crown_widest_position"),
    ("crown_fullness", "architecture.crown_fullness"),
    ("leaf_area_index", "architecture.leaf_area_index"),
    ("leaf_angle", "leaf_morphology.mean_leaf_angle_deg"),
    ("leaf_archetype", "leaf_morphology.archetype"),
    ("margin_type", "leaf_morphology.margin_type"),
    ("leaf_length_cm", "leaf_morphology.blade_length_cm"),
    ("leaf_aspect_ratio", "leaf_morphology.aspect_ratio"),
    ("leaf_widest", "leaf_morphology.widest_position"),
    ("leaf_base_angle", "leaf_morphology.base_angle_deg"),
    ("leaf_apex_angle", "leaf_morphology.apex_angle_deg"),
    ("leaf_base_curvature", "leaf_morphology.base_curvature"),
    ("leaf_apex_curvature", "leaf_morphology.apex_curvature"),
    ("leaf_cordate", "leaf_morphology.cordate_depth"),
    ("leaf_asymmetry", "leaf_morphology.base_asymmetry"),
    ("leaf_notch", "leaf_morphology.apex_notch"),
    ("leaf_falcate", "leaf_morphology.falcate_bend"),
    ("lobe_type", "leaf_morphology.lobe_type"),
    ("lobe_count", "leaf_morphology.lobe_count"),
    ("lobe_depth", "leaf_morphology.lobe_depth"),
    ("lobe_angle", "leaf_morphology.lobe_angle_deg"),
    ("lobe_spread", "leaf_morphology.lobe_spread_deg"),
    ("lobe_width", "leaf_morphology.lobe_width"),
    ("lobe_roundness", "leaf_morphology.lobe_roundness"),
    ("lobe_apex_angle", "leaf_morphology.lobe_apex_angle_deg"),
    ("teeth_count", "leaf_morphology.teeth_count"),
    ("tooth_height", "leaf_morphology.tooth_height_ratio"),
    ("tooth_skew", "leaf_morphology.tooth_skew"),
    ("compound_type", "leaf_morphology.compound_type"),
    ("leaflet_count", "leaf_morphology.leaflet_count"),
    ("leaflet_angle", "leaf_morphology.leaflet_angle_deg"),
    ("rachis_ratio", "leaf_morphology.rachis_length_ratio"),
    ("terminal_leaflet", "leaf_morphology.terminal_leaflet"),
    ("leaflet_gradient", "leaf_morphology.leaflet_size_gradient"),
    ("petiole_ratio", "leaf_morphology.petiole_length_ratio"),
    ("petiole_angle", "leaf_morphology.petiole_angle_deg"),
    ("leaf_curl", "leaf_morphology.transverse_curl"),
    ("leaf_droop", "leaf_morphology.longitudinal_droop"),
    ("leaf_undulation", "leaf_morphology.undulation_amplitude"),
    ("leaf_gloss", "leaf_morphology.gloss"),
    ("leaf_thickness", "leaf_morphology.thickness_mm"),
    ("color_adaxial", "leaf_morphology.adaxial_color"),
    ("color_abaxial", "leaf_morphology.abaxial_color"),
    ("color_vein", "leaf_morphology.vein_color"),
    ("color_autumn", "leaf_morphology.autumn_color"),
    ("vein_pattern", "venation.pattern"),
    ("vein_vla", "venation.vla_mm_per_mm2"),
    ("vein_pairs", "venation.secondary_vein_pairs"),
    ("vein_angle", "venation.divergence_angle_deg"),
    ("vein_curvature", "venation.secondary_curvature"),
    ("vein_reticulation", "venation.reticulation_density"),
    ("vein_contrast", "venation.vein_contrast"),
    ("bark_pattern", "bark.pattern"),
    ("bark_color", "bark.base_color"),
    ("bark_color2", "bark.secondary_color"),
    ("bark_scale", "bark.feature_scale_m"),
    ("bark_relief", "bark.relief"),
    ("bark_color_young", "bark.young_color"),
    ("bark_onset_cm", "bark.onset_radius_cm"),
    ("bark_weathering", "bark.weathering"),
    ("bark_blockiness", "bark.blockiness"),
    ("bark_segments", "bark.segments"),
    ("bark_plate_tilt", "bark.plate_tilt"),
    ("bark_warp", "bark.warp"),
    ("bark_moss", "bark.moss"),
    ("bark_lichen", "bark.lichen"),
    ("root_system", "roots.system"),
    ("root_laterals", "roots.lateral_count"),
    ("root_spread", "roots.spread_crown_ratio"),
    ("root_max_depth", "roots.max_depth_m"),
    ("root_beta", "roots.beta"),
    ("root_taproot_share", "roots.taproot_share"),
    ("root_zrt", "roots.zrt_dbh_ratio"),
    ("root_sinker_spacing", "roots.sinker_spacing_m"),
    ("root_exposure", "roots.surface_exposure"),
    ("root_plank", "roots.plank"),
    ("root_buttress_height", "roots.buttress_height_dbh"),
    ("root_knees", "roots.knees"),
]

_IS_UPDATING = False
_PENDING = False


def is_updating() -> bool:
    return _IS_UPDATING


def set_updating(val: bool):
    global _IS_UPDATING
    _IS_UPDATING = val


def _to_prop(value):
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, tuple):
        return tuple(float(v) for v in value)
    return value


def _from_prop(current, value):
    if isinstance(current, Enum):
        return type(current)(value)
    if isinstance(current, tuple):
        return tuple(float(v) for v in value)
    if isinstance(current, bool):
        return bool(value)
    if isinstance(current, int):
        return int(value)
    if isinstance(current, float):
        return float(value)
    return value


def write_preset_to_props(props, preset, context=None, include_dimensions: bool = True):
    """Copies preset traits into the UI properties without triggering updates."""
    global _IS_UPDATING
    was = _IS_UPDATING
    _IS_UPDATING = True
    try:
        for prop, path in PROP_MAP:
            if hasattr(props, prop):
                try:
                    setattr(props, prop, _to_prop(get_path(preset, path)))
                except (TypeError, ValueError) as e:
                    print(f"[PPG] Could not set {prop}: {e}")
        if include_dimensions:
            props.dbh_m = preset.allometry.dbh_default_m
            apply_allometry(props, preset)
    finally:
        _IS_UPDATING = was


def apply_allometry(props, preset=None):
    """Sets height, crown radius and crown base from the allometric model for the current DBH."""
    preset = preset or get_species_preset(props.species_enum)
    h, cr, cd = BotanicalPlantPipeline(preset).dimensions(props.dbh_m)
    props.tree_height_m = h
    props.crown_radius_m = cr
    props.crown_base_height_m = max(0.2, h - cd)


def apply_species_preset_to_props(props, species_key: str, context=None):
    write_preset_to_props(props, get_species_preset(species_key), context)
    if context and getattr(props, "auto_update", True):
        schedule_update()


def build_custom_preset_from_props(props):
    """Builds a full species preset from the current UI properties."""
    preset = copy.deepcopy(get_species_preset(props.species_enum))
    for prop, path in PROP_MAP:
        if hasattr(props, prop):
            current = get_path(preset, path)
            set_path(preset, path, _from_prop(current, getattr(props, prop)))
    preset.allometry.wood_density_g_cm3 = preset.biomechanics.wood_density_g_cm3
    preset.allometry.dbh_default_m = props.dbh_m
    gielis = GielisProfile(m=float(props.buttress_m), n1=props.gielis_n1, n2=props.gielis_n2,
                           n3=props.gielis_n3, a=1.0, b=1.0)
    return preset, gielis


def _succulent_key(props):
    form = GrowthForm(props.growth_form)
    return form, (props.cactus_species if form == GrowthForm.CACTUS else props.rosette_species)


def apply_succulent_preset(props, context=None):
    """Loads the selected cactus / rosette preset into its sliders."""
    from .succulents import write_profile_to_props
    form, key = _succulent_key(props)
    if key not in CATALOGS[form]:          # Stale selection (removed preset, older file): nothing to load
        return
    global _IS_UPDATING
    was = _IS_UPDATING
    _IS_UPDATING = True
    try:
        write_profile_to_props(props, form, CATALOGS[form][key].profile)
    finally:
        _IS_UPDATING = was
    if context and getattr(props, "auto_update", True):
        schedule_update()


def apply_variant_to_props(props, context=None):
    """Writes a blended and/or mutated variant (trait-space arithmetic) into the sliders."""
    if props.growth_form != 'Tree':
        from .succulents import profile_from_props, write_profile_to_props
        form, key = _succulent_key(props)
        prof = profile_from_props(props, form, key)
        other_key = props.cactus_blend if form == GrowthForm.CACTUS else props.rosette_blend
        if props.blend_factor > 0.0 and other_key != key:
            prof = blend_profile(prof, CATALOGS[form][other_key].profile, props.blend_factor, RANGES[form])
        if props.variation_amount > 0.0:
            prof = mutate_profile(prof, props.variation_amount, props.variation_seed, RANGES[form])
        global _IS_UPDATING
        was = _IS_UPDATING
        _IS_UPDATING = True
        try:
            write_profile_to_props(props, form, prof)
        finally:
            _IS_UPDATING = was
        if getattr(props, "auto_update", True):
            schedule_update()
        return
    base, _ = build_custom_preset_from_props(props)
    if props.blend_factor > 0.0 and props.blend_species != props.species_enum:
        other = get_species_preset(props.blend_species)
        base = blend(base, other, props.blend_factor)
        base.allometry.dbh_default_m = (1 - props.blend_factor) * props.dbh_m + \
            props.blend_factor * other.allometry.dbh_default_m
    if props.variation_amount > 0.0:
        base = mutate(base, props.variation_amount, seed=props.variation_seed)
    write_preset_to_props(props, base, context)
    if getattr(props, "auto_update", True):
        schedule_update()


def apply_leaf_template(props):
    tpl = ARCHETYPE_TEMPLATES.get(LeafArchetype(props.leaf_archetype), {})
    preset, _ = build_custom_preset_from_props(props)
    for k, v in tpl.items():
        setattr(preset.leaf_morphology, k, v)
    global _IS_UPDATING
    was = _IS_UPDATING
    _IS_UPDATING = True
    try:
        for prop, path in PROP_MAP:
            if path.startswith("leaf_morphology.") and hasattr(props, prop):
                setattr(props, prop, _to_prop(get_path(preset, path)))
    finally:
        _IS_UPDATING = was


def apply_crown_shape(props):
    p, k = CROWN_SHAPE_PARAMS[CrownShape(props.crown_shape)]
    global _IS_UPDATING
    was = _IS_UPDATING
    _IS_UPDATING = True
    try:
        props.crown_widest = p
        props.crown_fullness = k
    finally:
        _IS_UPDATING = was


# -----------------------------------------------------------------------------
# Debounced updates
# -----------------------------------------------------------------------------
def _deferred_update():
    global _PENDING
    _PENDING = False
    if _IS_UPDATING:
        return 0.05
    set_updating(True)
    try:
        update_tree_geometry(bpy.context)
    finally:
        set_updating(False)
    return None


def schedule_update(delay: float = 0.06):
    """Coalesces rapid slider changes into a single regeneration."""
    global _PENDING
    if not BLENDER_AVAILABLE:
        return
    if _PENDING:
        return
    _PENDING = True
    bpy.app.timers.register(_deferred_update, first_interval=delay)


# -----------------------------------------------------------------------------
# Geometry + materials
# -----------------------------------------------------------------------------
def _key(*parts) -> str:
    return hashlib.sha1(repr(parts).encode()).hexdigest()[:12]


def _ensure_leaf_material(root, preset, props, model, shoot_leaves):
    res = int(props.texture_resolution)
    key = _key(preset.leaf_morphology, preset.venation, res, round(props.senescence, 3), shoot_leaves)
    mat_name = f"PPG_Leaf_{key}"          # Shared by every plant with the same leaf parameters
    mat = bpy.data.materials.get(mat_name)
    if mat is not None and mat.get("ppg_key") == key:
        return mat
    tex = LeafTextureEngine(preset.leaf_morphology, preset.venation).render(
        resolution=res, senescence=props.senescence, model=model)
    color_img, height_img = leaf_images_from_texture(mat_name, tex)
    mat = create_leaf_material(mat_name, color_img, height_img, preset.leaf_morphology)
    mat["ppg_key"] = key
    return mat


def _bark_displacement_modifier(obj, enabled: bool):
    """True bark displacement needs geometry: an adaptive subdivision modifier (Cycles dices by screen size)."""
    if obj is None:
        return
    mod = obj.modifiers.get("PPG_BarkDisplace")
    if enabled:
        if mod is None:
            mod = obj.modifiers.new("PPG_BarkDisplace", 'SUBSURF')
        mod.levels = 0
        mod.render_levels = 2
        if hasattr(mod, "use_adaptive_subdivision"):
            mod.use_adaptive_subdivision = True
            if hasattr(mod, "adaptive_pixel_size"):
                mod.adaptive_pixel_size = 2.0
    elif mod is not None:
        obj.modifiers.remove(mod)


def _ensure_bark_material(root, preset, displacement=False):
    key = _key(preset.bark, "bark-3d-v5", displacement)
    name = f"PPG_Bark_{key}"               # Shared by every plant with the same bark parameters
    mat = bpy.data.materials.get(name)
    if mat is not None and mat.get("ppg_key") == key:
        return mat
    mat = create_bark_material(name, preset.bark, displacement=displacement)
    mat["ppg_key"] = key
    return mat


def _assign(obj, mat):
    if obj is None or mat is None:
        return
    mats = obj.data.materials
    if len(mats) == 0:
        mats.append(mat)
    elif mats[0] != mat:
        mats[0] = mat


def _tree_flowers(context, root, props, result):
    from .flowers import update_flowers_on_plant, hide_flowers, flower_from_props, flower_site_count
    from ..core.inflorescence import tree_flower_sites
    for c in root.children:
        if c.name.endswith("_Bloom"):
            c.hide_viewport = c.hide_render = True
    if not props.show_flowers:
        hide_flowers(root)
        return
    _, ip = flower_from_props(props)
    every = tree_flower_sites(result.skeleton_graph, ip, 10 ** 9, seed=props.seed)
    sites = tree_flower_sites(result.skeleton_graph, ip, flower_site_count(props, len(every)), seed=props.seed)
    update_flowers_on_plant(context, root, props, sites)


def update_tree_geometry(context):
    """Regenerates the active plant in place (no bpy.ops, safe inside update callbacks)."""
    if not BLENDER_AVAILABLE:
        return
    props = getattr(context.scene, "ppg_properties", None)
    if not props:
        return
    find = lambda ctx, name: BlenderMeshBuilder._find_root(None, ctx, name)  # noqa: E731
    if getattr(props, "growth_form", 'Tree') == 'Flower':
        try:
            from .flowers import update_flower_geometry
            update_flower_geometry(context, props, find)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[PPG Error] Failed to update flower geometry: {e}")
        return
    if getattr(props, "growth_form", 'Tree') != 'Tree':
        try:
            from .succulents import update_succulent_geometry
            update_succulent_geometry(context, props, find)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[PPG Error] Failed to update succulent geometry: {e}")
        return
    try:
        preset, gielis = build_custom_preset_from_props(props)
        pipeline = BotanicalPlantPipeline(preset)
        result = pipeline.generate(
            dbh_m=props.dbh_m,
            leaf_density=props.leaf_density if props.show_leaves else 0.0,
            seed=props.seed,
            height_m=props.tree_height_m,
            crown_radius_m=props.crown_radius_m,
            crown_depth_m=max(0.3, props.tree_height_m - props.crown_base_height_m),
            leaf_budget=props.leaf_budget,
            shoot_leaves=props.shoot_leaves if props.foliage_unit == 'SHOOT' else 0,
            roots=props.show_roots,
            root_display_depth_m=props.root_display_depth,
        )
        builder = BlenderMeshBuilder(result, radial_resolution=props.radial_resolution,
                                     buttress_profile=gielis, twig_resolution=props.twig_resolution)
        built = builder.build_or_update_plant(
            context=context, leaf_density=props.leaf_density, leaf_scale=props.leaf_scale,
            show_leaves=props.show_leaves, use_subsurf=props.use_subsurf, show_roots=props.show_roots,
            junction_quality=props.junction_quality, fuse_detail=props.fuse_detail,
            fuse_smoothing=props.fuse_smoothing, hero_min_radius=props.hero_min_radius_cm * 0.01,
            sleeve_detail=props.sleeve_detail, max_sleeves=props.hero_max_junctions,
            foliage_instanced=getattr(props, "foliage_instancing", False))

        if props.assign_materials:
            root = built["root"]
            disp = bool(getattr(props, "bark_displacement", False))
            bark = _ensure_bark_material(root, preset, disp)
            _assign(built["wood"], bark)
            _assign(built.get("roots"), bark)
            for obj in (built["wood"], built.get("roots")):
                _bark_displacement_modifier(obj, disp)
            if props.show_leaves:
                leaf_mat = _ensure_leaf_material(root, preset, props, result.leaf_engine.shape_model,
                                                 result.leaf_engine.shoot_leaves)
                _assign(built["foliage"], leaf_mat)
                _assign(built.get("leaf_card"), leaf_mat)
        _tree_flowers(context, built["root"], props, result)
    except Exception as e:  # Keep the UI responsive; report in console
        import traceback
        traceback.print_exc()
        print(f"[PPG Error] Failed to update plant geometry: {e}")
