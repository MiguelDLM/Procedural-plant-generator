"""
Procedural Plant Generator Runtime Engine.
Manages interactive state, re-entrancy locking, fast geometry updates, and empirical presets.
Guarantees zero UI deadlocks and instant responsive scrubbing.
"""

import time
import numpy as np

try:
    import bpy
    BLENDER_AVAILABLE = True
except ImportError:
    BLENDER_AVAILABLE = False

try:
    from ..core.species_db import get_species_preset, SPECIES_CATALOG
    from ..core.allometry import AllometricProfile
    from ..core.architecture import ArchitectureProfile, HalleOldemanModel, PhyllotaxisType
    from ..core.leaf_morphology import LeafMorphologyProfile, LeafArchetype, MarginType
    from ..core.leaf_venation import VenationProfile, VenationPattern
    from ..core.biomechanics import BiomechanicalProfile
    from ..core.species_preset import BotanicalSpeciesPreset
    from ..core.gielis import GielisProfile
    from ..core.plant_pipeline import BotanicalPlantPipeline
    from .mesh_builder import BlenderMeshBuilder
    from .materials import create_bark_material, create_foliage_material
except (ImportError, ValueError):
    from core.species_db import get_species_preset, SPECIES_CATALOG
    from core.allometry import AllometricProfile
    from core.architecture import ArchitectureProfile, HalleOldemanModel, PhyllotaxisType
    from core.leaf_morphology import LeafMorphologyProfile, LeafArchetype, MarginType
    from core.leaf_venation import VenationProfile, VenationPattern
    from core.biomechanics import BiomechanicalProfile
    from core.species_preset import BotanicalSpeciesPreset
    from core.gielis import GielisProfile
    from core.plant_pipeline import BotanicalPlantPipeline
    from blender.mesh_builder import BlenderMeshBuilder
    from blender.materials import create_bark_material, create_foliage_material


_IS_UPDATING = False


def is_updating() -> bool:
    """Check if a property update or preset load is currently running."""
    global _IS_UPDATING
    return _IS_UPDATING


def set_updating(val: bool):
    """Set the re-entrancy update lock."""
    global _IS_UPDATING
    _IS_UPDATING = val


def build_custom_preset_from_props(props) -> tuple[BotanicalSpeciesPreset, GielisProfile]:
    """Constructs dynamic botanical profiles directly from the user's interactive sliders."""
    base_preset = get_species_preset(props.species_enum)

    # User-adjusted allometry
    allometry = AllometricProfile(
        dbh_min_m=0.02,
        dbh_max_m=props.dbh_m * 2.0,
        dbh_default_m=props.dbh_m,
        height_max_m=max(props.tree_height_m, base_preset.allometry.height_max_m),
        pipe_exponent_delta=props.pipe_delta,
        crown_depth_ratio=max(0.1, min(0.9, (props.tree_height_m - props.crown_base_height_m) / max(0.5, props.tree_height_m))),
        buttress_amplitude=props.buttress_amplitude,
        buttress_decay=props.buttress_decay,
        wood_density_g_cm3=props.wood_density
    )

    # User-adjusted architecture
    architecture = ArchitectureProfile(
        model=base_preset.architecture.model,
        phyllotaxis=PhyllotaxisType.SPIRAL if abs(props.phyllotaxis_angle_deg - 137.5) < 10.0 else (
            PhyllotaxisType.DECUSSATE if abs(props.phyllotaxis_angle_deg - 90.0) < 10.0 else PhyllotaxisType.DISTICHOUS
        ),
        max_order=props.branch_levels,
        branch_angle_mean_deg=props.branch_angle_deg,
        branch_angle_std_deg=5.0,
        divergence_angle_deg=props.phyllotaxis_angle_deg,
        apical_dominance=props.apical_dominance,
        gravitropism=props.branch_gravity,
        internode_decay_per_order=props.branch_length_decay,
        crookedness=props.crookedness
    )

    # User-adjusted leaf morphology
    leaf_morph = LeafMorphologyProfile(
        archetype=LeafArchetype(props.leaf_archetype),
        margin_type=MarginType(props.margin_type),
        blade_length_cm=props.leaf_length_cm,
        aspect_ratio=props.leaf_aspect_ratio,
        teeth_count=props.teeth_count,
        transverse_curl=0.20,
        longitudinal_droop=props.leaf_droop
    )

    # Gielis buttress cross section profile
    gielis = GielisProfile(
        m=float(props.buttress_m),
        n1=props.gielis_n1,
        n2=props.gielis_n2,
        n3=props.gielis_n3,
        a=1.0,
        b=1.0
    )

    preset = BotanicalSpeciesPreset(
        scientific_name=base_preset.scientific_name,
        common_name=base_preset.common_name,
        family=base_preset.family,
        biome=base_preset.biome,
        growth_habit=base_preset.growth_habit,
        allometry=allometry,
        architecture=architecture,
        leaf_morphology=leaf_morph,
        venation=base_preset.venation,
        biomechanics=base_preset.biomechanics,
        po_growth_form=base_preset.po_growth_form
    )

    return preset, gielis


def apply_species_preset_to_props(props, species_key: str, context=None):
    """
    Safely writes preset values to scene properties without triggering
    recursive UI callbacks or freezing Blender.
    """
    global _IS_UPDATING
    was_updating = _IS_UPDATING
    _IS_UPDATING = True
    try:
        preset = get_species_preset(species_key)

        # Trunk & Allometry
        props.dbh_m = preset.allometry.dbh_default_m
        props.tree_height_m = preset.allometry.height_a * ((preset.allometry.dbh_default_m * 100) ** preset.allometry.height_b)
        props.crown_base_height_m = props.tree_height_m * (1.0 - preset.allometry.crown_depth_ratio)
        props.pipe_delta = preset.allometry.pipe_exponent_delta
        props.buttress_amplitude = preset.allometry.buttress_amplitude
        props.buttress_decay = preset.allometry.buttress_decay
        props.wood_density = preset.allometry.wood_density_g_cm3

        # Architecture
        props.branch_levels = preset.architecture.max_order
        props.branch_angle_deg = preset.architecture.branch_angle_mean_deg
        props.phyllotaxis_angle_deg = preset.architecture.divergence_angle_deg
        props.apical_dominance = preset.architecture.apical_dominance
        props.branch_gravity = preset.architecture.gravitropism
        props.branch_length_decay = preset.architecture.internode_decay_per_order
        props.crookedness = preset.architecture.crookedness

        # Foliage
        props.leaf_archetype = preset.leaf_morphology.archetype.value
        props.margin_type = preset.leaf_morphology.margin_type.value
        props.leaf_length_cm = preset.leaf_morphology.blade_length_cm
        props.leaf_aspect_ratio = preset.leaf_morphology.aspect_ratio
        props.teeth_count = preset.leaf_morphology.teeth_count
        props.leaf_droop = preset.leaf_morphology.longitudinal_droop

        # Gielis buttress defaults per species
        if "ficus" in species_key or "sequoia" in species_key:
            props.buttress_m = 6
        elif "quercus" in species_key:
            props.buttress_m = 4
        else:
            props.buttress_m = 0

    finally:
        _IS_UPDATING = was_updating

    # Trigger a single geometry update if auto_update is active
    if context and getattr(props, "auto_update", True):
        update_tree_geometry(context)


def update_tree_geometry(context):
    """
    Direct in-place update of tree mesh in active scene.
    Does NOT call any bpy.ops operators, avoiding undo deadlocks.
    """
    if not BLENDER_AVAILABLE:
        return

    props = getattr(context.scene, "ppg_properties", None)
    if not props:
        return

    try:
        preset, gielis = build_custom_preset_from_props(props)
        pipeline = BotanicalPlantPipeline(preset)

        # Generate procedural skeleton
        result = pipeline.generate(
            dbh_m=props.dbh_m,
            leaf_density=props.leaf_density,
            seed=props.seed
        )
        result.total_height_m = props.tree_height_m
        result.crown_depth_m = max(0.5, props.tree_height_m - props.crown_base_height_m)

        # Build or update mesh in viewport
        builder = BlenderMeshBuilder(
            result,
            radial_resolution=props.radial_resolution,
            buttress_profile=gielis
        )

        built = builder.build_or_update_plant(
            context=context,
            leaf_density=props.leaf_density,
            leaf_scale=props.leaf_scale,
            show_leaves=props.show_leaves,
            use_subsurf=props.use_subsurf
        )

        # Materials
        if props.assign_materials:
            bark_mat = create_bark_material()
            foliage_mat = create_foliage_material()

            if built.get("wood") and not built["wood"].data.materials:
                built["wood"].data.materials.append(bark_mat)
            if built.get("foliage") and not built["foliage"].data.materials:
                built["foliage"].data.materials.append(foliage_mat)

    except Exception as e:
        print(f"[PPG Error] Failed to update plant geometry: {e}")
