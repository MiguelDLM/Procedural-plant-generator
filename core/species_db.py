"""
Botanical species catalog.

Each species is declared by measurable traits: a typical mature stem diameter
(DBH), the height and crown radius observed at that DBH, an asymptotic maximum
height, crown shape and branching traits, and leaf/venation/bark descriptors.
Allometric coefficients are fitted from these anchor values, so adding a
species only requires numbers that can be read from a flora, a forest
inventory (e.g. TALLO, Jucker et al. 2022) or a field measurement.

Values are typical adult, open-grown ranges compiled from general botanical
references; they are approximations meant to reproduce each species' habit and
leaf form, not statistical estimates from a specific dataset.
"""

import math

from .allometry import AllometricProfile
from .architecture import ArchitectureProfile, HalleOldemanModel, PhyllotaxisType, CROWN_SHAPE_PARAMS, CrownShape
from .leaf_morphology import (LeafMorphologyProfile, LeafArchetype, MarginType, LobeType, CompoundType,
                              ARCHETYPE_TEMPLATES)
from .leaf_venation import VenationProfile, VenationPattern
from .biomechanics import BiomechanicalProfile
from .bark import BarkProfile, BarkPattern
from .roots import RootProfile, RootSystemType, JACKSON_BETA
from .species_preset import BotanicalSpeciesPreset

M = HalleOldemanModel
P = PhyllotaxisType
A = LeafArchetype
MT = MarginType
V = VenationPattern
BP = BarkPattern
C = CrownShape


def _mix(a, b, t):
    return tuple(round(x * (1 - t) + y * t, 3) for x, y in zip(a, b))


def species(sci, common, family, biome, habit, *, dbh, height, crown_r, hmax, crown_depth=0.6,
            b=0.6, d=0.55, delta=2.3, buttress=(0.4, 12.0), buttress_lobes=0, crown=C.OVOID,
            arch=None, leaf=None, vein=None, bark=None, wood=(0.6, 11.0), lma=85.0,
            deciduous=True, notes="") -> BotanicalSpeciesPreset:
    d_cm = dbh * 100.0
    height = min(height, hmax * 0.97)
    height_a = -hmax * math.log(1.0 - height / hmax) / (d_cm ** b)
    allometry = AllometricProfile(
        dbh_min_m=0.03, dbh_max_m=round(dbh * 2.5, 3), dbh_default_m=dbh,
        height_a=round(height_a, 4), height_b=b, height_max_m=hmax,
        crown_radius_c=round(crown_r / (d_cm ** d), 4), crown_radius_d=d,
        crown_depth_ratio=crown_depth, pipe_exponent_delta=delta,
        buttress_amplitude=buttress[0], buttress_decay=buttress[1], buttress_lobes=buttress_lobes,
        wood_density_g_cm3=wood[0])

    p, k = CROWN_SHAPE_PARAMS[crown]
    arch_kw = dict(crown_widest_position=p, crown_fullness=k)
    arch_kw.update(arch or {})
    architecture = ArchitectureProfile(**arch_kw)
    if architecture.phyllotaxis == P.DECUSSATE:
        architecture.divergence_angle_deg = 90.0
    elif architecture.phyllotaxis == P.DISTICHOUS:
        architecture.divergence_angle_deg = 180.0

    leaf = dict(leaf or {})
    archetype = leaf.pop("archetype", A.OVATE)
    lkw = dict(ARCHETYPE_TEMPLATES.get(archetype, {}))
    lkw.update(leaf)
    adaxial = lkw.get("adaxial_color", (0.20, 0.36, 0.10))
    lkw.setdefault("abaxial_color", _mix(adaxial, (0.62, 0.68, 0.50), 0.4))
    lkw.setdefault("vein_color", _mix(adaxial, (0.78, 0.82, 0.42), 0.5))
    leaf_morph = LeafMorphologyProfile(archetype=archetype, **lkw)

    venation = VenationProfile(**(vein or {}))
    bark_profile = BarkProfile(**(bark or {}))
    mech = BiomechanicalProfile(wood_density_g_cm3=wood[0], youngs_modulus_gpa=wood[1],
                                leaf_mass_per_area_g_m2=lma)
    return BotanicalSpeciesPreset(
        scientific_name=sci, common_name=common, family=family, biome=biome, growth_habit=habit,
        allometry=allometry, architecture=architecture, leaf_morphology=leaf_morph, venation=venation,
        biomechanics=mech, bark=bark_profile, notes=notes, deciduous=deciduous)


# Recurring trait bundles
CONIFER_ARCH = dict(model=M.RAUH, phyllotaxis=P.WHORLED, whorl_size=5, apical_dominance=0.95,
                    branch_angle_mean_deg=82, twig_angle_mean_deg=55, plagiotropy=0.75, phototropism=0.2,
                    branch_frequency_per_meter=1.4, internode_length_base_m=0.6, internode_decay_per_order=0.55,
                    crookedness=0.06)
NEEDLE_VEIN = dict(pattern=V.PARALLELODROMOUS, vla_mm_per_mm2=2.5, secondary_vein_pairs=0,
                   primary_radius_mm=0.25, secondary_radius_mm=0.0, vein_contrast=0.25)


SPECIES_CATALOG: dict[str, BotanicalSpeciesPreset] = {
    # ------------------------------------------------------------------ Fagaceae
    "quercus_robur": species(
        "Quercus robur", "English Oak", "Fagaceae", "Temperate Deciduous", "Canopy Tree",
        dbh=0.8, height=24, crown_r=9.5, hmax=40, crown_depth=0.72, crown=C.SPHERICAL,
        buttress=(0.5, 10.0), buttress_lobes=4,
        arch=dict(leaf_area_index=5.0, model=M.RAUH, apical_dominance=0.3, branch_angle_mean_deg=58, twig_angle_mean_deg=55,
                  gravitropism=-0.05, crookedness=0.32, branch_frequency_per_meter=2.0, plagiotropy=0.2),
        leaf=dict(archetype=A.PINNATE_LOBED, margin_type=MT.ENTIRE, blade_length_cm=10, aspect_ratio=1.9,
                  widest_position=0.66, lobe_count=5, lobe_depth=0.5, lobe_roundness=0.9, lobe_apex_angle_deg=130,
                  cordate_depth=0.15, petiole_length_ratio=0.05, adaxial_color=(0.19, 0.31, 0.09),
                  autumn_color=(0.55, 0.38, 0.15), gloss=0.3),
        vein=dict(pattern=V.CRASPEDODROMOUS, vla_mm_per_mm2=7.4, secondary_vein_pairs=7, divergence_angle_deg=50,
                  primary_radius_mm=0.9, secondary_radius_mm=0.32),
        bark=dict(pattern=BP.FISSURED, base_color=(0.36, 0.33, 0.29), secondary_color=(0.14, 0.12, 0.10),
                  feature_scale_m=0.06, relief=0.85),
        wood=(0.71, 12.5), lma=95,
        notes="Broad decurrent crown with tortuous limbs; obovate pinnately lobed leaves with auriculate base."),
    "quercus_rubra": species(
        "Quercus rubra", "Northern Red Oak", "Fagaceae", "Temperate Deciduous", "Canopy Tree",
        dbh=0.7, height=25, crown_r=8.0, hmax=38, crown_depth=0.65, crown=C.SPHERICAL,
        arch=dict(leaf_area_index=5.0, model=M.RAUH, apical_dominance=0.45, branch_angle_mean_deg=55, crookedness=0.2,
                  branch_frequency_per_meter=2.2),
        leaf=dict(archetype=A.PINNATE_LOBED, margin_type=MT.SPINOSE, blade_length_cm=17, aspect_ratio=1.6,
                  widest_position=0.6, lobe_count=4, lobe_depth=0.58, lobe_roundness=0.35, lobe_apex_angle_deg=35,
                  lobe_angle_deg=60, teeth_count=4, tooth_height_ratio=0.03, petiole_length_ratio=0.2,
                  adaxial_color=(0.18, 0.33, 0.10), autumn_color=(0.62, 0.12, 0.06)),
        vein=dict(pattern=V.CRASPEDODROMOUS, vla_mm_per_mm2=6.5, secondary_vein_pairs=8, divergence_angle_deg=55),
        bark=dict(pattern=BP.FISSURED, base_color=(0.40, 0.38, 0.36), secondary_color=(0.18, 0.16, 0.15),
                  feature_scale_m=0.09, relief=0.6),
        wood=(0.63, 12.5), notes="Bristle-tipped acute lobes; brilliant red autumn colour."),
    "quercus_agrifolia": species(
        "Quercus agrifolia", "Coast Live Oak", "Fagaceae", "Mediterranean / California", "Canopy Tree",
        dbh=0.6, height=14, crown_r=9.0, hmax=25, crown_depth=0.8, crown=C.UMBRELLA,
        arch=dict(leaf_area_index=4.0, model=M.TROLL, apical_dominance=0.2, branch_angle_mean_deg=64, crookedness=0.45,
                  gravitropism=0.05, plagiotropy=0.45, branch_frequency_per_meter=2.4),
        leaf=dict(archetype=A.ELLIPTIC, mean_leaf_angle_deg=45, margin_type=MT.SPINOSE, blade_length_cm=5, aspect_ratio=1.5,
                  teeth_count=6, tooth_height_ratio=0.05, transverse_curl=0.6, petiole_length_ratio=0.12,
                  adaxial_color=(0.13, 0.25, 0.08), gloss=0.75, autumn_color=(0.45, 0.40, 0.18)),
        vein=dict(pattern=V.CRASPEDODROMOUS, vla_mm_per_mm2=9.8, secondary_vein_pairs=6, divergence_angle_deg=50),
        bark=dict(pattern=BP.FISSURED, base_color=(0.45, 0.43, 0.40), secondary_color=(0.22, 0.20, 0.18),
                  feature_scale_m=0.07, relief=0.5),
        wood=(0.76, 13.8), lma=145, deciduous=False,
        notes="Evergreen sclerophyll; convex spiny leaves; gnarled spreading limbs."),
    "fagus_sylvatica": species(
        "Fagus sylvatica", "European Beech", "Fagaceae", "Temperate Broadleaf", "Canopy Tree",
        dbh=0.7, height=28, crown_r=8.0, hmax=40, crown_depth=0.65, crown=C.OVOID,
        arch=dict(leaf_area_index=6.5, model=M.TROLL, phyllotaxis=P.DISTICHOUS, apical_dominance=0.5, branch_angle_mean_deg=55,
                  plagiotropy=0.85, gravitropism=0.05, crookedness=0.1, branch_frequency_per_meter=2.4),
        leaf=dict(archetype=A.ELLIPTIC, mean_leaf_angle_deg=25, margin_type=MT.SINUATE, blade_length_cm=8, aspect_ratio=1.6,
                  widest_position=0.45, teeth_count=7, tooth_height_ratio=0.04, petiole_length_ratio=0.12,
                  undulation_amplitude=0.12, adaxial_color=(0.20, 0.38, 0.10), autumn_color=(0.65, 0.35, 0.10)),
        vein=dict(pattern=V.CRASPEDODROMOUS, vla_mm_per_mm2=6.8, secondary_vein_pairs=7, divergence_angle_deg=42,
                  secondary_curvature=0.15, reticulation_density=0.3),
        bark=dict(pattern=BP.SMOOTH, base_color=(0.55, 0.55, 0.52), secondary_color=(0.40, 0.40, 0.38),
                  feature_scale_m=0.3, relief=0.12, roughness=0.6),
        wood=(0.72, 13.2), notes="Flat distichous leaf sprays; smooth silver-grey bark."),
    "castanea_sativa": species(
        "Castanea sativa", "Sweet Chestnut", "Fagaceae", "Temperate Deciduous", "Canopy Tree",
        dbh=1.0, height=25, crown_r=10.0, hmax=35, crown_depth=0.7, crown=C.SPHERICAL,
        arch=dict(leaf_area_index=5.0, model=M.RAUH, apical_dominance=0.4, branch_angle_mean_deg=52, crookedness=0.25),
        leaf=dict(archetype=A.LANCEOLATE, margin_type=MT.SERRATE, blade_length_cm=18, aspect_ratio=3.6,
                  widest_position=0.5, base_angle_deg=80, apex_angle_deg=40, teeth_count=20,
                  tooth_height_ratio=0.05, tooth_skew=0.82, petiole_length_ratio=0.1,
                  adaxial_color=(0.18, 0.34, 0.09), autumn_color=(0.70, 0.52, 0.15), gloss=0.55),
        vein=dict(pattern=V.CRASPEDODROMOUS, vla_mm_per_mm2=7.0, secondary_vein_pairs=18, divergence_angle_deg=55,
                  secondary_curvature=0.1, reticulation_density=0.35),
        bark=dict(pattern=BP.FISSURED, base_color=(0.40, 0.34, 0.28), secondary_color=(0.17, 0.14, 0.11),
                  feature_scale_m=0.08, relief=0.85),
        wood=(0.56, 9.5), notes="Aristate serrate margin; straight parallel secondaries ending in teeth."),
    # ------------------------------------------------------------------ Sapindaceae / Hippocastanaceae
    "acer_palmatum": species(
        "Acer palmatum", "Japanese Maple", "Sapindaceae", "Temperate Broadleaf", "Sub-canopy Tree",
        dbh=0.22, height=7, crown_r=4.0, hmax=12, crown_depth=0.85, crown=C.UMBRELLA,
        arch=dict(leaf_area_index=4.0, model=M.TROLL, phyllotaxis=P.DECUSSATE, apical_dominance=0.25, branch_angle_mean_deg=62,
                  plagiotropy=0.6, crookedness=0.22, branch_frequency_per_meter=3.0),
        leaf=dict(archetype=A.PALMATE_LOBED, mean_leaf_angle_deg=30, margin_type=MT.DOUBLY_SERRATE, blade_length_cm=6, lobe_count=7,
                  lobe_depth=0.78, lobe_spread_deg=270, lobe_width=0.45, lobe_apex_angle_deg=30,
                  teeth_count=16, tooth_height_ratio=0.05, petiole_length_ratio=0.65,
                  adaxial_color=(0.24, 0.40, 0.11), autumn_color=(0.72, 0.08, 0.06)),
        vein=dict(pattern=V.ACTINODROMOUS, vla_mm_per_mm2=8.8, secondary_vein_pairs=6, divergence_angle_deg=48),
        bark=dict(pattern=BP.SMOOTH, base_color=(0.42, 0.42, 0.36), secondary_color=(0.30, 0.32, 0.26),
                  feature_scale_m=0.2, relief=0.1, roughness=0.6),
        wood=(0.64, 10.8), lma=55, notes="Deeply palmatisect 7-lobed leaves; layered umbrella crown."),
    "acer_pseudoplatanus": species(
        "Acer pseudoplatanus", "Sycamore Maple", "Sapindaceae", "Temperate Deciduous", "Canopy Tree",
        dbh=0.7, height=25, crown_r=8.5, hmax=35, crown_depth=0.62, crown=C.SPHERICAL,
        arch=dict(leaf_area_index=5.5, model=M.RAUH, phyllotaxis=P.DECUSSATE, apical_dominance=0.5, branch_angle_mean_deg=50),
        leaf=dict(archetype=A.PALMATE_LOBED, mean_leaf_angle_deg=32, margin_type=MT.CRENATE, blade_length_cm=14, lobe_count=5,
                  lobe_depth=0.4, lobe_spread_deg=215, lobe_width=0.7, lobe_apex_angle_deg=60, teeth_count=10,
                  tooth_height_ratio=0.04, petiole_length_ratio=0.8, adaxial_color=(0.16, 0.30, 0.08),
                  autumn_color=(0.65, 0.50, 0.15)),
        vein=dict(pattern=V.ACTINODROMOUS, vla_mm_per_mm2=7.6, secondary_vein_pairs=7, divergence_angle_deg=46,
                  primary_radius_mm=1.05, secondary_radius_mm=0.32),
        bark=dict(pattern=BP.PLATED, base_color=(0.52, 0.47, 0.44), secondary_color=(0.36, 0.30, 0.27),
                  feature_scale_m=0.1, relief=0.45),
        wood=(0.63, 11.2), notes="Five shallow coarsely crenate lobes; flaking plated bark."),
    "acer_saccharum": species(
        "Acer saccharum", "Sugar Maple", "Sapindaceae", "Temperate Deciduous", "Canopy Tree",
        dbh=0.6, height=25, crown_r=7.5, hmax=35, crown_depth=0.65, crown=C.OVOID,
        arch=dict(leaf_area_index=6.0, model=M.RAUH, phyllotaxis=P.DECUSSATE, apical_dominance=0.55, branch_angle_mean_deg=45),
        leaf=dict(archetype=A.PALMATE_LOBED, mean_leaf_angle_deg=30, margin_type=MT.DENTATE, blade_length_cm=11, lobe_count=5,
                  lobe_depth=0.52, lobe_spread_deg=220, lobe_width=0.6, lobe_roundness=1.0,
                  lobe_apex_angle_deg=40, teeth_count=3, tooth_height_ratio=0.09, petiole_length_ratio=0.75,
                  adaxial_color=(0.18, 0.34, 0.10), autumn_color=(0.88, 0.36, 0.05)),
        vein=dict(pattern=V.ACTINODROMOUS, vla_mm_per_mm2=8.0, secondary_vein_pairs=6, divergence_angle_deg=50),
        bark=dict(pattern=BP.FISSURED, base_color=(0.45, 0.42, 0.38), secondary_color=(0.22, 0.20, 0.18),
                  feature_scale_m=0.1, relief=0.6),
        wood=(0.63, 12.6), notes="U-shaped rounded sinuses, few large teeth; orange-red autumn colour."),
    "aesculus_hippocastanum": species(
        "Aesculus hippocastanum", "Horse Chestnut", "Sapindaceae", "Temperate Deciduous", "Canopy Tree",
        dbh=0.8, height=24, crown_r=8.0, hmax=32, crown_depth=0.72, crown=C.OVOID,
        arch=dict(leaf_area_index=6.0, model=M.RAUH, phyllotaxis=P.DECUSSATE, apical_dominance=0.5, branch_angle_mean_deg=55,
                  gravitropism=0.15, branch_frequency_per_meter=2.0),
        leaf=dict(archetype=A.PALMATE_COMPOUND, mean_leaf_angle_deg=30, margin_type=MT.DOUBLY_SERRATE, blade_length_cm=20,
                  aspect_ratio=2.8, leaflet_count=7, teeth_count=30, tooth_height_ratio=0.025,
                  petiole_length_ratio=0.8, adaxial_color=(0.16, 0.32, 0.08), autumn_color=(0.70, 0.40, 0.10)),
        vein=dict(pattern=V.CRASPEDODROMOUS, vla_mm_per_mm2=6.5, secondary_vein_pairs=18, divergence_angle_deg=55,
                  secondary_curvature=0.2),
        bark=dict(pattern=BP.PLATED, base_color=(0.42, 0.36, 0.30), secondary_color=(0.24, 0.20, 0.16),
                  feature_scale_m=0.12, relief=0.5),
        wood=(0.51, 8.5), notes="Palmately compound leaves with 7 obovate leaflets."),
    # ------------------------------------------------------------------ Betulaceae / Salicaceae
    "betula_pendula": species(
        "Betula pendula", "Silver Birch", "Betulaceae", "Boreal / Temperate", "Canopy Tree",
        dbh=0.3, height=20, crown_r=4.2, hmax=28, crown_depth=0.68, crown=C.OVOID,
        arch=dict(leaf_area_index=3.2, model=M.RAUH, apical_dominance=0.75, branch_angle_mean_deg=42, twig_angle_mean_deg=40,
                  gravitropism=0.45, crookedness=0.12, branch_frequency_per_meter=3.0),
        leaf=dict(archetype=A.DELTOID, mean_leaf_angle_deg=50, margin_type=MT.DOUBLY_SERRATE, blade_length_cm=5, aspect_ratio=1.35,
                  widest_position=0.25, base_angle_deg=165, apex_angle_deg=50, apex_curvature=-0.6,
                  teeth_count=16, tooth_height_ratio=0.05, petiole_length_ratio=0.45,
                  adaxial_color=(0.24, 0.42, 0.12), autumn_color=(0.88, 0.72, 0.15)),
        vein=dict(pattern=V.CRASPEDODROMOUS, vla_mm_per_mm2=9.6, secondary_vein_pairs=7, divergence_angle_deg=42,
                  secondary_curvature=0.2),
        bark=dict(pattern=BP.LENTICELLED, base_color=(0.88, 0.86, 0.82), secondary_color=(0.12, 0.11, 0.10),
                  feature_scale_m=0.05, relief=0.3, roughness=0.55),
        wood=(0.62, 14.0), lma=72, notes="White lenticelled bark; pendulous twigs; rhombic-deltoid leaves."),
    "populus_tremula": species(
        "Populus tremula", "Common Aspen", "Salicaceae", "Temperate / Boreal", "Canopy Tree",
        dbh=0.35, height=20, crown_r=4.0, hmax=30, crown_depth=0.5, crown=C.OVOID,
        arch=dict(leaf_area_index=3.5, model=M.RAUH, apical_dominance=0.75, branch_angle_mean_deg=45),
        leaf=dict(archetype=A.ORBICULAR, mean_leaf_angle_deg=50, margin_type=MT.CRENATE, blade_length_cm=6, aspect_ratio=1.05,
                  teeth_count=12, tooth_height_ratio=0.05, petiole_length_ratio=1.0,
                  adaxial_color=(0.22, 0.40, 0.14), autumn_color=(0.95, 0.75, 0.15)),
        vein=dict(pattern=V.ACTINODROMOUS, vla_mm_per_mm2=8.2, secondary_vein_pairs=5, divergence_angle_deg=50,
                  secondary_curvature=0.55),
        bark=dict(pattern=BP.LENTICELLED, base_color=(0.62, 0.64, 0.55), secondary_color=(0.20, 0.20, 0.18),
                  feature_scale_m=0.08, relief=0.2, roughness=0.6),
        wood=(0.48, 10.5), lma=68, notes="Long flattened petiole (trembling leaves); coarsely crenate margin."),
    "populus_nigra_italica": species(
        "Populus nigra 'Italica'", "Lombardy Poplar", "Salicaceae", "Temperate (cultivated)", "Columnar Tree",
        dbh=0.6, height=28, crown_r=2.6, hmax=35, crown_depth=0.9, crown=C.COLUMNAR,
        arch=dict(leaf_area_index=4.5, model=M.RAUH, apical_dominance=0.7, branch_angle_mean_deg=16, twig_angle_mean_deg=25,
                  gravitropism=-0.4, branch_frequency_per_meter=3.5),
        leaf=dict(archetype=A.DELTOID, mean_leaf_angle_deg=55, margin_type=MT.CRENATE, blade_length_cm=7, teeth_count=20,
                  tooth_height_ratio=0.03, petiole_length_ratio=0.6, adaxial_color=(0.18, 0.36, 0.10),
                  autumn_color=(0.90, 0.75, 0.18)),
        vein=dict(pattern=V.CRASPEDODROMOUS, vla_mm_per_mm2=7.5, secondary_vein_pairs=7),
        bark=dict(pattern=BP.FISSURED, base_color=(0.30, 0.28, 0.25), secondary_color=(0.12, 0.11, 0.10),
                  feature_scale_m=0.08, relief=0.8),
        wood=(0.45, 8.5), notes="Fastigiate columnar habit with near-vertical branches."),
    "salix_babylonica": species(
        "Salix babylonica", "Weeping Willow", "Salicaceae", "Riparian Temperate", "Canopy Tree",
        dbh=0.6, height=12, crown_r=7.0, hmax=18, crown_depth=0.85, crown=C.WEEPING,
        arch=dict(leaf_area_index=4.0, model=M.RAUH, apical_dominance=0.25, branch_angle_mean_deg=45, twig_angle_mean_deg=28,
                  gravitropism=0.8, crookedness=0.18, branch_frequency_per_meter=3.0,
                  internode_decay_per_order=0.85),
        leaf=dict(archetype=A.LINEAR, mean_leaf_angle_deg=65, margin_type=MT.SERRATE, blade_length_cm=11, aspect_ratio=7.5,
                  teeth_count=40, tooth_height_ratio=0.02, petiole_length_ratio=0.06,
                  adaxial_color=(0.32, 0.46, 0.16), autumn_color=(0.85, 0.75, 0.25)),
        vein=dict(pattern=V.EUCAMPTODROMOUS, vla_mm_per_mm2=8.5, secondary_vein_pairs=16, divergence_angle_deg=40),
        bark=dict(pattern=BP.FISSURED, base_color=(0.38, 0.34, 0.28), secondary_color=(0.16, 0.14, 0.12),
                  feature_scale_m=0.07, relief=0.75),
        wood=(0.42, 7.0), notes="Long pendulous whip-like shoots hanging to the ground."),
    # ------------------------------------------------------------------ Other broadleaves
    "tilia_cordata": species(
        "Tilia cordata", "Small-leaved Lime", "Malvaceae", "Temperate Deciduous", "Canopy Tree",
        dbh=0.6, height=22, crown_r=7.0, hmax=32, crown_depth=0.72, crown=C.OVOID,
        arch=dict(leaf_area_index=6.5, model=M.TROLL, phyllotaxis=P.DISTICHOUS, apical_dominance=0.65, branch_angle_mean_deg=50,
                  plagiotropy=0.75, branch_frequency_per_meter=2.8),
        leaf=dict(archetype=A.CORDATE, mean_leaf_angle_deg=30, margin_type=MT.SERRATE, blade_length_cm=6, aspect_ratio=1.1,
                  base_asymmetry=0.3, cordate_depth=0.3, teeth_count=20, tooth_height_ratio=0.03,
                  petiole_length_ratio=0.6, adaxial_color=(0.17, 0.33, 0.09), autumn_color=(0.85, 0.70, 0.18)),
        vein=dict(pattern=V.ACTINODROMOUS, vla_mm_per_mm2=8.0, secondary_vein_pairs=6, divergence_angle_deg=45,
                  secondary_curvature=0.5),
        bark=dict(pattern=BP.FISSURED, base_color=(0.42, 0.40, 0.36), secondary_color=(0.20, 0.19, 0.17),
                  feature_scale_m=0.06, relief=0.55),
        wood=(0.49, 9.0), notes="Oblique cordate leaves with palinactinodromous basal veins."),
    "platanus_hispanica": species(
        "Platanus x hispanica", "London Plane", "Platanaceae", "Urban / Temperate", "Canopy Tree",
        dbh=0.9, height=30, crown_r=10.0, hmax=40, crown_depth=0.68, crown=C.SPHERICAL,
        arch=dict(leaf_area_index=5.0, model=M.RAUH, apical_dominance=0.45, branch_angle_mean_deg=55, crookedness=0.25,
                  branch_frequency_per_meter=1.8),
        leaf=dict(archetype=A.PALMATE_LOBED, margin_type=MT.DENTATE, blade_length_cm=18, lobe_count=5,
                  lobe_depth=0.35, lobe_spread_deg=200, lobe_width=0.8, lobe_apex_angle_deg=45, teeth_count=4,
                  tooth_height_ratio=0.06, petiole_length_ratio=0.4, adaxial_color=(0.20, 0.36, 0.10),
                  autumn_color=(0.62, 0.48, 0.18)),
        vein=dict(pattern=V.ACTINODROMOUS, vla_mm_per_mm2=6.0, secondary_vein_pairs=7),
        bark=dict(pattern=BP.PEELING, base_color=(0.72, 0.70, 0.55), secondary_color=(0.45, 0.42, 0.33),
                  feature_scale_m=0.15, relief=0.3, roughness=0.7),
        wood=(0.56, 9.8), notes="Camouflage exfoliating bark; broad shallowly 5-lobed leaves."),
    "liriodendron_tulipifera": species(
        "Liriodendron tulipifera", "Tulip Tree", "Magnoliaceae", "Temperate Deciduous", "Canopy Tree",
        dbh=0.8, height=35, crown_r=7.0, hmax=50, crown_depth=0.6, crown=C.OVOID,
        arch=dict(leaf_area_index=5.0, model=M.RAUH, apical_dominance=0.88, branch_angle_mean_deg=50, crookedness=0.1),
        leaf=dict(archetype=A.TULIP, margin_type=MT.ENTIRE, blade_length_cm=12, petiole_length_ratio=0.8,
                  adaxial_color=(0.20, 0.38, 0.12), autumn_color=(0.92, 0.75, 0.20)),
        vein=dict(pattern=V.BROCHIDODROMOUS, vla_mm_per_mm2=7.0, secondary_vein_pairs=6, divergence_angle_deg=55),
        bark=dict(pattern=BP.FISSURED, base_color=(0.45, 0.43, 0.38), secondary_color=(0.22, 0.20, 0.17),
                  feature_scale_m=0.07, relief=0.7),
        wood=(0.42, 10.9), notes="Truncate-emarginate four-lobed leaves; tall excurrent stem."),
    "liquidambar_styraciflua": species(
        "Liquidambar styraciflua", "Sweetgum", "Altingiaceae", "Temperate / Subtropical", "Canopy Tree",
        dbh=0.6, height=25, crown_r=6.0, hmax=40, crown_depth=0.75, crown=C.CONICAL,
        arch=dict(leaf_area_index=5.0, model=M.RAUH, apical_dominance=0.8, branch_angle_mean_deg=55),
        leaf=dict(archetype=A.PALMATE_LOBED, margin_type=MT.SERRATE, blade_length_cm=12, lobe_count=5,
                  lobe_depth=0.62, lobe_spread_deg=230, lobe_width=0.55, lobe_apex_angle_deg=28,
                  lobe_roundness=0.2, teeth_count=22, tooth_height_ratio=0.02, petiole_length_ratio=0.9,
                  gloss=0.65, adaxial_color=(0.16, 0.34, 0.10), autumn_color=(0.55, 0.08, 0.18)),
        vein=dict(pattern=V.ACTINODROMOUS, vla_mm_per_mm2=7.2, secondary_vein_pairs=8),
        bark=dict(pattern=BP.FISSURED, base_color=(0.40, 0.36, 0.32), secondary_color=(0.18, 0.15, 0.13),
                  feature_scale_m=0.05, relief=0.8),
        wood=(0.46, 9.0), notes="Star-shaped glossy leaves; purple-red autumn colour; pyramidal crown."),
    "ulmus_minor": species(
        "Ulmus minor", "Field Elm", "Ulmaceae", "Temperate Deciduous", "Canopy Tree",
        dbh=0.6, height=25, crown_r=7.0, hmax=32, crown_depth=0.7, crown=C.OVOID,
        arch=dict(leaf_area_index=5.0, model=M.TROLL, phyllotaxis=P.DISTICHOUS, apical_dominance=0.5, plagiotropy=0.85,
                  branch_angle_mean_deg=50, branch_frequency_per_meter=2.6),
        leaf=dict(archetype=A.ELLIPTIC, mean_leaf_angle_deg=32, margin_type=MT.DOUBLY_SERRATE, blade_length_cm=7, aspect_ratio=1.8,
                  base_asymmetry=0.7, apex_angle_deg=50, apex_curvature=-0.4, teeth_count=18,
                  tooth_height_ratio=0.05, petiole_length_ratio=0.08, adaxial_color=(0.16, 0.32, 0.08),
                  autumn_color=(0.80, 0.65, 0.18), gloss=0.5),
        vein=dict(pattern=V.CRASPEDODROMOUS, vla_mm_per_mm2=8.0, secondary_vein_pairs=12, divergence_angle_deg=45,
                  secondary_curvature=0.1, reticulation_density=0.3),
        bark=dict(pattern=BP.FISSURED, base_color=(0.36, 0.32, 0.28), secondary_color=(0.15, 0.13, 0.11),
                  feature_scale_m=0.06, relief=0.8),
        wood=(0.55, 9.3), notes="Strongly oblique leaf base; doubly serrate; straight secondaries."),
    "prunus_avium": species(
        "Prunus avium", "Wild Cherry", "Rosaceae", "Temperate Deciduous", "Canopy Tree",
        dbh=0.45, height=18, crown_r=6.0, hmax=25, crown_depth=0.65, crown=C.OVOID,
        arch=dict(leaf_area_index=4.0, model=M.RAUH, phyllotaxis=P.SPIRAL, whorl_size=3, apical_dominance=0.65,
                  branch_angle_mean_deg=48, branch_frequency_per_meter=1.6),
        leaf=dict(archetype=A.OBOVATE, margin_type=MT.DOUBLY_SERRATE, blade_length_cm=11, aspect_ratio=2.1,
                  widest_position=0.6, apex_angle_deg=45, apex_curvature=-0.7, teeth_count=26,
                  tooth_height_ratio=0.025, petiole_length_ratio=0.3, adaxial_color=(0.18, 0.34, 0.10),
                  autumn_color=(0.85, 0.30, 0.10)),
        vein=dict(pattern=V.BROCHIDODROMOUS, vla_mm_per_mm2=8.5, secondary_vein_pairs=10, divergence_angle_deg=50),
        bark=dict(pattern=BP.LENTICELLED, base_color=(0.40, 0.22, 0.18), secondary_color=(0.20, 0.12, 0.10),
                  feature_scale_m=0.06, relief=0.25, roughness=0.45),
        wood=(0.60, 10.2), notes="Glossy reddish bark with horizontal lenticel bands; drooping acuminate leaves."),
    "malus_domestica": species(
        "Malus domestica", "Apple", "Rosaceae", "Temperate (cultivated)", "Small Tree",
        dbh=0.3, height=7, crown_r=4.0, hmax=10, crown_depth=0.75, crown=C.SPHERICAL,
        arch=dict(leaf_area_index=3.5, model=M.RAUH, apical_dominance=0.2, branch_angle_mean_deg=58, crookedness=0.35,
                  branch_frequency_per_meter=2.6),
        leaf=dict(archetype=A.OVATE, margin_type=MT.CRENATE, blade_length_cm=7, aspect_ratio=1.6, teeth_count=22,
                  tooth_height_ratio=0.03, petiole_length_ratio=0.35, adaxial_color=(0.20, 0.36, 0.12),
                  autumn_color=(0.75, 0.62, 0.20)),
        vein=dict(pattern=V.BROCHIDODROMOUS, vla_mm_per_mm2=8.0, secondary_vein_pairs=7),
        bark=dict(pattern=BP.PLATED, base_color=(0.42, 0.38, 0.32), secondary_color=(0.25, 0.21, 0.17),
                  feature_scale_m=0.05, relief=0.5),
        wood=(0.68, 9.0), notes="Low spreading decurrent crown; scaly bark."),
    "pyrus_communis": species(
        "Pyrus communis", "Pear (peral)", "Rosaceae", "Temperate (cultivated)", "Small Tree",
        dbh=0.35, height=10, crown_r=3.5, hmax=15, crown_depth=0.75, crown=C.OVOID,
        arch=dict(leaf_area_index=3.5, model=M.RAUH, apical_dominance=0.45, branch_angle_mean_deg=40, crookedness=0.3,
                  branch_frequency_per_meter=2.4),
        leaf=dict(archetype=A.OVATE, margin_type=MT.CRENATE, blade_length_cm=6, aspect_ratio=1.5, teeth_count=18,
                  tooth_height_ratio=0.02, petiole_length_ratio=0.6, adaxial_color=(0.16, 0.32, 0.10), gloss=0.5,
                  autumn_color=(0.70, 0.35, 0.12)),
        vein=dict(pattern=V.BROCHIDODROMOUS, vla_mm_per_mm2=8.0, secondary_vein_pairs=7),
        bark=dict(pattern=BP.PLATED, base_color=(0.36, 0.32, 0.28), secondary_color=(0.16, 0.14, 0.12),
                  feature_scale_m=0.04, relief=0.8),
        wood=(0.69, 9.5), notes="Narrower, more upright crown than apple; bark in small square plates."),
    "punica_granatum": species(
        "Punica granatum", "Pomegranate (granado)", "Lythraceae", "Mediterranean / Western Asia", "Shrub",
        dbh=0.12, height=4, crown_r=2.0, hmax=6, crown_depth=0.85, crown=C.SPHERICAL,
        arch=dict(leaf_area_index=3.0, model=M.ATTIMS, phyllotaxis=P.DECUSSATE, apical_dominance=0.1,
                  branch_angle_mean_deg=45, crookedness=0.45, branch_frequency_per_meter=4.0, gravitropism=0.15),
        leaf=dict(archetype=A.LANCEOLATE, margin_type=MT.ENTIRE, blade_length_cm=5, aspect_ratio=3.5,
                  petiole_length_ratio=0.05, adaxial_color=(0.18, 0.36, 0.10), gloss=0.6,
                  autumn_color=(0.85, 0.70, 0.15)),
        vein=dict(pattern=V.BROCHIDODROMOUS, vla_mm_per_mm2=8.5, secondary_vein_pairs=8),
        bark=dict(pattern=BP.SMOOTH, base_color=(0.45, 0.40, 0.34), secondary_color=(0.30, 0.26, 0.22),
                  feature_scale_m=0.05, relief=0.3),
        wood=(0.80, 10.0), notes="Multi-stemmed shrub with opposite glossy leaves and drooping fruiting twigs."),
    "citrus_sinensis": species(
        "Citrus × sinensis", "Sweet orange (naranjo)", "Rutaceae", "Subtropical (cultivated)", "Small Tree",
        dbh=0.25, height=7, crown_r=3.2, hmax=10, crown_depth=0.85, crown=C.SPHERICAL,
        arch=dict(leaf_area_index=4.5, model=M.RAUH, apical_dominance=0.25, branch_angle_mean_deg=50, crookedness=0.3,
                  branch_frequency_per_meter=3.0),
        leaf=dict(archetype=A.ELLIPTIC, margin_type=MT.ENTIRE, blade_length_cm=9, aspect_ratio=2.0,
                  petiole_length_ratio=0.15, adaxial_color=(0.12, 0.28, 0.08), gloss=0.7),
        vein=dict(pattern=V.BROCHIDODROMOUS, vla_mm_per_mm2=7.5, secondary_vein_pairs=9),
        bark=dict(pattern=BP.SMOOTH, base_color=(0.40, 0.38, 0.32), secondary_color=(0.25, 0.24, 0.20),
                  feature_scale_m=0.05, relief=0.3),
        wood=(0.75, 11.0), lma=120, deciduous=False, notes="Dense rounded evergreen crown with glossy leaves."),
    "magnolia_grandiflora": species(
        "Magnolia grandiflora", "Southern Magnolia", "Magnoliaceae", "Subtropical Humid", "Canopy Tree",
        dbh=0.6, height=20, crown_r=6.0, hmax=28, crown_depth=0.82, crown=C.OVOID,
        arch=dict(leaf_area_index=5.5, model=M.RAUH, apical_dominance=0.75, branch_angle_mean_deg=52),
        leaf=dict(archetype=A.ELLIPTIC, mean_leaf_angle_deg=45, margin_type=MT.ENTIRE, blade_length_cm=18, aspect_ratio=2.2,
                  petiole_length_ratio=0.12, thickness_mm=0.5, transverse_curl=0.3, gloss=0.92,
                  adaxial_color=(0.08, 0.20, 0.06), abaxial_color=(0.55, 0.38, 0.22),
                  vein_color=(0.40, 0.48, 0.20), autumn_color=(0.55, 0.42, 0.18)),
        vein=dict(pattern=V.BROCHIDODROMOUS, vla_mm_per_mm2=7.0, secondary_vein_pairs=11, divergence_angle_deg=60,
                  vein_contrast=0.45),
        bark=dict(pattern=BP.SMOOTH, base_color=(0.45, 0.43, 0.40), secondary_color=(0.32, 0.30, 0.27),
                  feature_scale_m=0.15, relief=0.2),
        wood=(0.50, 9.7), lma=180, deciduous=False, notes="Leathery glossy leaves with rusty-tomentose underside."),
    "ficus_elastica": species(
        "Ficus elastica", "Rubber Fig", "Moraceae", "Tropical Rainforest", "Canopy Tree",
        dbh=0.6, height=24, crown_r=9.0, hmax=35, crown_depth=0.7, crown=C.UMBRELLA,
        buttress=(1.0, 8.0), buttress_lobes=5,
        arch=dict(leaf_area_index=4.5, model=M.RAUH, apical_dominance=0.4, branch_angle_mean_deg=55, max_order=2,
                  branch_frequency_per_meter=1.6),
        leaf=dict(archetype=A.ELLIPTIC, mean_leaf_angle_deg=45, margin_type=MT.ENTIRE, blade_length_cm=25, aspect_ratio=2.2,
                  apex_angle_deg=55, apex_curvature=-0.5, petiole_length_ratio=0.15, thickness_mm=0.6,
                  gloss=0.9, adaxial_color=(0.07, 0.20, 0.06), vein_color=(0.62, 0.62, 0.30)),
        vein=dict(pattern=V.BROCHIDODROMOUS, vla_mm_per_mm2=11.2, secondary_vein_pairs=20, divergence_angle_deg=75,
                  secondary_curvature=0.3, primary_radius_mm=1.4, vein_contrast=0.5),
        bark=dict(pattern=BP.SMOOTH, base_color=(0.48, 0.47, 0.43), secondary_color=(0.35, 0.34, 0.30),
                  feature_scale_m=0.2, relief=0.15),
        wood=(0.52, 9.5), lma=165, deciduous=False,
        notes="Fluted buttressed base; large leathery leaves with many near-perpendicular secondaries."),
    "olea_europaea": species(
        "Olea europaea", "Olive", "Oleaceae", "Mediterranean", "Small Tree",
        dbh=0.5, height=8, crown_r=4.0, hmax=12, crown_depth=0.8, crown=C.SPHERICAL,
        buttress=(0.6, 8.0), buttress_lobes=3,
        arch=dict(leaf_area_index=3.0, model=M.RAUH, phyllotaxis=P.DECUSSATE, apical_dominance=0.15, branch_angle_mean_deg=55,
                  crookedness=0.55, branch_frequency_per_meter=3.0, gravitropism=0.1),
        leaf=dict(archetype=A.LINEAR, mean_leaf_angle_deg=55, margin_type=MT.ENTIRE, blade_length_cm=6, aspect_ratio=6.0,
                  petiole_length_ratio=0.08, adaxial_color=(0.24, 0.31, 0.20), abaxial_color=(0.66, 0.68, 0.60),
                  vein_color=(0.40, 0.45, 0.32), gloss=0.4),
        vein=dict(pattern=V.BROCHIDODROMOUS, vla_mm_per_mm2=9.0, secondary_vein_pairs=8, vein_contrast=0.2),
        bark=dict(pattern=BP.FISSURED, base_color=(0.48, 0.46, 0.42), secondary_color=(0.22, 0.21, 0.19),
                  feature_scale_m=0.05, relief=0.9),
        wood=(0.85, 15.0), lma=200, deciduous=False, notes="Gnarled hollowing trunk; grey-green leaves silvery beneath."),
    "cercis_canadensis": species(
        "Cercis canadensis", "Eastern Redbud", "Fabaceae", "Temperate Deciduous", "Small Tree",
        dbh=0.25, height=8, crown_r=4.5, hmax=12, crown_depth=0.8, crown=C.UMBRELLA,
        arch=dict(leaf_area_index=3.5, model=M.TROLL, phyllotaxis=P.DISTICHOUS, apical_dominance=0.25, plagiotropy=0.7,
                  crookedness=0.25),
        leaf=dict(archetype=A.CORDATE, mean_leaf_angle_deg=30, margin_type=MT.ENTIRE, blade_length_cm=9, aspect_ratio=1.0,
                  apex_angle_deg=80, apex_curvature=0.0, cordate_depth=0.4, petiole_length_ratio=0.45,
                  adaxial_color=(0.20, 0.38, 0.12), autumn_color=(0.88, 0.75, 0.25)),
        vein=dict(pattern=V.ACTINODROMOUS, vla_mm_per_mm2=7.0, secondary_vein_pairs=5, secondary_curvature=0.6),
        bark=dict(pattern=BP.SMOOTH, base_color=(0.28, 0.24, 0.22), secondary_color=(0.18, 0.15, 0.13),
                  feature_scale_m=0.1, relief=0.25),
        wood=(0.63, 9.0), notes="Heart-shaped entire leaves with 5-7 palmate primary veins."),
    "eucalyptus_globulus": species(
        "Eucalyptus globulus", "Tasmanian Blue Gum", "Myrtaceae", "Mediterranean / Subtropical", "Canopy Tree",
        dbh=1.0, height=45, crown_r=9.0, hmax=65, crown_depth=0.5, crown=C.OVOID,
        arch=dict(leaf_area_index=2.5, model=M.ATTIMS, apical_dominance=0.8, branch_angle_mean_deg=40, gravitropism=0.3,
                  crookedness=0.2, branch_frequency_per_meter=1.6),
        leaf=dict(archetype=A.FALCATE, mean_leaf_angle_deg=72, margin_type=MT.ENTIRE, blade_length_cm=20, aspect_ratio=6.0,
                  petiole_length_ratio=0.15, petiole_angle_deg=80, longitudinal_droop=0.6,
                  adaxial_color=(0.26, 0.36, 0.30), abaxial_color=(0.30, 0.40, 0.33), vein_color=(0.62, 0.62, 0.40),
                  gloss=0.45),
        vein=dict(pattern=V.BROCHIDODROMOUS, vla_mm_per_mm2=12.4, secondary_vein_pairs=22, divergence_angle_deg=40,
                  secondary_curvature=0.2, vein_contrast=0.4),
        bark=dict(pattern=BP.PEELING, base_color=(0.78, 0.75, 0.66), secondary_color=(0.52, 0.48, 0.40),
                  feature_scale_m=0.25, relief=0.25, roughness=0.6),
        wood=(0.78, 16.5), lma=185, deciduous=False,
        notes="Pendulous falcate adult leaves with intramarginal vein; ribbon-peeling bark."),
    "ceiba_pentandra": species(
        "Ceiba pentandra", "Kapok / Ceiba", "Malvaceae", "Tropical (Mesoamerica)", "Emergent Tree",
        dbh=1.5, height=45, crown_r=15.0, hmax=60, crown_depth=0.4, crown=C.UMBRELLA,
        buttress=(1.8, 6.0), buttress_lobes=6,
        arch=dict(leaf_area_index=3.0, model=M.MASSART, phyllotaxis=P.WHORLED, whorl_size=4, apical_dominance=0.85,
                  branch_angle_mean_deg=80, plagiotropy=0.8, branch_frequency_per_meter=0.8),
        leaf=dict(archetype=A.PALMATE_COMPOUND, margin_type=MT.ENTIRE, blade_length_cm=12, aspect_ratio=4.0,
                  widest_position=0.5, leaflet_count=7, lobe_spread_deg=260, petiole_length_ratio=1.2,
                  adaxial_color=(0.18, 0.36, 0.10)),
        vein=dict(pattern=V.BROCHIDODROMOUS, vla_mm_per_mm2=7.0, secondary_vein_pairs=12),
        bark=dict(pattern=BP.SMOOTH, base_color=(0.55, 0.55, 0.50), secondary_color=(0.40, 0.40, 0.36),
                  feature_scale_m=0.2, relief=0.2),
        wood=(0.29, 4.5), lma=70, notes="Sacred Maya tree; huge plank buttresses and tiered horizontal limbs."),
    # ------------------------------------------------------------------ Compound-leaved
    "fraxinus_excelsior": species(
        "Fraxinus excelsior", "European Ash", "Oleaceae", "Temperate Deciduous", "Canopy Tree",
        dbh=0.6, height=28, crown_r=7.0, hmax=40, crown_depth=0.6, crown=C.OVOID,
        arch=dict(leaf_area_index=4.0, model=M.RAUH, phyllotaxis=P.DECUSSATE, apical_dominance=0.65, branch_angle_mean_deg=45,
                  branch_frequency_per_meter=1.8),
        leaf=dict(archetype=A.PINNATE_COMPOUND, margin_type=MT.SERRATE, blade_length_cm=8, aspect_ratio=3.2,
                  leaflet_count=5, teeth_count=18, tooth_height_ratio=0.03, rachis_length_ratio=3.0,
                  petiole_length_ratio=0.5, adaxial_color=(0.18, 0.36, 0.10), autumn_color=(0.70, 0.70, 0.30)),
        vein=dict(pattern=V.BROCHIDODROMOUS, vla_mm_per_mm2=8.0, secondary_vein_pairs=11),
        bark=dict(pattern=BP.FISSURED, base_color=(0.48, 0.46, 0.42), secondary_color=(0.24, 0.22, 0.20),
                  feature_scale_m=0.05, relief=0.6),
        wood=(0.68, 12.3), notes="Opposite odd-pinnate leaves with 9-13 serrate leaflets."),
    "juglans_regia": species(
        "Juglans regia", "Common Walnut", "Juglandaceae", "Temperate Deciduous", "Canopy Tree",
        dbh=0.8, height=20, crown_r=9.0, hmax=30, crown_depth=0.72, crown=C.SPHERICAL,
        arch=dict(leaf_area_index=4.0, model=M.RAUH, apical_dominance=0.35, branch_angle_mean_deg=58, crookedness=0.22,
                  branch_frequency_per_meter=1.6),
        leaf=dict(archetype=A.PINNATE_COMPOUND, margin_type=MT.ENTIRE, blade_length_cm=11, aspect_ratio=2.0,
                  leaflet_count=3, leaflet_size_gradient=0.45, rachis_length_ratio=2.2, petiole_length_ratio=0.5,
                  adaxial_color=(0.18, 0.34, 0.10), autumn_color=(0.60, 0.48, 0.18)),
        vein=dict(pattern=V.BROCHIDODROMOUS, vla_mm_per_mm2=7.0, secondary_vein_pairs=14, divergence_angle_deg=60),
        bark=dict(pattern=BP.FISSURED, base_color=(0.55, 0.53, 0.50), secondary_color=(0.28, 0.27, 0.25),
                  feature_scale_m=0.07, relief=0.6),
        wood=(0.64, 11.0), notes="Aromatic pinnate leaves with 5-9 entire leaflets; silvery fissured bark."),
    "robinia_pseudoacacia": species(
        "Robinia pseudoacacia", "Black Locust", "Fabaceae", "Temperate Deciduous", "Canopy Tree",
        dbh=0.5, height=20, crown_r=6.0, hmax=28, crown_depth=0.6, crown=C.OVOID,
        arch=dict(leaf_area_index=3.5, model=M.RAUH, apical_dominance=0.4, branch_angle_mean_deg=50, crookedness=0.35),
        leaf=dict(archetype=A.PINNATE_COMPOUND, margin_type=MT.ENTIRE, blade_length_cm=3.5, aspect_ratio=1.8,
                  widest_position=0.5, apex_notch=0.06, apex_angle_deg=130, apex_curvature=0.7,
                  leaflet_count=7, leaflet_angle_deg=80, leaflet_size_gradient=0.15, rachis_length_ratio=5.5,
                  petiole_length_ratio=0.6, adaxial_color=(0.24, 0.42, 0.14), autumn_color=(0.85, 0.80, 0.30)),
        vein=dict(pattern=V.BROCHIDODROMOUS, vla_mm_per_mm2=8.0, secondary_vein_pairs=6),
        bark=dict(pattern=BP.FIBROUS, base_color=(0.42, 0.36, 0.30), secondary_color=(0.18, 0.15, 0.12),
                  feature_scale_m=0.06, relief=0.9),
        wood=(0.72, 14.0), notes="Odd-pinnate leaves with 11-21 small elliptic leaflets; deeply furrowed bark."),
    # ------------------------------------------------------------------ Gymnosperms
    "ginkgo_biloba": species(
        "Ginkgo biloba", "Maidenhair Tree", "Ginkgoaceae", "Temperate Relict", "Canopy Tree",
        dbh=0.6, height=25, crown_r=6.0, hmax=35, crown_depth=0.7, crown=C.OVOID,
        arch=dict(leaf_area_index=4.0, model=M.RAUH, apical_dominance=0.7, branch_angle_mean_deg=48, max_order=2,
                  branch_frequency_per_meter=1.8),
        leaf=dict(archetype=A.FLABELLATE, mean_leaf_angle_deg=45, margin_type=MT.ENTIRE, blade_length_cm=6, petiole_length_ratio=0.9,
                  adaxial_color=(0.26, 0.42, 0.14), autumn_color=(0.95, 0.80, 0.15)),
        vein=dict(pattern=V.FLABELLATE, vla_mm_per_mm2=5.2, secondary_radius_mm=0.15, vein_contrast=0.45),
        bark=dict(pattern=BP.FISSURED, base_color=(0.45, 0.42, 0.38), secondary_color=(0.22, 0.20, 0.18),
                  feature_scale_m=0.06, relief=0.65),
        wood=(0.54, 9.8), lma=110, notes="Fan-shaped bilobed leaves with open dichotomous venation."),
    "pinus_sylvestris": species(
        "Pinus sylvestris", "Scots Pine", "Pinaceae", "Boreal / Montane Conifer", "Canopy Tree",
        dbh=0.5, height=25, crown_r=4.5, hmax=36, crown_depth=0.4, crown=C.UMBRELLA,
        arch=dict(CONIFER_ARCH, leaf_area_index=3.5, apical_dominance=0.8, branch_angle_mean_deg=70, gravitropism=-0.1,
                  crookedness=0.2, crown_widest_position=0.6, crown_fullness=1.2),
        leaf=dict(archetype=A.NEEDLE_SPRAY, mean_leaf_angle_deg=57, petiole_radius_mm=0.9, blade_length_cm=5, aspect_ratio=50, leaflet_count=18,
                  leaflet_angle_deg=45, rachis_length_ratio=3.0, petiole_length_ratio=0.1,
                  adaxial_color=(0.24, 0.35, 0.22), autumn_color=(0.55, 0.45, 0.20)),
        vein=NEEDLE_VEIN,
        bark=dict(pattern=BP.PLATED, base_color=(0.62, 0.36, 0.22), secondary_color=(0.30, 0.22, 0.17),
                  feature_scale_m=0.08, relief=0.6),
        wood=(0.51, 12.0), lma=230, deciduous=False,
        notes="Needles in pairs; orange flaking upper bark; flat-topped crown with age."),
    "pinus_pinea": species(
        "Pinus pinea", "Stone Pine", "Pinaceae", "Mediterranean", "Canopy Tree",
        dbh=0.7, height=18, crown_r=9.0, hmax=25, crown_depth=0.32, crown=C.UMBRELLA,
        arch=dict(CONIFER_ARCH, leaf_area_index=4.0, apical_dominance=0.25, leader_count=4, branch_angle_mean_deg=55,
                  gravitropism=-0.15, plagiotropy=0.3, crown_widest_position=0.75, crown_fullness=1.8),
        leaf=dict(archetype=A.NEEDLE_SPRAY, mean_leaf_angle_deg=57, petiole_radius_mm=1.0, blade_length_cm=14, aspect_ratio=90, leaflet_count=14,
                  leaflet_angle_deg=38, rachis_length_ratio=1.4, petiole_length_ratio=0.05,
                  adaxial_color=(0.22, 0.36, 0.15)),
        vein=NEEDLE_VEIN,
        bark=dict(pattern=BP.PLATED, base_color=(0.50, 0.32, 0.24), secondary_color=(0.20, 0.14, 0.11),
                  feature_scale_m=0.12, relief=0.8),
        wood=(0.55, 10.0), lma=250, deciduous=False, notes="Umbrella crown of several ascending leaders."),
    "picea_abies": species(
        "Picea abies", "Norway Spruce", "Pinaceae", "Boreal / Montane Conifer", "Canopy Tree",
        dbh=0.5, height=32, crown_r=3.8, hmax=50, crown_depth=0.85, crown=C.CONICAL,
        arch=dict(CONIFER_ARCH, leaf_area_index=8.0, gravitropism=0.25, branch_angle_mean_deg=85, twig_angle_mean_deg=60,
                  internode_decay_per_order=0.6, branch_frequency_per_meter=1.6),
        leaf=dict(archetype=A.NEEDLE_SPRAY, mean_leaf_angle_deg=55, petiole_radius_mm=0.7, blade_length_cm=2, aspect_ratio=18, leaflet_count=24,
                  leaflet_angle_deg=60, rachis_length_ratio=8.0, petiole_length_ratio=0.1,
                  adaxial_color=(0.12, 0.25, 0.10), autumn_color=(0.40, 0.35, 0.15)),
        vein=NEEDLE_VEIN,
        bark=dict(pattern=BP.PLATED, base_color=(0.42, 0.30, 0.24), secondary_color=(0.24, 0.18, 0.15),
                  feature_scale_m=0.04, relief=0.4),
        wood=(0.43, 11.0), lma=220, deciduous=False, notes="Narrow cone; drooping comb-like branchlets."),
    "sequoiadendron_giganteum": species(
        "Sequoiadendron giganteum", "Giant Sequoia", "Cupressaceae", "Montane Sierra Nevada", "Canopy Giant",
        dbh=3.0, height=75, crown_r=9.0, hmax=95, crown_depth=0.55, crown=C.OVOID, delta=2.5,
        buttress=(0.95, 8.0), buttress_lobes=8,
        arch=dict(CONIFER_ARCH, leaf_area_index=6.0, apical_dominance=0.92, branch_angle_mean_deg=80, gravitropism=0.1,
                  internode_length_base_m=0.9, branch_frequency_per_meter=0.9),
        leaf=dict(archetype=A.SCALE_SPRAY, mean_leaf_angle_deg=60, petiole_radius_mm=0.5, blade_length_cm=0.6, aspect_ratio=4.0, leaflet_count=40,
                  leaflet_angle_deg=25, rachis_length_ratio=15.0, petiole_length_ratio=0.0,
                  adaxial_color=(0.24, 0.38, 0.26)),
        vein=NEEDLE_VEIN,
        bark=dict(pattern=BP.FIBROUS, base_color=(0.55, 0.30, 0.20), secondary_color=(0.30, 0.16, 0.11),
                  feature_scale_m=0.12, relief=0.95),
        wood=(0.38, 9.2), lma=260, deciduous=False,
        notes="Largest tree by volume; fluted base; thick fibrous fire-resistant bark."),
    "taxodium_mucronatum": species(
        "Taxodium mucronatum", "Ahuehuete / Montezuma Cypress", "Cupressaceae", "Riparian Mexico", "Canopy Giant",
        dbh=2.0, height=30, crown_r=12.0, hmax=45, crown_depth=0.75, crown=C.SPHERICAL,
        buttress=(1.2, 6.0), buttress_lobes=6,
        arch=dict(leaf_area_index=5.0, model=M.RAUH, apical_dominance=0.3, branch_angle_mean_deg=62, gravitropism=0.45,
                  crookedness=0.3, branch_frequency_per_meter=1.6, internode_decay_per_order=0.75),
        leaf=dict(archetype=A.NEEDLE_SPRAY, mean_leaf_angle_deg=55, petiole_radius_mm=0.4, blade_length_cm=1.5, aspect_ratio=12, leaflet_count=22,
                  leaflet_angle_deg=78, rachis_length_ratio=7.0, petiole_length_ratio=0.05,
                  adaxial_color=(0.25, 0.42, 0.15), autumn_color=(0.60, 0.40, 0.18)),
        vein=NEEDLE_VEIN,
        bark=dict(pattern=BP.FIBROUS, base_color=(0.48, 0.34, 0.26), secondary_color=(0.25, 0.17, 0.13),
                  feature_scale_m=0.08, relief=0.85),
        wood=(0.45, 8.0), lma=120,
        notes="National tree of Mexico (Arbol del Tule); flat feathery sprays on weeping branchlets."),
    "cupressus_sempervirens": species(
        "Cupressus sempervirens", "Italian Cypress", "Cupressaceae", "Mediterranean", "Columnar Tree",
        dbh=0.4, height=20, crown_r=1.6, hmax=30, crown_depth=0.92, crown=C.COLUMNAR,
        arch=dict(CONIFER_ARCH, leaf_area_index=8.0, phyllotaxis=P.DECUSSATE, whorl_size=2, apical_dominance=0.9,
                  branch_angle_mean_deg=15, twig_angle_mean_deg=25, gravitropism=-0.4, plagiotropy=0.1,
                  branch_frequency_per_meter=4.0),
        leaf=dict(archetype=A.SCALE_SPRAY, mean_leaf_angle_deg=65, petiole_radius_mm=0.4, blade_length_cm=0.3, aspect_ratio=2.5, leaflet_count=40,
                  leaflet_angle_deg=20, rachis_length_ratio=18.0, petiole_length_ratio=0.0,
                  adaxial_color=(0.12, 0.24, 0.10)),
        vein=NEEDLE_VEIN,
        bark=dict(pattern=BP.FIBROUS, base_color=(0.42, 0.36, 0.30), secondary_color=(0.22, 0.18, 0.15),
                  feature_scale_m=0.05, relief=0.6),
        wood=(0.55, 9.0), lma=240, deciduous=False, notes="Fastigiate dark-green column of scale-leaf sprays."),
    # ------------------------------------------------------------------ Monocots
    "rhizophora_mangle": species(
        "Rhizophora mangle", "Red mangrove (mangle rojo)", "Rhizophoraceae", "Tropical Coast (mangrove)", "Tree",
        dbh=0.25, height=10, crown_r=4.0, hmax=20, crown_depth=0.6, crown=C.SPHERICAL,
        arch=dict(leaf_area_index=3.5, model=M.RAUH, phyllotaxis=P.DECUSSATE, apical_dominance=0.3,
                  branch_angle_mean_deg=55, crookedness=0.35, branch_frequency_per_meter=2.0),
        leaf=dict(archetype=A.ELLIPTIC, mean_leaf_angle_deg=45, margin_type=MT.ENTIRE, blade_length_cm=11,
                  aspect_ratio=2.3, petiole_length_ratio=0.15, adaxial_color=(0.16, 0.32, 0.12),
                  abaxial_color=(0.40, 0.48, 0.25), gloss=0.6),
        vein=dict(pattern=V.BROCHIDODROMOUS, vla_mm_per_mm2=6.0, secondary_vein_pairs=10, vein_contrast=0.1),
        bark=dict(pattern=BP.FISSURED, base_color=(0.42, 0.36, 0.30), secondary_color=(0.25, 0.20, 0.16),
                  feature_scale_m=0.04, relief=0.4),
        wood=(0.9, 15.0), lma=180, deciduous=False,
        notes="Arching stilt roots (rhizophores) prop the trunk up to 10-33 % of its height and drop roots "
              "hang from the branches into the mud (Mendez-Alonzo et al. 2015, Ann. Bot. 115: 833)."),
    "avicennia_germinans": species(
        "Avicennia germinans", "Black mangrove (mangle negro)", "Acanthaceae", "Tropical Coast (mangrove)", "Tree",
        dbh=0.3, height=9, crown_r=4.0, hmax=20, crown_depth=0.65, crown=C.SPHERICAL,
        arch=dict(leaf_area_index=3.0, model=M.RAUH, phyllotaxis=P.DECUSSATE, apical_dominance=0.25,
                  branch_angle_mean_deg=50, crookedness=0.4, branch_frequency_per_meter=2.5),
        leaf=dict(archetype=A.ELLIPTIC, mean_leaf_angle_deg=50, margin_type=MT.ENTIRE, blade_length_cm=8,
                  aspect_ratio=2.8, petiole_length_ratio=0.12, adaxial_color=(0.20, 0.33, 0.16),
                  abaxial_color=(0.62, 0.64, 0.58), gloss=0.4),
        vein=dict(pattern=V.BROCHIDODROMOUS, vla_mm_per_mm2=6.0, secondary_vein_pairs=8, vein_contrast=0.1),
        bark=dict(pattern=BP.FISSURED, base_color=(0.30, 0.26, 0.22), secondary_color=(0.16, 0.13, 0.10),
                  feature_scale_m=0.03, relief=0.5),
        wood=(0.85, 14.0), lma=170, deciduous=False,
        notes="Shallow cable roots radiating from the trunk carry hundreds of pencil-like pneumatophores "
              "rising 10-30 cm from the mud; leaves salt-crusted and pale beneath."),
    "phoenix_canariensis": species(
        "Phoenix canariensis", "Canary Island Date Palm", "Arecaceae", "Subtropical / Mediterranean", "Palm",
        dbh=0.8, height=15, crown_r=5.0, hmax=20, crown_depth=0.3, crown=C.SPHERICAL, buttress=(0.25, 15.0),
        arch=dict(leaf_area_index=2.0, model=M.CORNER, max_order=0, apical_dominance=1.0, crookedness=0.05),
        leaf=dict(archetype=A.PALM_FROND, blade_length_cm=40, aspect_ratio=14, leaflet_count=85,
                  leaflet_angle_deg=48, petiole_radius_mm=12.0, rachis_length_ratio=10.0, petiole_length_ratio=1.2, longitudinal_droop=0.7,
                  adaxial_color=(0.20, 0.36, 0.12)),
        vein=dict(pattern=V.PARALLELODROMOUS, vla_mm_per_mm2=4.0, vein_contrast=0.3),
        bark=dict(pattern=BP.ANNULATED, base_color=(0.45, 0.38, 0.28), secondary_color=(0.25, 0.20, 0.14),
                  feature_scale_m=0.15, relief=0.8),
        wood=(0.40, 8.0), lma=200, deciduous=False, notes="Unbranched Corner-model stem with a pinnate frond crown."),
}


# -----------------------------------------------------------------------------
# Root systems
# -----------------------------------------------------------------------------
RT = RootSystemType
ROOT_SYSTEM_DEFAULTS = {
    RT.TAPROOT: dict(taproot_share=0.35, lateral_count=6, plank=0.4, surface_exposure=0.10, buttress_height_dbh=0.5),
    RT.HEART: dict(taproot_share=0.30, lateral_count=7, plank=0.5, surface_exposure=0.15, buttress_height_dbh=0.6),
    RT.PLATE: dict(taproot_share=0.0, lateral_count=8, plank=0.7, surface_exposure=0.25, sinker_spacing_m=0.9,
                   buttress_height_dbh=0.5),
    RT.BUTTRESS: dict(taproot_share=0.05, plank=3.0, surface_exposure=0.6, buttress_height_dbh=1.5,
                      sinker_spacing_m=1.5),
    RT.FIBROUS: dict(taproot_share=0.0, fibrous_count=70, fibrous_radius_m=0.005),
    RT.STILT: dict(taproot_share=0.0, lateral_count=10, plank=0.2, surface_exposure=0.1, sinker_spacing_m=0.8,
                   buttress_height_dbh=0.3),
}

# (system, biome for Jackson et al. 1996 beta, max rooting depth m, lateral spread / crown radius, extras)
# Systems follow Koestler et al. (1968) where the species is covered; depths are typical maxima within the
# ranges compiled by Canadell et al. (1996). Values are approximate, for morphology rather than inventory.
ROOT_TRAITS = {
    "quercus_robur": (RT.TAPROOT, "temperate_deciduous", 5.0, 1.5, {}),
    "quercus_rubra": (RT.HEART, "temperate_deciduous", 4.0, 1.4, {}),
    "quercus_agrifolia": (RT.TAPROOT, "sclerophyllous_shrubs", 7.0, 1.6, {}),
    "fagus_sylvatica": (RT.HEART, "temperate_deciduous", 2.0, 1.2, {"surface_exposure": 0.3}),
    "castanea_sativa": (RT.TAPROOT, "temperate_deciduous", 3.5, 1.4, {}),
    "acer_palmatum": (RT.HEART, "temperate_deciduous", 1.5, 1.2, {"lateral_count": 6}),
    "acer_pseudoplatanus": (RT.HEART, "temperate_deciduous", 3.0, 1.3, {}),
    "acer_saccharum": (RT.HEART, "temperate_deciduous", 2.5, 1.3, {}),
    "aesculus_hippocastanum": (RT.HEART, "temperate_deciduous", 2.5, 1.3, {}),
    "betula_pendula": (RT.HEART, "boreal_forest", 2.0, 1.5, {"lateral_count": 6}),
    "populus_tremula": (RT.PLATE, "boreal_forest", 2.0, 2.5, {}),
    "populus_nigra_italica": (RT.PLATE, "temperate_deciduous", 2.5, 3.0, {}),
    "salix_babylonica": (RT.PLATE, "temperate_deciduous", 2.5, 1.8, {"surface_exposure": 0.35}),
    "tilia_cordata": (RT.HEART, "temperate_deciduous", 3.0, 1.3, {}),
    "platanus_hispanica": (RT.HEART, "temperate_deciduous", 3.5, 1.4, {}),
    "liriodendron_tulipifera": (RT.HEART, "temperate_deciduous", 3.0, 1.3, {}),
    "liquidambar_styraciflua": (RT.TAPROOT, "temperate_deciduous", 3.5, 1.4, {}),
    "ulmus_minor": (RT.HEART, "temperate_deciduous", 3.0, 1.6, {}),
    "prunus_avium": (RT.HEART, "temperate_deciduous", 2.0, 1.2, {}),
    "malus_domestica": (RT.HEART, "temperate_deciduous", 2.0, 1.2, {}),
    "magnolia_grandiflora": (RT.HEART, "temperate_deciduous", 2.5, 1.2, {}),
    "ficus_elastica": (RT.BUTTRESS, "tropical_evergreen", 4.9, 2.0, {"surface_exposure": 0.75}),
    "olea_europaea": (RT.HEART, "sclerophyllous_shrubs", 6.0, 1.6, {"surface_exposure": 0.35, "plank": 0.9}),
    "cercis_canadensis": (RT.TAPROOT, "temperate_deciduous", 3.0, 1.2, {}),
    "eucalyptus_globulus": (RT.TAPROOT, "sclerophyllous_shrubs", 15.0, 1.5, {}),
    "ceiba_pentandra": (RT.BUTTRESS, "tropical_deciduous", 4.0, 1.4, {"buttress_height_dbh": 2.2}),
    "fraxinus_excelsior": (RT.HEART, "temperate_deciduous", 3.5, 1.4, {}),
    "juglans_regia": (RT.TAPROOT, "temperate_deciduous", 6.0, 1.5, {}),
    "robinia_pseudoacacia": (RT.TAPROOT, "temperate_deciduous", 4.0, 2.0, {"taproot_share": 0.25}),
    "ginkgo_biloba": (RT.HEART, "temperate_deciduous", 3.0, 1.3, {}),
    "pinus_sylvestris": (RT.TAPROOT, "temperate_coniferous", 4.0, 1.6, {"taproot_share": 0.4}),
    "pinus_pinea": (RT.TAPROOT, "sclerophyllous_shrubs", 5.0, 1.3, {}),
    "picea_abies": (RT.PLATE, "boreal_forest", 1.5, 1.6, {}),
    "sequoiadendron_giganteum": (RT.PLATE, "temperate_coniferous", 2.5, 2.5, {"lateral_count": 8}),
    "taxodium_mucronatum": (RT.BUTTRESS, "temperate_coniferous", 3.0, 1.5,
                            {"plank": 1.6, "surface_exposure": 0.55, "buttress_height_dbh": 1.2}),
    "cupressus_sempervirens": (RT.TAPROOT, "sclerophyllous_shrubs", 3.0, 1.6, {}),
    "phoenix_canariensis": (RT.FIBROUS, "sclerophyllous_shrubs", 3.0, 0.6, {}),
    "rhizophora_mangle": (RT.STILT, "tropical_evergreen", 1.2, 1.3,
                          {"lateral_count": 14, "stilt_height_dbh": 7.0, "drop_roots": 6, "deep_roots": 0.4}),
    "avicennia_germinans": (RT.PLATE, "tropical_evergreen", 1.0, 2.0,
                            {"lateral_count": 10, "pneumatophores": 30, "surface_exposure": 0.0}),
}

for _key, (_system, _biome, _depth, _spread, _extra) in ROOT_TRAITS.items():
    _kw = dict(ROOT_SYSTEM_DEFAULTS[_system])
    _kw.update(system=_system, beta=JACKSON_BETA[_biome], max_depth_m=_depth, spread_crown_ratio=_spread)
    _kw.update(_extra)
    _preset = SPECIES_CATALOG[_key]
    if _preset.allometry.buttress_lobes > 0 and _system != RT.FIBROUS:
        # Stem flutes sit on the main structural roots: one lateral per lobe
        _kw["lateral_count"] = max(_kw.get("lateral_count", 6), _preset.allometry.buttress_lobes)
    _preset.roots = RootProfile(**_kw)


# Bark ageing: radius (cm) at which the rhytidome pattern replaces the smooth young periderm, and the
# periderm colour of young axes (sRGB). Field-guide descriptions: smooth silvery young oaks and ashes,
# orange upper trunk of Scots pine, reddish-brown birch and cherry twigs, early-furrowed Robinia.
BARK_AGE = {
    "quercus_robur": dict(onset_radius_cm=5.0, young_color=(0.47, 0.47, 0.42), weathering=0.45),
    "quercus_rubra": dict(onset_radius_cm=6.0, young_color=(0.50, 0.50, 0.47)),
    "quercus_agrifolia": dict(onset_radius_cm=6.0, young_color=(0.50, 0.49, 0.45)),
    "castanea_sativa": dict(onset_radius_cm=6.0, young_color=(0.45, 0.36, 0.29)),
    "acer_pseudoplatanus": dict(onset_radius_cm=8.0, young_color=(0.52, 0.50, 0.45)),
    "acer_saccharum": dict(onset_radius_cm=6.0, young_color=(0.50, 0.48, 0.44)),
    "aesculus_hippocastanum": dict(onset_radius_cm=8.0, young_color=(0.46, 0.41, 0.36)),
    "betula_pendula": dict(onset_radius_cm=1.0, young_color=(0.40, 0.22, 0.16), weathering=0.1),
    "populus_tremula": dict(onset_radius_cm=12.0, young_color=(0.58, 0.60, 0.50)),
    "populus_nigra_italica": dict(onset_radius_cm=5.0, young_color=(0.55, 0.55, 0.48)),
    "salix_babylonica": dict(onset_radius_cm=4.0, young_color=(0.52, 0.48, 0.30)),
    "prunus_avium": dict(onset_radius_cm=1.5, young_color=(0.42, 0.24, 0.19), weathering=0.15),
    "malus_domestica": dict(onset_radius_cm=5.0, young_color=(0.45, 0.35, 0.28)),
    "pinus_sylvestris": dict(onset_radius_cm=12.0, young_color=(0.78, 0.45, 0.25), weathering=0.25,
                             base_color=(0.44, 0.34, 0.28), secondary_color=(0.24, 0.15, 0.11),
                             inner_color=(0.58, 0.30, 0.16)),   # Grey-brown plates, orange-red furrows
    "pinus_pinea": dict(onset_radius_cm=8.0, young_color=(0.55, 0.38, 0.28)),
    "picea_abies": dict(onset_radius_cm=10.0, young_color=(0.50, 0.32, 0.22)),
    "sequoiadendron_giganteum": dict(onset_radius_cm=6.0, young_color=(0.50, 0.30, 0.22)),
    "taxodium_mucronatum": dict(onset_radius_cm=3.0, young_color=(0.48, 0.34, 0.26)),
    "cupressus_sempervirens": dict(onset_radius_cm=3.0, young_color=(0.45, 0.36, 0.28)),
    "platanus_hispanica": dict(onset_radius_cm=5.0, young_color=(0.50, 0.45, 0.35)),
    "eucalyptus_globulus": dict(onset_radius_cm=4.0, young_color=(0.62, 0.62, 0.52)),
    "tilia_cordata": dict(onset_radius_cm=6.0, young_color=(0.45, 0.38, 0.30)),
    "ulmus_minor": dict(onset_radius_cm=4.0, young_color=(0.44, 0.38, 0.32)),
    "fraxinus_excelsior": dict(onset_radius_cm=8.0, young_color=(0.56, 0.56, 0.51)),
    "juglans_regia": dict(onset_radius_cm=7.0, young_color=(0.60, 0.60, 0.56)),
    "robinia_pseudoacacia": dict(onset_radius_cm=2.5, young_color=(0.45, 0.40, 0.33), weathering=0.5),
    "olea_europaea": dict(onset_radius_cm=6.0, young_color=(0.55, 0.55, 0.50), weathering=0.5),
    "liriodendron_tulipifera": dict(onset_radius_cm=8.0, young_color=(0.50, 0.50, 0.42)),
    "liquidambar_styraciflua": dict(onset_radius_cm=3.0, young_color=(0.48, 0.42, 0.32)),
    "phoenix_canariensis": dict(onset_radius_cm=0.1),
    "rhizophora_mangle": dict(onset_radius_cm=4.0, young_color=(0.50, 0.42, 0.32)),
    "avicennia_germinans": dict(onset_radius_cm=4.0, young_color=(0.40, 0.36, 0.30)),
}
for _key, _kw in BARK_AGE.items():
    _b = SPECIES_CATALOG[_key].bark
    for _f, _v in _kw.items():
        setattr(_b, _f, _v)
# Macroscopic bark texture of mature trunks, in Junikka's (1994) terms, with values read from field
# descriptions (Vaucher 2003): b = blockiness (rectangular blocks), s = transverse segments, t = plate
# tilt, w = warp (interlacing ridges), m = moss, l = lichen (humid temperate barks carry more epiphytes).
BARK_TEXTURE = {
    # Fagaceae
    "quercus_robur": dict(blockiness=0.7, segments=0.6, plate_tilt=0.35, warp=0.45, moss=0.35, lichen=0.25),  # deeply furrowed, blocky ridges
    "quercus_rubra": dict(blockiness=0.5, segments=0.15, plate_tilt=0.2, warp=0.3, moss=0.25, lichen=0.2),   # flat-topped shiny ridge stripes
    "quercus_agrifolia": dict(blockiness=0.5, segments=0.35, plate_tilt=0.25, warp=0.45, moss=0.1, lichen=0.3),
    "fagus_sylvatica": dict(blockiness=0.2, segments=0.0, plate_tilt=0.0, warp=0.2, moss=0.3, lichen=0.45),    # smooth grey, lichen-rich
    "castanea_sativa": dict(blockiness=0.45, segments=0.2, plate_tilt=0.2, warp=0.6, moss=0.25, lichen=0.2),  # long, often spiral furrows
    # Sapindaceae
    "acer_palmatum": dict(blockiness=0.2, segments=0.0, plate_tilt=0.0, warp=0.2, moss=0.15, lichen=0.2),
    "acer_pseudoplatanus": dict(blockiness=0.75, segments=0.5, plate_tilt=0.6, warp=0.3, moss=0.3, lichen=0.3),  # flaking rectangular scales
    "acer_saccharum": dict(blockiness=0.65, segments=0.45, plate_tilt=0.55, warp=0.35, moss=0.2, lichen=0.25),  # furrowed, curling plates
    "aesculus_hippocastanum": dict(blockiness=0.6, segments=0.5, plate_tilt=0.55, warp=0.35, moss=0.25, lichen=0.2),  # scaly plates
    # Betulaceae / Salicaceae
    "betula_pendula": dict(blockiness=0.3, segments=0.1, plate_tilt=0.1, warp=0.3, moss=0.15, lichen=0.3),
    "populus_tremula": dict(blockiness=0.3, segments=0.0, plate_tilt=0.0, warp=0.25, moss=0.15, lichen=0.35),
    "populus_nigra_italica": dict(blockiness=0.35, segments=0.15, plate_tilt=0.15, warp=0.7, moss=0.2, lichen=0.2),  # deep interlacing furrows
    "salix_babylonica": dict(blockiness=0.3, segments=0.1, plate_tilt=0.1, warp=0.75, moss=0.3, lichen=0.15),
    # Others, broadleaved
    "tilia_cordata": dict(blockiness=0.3, segments=0.1, plate_tilt=0.1, warp=0.7, moss=0.25, lichen=0.2),     # narrow interlacing ridges
    "platanus_hispanica": dict(blockiness=0.3, segments=0.0, plate_tilt=0.2, warp=0.35, moss=0.05, lichen=0.1),
    "liriodendron_tulipifera": dict(blockiness=0.35, segments=0.1, plate_tilt=0.1, warp=0.65, moss=0.2, lichen=0.2),  # diamond-patterned furrows
    "liquidambar_styraciflua": dict(blockiness=0.55, segments=0.45, plate_tilt=0.3, warp=0.45, moss=0.2, lichen=0.15),
    "ulmus_minor": dict(blockiness=0.6, segments=0.55, plate_tilt=0.3, warp=0.5, moss=0.25, lichen=0.2),
    "prunus_avium": dict(blockiness=0.25, segments=0.0, plate_tilt=0.15, warp=0.2, moss=0.1, lichen=0.25),
    "malus_domestica": dict(blockiness=0.6, segments=0.45, plate_tilt=0.6, warp=0.35, moss=0.25, lichen=0.4),  # scaly, flaking, lichen-covered
    "magnolia_grandiflora": dict(blockiness=0.35, segments=0.2, plate_tilt=0.25, warp=0.3, moss=0.1, lichen=0.25),
    "ficus_elastica": dict(blockiness=0.2, segments=0.0, plate_tilt=0.0, warp=0.2, moss=0.05, lichen=0.1),
    "olea_europaea": dict(blockiness=0.6, segments=0.55, plate_tilt=0.35, warp=0.75, moss=0.05, lichen=0.35),  # twisted, cracked into small blocks
    "cercis_canadensis": dict(blockiness=0.5, segments=0.35, plate_tilt=0.4, warp=0.35, moss=0.1, lichen=0.2),
    "eucalyptus_globulus": dict(blockiness=0.2, segments=0.0, plate_tilt=0.3, warp=0.3, moss=0.0, lichen=0.05),
    "ceiba_pentandra": dict(blockiness=0.2, segments=0.0, plate_tilt=0.0, warp=0.2, moss=0.15, lichen=0.25),
    "fraxinus_excelsior": dict(blockiness=0.35, segments=0.15, plate_tilt=0.1, warp=0.75, moss=0.3, lichen=0.25),  # diamond (anastomosing) ridges
    "juglans_regia": dict(blockiness=0.45, segments=0.2, plate_tilt=0.15, warp=0.6, moss=0.2, lichen=0.2),
    "robinia_pseudoacacia": dict(blockiness=0.3, segments=0.15, plate_tilt=0.1, warp=0.8, moss=0.15, lichen=0.15),  # ropy interlacing ridges
    "ginkgo_biloba": dict(blockiness=0.5, segments=0.35, plate_tilt=0.2, warp=0.55, moss=0.15, lichen=0.15),
    # Conifers and palm
    "pinus_sylvestris": dict(blockiness=0.6, segments=0.4, plate_tilt=0.55, warp=0.35, moss=0.15, lichen=0.15),
    "pinus_pinea": dict(blockiness=0.65, segments=0.45, plate_tilt=0.5, warp=0.3, moss=0.05, lichen=0.1),     # large flat reddish plates
    "picea_abies": dict(blockiness=0.4, segments=0.5, plate_tilt=0.7, warp=0.3, moss=0.3, lichen=0.35),       # thin, small rounded scales
    "sequoiadendron_giganteum": dict(blockiness=0.2, segments=0.05, plate_tilt=0.1, warp=0.55, moss=0.05, lichen=0.05),
    "taxodium_mucronatum": dict(blockiness=0.2, segments=0.05, plate_tilt=0.1, warp=0.5, moss=0.3, lichen=0.1),
    "cupressus_sempervirens": dict(blockiness=0.2, segments=0.05, plate_tilt=0.1, warp=0.45, moss=0.1, lichen=0.15),
    "rhizophora_mangle": dict(blockiness=0.3, segments=0.2, plate_tilt=0.1, warp=0.3, moss=0.05, lichen=0.3),
    "avicennia_germinans": dict(blockiness=0.5, segments=0.4, plate_tilt=0.2, warp=0.3, moss=0.05, lichen=0.3),
    "phoenix_canariensis": dict(blockiness=0.5, segments=0.0, plate_tilt=0.3, warp=0.2, moss=0.05, lichen=0.05),
}
for _key, _kw in BARK_TEXTURE.items():
    _b = SPECIES_CATALOG[_key].bark
    for _f, _v in _kw.items():
        setattr(_b, _f, _v)
for _preset in SPECIES_CATALOG.values():      # Smooth barks stay as they are at every age
    if _preset.bark.pattern == BP.SMOOTH and _preset.bark.young_color is not None:
        _preset.bark.onset_radius_cm = 0.1


def get_preset_names() -> list[tuple[str, str, str]]:
    """Returns (id, label, description) tuples for a Blender enum, sorted by family then name."""
    items = []
    for key, spec in sorted(SPECIES_CATALOG.items(), key=lambda kv: (kv[1].family, kv[1].scientific_name)):
        desc = f"{spec.scientific_name} ({spec.family}) - {spec.biome}. {spec.notes}"
        items.append((key, f"{spec.common_name} ({spec.scientific_name})", desc))
    return items


def get_species_preset(key: str) -> BotanicalSpeciesPreset:
    """Retrieves a botanical preset by ID, falling back to English Oak."""
    return SPECIES_CATALOG.get(key, SPECIES_CATALOG["quercus_robur"])
