"""
Succulent catalogs: stem succulents (Cactaceae and a convergent Euphorbia) and
rosette leaf succulents.

Values are typical adult ranges from general botanical references (rib and
spine counts, areole spacing, leaf dimensions); they are approximations meant
to reproduce each species' habit, not measurements of particular specimens.
Rib counts favour Fibonacci numbers where the species range allows it
(Robberecht & Nobel 1983).
"""

from dataclasses import dataclass, field, fields
from enum import Enum

from .cactus import CactusProfile, CactusHabit, AreoleArrangement
from .rosette import RosetteProfile, RosettePhyllotaxis
from .roots import RootSystemType


class GrowthForm(str, Enum):
    TREE = "Tree"
    CACTUS = "Cactus"
    ROSETTE = "Rosette"


@dataclass
class SucculentPreset:
    scientific_name: str
    common_name: str
    family: str
    biome: str
    form: GrowthForm
    profile: object
    notes: str = ""


H, A = CactusHabit, AreoleArrangement


def cactus(sci, common, family, biome, notes="", **kw) -> SucculentPreset:
    return SucculentPreset(sci, common, family, biome, GrowthForm.CACTUS, CactusProfile(**kw), notes)


def rosette(sci, common, family, biome, notes="", **kw) -> SucculentPreset:
    return SucculentPreset(sci, common, family, biome, GrowthForm.ROSETTE, RosetteProfile(**kw), notes)


GREY_SPINE = dict(spine_color=(0.70, 0.66, 0.58), spine_tip_color=(0.25, 0.20, 0.17))

CACTUS_CATALOG: dict[str, SucculentPreset] = {
    "carnegiea_gigantea": cactus(
        "Carnegiea gigantea", "Saguaro", "Cactaceae", "Sonoran Desert",
        "Giant columnar cactus; arms from mid-height curving upward.",
        habit=H.COLUMNAR, height_m=9.0, diameter_m=0.55, rib_count=21, rib_depth=0.12, rib_sharpness=0.6,
        apex_dome=0.9, areole_spacing_cm=2.5, radial_spines=15, radial_length_cm=2.0, central_spines=4,
        central_length_cm=5.0, spine_thickness_mm=1.0, arm_count=4, arm_height_min=0.35, arm_height_max=0.6,
        arm_reach_m=0.6, arm_length_ratio=0.3, stem_color=(0.30, 0.45, 0.28), groove_color=(0.20, 0.32, 0.18),
        wool=0.25, **GREY_SPINE,
        root_system=RootSystemType.TAPROOT, taproot_share=0.25, taproot_depth_m=0.8, root_count=12, root_spread_ratio=1.1, root_depth_m=0.3,
        browning_height_m=1.2, scars=0.25, areole_stain=0.35),
    "pachycereus_pringlei": cactus(
        "Pachycereus pringlei", "Cardón", "Cactaceae", "Sonoran Desert / Baja California",
        "Tallest cactus; massive trunk with many erect branches from low on the stem.",
        habit=H.COLUMNAR, height_m=12.0, diameter_m=0.8, rib_count=13, rib_depth=0.15, apex_dome=0.8,
        areole_spacing_cm=2.5, radial_spines=10, radial_length_cm=1.5, central_spines=2, central_length_cm=3.0,
        arm_count=8, arm_height_min=0.12, arm_height_max=0.35, arm_reach_m=0.5, arm_length_ratio=0.6,
        stem_color=(0.30, 0.43, 0.30), glaucous=0.4, wool=0.35, **GREY_SPINE,
        root_system=RootSystemType.TAPROOT, taproot_share=0.25, taproot_depth_m=1.0, root_count=14, root_spread_ratio=1.0, root_depth_m=0.35,
        browning_height_m=1.0, scars=0.25),
    "pachycereus_weberi": cactus(
        "Pachycereus weberi", "Candelabro (giant Mexican candelabra)", "Cactaceae", "Puebla, Oaxaca & Guerrero, Mexico",
        "Short trunk (to ~2 m) crowned by numerous erect glaucous branches, up to ~11 m tall; black flattened "
        "central spine to 10 cm, 6-12 reddish radials, white-felted areoles.",
        habit=H.COLUMNAR, height_m=2.5, diameter_m=1.0, rib_count=8, rib_depth=0.18, rib_sharpness=0.9,
        apex_dome=0.6, areole_spacing_cm=2.5, radial_spines=9, radial_length_cm=2.5, central_spines=1,
        central_length_cm=10.0, spine_thickness_mm=1.2, wool=0.7, wool_color=(0.9, 0.9, 0.86),
        arm_count=40, arm_height_min=0.6, arm_height_max=0.98, arm_radius_ratio=0.24, arm_reach_m=2.4,
        arm_length_ratio=1.9, arm_lean_deg=6.0, arm_branching=0.8, crown_fill=0.8, stem_color=(0.36, 0.50, 0.50),
        groove_color=(0.26, 0.40, 0.40), glaucous=0.7, spine_color=(0.55, 0.30, 0.22),
        spine_tip_color=(0.10, 0.08, 0.07),
        root_system=RootSystemType.TAPROOT, taproot_share=0.2, taproot_depth_m=0.8, root_count=14, root_spread_ratio=0.8, root_depth_m=0.3,
        browning_height_m=1.8, barking_color=(0.36, 0.34, 0.31), scaling_color=(0.55, 0.50, 0.42), scars=0.2),
    "pachycereus_marginatus": cactus(
        "Pachycereus marginatus", "Órgano (Mexican fence post)", "Cactaceae", "Central Mexico",
        "Unbranched columns in clumps; 5 prominent ribs with felted, nearly confluent areoles.",
        habit=H.COLUMNAR, height_m=4.0, diameter_m=0.14, rib_count=5, rib_depth=0.3, rib_sharpness=1.3,
        apex_dome=0.8, areole_spacing_cm=1.0, radial_spines=6, radial_length_cm=0.3, central_spines=1,
        central_length_cm=0.5, spine_thickness_mm=0.5, wool=0.9, wool_color=(0.75, 0.75, 0.72),
        offsets=7, offset_scale=0.95, stem_color=(0.22, 0.40, 0.22), groove_color=(0.14, 0.30, 0.14),
        spine_color=(0.85, 0.82, 0.75), spine_tip_color=(0.3, 0.25, 0.2),
        root_system=RootSystemType.PLATE, root_count=8, root_spread_ratio=0.8, root_depth_m=0.3,
        browning_height_m=0.4, scars=0.15),
    "myrtillocactus_geometrizans": cactus(
        "Myrtillocactus geometrizans", "Garambullo (blue candle)", "Cactaceae", "Central & Northern Mexico",
        "Candelabra of glaucous blue-green branches from a short trunk.",
        habit=H.COLUMNAR, height_m=4.0, diameter_m=0.12, rib_count=5, rib_depth=0.28, rib_sharpness=1.1,
        apex_dome=0.8, areole_spacing_cm=2.0, radial_spines=5, radial_length_cm=0.5, central_spines=1,
        central_length_cm=2.0, arm_count=12, arm_height_min=0.15, arm_height_max=0.4, arm_reach_m=0.25,
        arm_length_ratio=0.58, arm_radius_ratio=0.9, stem_color=(0.34, 0.50, 0.50), groove_color=(0.25, 0.40, 0.40),
        glaucous=0.75, spine_color=(0.25, 0.2, 0.18), spine_tip_color=(0.1, 0.08, 0.08), wool=0.2,
        root_system=RootSystemType.PLATE, root_count=10, root_spread_ratio=0.9, root_depth_m=0.3,
        browning_height_m=0.4, scars=0.2),
    "echinopsis_pachanoi": cactus(
        "Echinopsis pachanoi", "San Pedro", "Cactaceae", "Andes",
        "Columnar, branching from the base; broad rounded ribs and short dark spines.",
        habit=H.COLUMNAR, height_m=3.5, diameter_m=0.13, rib_count=7, rib_depth=0.2, rib_sharpness=0.6,
        areole_spacing_cm=2.0, radial_spines=4, radial_length_cm=0.8, central_spines=1, central_length_cm=1.0,
        offsets=5, offset_scale=0.85, stem_color=(0.30, 0.50, 0.40), glaucous=0.5,
        spine_color=(0.45, 0.32, 0.22), spine_tip_color=(0.2, 0.14, 0.1),
        root_system=RootSystemType.PLATE, root_count=8, root_spread_ratio=0.8, root_depth_m=0.3,
        browning_height_m=0.25, scars=0.15),
    "cephalocereus_senilis": cactus(
        "Cephalocereus senilis", "Viejito (old man cactus)", "Cactaceae", "Hidalgo, Mexico",
        "Column clothed in long white hair-like spines.",
        habit=H.COLUMNAR, height_m=3.0, diameter_m=0.25, rib_count=25, rib_depth=0.06, areole_spacing_cm=1.2,
        radial_spines=30, radial_length_cm=10.0, spine_thickness_mm=0.5, spine_curvature=1.1, radial_lift_deg=30,
        central_spines=2, central_length_cm=2.0, wool=0.5, stem_color=(0.32, 0.45, 0.30),
        spine_color=(0.96, 0.95, 0.92), spine_tip_color=(0.9, 0.88, 0.84),
        root_system=RootSystemType.PLATE, root_count=8, root_spread_ratio=0.8, root_depth_m=0.3,
        browning_height_m=0.5, scars=0.1),
    "ferocactus_wislizeni": cactus(
        "Ferocactus wislizeni", "Fishhook barrel", "Cactaceae", "Chihuahuan & Sonoran Deserts",
        "Barrel with hooked red central spines.",
        habit=H.BARREL, height_m=1.2, diameter_m=0.55, rib_count=21, rib_depth=0.28, rib_sharpness=0.9,
        apex_dome=1.0, apex_depression=0.1, tubercle_height=0.06, areole_spacing_cm=3.0, radial_spines=15,
        radial_length_cm=4.0, spine_thickness_mm=0.5, central_spines=4, central_length_cm=6.0, central_hook=1.0,
        stem_color=(0.25, 0.40, 0.22), spine_color=(0.75, 0.35, 0.22), spine_tip_color=(0.45, 0.2, 0.12),
        wool=0.3,
        root_system=RootSystemType.TAPROOT, taproot_share=0.35, taproot_depth_m=0.4, root_count=10, root_spread_ratio=2.0, root_depth_m=0.25,
        browning_height_m=0.15, scars=0.2),
    "echinocactus_grusonii": cactus(
        "Echinocactus grusonii", "Biznaga dorada (golden barrel)", "Cactaceae", "Querétaro & Hidalgo, Mexico",
        "Globose barrel with many sharp ribs, golden spines and a woolly apex.",
        habit=H.GLOBOSE, height_m=0.6, diameter_m=0.6, rib_count=34, rib_depth=0.18, rib_sharpness=1.2,
        apex_dome=1.0, apex_depression=0.15, areole_spacing_cm=1.5, radial_spines=9, radial_length_cm=3.0,
        central_spines=4, central_length_cm=4.0, spine_thickness_mm=1.0, apical_wool=1.8, wool=0.4,
        stem_color=(0.25, 0.42, 0.20), spine_color=(0.97, 0.82, 0.30), spine_tip_color=(0.85, 0.62, 0.2),
        wool_color=(0.97, 0.93, 0.80),
        root_system=RootSystemType.PLATE, root_count=10, root_spread_ratio=2.0, root_depth_m=0.25,
        browning_height_m=0.08, scars=0.1),
    "mammillaria_hahniana": cactus(
        "Mammillaria hahniana", "Old lady cactus", "Cactaceae", "Guanajuato & Querétaro, Mexico",
        "Clustering globose stems; spiral tubercles with white hair-like radial spines.",
        habit=H.GLOBOSE, arrangement=A.SPIRAL, height_m=0.12, diameter_m=0.10, apex_dome=0.9,
        tubercle_height=0.3, areole_spacing_cm=0.7, radial_spines=25, radial_length_cm=1.2,
        spine_thickness_mm=0.05, spine_curvature=0.6, central_spines=1, central_length_cm=0.5, wool=0.6,
        apical_wool=0.8, offsets=4, offset_scale=0.85, stem_color=(0.28, 0.45, 0.28),
        spine_color=(0.97, 0.97, 0.95), spine_tip_color=(0.92, 0.9, 0.88),
        root_system=RootSystemType.PLATE, root_count=8, root_spread_ratio=1.5, root_depth_m=0.15,
        scars=0.05),
    "lophophora_williamsii": cactus(
        "Lophophora williamsii", "Peyote", "Cactaceae", "Chihuahuan Desert, Mexico & Texas",
        "Spineless flat-topped button with tufts of hairs on the areoles and a large napiform taproot.",
        habit=H.GLOBOSE, height_m=0.05, diameter_m=0.08, apex_dome=0.6, apex_roundness=4.0, apex_depression=0.08,
        rib_count=8, rib_depth=0.22, rib_sharpness=0.7, tubercle_height=0.12, areole_spacing_cm=1.4,
        radial_spines=0, central_spines=0, wool=1.0, wool_color=(0.85, 0.83, 0.78),
        stem_color=(0.38, 0.52, 0.50), groove_color=(0.30, 0.44, 0.42), glaucous=0.7,
        root_system=RootSystemType.TUBEROUS, tuber_length_cm=12.0, tuber_radius_ratio=0.75, root_count=10,
        root_spread_ratio=3.0, root_depth_m=0.15,
        scars=0.05, areole_stain=0.2),
    "astrophytum_myriostigma": cactus(
        "Astrophytum myriostigma", "Bonete de obispo (bishop's cap)", "Cactaceae", "Chihuahuan Desert, Mexico",
        "Spineless, five sharp ribs, flecked with white trichome scales.",
        habit=H.GLOBOSE, height_m=0.15, diameter_m=0.15, rib_count=5, rib_depth=0.45, rib_sharpness=1.6,
        apex_dome=0.8, areole_spacing_cm=1.2, radial_spines=0, central_spines=0, wool=0.7,
        stem_color=(0.40, 0.48, 0.38), groove_color=(0.30, 0.38, 0.28), flecks=0.9,
        wool_color=(0.70, 0.62, 0.50),
        root_system=RootSystemType.TAPROOT, taproot_share=0.35, taproot_depth_m=0.25, root_count=6, root_spread_ratio=1.5, root_depth_m=0.15,
        scars=0.1),
    "opuntia_ficus_indica": cactus(
        "Opuntia ficus-indica", "Nopal", "Cactaceae", "Mexico (cultivated worldwide)",
        "Shrubby chains of large flattened cladodes; few spines and tufts of glochids.",
        habit=H.CLADODE, pad_length_cm=40.0, pad_width_ratio=0.62, pad_thickness_ratio=0.06, pad_levels=5,
        pad_branching=1.8, areole_spacing_cm=3.5, radial_spines=1, radial_length_cm=1.0, central_spines=0,
        wool=0.25, wool_color=(0.75, 0.65, 0.35), stem_color=(0.32, 0.50, 0.28), glaucous=0.3,
        spine_color=(0.9, 0.88, 0.8), spine_tip_color=(0.6, 0.55, 0.45),
        root_system=RootSystemType.PLATE, root_count=12, root_spread_ratio=1.2, root_depth_m=0.15,
        browning_height_m=0.3, scars=0.3, areole_stain=0.5, scaling_color=(0.55, 0.50, 0.40), barking_color=(0.40, 0.36, 0.30)),
    "opuntia_microdasys": cactus(
        "Opuntia microdasys", "Bunny ears", "Cactaceae", "Chihuahuan Desert, Mexico",
        "Small pads without spines, dotted with dense glochid tufts.",
        habit=H.CLADODE, pad_length_cm=12.0, pad_width_ratio=0.75, pad_thickness_ratio=0.1, pad_levels=3,
        pad_branching=1.7, areole_spacing_cm=1.0, radial_spines=0, central_spines=0, wool=0.7,
        wool_color=(0.95, 0.88, 0.55), stem_color=(0.30, 0.52, 0.25),
        root_system=RootSystemType.PLATE, root_count=8, root_spread_ratio=1.5, root_depth_m=0.12,
        browning_height_m=0.05, scars=0.1),
    "euphorbia_ingens": cactus(
        "Euphorbia ingens", "Candelabra tree (convergent, Euphorbiaceae)", "Euphorbiaceae", "Southern Africa",
        "Not a cactus: a stem succulent converging on the candelabra form; 4 winged ribs with paired spines.",
        habit=H.COLUMNAR, height_m=6.0, diameter_m=0.15, rib_count=4, rib_depth=0.4, rib_sharpness=2.0,
        areole_spacing_cm=1.5, radial_spines=2, radial_length_cm=0.5, radial_lift_deg=30, central_spines=0,
        wool=0.0, arm_count=16, arm_height_min=0.3, arm_height_max=0.5, arm_reach_m=0.7, arm_length_ratio=0.5,
        arm_lean_deg=8.0, arm_branching=0.6,
        arm_radius_ratio=0.9, stem_color=(0.25, 0.42, 0.20), spine_color=(0.35, 0.25, 0.2),
        spine_tip_color=(0.2, 0.15, 0.1),
        root_system=RootSystemType.TAPROOT, taproot_share=0.2, taproot_depth_m=1.0, root_count=12, root_spread_ratio=0.8, root_depth_m=0.4,
        browning_height_m=1.2, barking_color=(0.40, 0.38, 0.33), scaling_color=(0.55, 0.50, 0.40), scars=0.2),
}

ROSETTE_CATALOG: dict[str, SucculentPreset] = {
    "echeveria_elegans": rosette(
        "Echeveria elegans", "Mexican snowball", "Crassulaceae", "Hidalgo, Mexico",
        "Tight glaucous rosette of incurved spoon-shaped leaves with translucent margins.",
        leaf_count=60, leaf_length_cm=5.0, leaf_aspect=2.3, leaf_thickness=0.4, widest_position=0.7,
        apex_angle_deg=90, apex_curvature=0.5, terminal_spine_cm=0.15, channel=0.05, keel=0.25,
        elevation_outer_deg=12, elevation_inner_deg=72, curvature_deg=15, size_gradient=0.6,
        leaf_color=(0.62, 0.75, 0.76), glaucous=0.85, blush_color=(0.85, 0.60, 0.65), blush_amount=0.25,
        offsets=4, rosette_height_m=0.02,
        root_count=18, root_spread_ratio=1.2, root_depth_m=0.1, root_radius_mm=0.8,
        base_width=0.4, clasp=0.4, base_swell=0.3, dead_leaves=5),
    "echeveria_agavoides": rosette(
        "Echeveria agavoides", "Lipstick echeveria", "Crassulaceae", "San Luis Potosí & Guanajuato, Mexico",
        "Thick triangular leaves with red tips and margins.",
        leaf_count=25, leaf_length_cm=7.0, leaf_aspect=2.0, leaf_thickness=0.45, widest_position=0.3,
        apex_angle_deg=45, apex_curvature=-0.1, terminal_spine_cm=0.2, channel=0.08, keel=0.35,
        elevation_outer_deg=20, elevation_inner_deg=60, curvature_deg=5, leaf_color=(0.45, 0.62, 0.35),
        glaucous=0.2, blush_color=(0.8, 0.15, 0.1), blush_amount=0.6,
        root_count=18, root_spread_ratio=1.2, root_depth_m=0.1, root_radius_mm=0.8,
        base_width=0.5, clasp=0.4, base_swell=0.3, dead_leaves=4),
    "agave_americana": rosette(
        "Agave americana", "Maguey (century plant)", "Asparagaceae", "Mexico",
        "Large blue-grey rosette; recurved leaves with hooked marginal teeth and a terminal spine.",
        leaf_count=36, leaf_length_cm=150, leaf_aspect=5.5, leaf_thickness=0.22, widest_position=0.4,
        apex_angle_deg=20, base_angle_deg=40, channel=0.45, keel=0.4, terminal_spine_cm=4.0, teeth_count=18,
        teeth_size_cm=0.8, teeth_hook=0.5, elevation_outer_deg=18, elevation_inner_deg=88, curvature_deg=-45,
        stem_radius_m=0.2, rosette_height_m=0.35, size_gradient=0.2, leaf_color=(0.40, 0.53, 0.54),
        glaucous=0.55,
        root_count=60, root_spread_ratio=1.0, root_depth_m=0.3, root_radius_mm=4.0,
        base_width=1.1, clasp=0.9, base_swell=1.5, furl=0.9, margin_band=0.6, striation=0.3, imprints=0.7, dead_leaves=8, stem_scars=0.5),
    "agave_tequilana": rosette(
        "Agave tequilana", "Agave azul (blue agave)", "Asparagaceae", "Jalisco, Mexico",
        "Dense rosette of narrow rigid blue leaves around a large stem (piña).",
        leaf_count=50, leaf_length_cm=120, leaf_aspect=12.0, leaf_thickness=0.25, widest_position=0.45,
        apex_angle_deg=18, channel=0.2, keel=0.35, terminal_spine_cm=2.0, teeth_count=25, teeth_size_cm=0.3,
        elevation_outer_deg=20, elevation_inner_deg=88, curvature_deg=-5, stem_radius_m=0.25,
        rosette_height_m=0.3, size_gradient=0.3, leaf_color=(0.45, 0.60, 0.63), glaucous=0.9,
        root_count=60, root_spread_ratio=1.0, root_depth_m=0.3, root_radius_mm=4.0,
        base_width=0.9, clasp=0.9, base_swell=1.3, furl=0.85, margin_band=0.5, striation=0.25, imprints=0.3, dead_leaves=6, stem_scars=0.5),
    "agave_victoriae_reginae": rosette(
        "Agave victoriae-reginae", "Queen Victoria agave", "Asparagaceae", "Coahuila & Nuevo León, Mexico",
        "Compact dome of keeled dark leaves edged with white lines; short black spine.",
        leaf_count=100, leaf_length_cm=15, leaf_aspect=3.5, leaf_thickness=0.6, widest_position=0.35,
        apex_angle_deg=40, channel=0.05, keel=0.6, section_exponent=1.6, terminal_spine_cm=1.5,
        elevation_outer_deg=15, elevation_inner_deg=80, curvature_deg=10, size_gradient=0.5,
        leaf_color=(0.20, 0.32, 0.22), glaucous=0.1, blush_color=(0.95, 0.95, 0.9), blush_amount=0.8,
        armature_color=(0.08, 0.06, 0.05), stem_radius_m=0.04,
        root_count=30, root_spread_ratio=1.2, root_depth_m=0.15, root_radius_mm=2.0,
        base_width=0.9, clasp=0.8, base_swell=0.8, furl=0.5, blush_tip=0.0, dead_leaves=4, dead_color=(0.42, 0.36, 0.28)),
    "aloe_vera": rosette(
        "Aloe vera", "Sábila (aloe)", "Asphodelaceae", "Arabian Peninsula (cultivated)",
        "Open rosette of thick lanceolate leaves with soft pale teeth; young leaves spotted.",
        leaf_count=18, leaf_length_cm=50, leaf_aspect=6.5, leaf_thickness=0.4, widest_position=0.2,
        apex_angle_deg=25, channel=0.3, keel=0.35, teeth_count=18, teeth_size_cm=0.25, teeth_hook=0.3,
        elevation_outer_deg=25, elevation_inner_deg=80, curvature_deg=-5, stem_radius_m=0.04,
        rosette_height_m=0.1, leaf_color=(0.45, 0.55, 0.40), glaucous=0.4, spots=0.4,
        armature_color=(0.85, 0.85, 0.6),
        root_count=25, root_spread_ratio=0.8, root_depth_m=0.25, root_radius_mm=3.0,
        base_width=1.2, clasp=0.9, base_swell=0.8, furl=0.5, striation=0.15, dead_leaves=6, dead_color=(0.50, 0.38, 0.26)),
    "haworthiopsis_attenuata": rosette(
        "Haworthiopsis attenuata", "Zebra plant", "Asphodelaceae", "Eastern Cape, South Africa",
        "Narrow dark leaves banded with white tubercles on the lower face.",
        leaf_count=40, leaf_length_cm=8.0, leaf_aspect=5.0, leaf_thickness=0.35, widest_position=0.15,
        apex_angle_deg=25, channel=0.15, keel=0.4, terminal_spine_cm=0.1, elevation_outer_deg=30,
        elevation_inner_deg=85, curvature_deg=10, leaf_color=(0.12, 0.22, 0.12), glaucous=0.0, bands=0.9,
        offsets=3,
        root_count=15, root_spread_ratio=1.0, root_depth_m=0.1, root_radius_mm=1.2,
        base_width=0.8, clasp=0.7, base_swell=0.5, furl=0.3, dead_leaves=3),
    "sempervivum_tectorum": rosette(
        "Sempervivum tectorum", "Houseleek", "Crassulaceae", "European mountains",
        "Dense flat rosettes with red-tipped leaves and many offsets.",
        leaf_count=70, leaf_length_cm=4.0, leaf_aspect=2.8, leaf_thickness=0.3, widest_position=0.55,
        apex_angle_deg=40, terminal_spine_cm=0.1, elevation_outer_deg=15, elevation_inner_deg=75,
        curvature_deg=5, leaf_color=(0.40, 0.55, 0.30), glaucous=0.15, blush_color=(0.7, 0.2, 0.25),
        blush_amount=0.6, offsets=6, offset_scale=0.4,
        root_count=15, root_spread_ratio=1.0, root_depth_m=0.08, root_radius_mm=0.7,
        base_width=0.6, clasp=0.5, base_swell=0.3, dead_leaves=5),
    "aeonium_arboreum": rosette(
        "Aeonium arboreum", "Tree aeonium", "Crassulaceae", "Canary Islands",
        "Flat rosettes of thin spatulate leaves at the end of a woody stem.",
        leaf_count=50, leaf_length_cm=6.0, leaf_aspect=2.5, leaf_thickness=0.12, widest_position=0.75,
        apex_angle_deg=100, apex_curvature=0.5, elevation_outer_deg=5, elevation_inner_deg=60,
        curvature_deg=8, stem_height_m=0.6, stem_radius_m=0.02, leaf_color=(0.35, 0.55, 0.25), glaucous=0.1,
        root_count=20, root_spread_ratio=1.5, root_depth_m=0.2, root_radius_mm=1.5,
        base_width=0.5, clasp=0.5, stem_scars=1.0, scar_spacing_mm=5.0, stem_color=(0.55, 0.50, 0.42)),
    "graptopetalum_paraguayense": rosette(
        "Graptopetalum paraguayense", "Ghost plant", "Crassulaceae", "Tamaulipas, Mexico",
        "Pale pinkish-grey thick leaves on a sprawling stem.",
        leaf_count=25, leaf_length_cm=4.5, leaf_aspect=2.0, leaf_thickness=0.45, widest_position=0.6,
        apex_angle_deg=60, terminal_spine_cm=0.1, elevation_outer_deg=15, elevation_inner_deg=65,
        curvature_deg=10, stem_height_m=0.1, leaf_color=(0.75, 0.70, 0.72), glaucous=1.0,
        blush_color=(0.85, 0.65, 0.7), blush_amount=0.3,
        root_count=15, root_spread_ratio=1.2, root_depth_m=0.1, root_radius_mm=0.8,
        base_width=0.4, clasp=0.3, dead_leaves=3, stem_scars=0.8),
    "dudleya_brittonii": rosette(
        "Dudleya brittonii", "Giant chalk dudleya", "Crassulaceae", "Baja California, Mexico",
        "Chalk-white powdery rosette of long tapering leaves.",
        leaf_count=50, leaf_length_cm=15, leaf_aspect=4.5, leaf_thickness=0.3, widest_position=0.3,
        apex_angle_deg=35, elevation_outer_deg=20, elevation_inner_deg=75, curvature_deg=8,
        leaf_color=(0.85, 0.88, 0.85), glaucous=1.0, stem_radius_m=0.03, rosette_height_m=0.04,
        root_count=20, root_spread_ratio=1.2, root_depth_m=0.15, root_radius_mm=1.5,
        base_width=0.8, clasp=0.6, base_swell=0.5, furl=0.3, dead_leaves=14, stem_scars=0.3, dead_color=(0.62, 0.52, 0.38)),
}

CATALOGS = {GrowthForm.CACTUS: CACTUS_CATALOG, GrowthForm.ROSETTE: ROSETTE_CATALOG}


def preset_items(form: GrowthForm) -> list[tuple[str, str, str]]:
    cat = CATALOGS[form]
    return [(k, f"{v.common_name} ({v.scientific_name})", f"{v.family} - {v.biome}. {v.notes}")
            for k, v in sorted(cat.items(), key=lambda kv: kv[1].scientific_name)]


# -----------------------------------------------------------------------------
# Editable field ranges (UI sliders, trait-space normalisation)
# -----------------------------------------------------------------------------
CACTUS_RANGES = {
    "height_m": (0.03, 20.0), "diameter_m": (0.02, 1.5), "base_taper": (0.0, 0.8), "apex_dome": (0.2, 3.0),
    "apex_roundness": (1.2, 6.0), "apex_depression": (0.0, 0.5), "rib_count": (2, 60), "rib_depth": (0.0, 0.6),
    "rib_sharpness": (0.2, 3.0), "rib_twist_deg_per_m": (-200.0, 200.0), "tubercle_height": (0.0, 0.8),
    "areole_spacing_cm": (0.3, 8.0), "radial_spines": (0, 40), "radial_length_cm": (0.0, 15.0),
    "central_spines": (0, 10), "central_length_cm": (0.0, 15.0), "spine_thickness_mm": (0.03, 4.0),
    "spine_curvature": (0.0, 1.5), "central_hook": (0.0, 2.0), "radial_lift_deg": (0.0, 80.0),
    "spine_jitter": (0.0, 0.8), "wool": (0.0, 2.0), "apical_wool": (0.0, 3.0), "arm_count": (0, 60),
    "arm_height_min": (0.05, 0.95), "arm_height_max": (0.05, 1.0), "arm_radius_ratio": (0.1, 1.0),
    "arm_reach_m": (0.05, 4.0), "arm_length_ratio": (0.1, 5.0), "arm_lean_deg": (0.0, 30.0), "arm_branching": (0.0, 3.0), "browning_height_m": (0.0, 6.0), "equator_bias": (0.0, 1.0), "equator_azimuth_deg": (0.0, 360.0), "scars": (0.0, 1.0), "areole_stain": (0.0, 1.0), "crest_light": (0.0, 1.0), "groove_dust": (0.0, 1.0), "streaks": (0.0, 1.0), "crown_fill": (0.0, 1.0), "offsets": (0, 20), "offset_scale": (0.2, 1.0),
    "pad_length_cm": (3.0, 80.0), "pad_width_ratio": (0.2, 1.2), "pad_thickness_ratio": (0.02, 0.3),
    "pad_levels": (1, 8), "pad_branching": (0.0, 4.0), "root_count": (1, 40), "root_spread_ratio": (0.1, 5.0), "root_depth_m": (0.02, 2.0), "taproot_share": (0.0, 0.8), "taproot_depth_m": (0.05, 3.0), "root_core_ratio": (0.05, 0.6), "fine_roots": (0.0, 3.0), "tuber_length_cm": (1.0, 60.0), "tuber_radius_ratio": (0.1, 1.5), "glaucous": (0.0, 1.0), "flecks": (0.0, 1.0),
}

ROSETTE_RANGES = {
    "leaf_count": (3, 200), "stem_height_m": (0.0, 2.0), "stem_radius_m": (0.003, 0.5),
    "rosette_height_m": (0.0, 1.0), "leaf_length_cm": (0.5, 250.0), "leaf_aspect": (0.8, 20.0),
    "leaf_thickness": (0.03, 1.2), "thickness_taper": (0.0, 1.0), "size_gradient": (0.0, 0.95),
    "elevation_outer_deg": (-30.0, 90.0), "elevation_inner_deg": (0.0, 90.0), "elevation_power": (0.3, 4.0),
    "curvature_deg": (-90.0, 90.0), "widest_position": (0.05, 0.95), "base_angle_deg": (10.0, 179.0),
    "apex_angle_deg": (5.0, 179.0), "base_curvature": (-1.0, 1.0), "apex_curvature": (-1.0, 1.0),
    "channel": (0.0, 1.0), "keel": (0.0, 1.0), "section_exponent": (1.2, 6.0), "terminal_spine_cm": (0.0, 8.0),
    "teeth_count": (0, 60), "teeth_size_cm": (0.0, 3.0), "teeth_hook": (-1.0, 1.5), "offsets": (0, 20),
    "offset_scale": (0.1, 1.0), "blush_amount": (0.0, 1.0), "glaucous": (0.0, 1.0), "spots": (0.0, 1.0),
    "bands": (0.0, 1.0), "base_width": (0.0, 1.6), "clasp": (0.0, 1.0), "base_swell": (0.0, 4.0), "furl": (0.0, 1.0), "blush_tip": (0.0, 1.0), "margin_band": (0.0, 1.0), "striation": (0.0, 1.0), "imprints": (0.0, 1.0), "dead_leaves": (0, 40), "stem_scars": (0.0, 1.0), "scar_spacing_mm": (1.0, 30.0), "root_count": (1, 150), "root_spread_ratio": (0.1, 4.0),
    "root_depth_m": (0.02, 2.0), "root_radius_mm": (0.2, 10.0), "fine_roots": (0.0, 3.0),
}

RANGES = {GrowthForm.CACTUS: CACTUS_RANGES, GrowthForm.ROSETTE: ROSETTE_RANGES}
PROFILE_CLASSES = {GrowthForm.CACTUS: CactusProfile, GrowthForm.ROSETTE: RosetteProfile}
