"""
Climbing and trailing plant presets.

Habit values (climbing mode, helix handedness, internode and leaf sizes, tendril type, fruit dimensions) are
typical adult values from general botanical references and crop descriptions; they reproduce each
species' habit rather than a particular specimen. Twining handedness follows Darwin (1875) and Edwards,
Moles & Franks (2007, "The global trend in plant twining direction", Global Ecol. Biogeogr.): about 90 % of
twiners are right-handed (Ipomoea, Phaseolus, Wisteria sinensis); Humulus is left-handed. Tendril types
(Sousa-Baena et al. 2018, Ann. Bot.): leaf-tip tendrils in Pisum, leaf-opposed branched tendrils in Vitis,
axillary tendrils in Cucurbitaceae and Passiflora.
"""

from dataclasses import dataclass, field

from .vine import VineProfile, ClimbingMode, Chirality, LeafArrangement, TendrilMode
from .leaf_morphology import LeafMorphologyProfile, LeafArchetype, MarginType, ARCHETYPE_TEMPLATES
from .leaf_venation import VenationProfile, VenationPattern


@dataclass
class VinePreset:
    scientific_name: str
    common_name: str
    family: str
    biome: str
    notes: str
    profile: VineProfile
    leaf: LeafMorphologyProfile = field(default_factory=LeafMorphologyProfile)
    venation: VenationProfile = field(default_factory=VenationProfile)


M, C, LA, TM = ClimbingMode, Chirality, LeafArrangement, TendrilMode
A, MT, V = LeafArchetype, MarginType, VenationPattern


def _mix(a, b, t):
    return tuple(round(x * (1 - t) + y * t, 3) for x, y in zip(a, b))


def vine(sci, common, family, biome, notes, leaf=None, vein=None, **kw) -> VinePreset:
    leaf = dict(leaf or {})
    archetype = leaf.pop("archetype", A.CORDATE)
    lkw = dict(ARCHETYPE_TEMPLATES.get(archetype, {}))
    lkw.update(leaf)
    adaxial = lkw.get("adaxial_color", (0.20, 0.36, 0.10))
    lkw.setdefault("abaxial_color", _mix(adaxial, (0.62, 0.68, 0.50), 0.4))
    lkw.setdefault("vein_color", _mix(adaxial, (0.78, 0.82, 0.42), 0.5))
    return VinePreset(sci, common, family, biome, notes, VineProfile(**kw),
                      LeafMorphologyProfile(archetype=archetype, **lkw), VenationProfile(**(vein or {})))


PALMATE_VEINS = dict(pattern=V.ACTINODROMOUS, vla_mm_per_mm2=6.5, secondary_vein_pairs=5, divergence_angle_deg=45)

VINE_CATALOG: dict[str, VinePreset] = {
    "ipomoea_purpurea": vine(
        "Ipomoea purpurea", "Morning glory (manto de la virgen)", "Convolvulaceae", "Mexico; cultivated worldwide",
        "Annual right-handed twiner; heart-shaped entire leaves on long petioles; hairy stems.",
        leaf=dict(archetype=A.CORDATE, margin_type=MT.ENTIRE, blade_length_cm=8.0, aspect_ratio=1.05,
                  cordate_depth=0.45, apex_curvature=-0.5, petiole_length_ratio=0.8, petiole_angle_deg=60,
                  mean_leaf_angle_deg=30, adaxial_color=(0.18, 0.36, 0.12), gloss=0.25),
        vein=dict(PALMATE_VEINS, secondary_vein_pairs=6),
        mode=M.TWINING, chirality=C.RIGHT, coil_radius_cm=1.2, coil_pitch_cm=10.0, stem_radius_mm=2.0,
        internode_cm=8.0, tip_length_cm=30.0, tip_hook=0.6, branch_probability=0.08, branch_length_cm=60.0,
        stem_color=(0.32, 0.36, 0.16), hairiness=0.5),
    "phaseolus_coccineus": vine(
        "Phaseolus coccineus", "Runner bean (ayocote)", "Fabaceae", "Mexican highlands; cultivated",
        "Vigorous right-handed twiner; trifoliate leaves, scarlet racemes and long pods.",
        leaf=dict(archetype=A.PALMATE_COMPOUND, margin_type=MT.ENTIRE, blade_length_cm=10.0, aspect_ratio=1.3,
                  widest_position=0.4, leaflet_count=3, lobe_spread_deg=140, petiole_length_ratio=1.0,
                  petiole_angle_deg=55, adaxial_color=(0.16, 0.34, 0.10)),
        vein=dict(pattern=V.BROCHIDODROMOUS, secondary_vein_pairs=6),
        mode=M.TWINING, chirality=C.RIGHT, coil_radius_cm=1.5, coil_pitch_cm=14.0, stem_radius_mm=3.0,
        internode_cm=15.0, tip_length_cm=35.0, leaf_size=1.0, branch_probability=0.1, hairiness=0.3,
        stem_color=(0.26, 0.40, 0.14),
        fruit_count=6, fruit_length_cm=25.0, fruit_diameter_cm=1.8, fruit_widest_position=0.6, fruit_stalk_cm=4.0,
        fruit_color=(0.30, 0.45, 0.15), fruit_mottle=0.1, fruit_gloss=0.3),
    "humulus_lupulus": vine(
        "Humulus lupulus", "Hop (lúpulo)", "Cannabaceae", "Temperate Europe, Asia and North America",
        "Left-handed (clockwise) twiner climbing with hooked hairs; opposite, 3-5-lobed serrate leaves.",
        leaf=dict(archetype=A.PALMATE_LOBED, margin_type=MT.SERRATE, blade_length_cm=12.0, lobe_count=3,
                  lobe_depth=0.5, cordate_depth=0.25, teeth_count=18, tooth_height_ratio=0.05,
                  petiole_length_ratio=0.6, adaxial_color=(0.17, 0.33, 0.10)),
        vein=PALMATE_VEINS,
        mode=M.TWINING, chirality=C.LEFT, coil_radius_cm=2.0, coil_pitch_cm=15.0, stem_radius_mm=3.0,
        internode_cm=20.0, leaf_arrangement=LA.OPPOSITE, tip_length_cm=40.0, branch_probability=0.15,
        branch_length_cm=80.0, hairiness=0.6, stem_color=(0.30, 0.38, 0.18)),
    "wisteria_sinensis": vine(
        "Wisteria sinensis", "Chinese wisteria (glicinia)", "Fabaceae", "China; cultivated",
        "Woody right-handed twiner with thick grey trunks; odd-pinnate leaves; long pendulous lilac racemes.",
        leaf=dict(archetype=A.PINNATE_COMPOUND, margin_type=MT.ENTIRE, blade_length_cm=6.0, aspect_ratio=2.4,
                  leaflet_count=5, leaflet_angle_deg=70, rachis_length_ratio=4.0, petiole_length_ratio=0.3,
                  adaxial_color=(0.20, 0.38, 0.12)),
        vein=dict(pattern=V.BROCHIDODROMOUS, secondary_vein_pairs=7),
        mode=M.TWINING, chirality=C.RIGHT, coil_radius_cm=4.0, coil_pitch_cm=40.0, stem_radius_mm=25.0,
        stem_taper=0.85, internode_cm=20.0, tip_length_cm=60.0, branch_probability=0.25, branch_length_cm=150.0,
        branch_droop=0.7, woodiness=0.8, stem_color=(0.30, 0.36, 0.20), stem_color_old=(0.45, 0.42, 0.38),
        hairiness=0.2, fruit_count=3, fruit_length_cm=12.0, fruit_diameter_cm=2.0, fruit_widest_position=0.65,
        fruit_color=(0.55, 0.58, 0.45), fruit_mottle=0.1, fruit_gloss=0.1),
    "pisum_sativum": vine(
        "Pisum sativum", "Garden pea (chícharo)", "Fabaceae", "Mediterranean / Near East; cultivated",
        "Tendril climber: the terminal leaflets of each pinnate leaf are branched tendrils; glaucous foliage.",
        leaf=dict(archetype=A.PINNATE_COMPOUND, margin_type=MT.ENTIRE, blade_length_cm=4.0, aspect_ratio=1.6,
                  leaflet_count=2, terminal_leaflet=False, rachis_length_ratio=2.0, petiole_length_ratio=0.3,
                  adaxial_color=(0.35, 0.48, 0.32), gloss=0.15),
        vein=dict(pattern=V.BROCHIDODROMOUS, secondary_vein_pairs=5),
        mode=M.TENDRIL, stem_radius_mm=2.0, internode_cm=8.0, wander_cm=2.0, tip_length_cm=15.0,
        tendril_mode=TM.LEAF_TIP, tendril_branches=3, tendril_length_cm=8.0, tendril_coils=4.0,
        tendril_coil_mm=2.0, tendril_radius_mm=0.4, tendril_reach=0.6, stem_color=(0.38, 0.50, 0.32),
        hairiness=0.0, fruit_count=5, fruit_length_cm=8.0, fruit_diameter_cm=1.2, fruit_widest_position=0.6,
        fruit_stalk_cm=2.0, fruit_color=(0.32, 0.50, 0.20), fruit_mottle=0.05, fruit_gloss=0.4),
    "cucumis_sativus": vine(
        "Cucumis sativus", "Cucumber (pepino)", "Cucurbitaceae", "South Asia; cultivated",
        "Tendril climber with simple axillary tendrils; rough, shallowly lobed leaves; elongated fruits.",
        leaf=dict(archetype=A.PALMATE_LOBED, margin_type=MT.DENTATE, blade_length_cm=15.0, lobe_count=5,
                  lobe_depth=0.2, lobe_apex_angle_deg=60, cordate_depth=0.35, teeth_count=12,
                  tooth_height_ratio=0.03, petiole_length_ratio=0.8, adaxial_color=(0.18, 0.35, 0.12)),
        vein=PALMATE_VEINS,
        mode=M.TENDRIL, stem_radius_mm=4.0, internode_cm=9.0, wander_cm=3.0, tip_length_cm=20.0,
        tendril_mode=TM.NODE, tendril_branches=1, tendril_length_cm=20.0, tendril_coils=8.0,
        tendril_coil_mm=4.0, tendril_radius_mm=0.7, hairiness=0.7, stem_color=(0.28, 0.42, 0.16),
        fruit_count=4, fruit_length_cm=22.0, fruit_diameter_cm=4.0, fruit_widest_position=0.55,
        fruit_stalk_cm=3.0, fruit_color=(0.10, 0.25, 0.08), fruit_stripe_color=(0.45, 0.55, 0.25),
        fruit_stripes=0.3, fruit_mottle=0.2, fruit_gloss=0.5),
    "cucurbita_pepo": vine(
        "Cucurbita pepo", "Pumpkin (calabaza)", "Cucurbitaceae", "Mexico; cultivated",
        "Trailing runner rooting at the nodes; branched tendrils; very large, shallowly lobed, bristly leaves on "
        "erect petioles; large ribbed fruits lying on the soil.",
        leaf=dict(archetype=A.PALMATE_LOBED, margin_type=MT.DENTATE, blade_length_cm=28.0, lobe_count=5,
                  lobe_depth=0.3, cordate_depth=0.4, teeth_count=14, tooth_height_ratio=0.03,
                  petiole_length_ratio=1.1, mean_leaf_angle_deg=25, adaxial_color=(0.16, 0.32, 0.10)),
        vein=PALMATE_VEINS,
        mode=M.TRAILING, stem_radius_mm=9.0, stem_taper=0.5, internode_cm=14.0, wander_cm=8.0,
        tip_length_cm=30.0, tip_hook=0.3, leaf_facing=0.8, branch_probability=0.06, branch_length_cm=200.0,
        tendril_mode=TM.NODE, tendril_branches=3, tendril_length_cm=15.0, tendril_coils=5.0,
        tendril_coil_mm=5.0, tendril_radius_mm=1.0, tendril_reach=0.4, aerial_roots=0.5, rootlet_length_cm=6.0,
        hairiness=0.8, stem_color=(0.30, 0.42, 0.16),
        fruit_count=2, fruit_length_cm=22.0, fruit_diameter_cm=32.0, fruit_ribs=10, fruit_rib_depth=0.12,
        fruit_end_depression=0.35, fruit_stalk_cm=6.0, fruit_color=(0.88, 0.45, 0.06),
        fruit_stripe_color=(0.95, 0.62, 0.20), fruit_stripes=0.15, fruit_mottle=0.15, fruit_gloss=0.3),
    "citrullus_lanatus": vine(
        "Citrullus lanatus", "Watermelon (sandía)", "Cucurbitaceae", "Africa; cultivated",
        "Trailing runner with bifid tendrils and deeply pinnately lobed leaves; large striped fruits.",
        leaf=dict(archetype=A.PINNATE_LOBED, margin_type=MT.ENTIRE, blade_length_cm=15.0, lobe_count=3,
                  lobe_depth=0.75, lobe_roundness=0.8, petiole_length_ratio=0.4, mean_leaf_angle_deg=30,
                  adaxial_color=(0.25, 0.36, 0.18)),
        vein=dict(pattern=V.CRASPEDODROMOUS, secondary_vein_pairs=4),
        mode=M.TRAILING, stem_radius_mm=5.0, internode_cm=10.0, wander_cm=6.0, tip_length_cm=25.0, tip_hook=0.3,
        leaf_facing=0.7, branch_probability=0.12, branch_length_cm=150.0, tendril_mode=TM.NODE,
        tendril_branches=2, tendril_length_cm=10.0, tendril_coils=4.0, tendril_coil_mm=3.0,
        tendril_radius_mm=0.6, tendril_reach=0.3, aerial_roots=0.3, rootlet_length_cm=4.0, hairiness=0.6,
        stem_color=(0.30, 0.40, 0.18),
        fruit_count=1, fruit_length_cm=35.0, fruit_diameter_cm=28.0, fruit_stalk_cm=4.0,
        fruit_color=(0.50, 0.62, 0.30), fruit_stripe_color=(0.12, 0.28, 0.10), fruit_stripes=0.8,
        fruit_mottle=0.3, fruit_gloss=0.6),
    "vitis_vinifera": vine(
        "Vitis vinifera", "Grapevine (vid)", "Vitaceae", "Mediterranean / Caucasus; cultivated",
        "Woody tendril climber: forked tendrils opposite the leaves (two nodes of three); 5-lobed coarsely "
        "toothed leaves; shredding bark on old trunks. Fruit clusters are not modelled.",
        leaf=dict(archetype=A.PALMATE_LOBED, margin_type=MT.DENTATE, blade_length_cm=15.0, lobe_count=5,
                  lobe_depth=0.4, cordate_depth=0.4, teeth_count=10, tooth_height_ratio=0.07,
                  petiole_length_ratio=0.6, adaxial_color=(0.20, 0.36, 0.12)),
        vein=PALMATE_VEINS,
        mode=M.TENDRIL, stem_radius_mm=30.0, stem_taper=0.85, internode_cm=12.0, wander_cm=4.0, tip_length_cm=40.0,
        leaf_arrangement=LA.DISTICHOUS, branch_probability=0.3, branch_length_cm=120.0, branch_droop=0.5,
        tendril_mode=TM.NODE, tendril_branches=2, tendril_length_cm=18.0, tendril_coils=3.0,
        tendril_coil_mm=5.0, tendril_radius_mm=1.2, woodiness=0.7, stem_color=(0.40, 0.42, 0.20),
        stem_color_old=(0.42, 0.32, 0.24), hairiness=0.1),
    "hedera_helix": vine(
        "Hedera helix", "Ivy (hiedra)", "Araliaceae", "Europe; naturalised worldwide",
        "Root climber: adventitious rootlets on the shaded side press the stem to walls and trunks; dark glossy "
        "3-5-lobed leaves with pale veins facing away from the support.",
        leaf=dict(archetype=A.PALMATE_LOBED, margin_type=MT.ENTIRE, blade_length_cm=6.0, lobe_count=5,
                  lobe_depth=0.35, lobe_spread_deg=180, petiole_length_ratio=0.8, adaxial_color=(0.08, 0.18, 0.06),
                  vein_color=(0.60, 0.65, 0.55), gloss=0.7),
        vein=dict(PALMATE_VEINS, vein_contrast=0.8),
        mode=M.CLINGING, stem_radius_mm=3.0, internode_cm=4.0, wander_cm=3.0, tip_length_cm=15.0, tip_hook=0.2,
        leaf_arrangement=LA.DISTICHOUS, leaf_facing=0.9, branch_probability=0.2, branch_length_cm=50.0,
        branch_droop=0.2, aerial_roots=0.9, rootlet_length_cm=1.5, woodiness=0.6, stem_color=(0.30, 0.34, 0.18),
        stem_color_old=(0.40, 0.33, 0.25), hairiness=0.1),
    "passiflora_caerulea": vine(
        "Passiflora caerulea", "Blue passion flower (pasionaria)", "Passifloraceae", "South America; cultivated",
        "Tendril climber with simple axillary tendrils; deeply 5-lobed leaves; orange ovoid fruits.",
        leaf=dict(archetype=A.PALMATE_LOBED, margin_type=MT.ENTIRE, blade_length_cm=10.0, lobe_count=5,
                  lobe_depth=0.85, lobe_width=0.45, lobe_spread_deg=200, petiole_length_ratio=0.4,
                  adaxial_color=(0.16, 0.33, 0.10)),
        vein=PALMATE_VEINS,
        mode=M.TENDRIL, stem_radius_mm=3.0, internode_cm=10.0, wander_cm=3.0, tip_length_cm=30.0,
        tendril_mode=TM.NODE, tendril_branches=1, tendril_length_cm=20.0, tendril_coils=6.0,
        tendril_coil_mm=3.5, tendril_radius_mm=0.7, branch_probability=0.15, stem_color=(0.28, 0.40, 0.16),
        hairiness=0.0, fruit_count=2, fruit_length_cm=6.0, fruit_diameter_cm=4.0, fruit_stalk_cm=4.0,
        fruit_color=(0.95, 0.55, 0.05), fruit_mottle=0.1, fruit_gloss=0.6),
}

VINE_RANGES = {
    "coil_radius_cm": (0.2, 30.0), "coil_pitch_cm": (2.0, 150.0), "wander_cm": (0.0, 50.0),
    "tip_length_cm": (0.0, 300.0), "tip_hook": (0.0, 1.0), "stem_radius_mm": (0.3, 200.0),
    "stem_taper": (0.0, 0.95), "internode_cm": (0.5, 80.0), "leaf_size": (0.1, 3.0),
    "young_leaf_size": (0.05, 1.0), "expansion_zone_cm": (1.0, 300.0), "leaf_facing": (0.0, 1.0),
    "basal_leaf_loss": (0.0, 0.95), "branch_probability": (0.0, 1.0), "branch_length_cm": (0.0, 600.0),
    "branch_angle_deg": (5.0, 90.0), "branch_droop": (0.0, 1.0), "tendril_length_cm": (1.0, 60.0),
    "tendril_branches": (1, 6), "tendril_coils": (0.0, 30.0), "tendril_coil_mm": (0.5, 20.0),
    "tendril_radius_mm": (0.1, 3.0), "tendril_reach": (0.0, 1.0), "aerial_roots": (0.0, 1.0),
    "rootlet_length_cm": (0.2, 30.0), "woodiness": (0.0, 1.0), "hairiness": (0.0, 1.0),
    "fruit_count": (0, 100), "fruit_length_cm": (0.5, 120.0), "fruit_diameter_cm": (0.3, 100.0),
    "fruit_widest_position": (0.15, 0.85), "fruit_neck": (0.0, 1.0), "fruit_ribs": (0, 30),
    "fruit_rib_depth": (0.0, 0.4), "fruit_end_depression": (0.0, 0.6), "fruit_stalk_cm": (0.0, 40.0),
    "fruit_stripes": (0.0, 1.0), "fruit_mottle": (0.0, 1.0), "fruit_gloss": (0.0, 1.0),
}


def leaf_ranges() -> dict:
    """Leaf and venation ranges shared with the tree presets."""
    from .trait_space import TRAITS
    out = {}
    for t in TRAITS:
        sec, _, name = t.path.partition(".")
        if sec in ("leaf_morphology", "venation"):
            out[("leaf" if sec == "leaf_morphology" else "venation", name)] = (t.lo, t.hi)
    return out


def vine_items() -> list[tuple[str, str, str]]:
    return [(k, f"{v.scientific_name} ({v.common_name})", f"{v.family} · {v.profile.mode.value}. {v.notes}")
            for k, v in sorted(VINE_CATALOG.items(), key=lambda kv: kv[1].scientific_name)]
