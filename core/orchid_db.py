"""
Orchid presets. Measurements from:

- Phalaenopsis: USPTO plant patent PP25,332 ('Northstar', 2015): plant 59 cm, 6-8 leaves 17-24 x 7-11 cm,
  2 mm thick, two-ranked; fleshy roots 4-7 mm; 1-2 racemes 50-60 cm with 5-12 flowers, internodes 3-4 cm;
  flower 92 x 80 mm; petals 56 x 40 mm; sepals 45 x 30-35 mm; lip 3-lobed, 20 x 25 mm, lateral lobes folded up
  about the column, two cirrhi ~17 mm, callus 4 x 6 x 7 mm; pedicel 40 x 3 mm; column 10 x 6 mm. Spikes from
  the 3rd-4th leaf below the apex (Liu et al. 2022, Int. J. Mol. Sci. 23: 10461).
- Dendrobium nobile: Flora of China 25 (2009): stems 10-60 cm x 1.3 cm, internodes 2-4 cm; leaves 6-11 x 1-3
  cm; racemes 2-4 cm with 1-4 flowers on old stems; sepals 25-35 x 10-14 mm; petals 25-35 x 18-25 mm; lip
  broadly ovate 25-35 x 22-32 mm, clawed, embracing the column below its middle, purple disc spot; mentum ~6 mm.
- Paphiopedilum insigne: POWO / Flora of China: 5-6 leaves to 32 x 3 cm; scape to 25 cm, one flower 7-11.6
  cm; dorsal sepal 4-7 x 3-4.2 cm with spots; synsepal 2.2-5 x 1.4-2.5 cm; petals 5-6.3 x 1-1.8 cm, upper margin
  undulate; pouch 2.4-5 x 1.5-3 cm; staminode ~10 mm.
- Oncidium sphacelatum: Flora of the Bahamas / NYBG: pseudobulbs 8-12 x 1.5-2 cm, compressed, 2-3 linear
  leaves; arching panicle 1-1.5 m; flowers 2.5-3 cm; sepals ~16 mm barred brown and yellow; petals ~12 x 5 mm;
  lip ~15 mm, narrow base, two short side lobes, broad mid lobe 11-14 mm, white ridged callus.
- Cattleya labiata: RHS / Wikipedia: club-shaped unifoliate pseudobulbs with papery sheaths, leaf to 30 x 5 cm;
  2-5 flowers to 15 cm from a sheath at the apex; lip rolled around the column, frilled, dark-veined with a
  yellow throat.
- Cymbidium: Flora of China (C. erythraeum, C. wenshanense, C. lowianum): ovoid pseudobulbs hidden by leaf
  bases, strap-shaped leaves 40-90 x 1.5-2.5 cm; erect-arching racemes of 6-20 flowers; sepals 34-64 x 7-21
  mm; lip 3-lobed with two lamellae, lateral lobes erect, mid lobe recurved, spotted.
- Prosthechea cochleata: Flora of North America: pseudobulbs 3-15 x 1-3.5 cm, 1-3 leaves to 40 x 6 cm; raceme
  to 45 cm; flowers non-resupinate (lip uppermost); sepals and petals linear-lanceolate 25-35 x 3-6 mm, hanging
  and twisted; lip cordate, concave (shell-like), dark purple veined; column 6-10 mm.
- Vanilla planifolia: UF/IFAS HS1348 and Flora treatments: stems 5-10 mm thick, internodes to 12 cm; leaves to
  25 x 8 cm, fleshy; one aerial root per node opposite the leaf; axillary racemes ~5 cm with up to 15 flowers;
  tepals 3.5-5.5 x 1.1-1.3 cm yellow-green; lip 4-5 x 3 cm fused to the column for 1.5-2 cm (a trumpet);
  capsules ('beans') 15-25 x 0.8-1 cm.
- Angraecum sesquipedale (Darwin's orchid): Smithsonian Gardens, NParks, Wikipedia: monopodial epiphyte; leaves
  22-30 x 3 cm, two-ranked, leathery, unequally bilobed; inflorescence 30-46 cm with 1-4(6) waxy white star
  flowers 15-20 cm across; concave pointed lip; spur 27-43 cm (mean ~33 cm), nectar only at its tip.
- Stanhopea tigrina: Kew, RBG Victoria Horticultural Flora, RHS: clustered epiphyte, ovoid pseudobulbs with
  one leaf ~40 x 10 cm; pendent inflorescence ~20 cm growing down through the substrate with 1-2 heavily
  scented flowers to 18-20 cm, sepals and petals bent back, lip divided into a deeply pouched hypochile, a
  mesochile with two forward horns and a pointed epichile.
- Bulbophyllum rothschildianum: Flora of China, orchidspecies.com: ovoid pseudobulbs ~3 cm, well spaced, one
  leaf 9-10 x 2-2.5 cm; erect scape 20-24 cm with an umbel of 4-6 maroon flowers; dorsal sepal ~15 x 7 mm
  with fringed margins (fringes to 7 mm), petals ~10 x 4 mm fringed; lateral sepals fused into a long tail
  (flower ~2.5 x 15-17 cm).
"""

from dataclasses import dataclass, fields

from .orchid import OrchidProfile, OrchidHabit, LeafPlacement, InflOrigin, OrchidMount


@dataclass
class OrchidPreset:
    scientific_name: str
    common_name: str
    family: str
    notes: str
    profile: OrchidProfile


H, LP, IO, MO = OrchidHabit, LeafPlacement, InflOrigin, OrchidMount


def _typed(kw: dict) -> dict:
    floats = {f.name for f in fields(OrchidProfile) if isinstance(getattr(OrchidProfile(), f.name), float)}
    return {k: float(v) if k in floats and isinstance(v, int) and not isinstance(v, bool) else v for k, v in kw.items()}


def orchid(sci, common, notes, **kw) -> OrchidPreset:
    return OrchidPreset(sci, common, "Orchidaceae", notes, OrchidProfile(**_typed(kw)))


WHITE = (0.97, 0.96, 0.97)

ORCHID_CATALOG: dict[str, OrchidPreset] = {
    "phalaenopsis_hybrid": orchid(
        "Phalaenopsis hybrids", "Moth orchid (orquídea mariposa)",
        "Monopodial: short stem, two ranks of broad fleshy leaves, thick silvery aerial roots; arching racemes "
        "from the axils of the lower leaves; round white flowers, 3-lobed lip with erect side lobes, callus "
        "and two cirrhi.",
        habit=H.MONOPODIAL, stem_length_cm=4, stem_radius_mm=7, internode_cm=1.1, leaf_placement=LP.ALONG,
        leaves=7, leaf_length_cm=21, leaf_width_cm=9, leaf_thickness_mm=2.0, leaf_widest=0.62, leaf_apex_deg=130,
        leaf_angle_deg=75, leaf_droop=0.45, leaf_fold=0.35, leaf_twist_deg=12, leaf_size_gradient=0.35,
        leaf_color=(0.10, 0.28, 0.08), roots=9, root_diameter_mm=5.5, root_length_cm=24, root_wander=0.6,
        aerial_share=0.6, infl_origin=IO.AXIL, inflorescences=2, flowers=9, peduncle_cm=28, rachis_cm=30,
        spike_radius_mm=2.5, spike_angle_deg=35, spike_flex=0.62, divergence_deg=180, pedicel_cm=4.0,
        flower_facing=0.75, maturation=0.25, bract_mm=4, spike_color=(0.22, 0.30, 0.15),
        resupination=1.0, flower_tilt_deg=10, sepal_length_cm=4.5, sepal_width_cm=3.0, sepal_widest=0.55,
        sepal_apex_deg=95, lateral_sepal_deg=58, sepal_cup=0.12, sepal_reflex_deg=8,
        petal_length_cm=5.0, petal_width_cm=5.4, petal_widest=0.62, petal_apex_deg=170, petal_claw=0.18,
        petal_angle_deg=12, petal_cup=0.08, petal_reflex_deg=6, perianth_forward_deg=8,
        lip_length_cm=2.6, lip_width_cm=3.2, lip_angle_deg=62, lip_deflex_deg=25, lip_claw=0.2, side_lobes=0.95,
        side_lobe_pos=0.3, side_lobe_length=0.42, side_lobe_erect_deg=50, midlobe_width=0.55, isthmus=0.3,
        lip_widest=0.35, lip_apex_deg=55, callus_mm=4.5, callus_pos=0.42, cirrhi_mm=12,
        column_length_cm=1.0, column_width_mm=6, column_arch=0.35,
        sepal_color=WHITE, petal_color=WHITE, lip_color=(0.97, 0.94, 0.90), lip_throat_color=(0.96, 0.82, 0.25),
        callus_color=(0.97, 0.78, 0.18), column_color=(0.97, 0.96, 0.94), anther_color=(0.96, 0.90, 0.70),
        pattern_color=(0.70, 0.12, 0.30), lip_spots=0.45, lip_veins=0.4, sheen=0.4),
    "cattleya_labiata": orchid(
        "Cattleya labiata", "Corsage orchid (catleya)",
        "Sympodial: club-shaped unifoliate pseudobulbs in papery sheaths; 2-5 large fragrant flowers from a "
        "sheath at the apex; wavy petals much wider than the sepals; lip rolled into a tube around the column, "
        "flaring into a frilled, dark-veined front lobe with a yellow throat.",
        habit=H.SYMPODIAL, growths=4, rhizome_cm=4, growth_angle_deg=20, leafless_growths=1, stem_radius_mm=5,
        pseudobulb_cm=20, pseudobulb_diameter_cm=2.6, bulb_widest=0.72, bulb_fullness=0.55, bulb_flatten=0.25,
        bulb_ridges=5, bulb_lean_deg=12, sheath_cover=0.45, leaf_placement=LP.APEX, leaves=1, leaf_length_cm=25,
        leaf_width_cm=5, leaf_thickness_mm=2.5, leaf_widest=0.5, leaf_apex_deg=80, leaf_angle_deg=22,
        leaf_droop=0.15, leaf_fold=0.5, leaf_color=(0.22, 0.38, 0.14), roots=4, root_diameter_mm=4,
        root_length_cm=15, aerial_share=0.6, infl_origin=IO.APEX, inflorescences=1, flowers=3, peduncle_cm=8,
        rachis_cm=7, spike_radius_mm=3, spike_angle_deg=20, spike_flex=0.35, divergence_deg=150, pedicel_cm=7,
        flower_facing=0.7, maturation=0.0, bract_mm=3, spike_color=(0.35, 0.40, 0.22), resupination=1.0,
        sepal_length_cm=8, sepal_width_cm=1.9, sepal_widest=0.5, sepal_apex_deg=60, lateral_sepal_deg=40,
        sepal_cup=0.15, sepal_reflex_deg=15, petal_length_cm=8, petal_width_cm=5, petal_widest=0.62,
        petal_apex_deg=150, petal_claw=0.25, petal_angle_deg=22, petal_cup=0.1, petal_reflex_deg=15,
        petal_wave=0.9, perianth_forward_deg=10, lip_length_cm=6.5, lip_width_cm=6.5, lip_angle_deg=15,
        lip_deflex_deg=65, lip_claw=0.3, side_lobes=0.9, side_lobe_pos=0.32, side_lobe_length=0.62,
        side_lobe_erect_deg=150,
        midlobe_width=1.0, isthmus=0.1, lip_widest=0.5, lip_apex_deg=170, lip_roll=0.3, lip_roll_extent=0.5,
        lip_wave=1.4, lip_waves=7, lip_flare=0.8, callus_mm=1.5, callus_ridges=3, callus_pos=0.3, column_length_cm=3.0,
        column_width_mm=6, column_arch=0.25, sepal_color=(0.88, 0.62, 0.82), petal_color=(0.88, 0.62, 0.82),
        lip_color=(0.68, 0.10, 0.45), lip_throat_color=(0.97, 0.80, 0.25), callus_color=(0.97, 0.82, 0.30),
        column_color=(0.92, 0.80, 0.88), pattern_color=(0.45, 0.04, 0.25), lip_veins=0.7, sheen=0.45),
    "dendrobium_nobile": orchid(
        "Dendrobium nobile", "Noble dendrobium (caña de ámbar)",
        "Sympodial canes with swollen internodes and two-ranked leaves on the upper half, deciduous after a "
        "year; 1-4 waxy flowers from the nodes of the leafless canes; lip downy, rolled around the column at "
        "the base, with a dark maroon disc; conical mentum.",
        habit=H.SYMPODIAL, growths=5, rhizome_cm=1.5, growth_angle_deg=30, leafless_growths=2, stem_radius_mm=4,
        pseudobulb_cm=40, pseudobulb_diameter_cm=1.3, bulb_widest=0.6, bulb_fullness=0.25, bulb_flatten=0.1,
        bulb_ridges=8, bulb_lean_deg=12, sheath_cover=0.0, internode_cm=3, leaf_placement=LP.ALONG, leaves=8,
        leaf_length_cm=9, leaf_width_cm=2.2, leaf_thickness_mm=1.0, leaf_widest=0.45, leaf_apex_deg=70,
        leaf_angle_deg=55, leaf_droop=0.2, leaf_fold=0.3, leaf_color=(0.25, 0.42, 0.15), roots=4,
        root_diameter_mm=2.5, root_length_cm=12, infl_origin=IO.NODES, inflorescences=9, flowers=2,
        peduncle_cm=1.0, rachis_cm=2.0, spike_radius_mm=1.2, spike_angle_deg=60, spike_flex=0.2, pedicel_cm=3.5,
        flower_facing=0.6, maturation=0.0, bract_mm=5, spike_color=(0.55, 0.40, 0.45), resupination=1.0,
        sepal_length_cm=3.0, sepal_width_cm=1.2, sepal_widest=0.45, sepal_apex_deg=70, lateral_sepal_deg=45,
        sepal_cup=0.1, sepal_reflex_deg=10, petal_length_cm=3.0, petal_width_cm=2.1, petal_widest=0.55,
        petal_apex_deg=130, petal_claw=0.15, petal_angle_deg=25, petal_wave=0.2, perianth_forward_deg=12,
        lip_length_cm=3.0, lip_width_cm=2.7, lip_angle_deg=30, lip_deflex_deg=35, lip_claw=0.25, lip_widest=0.55,
        lip_apex_deg=140, lip_roll=0.75, lip_roll_extent=0.4, lip_wave=0.2, column_length_cm=0.5,
        column_width_mm=4, mentum_mm=6, sepal_color=(0.95, 0.90, 0.95), petal_color=(0.95, 0.88, 0.94),
        lip_color=(0.97, 0.94, 0.85), lip_throat_color=(0.40, 0.05, 0.12), callus_color=(0.40, 0.05, 0.12),
        column_color=(0.95, 0.90, 0.80), anther_color=(0.65, 0.20, 0.45), pattern_color=(0.70, 0.15, 0.55),
        tip_color=(0.72, 0.25, 0.62), tip_amount=0.45,
        sheen=0.5),
    "paphiopedilum_insigne": orchid(
        "Paphiopedilum insigne", "Lady's slipper (zapatito de Venus)",
        "Sympodial without pseudobulbs: fans of strap-shaped leaves; one flower on an erect scape; spotted "
        "dorsal sepal with a white margin, synsepal behind the pouch, drooping wavy petals, helmet-shaped "
        "pouch and a shield-like staminode with a central boss.",
        habit=H.SYMPODIAL, growths=3, rhizome_cm=2.5, growth_angle_deg=35, stem_radius_mm=5, pseudobulb_cm=0,
        leaf_placement=LP.BASE, leaves=5, leaf_length_cm=28, leaf_width_cm=3.0, leaf_thickness_mm=1.2,
        leaf_widest=0.6, leaf_apex_deg=90, leaf_angle_deg=40, leaf_droop=0.45, leaf_fold=0.6,
        leaf_color=(0.25, 0.42, 0.15), leaf_mottle=0.0, roots=5, root_diameter_mm=3, root_length_cm=14,
        aerial_share=0.2, infl_origin=IO.APEX, inflorescences=1, flowers=1, peduncle_cm=24, rachis_cm=1,
        spike_radius_mm=2.5, spike_angle_deg=8, spike_flex=0.15, pedicel_cm=4.5, flower_facing=0.85,
        maturation=0.0, bract_mm=12, spike_color=(0.30, 0.22, 0.20), resupination=1.0, sepal_length_cm=5.5,
        sepal_width_cm=3.8, sepal_widest=0.5, sepal_apex_deg=120, lateral_sepal_scale=0.7, synsepal=1.0,
        sepal_cup=0.35, sepal_reflex_deg=-10, petal_length_cm=5.8, petal_width_cm=1.5, petal_widest=0.5,
        petal_apex_deg=110, petal_angle_deg=-22, petal_cup=0.0, petal_reflex_deg=5, petal_twist_deg=10,
        petal_wave=0.5, perianth_forward_deg=10, lip_length_cm=4.2, lip_width_cm=2.8, lip_angle_deg=55,
        lip_deflex_deg=20, lip_claw=0.45, lip_widest=0.6, lip_apex_deg=175, lip_sac=1.0,
        column_length_cm=0.8, column_width_mm=6, column_arch=0.0, staminode_mm=10,
        sepal_color=(0.62, 0.72, 0.30), petal_color=(0.72, 0.55, 0.22), lip_color=(0.72, 0.55, 0.18),
        lip_throat_color=(0.75, 0.62, 0.25), column_color=(0.85, 0.80, 0.40), anther_color=(0.90, 0.85, 0.30),
        pattern_color=(0.42, 0.12, 0.10), spots=0.55, spot_size_mm=2.5, veins=0.6, lip_veins=0.3, sheen=0.7),
    "oncidium_sphacelatum": orchid(
        "Oncidium sphacelatum", "Dancing lady orchid (lluvia de oro)",
        "Sympodial with compressed ovoid pseudobulbs and 2-3 long linear leaves; a long arching panicle from "
        "the base of the newest bulb with many small flowers; sepals and petals barred brown on yellow; large "
        "yellow pandurate lip (the 'skirt') with a ridged white callus.",
        habit=H.SYMPODIAL, growths=4, rhizome_cm=3, growth_angle_deg=20, leafless_growths=1, stem_radius_mm=4,
        pseudobulb_cm=10, pseudobulb_diameter_cm=3.0, bulb_widest=0.35, bulb_fullness=0.8, bulb_flatten=0.45,
        bulb_ridges=2, bulb_lean_deg=8, sheath_cover=0.15, leaf_placement=LP.APEX, leaves=2,
        leaf_length_cm=45, leaf_width_cm=2.5, leaf_thickness_mm=1.0, leaf_widest=0.4, leaf_apex_deg=40,
        leaf_angle_deg=25, leaf_droop=0.4, leaf_fold=0.5, leaf_color=(0.22, 0.42, 0.14), roots=5,
        root_diameter_mm=2.5, root_length_cm=15, infl_origin=IO.BASE, inflorescences=1, flowers=9, peduncle_cm=55,
        rachis_cm=55, spike_radius_mm=2.5, spike_angle_deg=20, spike_flex=0.7, branches=8, branch_ratio=0.25,
        divergence_deg=137.5, pedicel_cm=2.0, flower_facing=0.6, maturation=0.15, bract_mm=3,
        spike_color=(0.30, 0.32, 0.18), resupination=1.0, sepal_length_cm=1.6, sepal_width_cm=0.55,
        sepal_widest=0.55, sepal_apex_deg=80, lateral_sepal_deg=55, sepal_wave=0.3, petal_length_cm=1.2,
        petal_width_cm=0.5, petal_widest=0.55, petal_apex_deg=90, petal_angle_deg=25, petal_wave=0.4,
        perianth_forward_deg=5, lip_length_cm=1.5, lip_width_cm=1.4, lip_angle_deg=50, lip_deflex_deg=25,
        lip_claw=0.18, side_lobes=0.35, side_lobe_pos=0.18, side_lobe_length=0.25, side_lobe_erect_deg=10,
        midlobe_width=1.0, isthmus=0.7, lip_widest=0.55, lip_apex_deg=170, lip_wave=0.35, lip_waves=3,
        callus_mm=1.5, callus_ridges=3, callus_pos=0.18, column_length_cm=0.35, column_width_mm=2.5,
        sepal_color=(0.88, 0.70, 0.10), petal_color=(0.90, 0.72, 0.10), lip_color=(0.98, 0.85, 0.10),
        lip_throat_color=(0.85, 0.55, 0.10), callus_color=(0.96, 0.94, 0.85), column_color=(0.95, 0.85, 0.30),
        pattern_color=(0.45, 0.22, 0.05), spots=0.8, spot_size_mm=1.4, spot_stretch=3.0, lip_spots=0.25,
        sheen=0.3),
    "cymbidium_hybrid": orchid(
        "Cymbidium hybrids", "Boat orchid (cimbidio)",
        "Sympodial with ovoid pseudobulbs hidden by the bases of long arching strap-shaped leaves; erect, "
        "arching racemes from the base of the bulbs; waxy tepals; 3-lobed spotted lip with erect side lobes, "
        "recurved mid lobe and two hairy lamellae.",
        habit=H.SYMPODIAL, growths=3, rhizome_cm=4, growth_angle_deg=30, leafless_growths=1, stem_radius_mm=6,
        pseudobulb_cm=7, pseudobulb_diameter_cm=4.5, bulb_widest=0.4, bulb_fullness=0.9, bulb_flatten=0.15,
        bulb_lean_deg=5, sheath_cover=0.6, leaf_placement=LP.BASE, leaves=7, leaf_length_cm=65,
        leaf_width_cm=2.0, leaf_thickness_mm=0.8, leaf_widest=0.45, leaf_apex_deg=35, leaf_angle_deg=20,
        leaf_droop=0.6, leaf_fold=0.6, leaf_twist_deg=20, leaf_color=(0.22, 0.42, 0.16), roots=6,
        root_diameter_mm=5, root_length_cm=18, aerial_share=0.15, infl_origin=IO.BASE, inflorescences=1,
        flowers=10, peduncle_cm=30, rachis_cm=35, spike_radius_mm=4, spike_angle_deg=15, spike_flex=0.55,
        divergence_deg=150, pedicel_cm=4, flower_facing=0.6, maturation=0.2, bract_mm=6,
        spike_color=(0.30, 0.40, 0.18), resupination=1.0, sepal_length_cm=5.0, sepal_width_cm=1.6,
        sepal_widest=0.6, sepal_apex_deg=70, lateral_sepal_deg=40, sepal_cup=0.2, sepal_reflex_deg=5,
        petal_length_cm=4.6, petal_width_cm=1.5, petal_widest=0.6, petal_apex_deg=75, petal_angle_deg=35,
        petal_cup=0.25, petal_reflex_deg=-10, perianth_forward_deg=18, lip_length_cm=3.8, lip_width_cm=2.6,
        lip_angle_deg=25, lip_deflex_deg=75, lip_claw=0.25, side_lobes=0.9, side_lobe_pos=0.3,
        side_lobe_length=0.55, side_lobe_erect_deg=75, midlobe_width=0.7, isthmus=0.2, lip_widest=0.4,
        lip_apex_deg=130, lip_wave=0.25, callus_mm=1.8, callus_ridges=2, callus_pos=0.35, column_length_cm=2.8,
        column_width_mm=5, column_arch=0.4, sepal_color=(0.80, 0.86, 0.45), petal_color=(0.80, 0.86, 0.45),
        lip_color=(0.97, 0.94, 0.85), lip_throat_color=(0.96, 0.85, 0.40), callus_color=(0.98, 0.88, 0.35),
        column_color=(0.85, 0.80, 0.55), anther_color=(0.95, 0.88, 0.40), pattern_color=(0.70, 0.10, 0.12),
        veins=0.25, lip_spots=0.75, spot_size_mm=1.8, sheen=0.6),
    "prosthechea_cochleata": orchid(
        "Prosthechea cochleata", "Clamshell orchid (pulpo, orquídea concha)",
        "Mexican and Caribbean epiphyte: pear-shaped compressed pseudobulbs with 2 leaves; erect raceme from the "
        "apex; flowers NOT resupinate: the shell-shaped, dark purple veined lip is uppermost and the narrow "
        "twisted yellow-green tepals hang below like octopus arms.",
        habit=H.SYMPODIAL, growths=4, rhizome_cm=2.5, growth_angle_deg=25, leafless_growths=1, stem_radius_mm=4,
        pseudobulb_cm=10, pseudobulb_diameter_cm=2.8, bulb_widest=0.3, bulb_fullness=0.85, bulb_flatten=0.4,
        bulb_ridges=4, sheath_cover=0.2, leaf_placement=LP.APEX, leaves=2, leaf_length_cm=25, leaf_width_cm=3.0,
        leaf_thickness_mm=1.2, leaf_widest=0.5, leaf_apex_deg=50, leaf_angle_deg=25, leaf_droop=0.3,
        leaf_fold=0.4, leaf_color=(0.20, 0.38, 0.14), roots=4, root_diameter_mm=2.5, root_length_cm=12,
        infl_origin=IO.APEX, inflorescences=1, flowers=6, peduncle_cm=15, rachis_cm=20, spike_radius_mm=2,
        spike_angle_deg=10, spike_flex=0.2, divergence_deg=137.5, pedicel_cm=2.5, flower_facing=0.6,
        maturation=0.3, bract_mm=4, spike_color=(0.30, 0.38, 0.18), resupination=0.0, flower_tilt_deg=0,
        sepal_length_cm=3.2, sepal_width_cm=0.45, sepal_widest=0.35, sepal_apex_deg=25, lateral_sepal_deg=-60,
        sepal_cup=0.1, sepal_reflex_deg=-25, sepal_twist_deg=200, petal_length_cm=3.0, petal_width_cm=0.35,
        petal_widest=0.35, petal_apex_deg=25, petal_angle_deg=62, petal_reflex_deg=-25, petal_twist_deg=240,
        perianth_forward_deg=-10, lip_length_cm=2.0, lip_width_cm=2.6, lip_angle_deg=40, lip_deflex_deg=10,
        lip_claw=0.2, lip_widest=0.55, lip_apex_deg=150, lip_roll=0.35, lip_roll_extent=1.0,
        column_length_cm=0.8, column_width_mm=4, column_arch=0.1, sepal_color=(0.75, 0.80, 0.35),
        petal_color=(0.75, 0.80, 0.35), lip_color=(0.22, 0.05, 0.12), lip_throat_color=(0.85, 0.80, 0.45),
        callus_color=(0.85, 0.80, 0.45), column_color=(0.85, 0.85, 0.65), anther_color=(0.90, 0.85, 0.50),
        pattern_color=(0.90, 0.85, 0.55), lip_veins=0.8, spots=0.15, sheen=0.4),
    "vanilla_planifolia": orchid(
        "Vanilla planifolia", "Vanilla (vainilla)",
        "Climbing orchid native to Mexico: a thick green vine with fleshy alternate leaves and one aerial root "
        "per node (opposite the leaf) clinging to the support; short axillary racemes whose flowers open one "
        "at a time; greenish-yellow tepals and a trumpet-shaped lip fused to the column; long pods (beans).",
        habit=H.CLIMBING, stem_length_cm=260, stem_radius_mm=4.5, internode_cm=9, support_height_m=1.5,
        support_radius_cm=6, leaf_placement=LP.ALONG, leaves=1, leaf_length_cm=17, leaf_width_cm=5.5,
        leaf_thickness_mm=2.2, leaf_widest=0.5, leaf_apex_deg=60, leaf_angle_deg=55, leaf_droop=0.3,
        leaf_fold=0.1, leaf_color=(0.18, 0.40, 0.12), stem_color=(0.30, 0.48, 0.18), roots=6,
        root_diameter_mm=2.5, root_length_cm=25, root_color=(0.62, 0.65, 0.50), infl_origin=IO.AXIL,
        inflorescences=3, flowers=10, peduncle_cm=2, rachis_cm=5, spike_radius_mm=2.5, spike_angle_deg=45,
        spike_flex=0.5, divergence_deg=137.5, pedicel_cm=4.5, flower_facing=0.5, maturation=0.75, bract_mm=6,
        spike_color=(0.35, 0.50, 0.20), fruit_set=0.3, capsule_cm=18, capsule_diameter_cm=0.9,
        capsule_color=(0.30, 0.45, 0.12), resupination=1.0, sepal_length_cm=4.8, sepal_width_cm=1.2,
        sepal_widest=0.6, sepal_apex_deg=55, lateral_sepal_deg=40, sepal_cup=0.2, sepal_reflex_deg=10,
        petal_length_cm=4.6, petal_width_cm=1.2, petal_widest=0.6, petal_apex_deg=55, petal_angle_deg=30,
        petal_cup=0.2, perianth_forward_deg=25, lip_length_cm=4.5, lip_width_cm=3.0, lip_angle_deg=15,
        lip_deflex_deg=30, lip_claw=0.4, lip_widest=0.75, lip_apex_deg=170, lip_roll=1.0, lip_roll_extent=0.65,
        lip_wave=0.5, lip_waves=5, callus_mm=1.2, callus_ridges=4, callus_pos=0.5, column_length_cm=3.0,
        column_width_mm=4, column_arch=0.1, sepal_color=(0.82, 0.88, 0.55), petal_color=(0.84, 0.88, 0.55),
        lip_color=(0.92, 0.90, 0.60), lip_throat_color=(0.95, 0.85, 0.35), callus_color=(0.96, 0.86, 0.40),
        column_color=(0.92, 0.92, 0.70), anther_color=(0.90, 0.85, 0.50), pattern_color=(0.85, 0.65, 0.15),
        lip_veins=0.3, sheen=0.5),
    "angraecum_sesquipedale": orchid(
        "Angraecum sesquipedale", "Darwin's orchid (comet orchid)",
        "Monopodial epiphyte of Madagascar with two ranks of leathery strap leaves; 1-4 waxy white star-shaped "
        "flowers with a concave pointed lip and a nectar spur about 30 cm long, made famous by Darwin's "
        "prediction of a long-tongued hawkmoth (Xanthopan praedicta).",
        habit=H.MONOPODIAL, mount=MO.BRANCH, branch_diameter_cm=14, stem_length_cm=20, stem_radius_mm=9,
        internode_cm=2.2, leaf_placement=LP.ALONG, leaves=10, leaf_length_cm=27, leaf_width_cm=3.2,
        leaf_thickness_mm=2.5, leaf_widest=0.5, leaf_apex_deg=90, leaf_angle_deg=70, leaf_droop=0.25,
        leaf_fold=0.5, leaf_size_gradient=0.2, leaf_color=(0.14, 0.32, 0.12), roots=8, root_diameter_mm=7,
        root_length_cm=35, root_wander=0.5, aerial_share=0.5, root_color=(0.70, 0.70, 0.62), inflorescences=2,
        flowers=3, peduncle_cm=12, rachis_cm=18, spike_radius_mm=3, spike_angle_deg=55, spike_flex=0.4,
        divergence_deg=180, pedicel_cm=6, flower_facing=0.7, maturation=0.0, bract_mm=10,
        spike_color=(0.30, 0.42, 0.20), sepal_length_cm=8, sepal_width_cm=2.4, sepal_widest=0.3,
        sepal_apex_deg=40, lateral_sepal_deg=35, sepal_cup=0.0, sepal_reflex_deg=10, petal_length_cm=7,
        petal_width_cm=2.2, petal_widest=0.3, petal_apex_deg=40, petal_angle_deg=20, perianth_forward_deg=0,
        lip_length_cm=7.5, lip_width_cm=4.5, lip_angle_deg=55, lip_deflex_deg=15, lip_claw=0.6, lip_widest=0.25,
        lip_apex_deg=40, lip_roll=0.25, lip_roll_extent=1.0, spur_cm=33, spur_diameter_mm=4, spur_curve=0.7,
        column_length_cm=0.6, column_width_mm=7, sepal_color=(0.97, 0.97, 0.92), petal_color=(0.98, 0.98, 0.95),
        lip_color=(0.98, 0.98, 0.96), lip_throat_color=(0.92, 0.95, 0.85), column_color=(0.96, 0.96, 0.92),
        sheen=0.7),
    "stanhopea_tigrina": orchid(
        "Stanhopea tigrina", "Tiger-spotted stanhopea (torito)",
        "Mexican epiphyte: ovoid pseudobulbs with one large pleated leaf; the inflorescence grows down through "
        "the substrate and hangs 1-2 huge, heavily scented, waxy flowers with reflexed tiger-blotched tepals "
        "and a three-part lip: pouched hypochile, horned mesochile and pointed epichile.",
        habit=H.SYMPODIAL, mount=MO.BRANCH, branch_diameter_cm=16, growths=4, rhizome_cm=3, growth_angle_deg=25,
        leafless_growths=1, stem_radius_mm=5, pseudobulb_cm=6, pseudobulb_diameter_cm=3.5, bulb_widest=0.4,
        bulb_fullness=0.9, bulb_ridges=6, sheath_cover=0.4, leaf_placement=LP.APEX, leaves=1, leaf_length_cm=40,
        leaf_width_cm=10, leaf_thickness_mm=0.8, leaf_widest=0.5, leaf_apex_deg=50, leaf_angle_deg=35,
        leaf_droop=0.4, leaf_fold=0.3, leaf_color=(0.20, 0.40, 0.14), roots=4, root_diameter_mm=3,
        root_length_cm=15, infl_origin=IO.BASE, inflorescences=1, flowers=2, peduncle_cm=14, rachis_cm=6,
        spike_radius_mm=4, spike_angle_deg=165, spike_flex=0.3, divergence_deg=150, pedicel_cm=5,
        flower_facing=0.6, maturation=0.0, bract_mm=25, spike_color=(0.55, 0.50, 0.30), resupination=1.0,
        sepal_length_cm=9, sepal_width_cm=4.5, sepal_widest=0.4, sepal_apex_deg=80, lateral_sepal_deg=10,
        sepal_cup=0.3, sepal_reflex_deg=110, petal_length_cm=7.5, petal_width_cm=2.2, petal_widest=0.35,
        petal_apex_deg=60, petal_angle_deg=40, petal_reflex_deg=120, petal_wave=0.4, perianth_forward_deg=-20,
        lip_length_cm=7, lip_width_cm=3.2, lip_angle_deg=5, lip_deflex_deg=40, lip_claw=0.45, lip_widest=0.45,
        lip_apex_deg=60, lip_sac=1.0, lip_sac_extent=0.42, lip_horns_mm=30, column_length_cm=5.5,
        column_width_mm=8, column_arch=0.4, sepal_color=(0.95, 0.80, 0.35), petal_color=(0.95, 0.80, 0.35),
        lip_color=(0.97, 0.88, 0.55), lip_throat_color=(0.55, 0.15, 0.10), callus_color=(0.95, 0.85, 0.45),
        column_color=(0.92, 0.88, 0.60), pattern_color=(0.45, 0.10, 0.08), spots=0.85, spot_size_mm=9,
        spot_stretch=1.6, lip_spots=0.6, sheen=0.6),
    "bulbophyllum_rothschildianum": orchid(
        "Bulbophyllum rothschildianum", "Rothschild's bulbophyllum",
        "Creeping epiphyte with well-spaced one-leaved pseudobulbs; an erect scape ends in an umbel of 4-6 "
        "maroon flowers spread like a fan: small fringed dorsal sepal and petals and long lateral sepals fused "
        "into a twisted tail.",
        habit=H.SYMPODIAL, mount=MO.BRANCH, branch_diameter_cm=8, growths=6, rhizome_cm=4, growth_angle_deg=15,
        leafless_growths=2, stem_radius_mm=2.5, pseudobulb_cm=3, pseudobulb_diameter_cm=2.8, bulb_widest=0.45,
        bulb_fullness=0.9, bulb_ridges=4, leaf_placement=LP.APEX, leaves=1, leaf_length_cm=10,
        leaf_width_cm=2.4, leaf_thickness_mm=2.0, leaf_widest=0.5, leaf_apex_deg=110, leaf_angle_deg=30,
        leaf_droop=0.1, leaf_fold=0.3, leaf_color=(0.25, 0.42, 0.16), roots=3, root_diameter_mm=1.5,
        root_length_cm=10, infl_origin=IO.BASE, inflorescences=1, flowers=5, peduncle_cm=21, rachis_cm=1,
        spike_radius_mm=1.5, spike_angle_deg=20, spike_flex=0.1, umbel=1.0, pedicel_cm=2.5, flower_facing=0.4,
        maturation=0.0, bract_mm=4, spike_color=(0.40, 0.25, 0.25), resupination=1.0, sepal_length_cm=1.5,
        sepal_width_cm=0.7, sepal_widest=0.4, sepal_apex_deg=40, sepal_fringe_mm=6, lateral_sepal_scale=9.0,
        synsepal=1.0, sepal_cup=0.2, sepal_reflex_deg=0, sepal_twist_deg=40, petal_length_cm=1.0,
        petal_width_cm=0.4, petal_widest=0.4, petal_apex_deg=40, petal_angle_deg=40,
        perianth_forward_deg=10, lip_length_cm=0.7, lip_width_cm=0.4, lip_angle_deg=40, lip_deflex_deg=40,
        lip_claw=0.3, lip_apex_deg=60, column_length_cm=0.3, column_width_mm=2.5,
        sepal_color=(0.55, 0.10, 0.18), petal_color=(0.55, 0.12, 0.20), lip_color=(0.85, 0.65, 0.20),
        lip_throat_color=(0.70, 0.40, 0.15), column_color=(0.70, 0.50, 0.40), pattern_color=(0.95, 0.80, 0.40),
        spots=0.35, spot_size_mm=1.0, veins=0.4, sheen=0.5),
}

ORCHID_RANGES = {
    "growths": (1, 12), "rhizome_cm": (0.0, 20.0), "growth_angle_deg": (0.0, 60.0), "leafless_growths": (0, 10),
    "stem_length_cm": (1.0, 600.0), "stem_radius_mm": (1.0, 20.0), "internode_cm": (0.3, 20.0),
    "pseudobulb_cm": (0.0, 80.0), "pseudobulb_diameter_cm": (0.3, 10.0), "bulb_widest": (0.1, 0.9),
    "bulb_fullness": (0.05, 1.5), "bulb_flatten": (0.0, 0.8), "bulb_ridges": (0, 12), "bulb_lean_deg": (0.0, 60.0),
    "sheath_cover": (0.0, 1.0), "support_height_m": (0.3, 4.0), "support_radius_cm": (1.0, 30.0),
    "leaves": (0, 30), "leaf_length_cm": (1.0, 120.0), "leaf_width_cm": (0.2, 20.0), "leaf_thickness_mm": (0.2, 8.0),
    "leaf_widest": (0.1, 0.9), "leaf_apex_deg": (10.0, 179.0), "leaf_angle_deg": (0.0, 110.0),
    "leaf_droop": (0.0, 1.0), "leaf_fold": (0.0, 1.0), "leaf_twist_deg": (0.0, 90.0),
    "leaf_size_gradient": (0.0, 0.9), "leaf_mottle": (0.0, 1.0), "roots": (0, 40), "root_diameter_mm": (0.5, 12.0),
    "root_length_cm": (1.0, 150.0), "root_wander": (0.0, 1.0), "aerial_share": (0.0, 1.0),
    "inflorescences": (0, 12), "flowers": (1, 60), "peduncle_cm": (0.0, 150.0), "rachis_cm": (0.0, 150.0),
    "spike_radius_mm": (0.5, 8.0), "spike_angle_deg": (0.0, 180.0), "spike_flex": (0.0, 1.0), "branches": (0, 30),
    "branch_ratio": (0.05, 1.0), "divergence_deg": (0.0, 180.0), "pedicel_cm": (0.2, 15.0),
    "flower_facing": (0.0, 1.0), "maturation": (0.0, 1.0), "bract_mm": (0.0, 40.0), "fruit_set": (0.0, 1.0),
    "capsule_cm": (0.5, 30.0), "capsule_diameter_cm": (0.2, 5.0), "resupination": (0.0, 1.0),
    "flower_tilt_deg": (-45.0, 60.0), "sepal_length_cm": (0.2, 50.0), "sepal_width_cm": (0.05, 10.0),
    "sepal_widest": (0.1, 0.9), "sepal_apex_deg": (10.0, 179.0), "lateral_sepal_deg": (-90.0, 90.0),
    "lateral_sepal_scale": (0.3, 15.0), "synsepal": (0.0, 1.0), "sepal_cup": (-0.5, 1.0),
    "sepal_reflex_deg": (-90.0, 120.0), "sepal_twist_deg": (0.0, 720.0), "sepal_wave": (0.0, 1.0),
    "petal_length_cm": (0.2, 90.0), "petal_width_cm": (0.05, 10.0), "petal_widest": (0.1, 0.9),
    "petal_apex_deg": (10.0, 179.0), "petal_claw": (0.0, 0.6), "petal_angle_deg": (-80.0, 80.0),
    "petal_cup": (-0.5, 1.0), "petal_reflex_deg": (-90.0, 120.0), "petal_twist_deg": (0.0, 720.0),
    "petal_wave": (0.0, 1.0), "perianth_forward_deg": (-40.0, 80.0), "lip_length_cm": (0.2, 20.0),
    "lip_width_cm": (0.1, 15.0), "lip_angle_deg": (-30.0, 100.0), "lip_deflex_deg": (-60.0, 150.0),
    "lip_claw": (0.02, 0.8), "side_lobes": (0.0, 1.5), "side_lobe_pos": (0.05, 0.8), "side_lobe_length": (0.1, 0.9),
    "side_lobe_erect_deg": (0.0, 180.0), "midlobe_width": (0.1, 1.5), "isthmus": (0.0, 0.95),
    "lip_widest": (0.1, 0.9), "lip_apex_deg": (10.0, 179.0), "lip_roll": (0.0, 1.2), "lip_roll_extent": (0.05, 1.0),
    "lip_sac": (0.0, 1.0), "lip_wave": (0.0, 1.5), "lip_waves": (1.0, 12.0), "callus_mm": (0.0, 10.0),
    "callus_ridges": (0, 9), "callus_pos": (0.0, 0.9), "cirrhi_mm": (0.0, 40.0), "column_length_cm": (0.1, 6.0),
    "column_width_mm": (1.0, 15.0), "column_arch": (-0.5, 1.0), "staminode_mm": (0.0, 25.0),
    "mentum_mm": (0.0, 30.0), "spots": (0.0, 1.0), "spot_size_mm": (0.3, 10.0), "spot_stretch": (1.0, 6.0),
    "veins": (0.0, 1.0), "tip_amount": (0.0, 1.0),
    "lip_spots": (0.0, 1.0), "lip_veins": (0.0, 1.0), "sheen": (0.0, 1.0),
    "pot_diameter_cm": (5.0, 60.0), "branch_diameter_cm": (2.0, 80.0), "lip_sac_extent": (0.1, 1.0),
    "lip_horns_mm": (0.0, 60.0), "lip_flare": (0.0, 1.5), "spur_cm": (0.0, 45.0), "spur_diameter_mm": (0.5, 15.0),
    "spur_curve": (0.0, 1.0), "umbel": (0.0, 1.0), "sepal_fringe_mm": (0.0, 15.0),
}


def orchid_items() -> list[tuple[str, str, str]]:
    return [(k, f"{v.scientific_name} ({v.common_name})", f"{v.profile.habit.value}. {v.notes}")
            for k, v in sorted(ORCHID_CATALOG.items(), key=lambda kv: kv[1].scientific_name)]
