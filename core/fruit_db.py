"""
Fruit presets and the default fruits of the tree and vine presets.

Sizes are typical ranges of common cultivars or wild forms (stalk-to-blossom length and largest width, cm);
colours are sRGB. Shape descriptors after Tomato Analyzer (Brewer et al. 2006); fruit types after
Spjut (1994, A systematic treatment of fruit types, Mem. N. Y. Bot. Gard. 70).
"""

from dataclasses import dataclass

from .fruit import FruitProfile, FruitKind, CrownPosition


@dataclass
class FruitPreset:
    scientific_name: str
    common_name: str
    family: str
    notes: str
    fruit: FruitProfile


K, CP = FruitKind, CrownPosition


def fr(sci, common, family, notes, **kw) -> FruitPreset:
    return FruitPreset(sci, common, family, notes, FruitProfile(**kw))


GREEN_STALK = (0.35, 0.42, 0.18)

FRUIT_CATALOG: dict[str, FruitPreset] = {
    # --- Pomes ----------------------------------------------------------------------------------------
    "malus_domestica": fr(
        "Malus domestica", "Apple (manzana)", "Rosaceae",
        "Pome: deep stalk cavity, calyx basin with dried sepals, five faint lobes; red blush and streaks over "
        "yellow-green, pale lenticels.",
        kind=K.POME, length_cm=7.6, diameter_cm=7.8, widest_position=0.42, bluntness=0.5, stalk_cavity=0.24,
        calyx_basin=0.09, ribs=5, rib_depth=0.03, lopsided=0.06, crown_lobes=5, crown_length_cm=0.45,
        crown_flare_deg=20, stalk_length_cm=2.5, stalk_radius_mm=1.3, color=(0.78, 0.72, 0.22),
        blush_color=(0.66, 0.07, 0.06), blush=0.75, stripes=0.5, stripe_count=34, stripe_width=0.18, stripe_color=(0.52, 0.04, 0.05), dots=0.35,
        dot_color=(0.90, 0.85, 0.60), gloss=0.6),
    "pyrus_communis": fr(
        "Pyrus communis", "Pear (pera)", "Rosaceae",
        "Pyriform pome: widest near the blossom end, tapering neck to a shallow stalk end; russeted skin with "
        "brown lenticels.",
        kind=K.POME, length_cm=10.0, diameter_cm=6.5, widest_position=0.72, bluntness=0.6, neck=0.45,
        stalk_cavity=0.02, calyx_basin=0.04, lopsided=0.05, crown_lobes=5, crown_length_cm=0.35,
        crown_flare_deg=25, stalk_length_cm=3.5, stalk_radius_mm=1.6, color=(0.74, 0.68, 0.26),
        blush_color=(0.70, 0.35, 0.12), blush=0.2, dots=0.6, dot_color=(0.45, 0.32, 0.18), russet=0.35, gloss=0.3),
    "cydonia_oblonga": fr(
        "Cydonia oblonga", "Quince (membrillo)", "Rosaceae",
        "Large lumpy golden pome with a prominent calyx crown; downy when young.",
        kind=K.POME, length_cm=10.0, diameter_cm=9.0, widest_position=0.55, bluntness=0.5, neck=0.2,
        stalk_cavity=0.05, calyx_basin=0.06, ribs=5, rib_depth=0.06, lopsided=0.1, crown_lobes=5,
        crown_length_cm=1.0, crown_flare_deg=40, stalk_length_cm=0.6, stalk_radius_mm=2.5,
        color=(0.88, 0.74, 0.18), dots=0.15, gloss=0.25),
    # --- Drupes ---------------------------------------------------------------------------------------
    "prunus_avium": fr(
        "Prunus avium", "Sweet cherry (cereza)", "Rosaceae",
        "Glossy dark-red drupes on long pedicels, in pairs or threes from the spurs.",
        kind=K.DRUPE, length_cm=2.2, diameter_cm=2.4, widest_position=0.45, bluntness=0.5, stalk_cavity=0.12,
        cluster_berries=3, cluster_length_cm=1.0, cluster_width_cm=4.0, compactness=0.1, pedicel_cm=4.0,
        peduncle_cm=0.5, stalk_radius_mm=0.7, stalk_color=GREEN_STALK, color=(0.55, 0.03, 0.06),
        blush_color=(0.32, 0.02, 0.04), blush=0.5, gloss=0.85),
    "prunus_persica": fr(
        "Prunus persica", "Peach (durazno)", "Rosaceae",
        "Velvety drupe with a suture groove, red blush over yellow-orange.",
        kind=K.DRUPE, length_cm=7.0, diameter_cm=7.0, widest_position=0.5, bluntness=0.5, distal_point=0.1,
        stalk_cavity=0.08, ribs=1, rib_depth=0.06, lopsided=0.05, stalk_length_cm=0.5, stalk_radius_mm=2.0,
        color=(0.95, 0.65, 0.25), blush_color=(0.80, 0.20, 0.12), blush=0.7, bloom=0.4, gloss=0.1),
    "olea_europaea": fr(
        "Olea europaea", "Olive (aceituna)", "Oleaceae",
        "Small ellipsoid drupe ripening from green to purple-black.",
        kind=K.DRUPE, length_cm=2.2, diameter_cm=1.6, widest_position=0.45, bluntness=0.55, distal_point=0.15,
        stalk_length_cm=1.0, stalk_radius_mm=0.6, stalk_color=GREEN_STALK, color=(0.14, 0.07, 0.12),
        blush_color=(0.30, 0.32, 0.12), blush=0.3, bloom=0.15, gloss=0.7),
    # --- Berries / balausta / hesperidium --------------------------------------------------------------
    "vitis_vinifera": fr(
        "Vitis vinifera", "Grape bunch (racimo de uvas)", "Vitaceae",
        "Conical panicle with shoulders; round berries with a waxy bloom.",
        kind=K.BERRY, length_cm=1.7, diameter_cm=1.6, widest_position=0.5, bluntness=0.5, cluster_berries=110,
        cluster_length_cm=18.0, cluster_width_cm=11.0, shoulders=0.4, compactness=0.65, pedicel_cm=0.5,
        peduncle_cm=4.0, stalk_radius_mm=1.6, stalk_color=GREEN_STALK, color=(0.20, 0.05, 0.17),
        blush_color=(0.30, 0.08, 0.25), blush=0.2, bloom=0.6, gloss=0.3),
    "vitis_vinifera_white": fr(
        "Vitis vinifera (white)", "White grape bunch (uva blanca)", "Vitaceae",
        "Green-gold berries, slightly ellipsoid, in a looser winged bunch.",
        kind=K.BERRY, length_cm=1.9, diameter_cm=1.6, widest_position=0.5, bluntness=0.55, cluster_berries=90,
        cluster_length_cm=17.0, cluster_width_cm=10.0, shoulders=0.6, compactness=0.5, pedicel_cm=0.6,
        peduncle_cm=4.0, stalk_radius_mm=1.6, stalk_color=GREEN_STALK, color=(0.62, 0.70, 0.28),
        blush_color=(0.80, 0.70, 0.25), blush=0.3, bloom=0.4, gloss=0.4),
    "solanum_lycopersicum": fr(
        "Solanum lycopersicum", "Tomato (jitomate)", "Solanaceae",
        "Lobed berry with a green calyx star on the stalk end.",
        kind=K.BERRY, length_cm=5.0, diameter_cm=6.0, widest_position=0.5, bluntness=0.45, stalk_cavity=0.08,
        ribs=6, rib_depth=0.06, lopsided=0.05, crown_lobes=5, crown_length_cm=1.4, crown_flare_deg=40,
        crown_position=CP.PROXIMAL, crown_color=(0.25, 0.45, 0.15), stalk_length_cm=1.2, stalk_radius_mm=1.2,
        stalk_color=GREEN_STALK, color=(0.85, 0.12, 0.06), gloss=0.8),
    "punica_granatum": fr(
        "Punica granatum", "Pomegranate (granada)", "Lythraceae",
        "Balausta: leathery, faintly angular rind crowned by the persistent calyx lobes.",
        kind=K.BALAUSTA, length_cm=8.0, diameter_cm=9.0, widest_position=0.5, bluntness=0.5, stalk_cavity=0.04,
        ribs=6, rib_depth=0.02, lopsided=0.05, crown_lobes=6, crown_length_cm=1.8, crown_flare_deg=25,
        crown_color=(0.60, 0.20, 0.14), stalk_length_cm=0.8, stalk_radius_mm=3.0, color=(0.75, 0.18, 0.12),
        blush_color=(0.52, 0.08, 0.07), blush=0.5, dots=0.1, dot_color=(0.85, 0.65, 0.40), gloss=0.5),
    "citrus_sinensis": fr(
        "Citrus × sinensis", "Orange (naranja)", "Rutaceae",
        "Hesperidium: rind dimpled by oil glands.",
        kind=K.HESPERIDIUM, length_cm=7.5, diameter_cm=7.8, widest_position=0.5, bluntness=0.5, stalk_cavity=0.03,
        calyx_basin=0.02, stalk_length_cm=1.5, stalk_radius_mm=1.6, stalk_color=GREEN_STALK,
        color=(0.95, 0.50, 0.05), dots=0.8, dot_color=(0.98, 0.62, 0.15), gloss=0.5),
    "citrus_limon": fr(
        "Citrus × limon", "Lemon (limón amarillo)", "Rutaceae",
        "Ellipsoid hesperidium with a nipple at the blossom end.",
        kind=K.HESPERIDIUM, length_cm=8.5, diameter_cm=6.0, widest_position=0.5, bluntness=0.5, distal_point=0.5,
        stalk_length_cm=1.2, stalk_radius_mm=1.4, stalk_color=GREEN_STALK, color=(0.97, 0.85, 0.15),
        dots=0.8, dot_color=(0.99, 0.92, 0.35), gloss=0.5),
    # --- Pepos and pods (vines) -------------------------------------------------------------------------
    "cucurbita_pepo": fr(
        "Cucurbita pepo", "Pumpkin (calabaza)", "Cucurbitaceae",
        "Large oblate ribbed pepo with sunken ends and a thick corky stalk.",
        kind=K.PEPO, length_cm=22.0, diameter_cm=32.0, widest_position=0.5, bluntness=0.45, stalk_cavity=0.18,
        calyx_basin=0.12, ribs=10, rib_depth=0.12, lopsided=0.05, stalk_length_cm=6.0, stalk_radius_mm=12.0,
        stalk_color=(0.45, 0.48, 0.25), color=(0.88, 0.45, 0.06), stripes=0.15, stripe_count=10, stripe_width=0.3,
        stripe_color=(0.95, 0.62, 0.20), gloss=0.3),
    "citrullus_lanatus": fr(
        "Citrullus lanatus", "Watermelon (sandía)", "Cucurbitaceae",
        "Large pepo with irregular dark stripes over pale green.",
        kind=K.PEPO, length_cm=35.0, diameter_cm=28.0, widest_position=0.5, bluntness=0.5, stalk_cavity=0.02,
        stalk_length_cm=4.0, stalk_radius_mm=4.0, stalk_color=GREEN_STALK, color=(0.50, 0.62, 0.30),
        stripes=0.85, stripe_count=16, stripe_width=0.45, stripe_color=(0.12, 0.28, 0.10), gloss=0.6),
    "cucumis_sativus": fr(
        "Cucumis sativus", "Cucumber (pepino)", "Cucurbitaceae",
        "Elongated dark-green pepo with pale stripes toward the blossom end.",
        kind=K.PEPO, length_cm=22.0, diameter_cm=4.0, widest_position=0.55, bluntness=0.4,
        stalk_length_cm=3.0, stalk_radius_mm=2.0, stalk_color=GREEN_STALK, color=(0.10, 0.25, 0.08),
        stripes=0.3, stripe_count=10, stripe_width=0.3, stripe_color=(0.45, 0.55, 0.25), dots=0.2, dot_color=(0.75, 0.80, 0.55), gloss=0.5),
    "passiflora_caerulea": fr(
        "Passiflora caerulea", "Passion fruit (pasionaria)", "Passifloraceae",
        "Ovoid orange berry on a long stalk.",
        kind=K.BERRY, length_cm=6.0, diameter_cm=4.0, widest_position=0.5, bluntness=0.55,
        stalk_length_cm=4.0, stalk_radius_mm=1.5, stalk_color=GREEN_STALK, color=(0.95, 0.55, 0.05), gloss=0.6),
    "phaseolus_coccineus": fr(
        "Phaseolus coccineus", "Runner bean pod (vaina de ayocote)", "Fabaceae",
        "Long, slightly curved, rough green pod.",
        kind=K.POD, length_cm=25.0, diameter_cm=1.8, widest_position=0.6, bluntness=0.35, distal_point=0.4,
        stalk_length_cm=4.0, stalk_radius_mm=1.5, stalk_color=GREEN_STALK, color=(0.30, 0.45, 0.15), gloss=0.3),
    "pisum_sativum": fr(
        "Pisum sativum", "Pea pod (vaina de chícharo)", "Fabaceae",
        "Smooth green pod.",
        kind=K.POD, length_cm=8.0, diameter_cm=1.2, widest_position=0.6, bluntness=0.4, distal_point=0.3,
        stalk_length_cm=2.0, stalk_radius_mm=1.0, stalk_color=GREEN_STALK, color=(0.32, 0.50, 0.20), gloss=0.4),
    "wisteria_sinensis": fr(
        "Wisteria sinensis", "Wisteria pod (vaina de glicinia)", "Fabaceae",
        "Velvety grey-green pod.",
        kind=K.POD, length_cm=12.0, diameter_cm=2.0, widest_position=0.65, bluntness=0.4, distal_point=0.3,
        stalk_length_cm=3.0, stalk_radius_mm=1.5, color=(0.55, 0.58, 0.45), bloom=0.3, gloss=0.1),
}

# Default fruits of the plant presets (missing = none shown by default)
TREE_FRUITS = {
    "malus_domestica": "malus_domestica", "prunus_avium": "prunus_avium", "olea_europaea": "olea_europaea",
    "pyrus_communis": "pyrus_communis", "punica_granatum": "punica_granatum", "citrus_sinensis": "citrus_sinensis",
}
VINE_FRUITS = {
    "vitis_vinifera": "vitis_vinifera", "cucurbita_pepo": "cucurbita_pepo", "citrullus_lanatus": "citrullus_lanatus",
    "cucumis_sativus": "cucumis_sativus", "passiflora_caerulea": "passiflora_caerulea",
    "phaseolus_coccineus": "phaseolus_coccineus", "pisum_sativum": "pisum_sativum",
    "wisteria_sinensis": "wisteria_sinensis",
}

FRUIT_RANGES = {
    "length_cm": (0.3, 120.0), "diameter_cm": (0.3, 100.0), "widest_position": (0.15, 0.85),
    "bluntness": (0.2, 1.5), "distal_point": (0.0, 1.0), "neck": (0.0, 1.0), "stalk_cavity": (0.0, 0.4),
    "calyx_basin": (0.0, 0.3), "ribs": (0, 30), "rib_depth": (0.0, 0.4), "lopsided": (0.0, 0.3),
    "crown_lobes": (0, 12), "crown_length_cm": (0.0, 5.0), "crown_flare_deg": (0.0, 90.0),
    "stalk_length_cm": (0.0, 40.0), "stalk_radius_mm": (0.2, 30.0), "cluster_berries": (1, 400),
    "cluster_length_cm": (0.5, 60.0), "cluster_width_cm": (0.5, 40.0), "shoulders": (0.0, 1.0),
    "compactness": (0.0, 1.0), "pedicel_cm": (0.0, 10.0), "peduncle_cm": (0.0, 30.0), "blush": (0.0, 1.0),
    "stripes": (0.0, 1.0), "stripe_count": (1, 80), "stripe_width": (0.05, 0.95), "dots": (0.0, 1.0), "russet": (0.0, 1.0), "bloom": (0.0, 1.0), "gloss": (0.0, 1.0),
}


def fruit_items() -> list[tuple[str, str, str]]:
    return [(k, f"{v.scientific_name} ({v.common_name})", f"{v.family} · {v.fruit.kind.value}. {v.notes}")
            for k, v in sorted(FRUIT_CATALOG.items(), key=lambda kv: kv[1].scientific_name)]
