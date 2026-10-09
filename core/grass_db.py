"""
Grass presets: cereals, forage, lawn and ornamental grasses.

Collars after extension identification keys (Oklahoma State, Victoria Agriculture, Oregon State); rice
ligule 10-25 mm. Sugarcane node relief after Artschwager & Brandes (1958).

Root lengths: the fibrous systems of grasses reach about 1 m (Poaceae mean maximum rooting depth 1.02 m,
Schenk & Jackson 2002, J. Ecol. 90: 480); cereals 1-1.5 m, sorghum deeper, paddy rice and mown lawns
shallow.

Typical adult values from crop and grass floras (culm height, leaf length and width, number of elongated
internodes, inflorescence type and size). Maize phytomer geometry after Fournier & Andrieu (1998) and Wen et
al. (2021); wheat tillering after Evers et al. (2005).
"""

from dataclasses import dataclass, fields

from .grass import GrassProfile, GrassHead, Ligule, Auricles


@dataclass
class GrassPreset:
    scientific_name: str
    common_name: str
    family: str
    notes: str
    profile: GrassProfile


H = GrassHead
L, AU = Ligule, Auricles


def _typed(kw: dict) -> dict:
    floats = {f.name for f in fields(GrassProfile) if isinstance(getattr(GrassProfile(), f.name), float)}
    return {k: float(v) if k in floats and isinstance(v, int) and not isinstance(v, bool) else v for k, v in kw.items()}


def grass(sci, common, notes, **kw) -> GrassPreset:
    return GrassPreset(sci, common, "Poaceae", notes, GrassProfile(**_typed(kw)))


GREEN = (0.20, 0.42, 0.12)

GRASS_CATALOG: dict[str, GrassPreset] = {
    "zea_mays": grass(
        "Zea mays", "Maize (maíz)",
        "Single tall culm with ~14 wide, wavy-margined leaves; terminal tassel; ear in a leaf axil at mid-height "
        "with husk leaves and silks; brace roots from the lowest nodes.",
        tillers=1, culm_height_m=2.0, culm_radius_mm=12, nodes=13, internode_gradient=1.2, leaf_length_cm=85,
        leaf_width_cm=8, leaf_peak=0.55, leaf_angle_deg=30, leaf_droop=0.6, leaf_twist_deg=40, leaf_fold=0.25,
        margin_wave=0.6, sheath_fraction=0.95, leaf_color=(0.18, 0.40, 0.10), midrib_color=(0.70, 0.78, 0.55),
        tip_dryness=0.1, head=H.MAIZE, head_length_cm=30, peduncle_cm=15, spikelets=140, spikelet_mm=8,
        branches=12, branch_angle_deg=35, nod=0.1, head_color=(0.70, 0.65, 0.40), ears=1, ear_node=0.55,
        ear_length_cm=20, ear_diameter_cm=5, kernel_rows=16, husk=0.0, silk_cm=12, crown_roots=16,
        root_length_cm=120, brace_roots=10, culm_color=(0.40, 0.55, 0.22), node_swell=0.05, bud_groove=0.1,
        bud_size_mm=3, zigzag_deg=3, node_color=(0.55, 0.62, 0.35),
        ligule=L.MEMBRANE, ligule_mm=3, auricles=AU.NONE),
    "triticum_aestivum": grass(
        "Triticum aestivum", "Bread wheat (trigo)",
        "Tillering cereal; flat leaves; dense distichous spike, short awns or none.",
        tillers=5, tiller_spread_deg=12, clump_radius_cm=2, culm_height_m=0.85, culm_radius_mm=2.2, nodes=5,
        internode_gradient=1.6, leaf_length_cm=25, leaf_width_cm=1.4, leaf_peak=0.4, leaf_angle_deg=28,
        leaf_droop=0.45, leaf_twist_deg=90, head=H.SPIKE, head_length_cm=9, head_width_cm=1.5, peduncle_cm=14,
        spikelets=20, spikelet_mm=11, awn_cm=1.5, nod=0.05, head_color=(0.45, 0.58, 0.25),
        awn_color=(0.70, 0.68, 0.45), crown_roots=14, root_length_cm=100, node_swell=0.18, bud_size_mm=0,
        zigzag_deg=1.5, node_color=(0.38, 0.46, 0.18),
        ligule=L.MEMBRANE, ligule_mm=1.5, auricles=AU.BLUNT, auricle_mm=1.5),
    "hordeum_vulgare": grass(
        "Hordeum vulgare", "Barley (cebada)",
        "Spike with very long awns that nods when ripe.",
        tillers=5, tiller_spread_deg=14, culm_height_m=0.8, culm_radius_mm=2.0, nodes=5, leaf_length_cm=22,
        leaf_width_cm=1.3, leaf_peak=0.4, leaf_droop=0.5, head=H.SPIKE, head_length_cm=7, head_width_cm=1.2,
        peduncle_cm=12, spikelets=24, spikelet_mm=9, awn_cm=12, nod=0.3, head_color=(0.50, 0.62, 0.28),
        awn_color=(0.72, 0.70, 0.45), crown_roots=14, root_length_cm=100, node_swell=0.18, bud_size_mm=0,
        zigzag_deg=1.5, node_color=(0.38, 0.46, 0.18),
        ligule=L.MEMBRANE, ligule_mm=1.5, auricles=AU.CLASPING, auricle_mm=5),
    "avena_sativa": grass(
        "Avena sativa", "Oat (avena)",
        "Open panicle of large pendulous spikelets in whorls.",
        tillers=4, culm_height_m=1.0, culm_radius_mm=2.5, nodes=5, leaf_length_cm=28, leaf_width_cm=1.5,
        leaf_droop=0.5, head=H.PANICLE, head_length_cm=25, head_width_cm=16, peduncle_cm=10, spikelets=45,
        spikelet_mm=22, branches=6, branch_angle_deg=45, nod=0.45, head_color=(0.55, 0.65, 0.35),
        crown_roots=14, root_length_cm=90, node_swell=0.18, bud_size_mm=0, zigzag_deg=1.5,
        node_color=(0.38, 0.46, 0.18),
        ligule=L.MEMBRANE, ligule_mm=4, auricles=AU.NONE),
    "oryza_sativa": grass(
        "Oryza sativa", "Rice (arroz)",
        "Many tillers; drooping panicle heavy with grains.",
        tillers=10, tiller_spread_deg=18, clump_radius_cm=3, culm_height_m=0.85, culm_radius_mm=2.5, nodes=5,
        leaf_length_cm=40, leaf_width_cm=1.2, leaf_angle_deg=20, leaf_droop=0.45, head=H.PANICLE,
        head_length_cm=22, head_width_cm=7, peduncle_cm=8, spikelets=100, spikelet_mm=8, branches=8,
        branch_angle_deg=25, nod=0.7, head_color=(0.62, 0.62, 0.30), crown_roots=20, root_length_cm=40,
        node_swell=0.15, bud_size_mm=0, zigzag_deg=1.5, node_color=(0.42, 0.50, 0.22),
        ligule=L.MEMBRANE, ligule_mm=15, auricles=AU.CLAW, auricle_mm=3),
    "sorghum_bicolor": grass(
        "Sorghum bicolor", "Sorghum (sorgo)",
        "Maize-like culm and leaves; compact terminal panicle of reddish grains.",
        tillers=1, culm_height_m=1.8, culm_radius_mm=11, nodes=10, internode_gradient=1.2, leaf_length_cm=70,
        leaf_width_cm=7, leaf_droop=0.6, leaf_twist_deg=40, margin_wave=0.3, sheath_fraction=0.8, head=H.PANICLE,
        head_length_cm=25, head_width_cm=10, peduncle_cm=20, spikelets=170, spikelet_mm=5, branches=10,
        branch_angle_deg=22, nod=0.05, head_color=(0.58, 0.30, 0.15), crown_roots=16, root_length_cm=150,
        brace_roots=4, node_swell=0.06, bud_groove=0.08, bud_size_mm=3, zigzag_deg=3, wax_band=0.6,
        node_color=(0.60, 0.55, 0.40),
        ligule=L.MEMBRANE, ligule_mm=2, auricles=AU.NONE),
    "saccharum_officinarum": grass(
        "Saccharum officinarum", "Sugarcane (caña de azúcar)",
        "Thick jointed culms with prominent nodes; long arching leaves; flowering rare in cultivation.",
        tillers=5, tiller_spread_deg=10, clump_radius_cm=8, culm_height_m=3.0, culm_radius_mm=15, nodes=18,
        internode_gradient=0.6, leaf_length_cm=120, leaf_width_cm=5, leaf_peak=0.85, leaf_angle_deg=35,
        leaf_droop=0.75, sheath_fraction=0.95, leaf_loss=0.5, head=H.NONE, culm_color=(0.62, 0.62, 0.30), crown_roots=20,
        root_length_cm=120, node_swell=0.05, growth_ring=1.0, internode_barrel=0.06, bud_groove=0.06,
        bud_size_mm=7, zigzag_deg=4, wax_band=0.8, node_color=(0.72, 0.70, 0.52),
        ligule=L.MEMBRANE, ligule_mm=2, auricles=AU.CLAW, auricle_mm=3, root_primordia=1.0),
    "lolium_perenne": grass(
        "Lolium perenne", "Perennial ryegrass, lawn (pasto inglés)",
        "Lawn tuft: many tillers with basal leaves, kept mown (no flowering culms).",
        tillers=25, tiller_spread_deg=30, tiller_variation=0.4, flowering_tillers=0.0, clump_radius_cm=3,
        culm_height_m=0.02,
        culm_radius_mm=0.8, nodes=1, basal_leaves=3, leaf_length_cm=14, leaf_width_cm=0.4, leaf_peak=0.5,
        leaf_angle_deg=12, leaf_droop=0.35, leaf_twist_deg=90, leaf_fold=0.5, sheath_fraction=0.9,
        leaf_color=(0.16, 0.42, 0.12), tip_dryness=0.05, head=H.NONE, crown_roots=10, root_length_cm=30,
        node_swell=0.0, bud_size_mm=0, zigzag_deg=0,
        ligule=L.MEMBRANE, ligule_mm=1, auricles=AU.CLAW, auricle_mm=1),
    "bouteloua_gracilis": grass(
        "Bouteloua gracilis", "Blue grama (navajita)",
        "Short tufted grass of the Mexican grasslands; fine curly leaves; one or two comb-like spikes held "
        "at an angle like flags or eyelashes.",
        tillers=20, tiller_spread_deg=25, tiller_variation=0.3, flowering_tillers=0.4, clump_radius_cm=4,
        culm_height_m=0.35,
        culm_radius_mm=0.8, nodes=2, basal_leaves=4, leaf_length_cm=12, leaf_width_cm=0.18, leaf_angle_deg=20,
        leaf_droop=0.55, leaf_twist_deg=220, leaf_color=(0.36, 0.48, 0.32), tip_dryness=0.3, head=H.ONE_SIDED,
        head_length_cm=3.5, peduncle_cm=10, spikelets=40, spikelet_mm=5, awn_cm=0.3, branches=2,
        branch_angle_deg=55, nod=0.2, head_color=(0.45, 0.30, 0.35), crown_roots=10, root_length_cm=70, node_swell=0.1, bud_size_mm=0,
        ligule=L.HAIRS, ligule_mm=0.5, auricles=AU.NONE),
    "pennisetum_setaceum": grass(
        "Cenchrus setaceus (Pennisetum setaceum)", "Fountain grass (pasto de la fuente)",
        "Arching fountain of narrow leaves; bristly pink-purple plumes nodding at the tips.",
        tillers=30, tiller_spread_deg=35, flowering_tillers=0.35, clump_radius_cm=8, culm_height_m=0.9,
        culm_radius_mm=1.5, nodes=2,
        basal_leaves=4, leaf_length_cm=60, leaf_width_cm=0.4, leaf_angle_deg=20, leaf_droop=0.8,
        leaf_twist_deg=120, head=H.PLUME, head_length_cm=25, head_width_cm=5, peduncle_cm=20, spikelets=300,
        spikelet_mm=8, branches=24, branch_angle_deg=35, nod=0.5, head_color=(0.68, 0.48, 0.52),
        awn_color=(0.75, 0.55, 0.60), crown_roots=14, root_length_cm=60, node_swell=0.1, bud_size_mm=0,
        ligule=L.HAIRS, ligule_mm=1, auricles=AU.NONE),
    "cortaderia_selloana": grass(
        "Cortaderia selloana", "Pampas grass (hierba de las pampas)",
        "Huge tussock of long arching leaves with tall culms carrying silky cream plumes.",
        tillers=40, tiller_spread_deg=20, flowering_tillers=0.2, clump_radius_cm=25, culm_height_m=2.3,
        culm_radius_mm=6, nodes=3,
        basal_leaves=5, leaf_length_cm=150, leaf_width_cm=1.0, leaf_angle_deg=25, leaf_droop=0.8,
        leaf_twist_deg=100, leaf_color=(0.30, 0.44, 0.25), tip_dryness=0.3, head=H.PLUME, head_length_cm=50,
        head_width_cm=14, peduncle_cm=30, spikelets=500, spikelet_mm=10, branches=30, branch_angle_deg=18,
        nod=0.12, head_color=(0.92, 0.88, 0.80), awn_color=(0.95, 0.92, 0.85), crown_roots=24, root_length_cm=120, node_swell=0.1, bud_size_mm=0,
        ligule=L.HAIRS, ligule_mm=3, auricles=AU.NONE),
}

GRASS_RANGES = {
    "tillers": (1, 200), "tiller_spread_deg": (0.0, 80.0), "tiller_variation": (0.0, 0.9),
    "node_swell": (0.0, 0.5), "growth_ring": (0.0, 1.0), "internode_barrel": (0.0, 0.3), "bud_groove": (0.0, 0.3),
    "bud_size_mm": (0.0, 15.0), "zigzag_deg": (0.0, 15.0), "wax_band": (0.0, 1.0),
    "flowering_tillers": (0.0, 1.0),
    "clump_radius_cm": (0.0, 100.0), "culm_height_m": (0.0, 6.0), "culm_radius_mm": (0.3, 40.0), "nodes": (1, 30),
    "internode_gradient": (0.2, 4.0), "basal_leaves": (0, 12), "leaf_length_cm": (1.0, 250.0),
    "leaf_width_cm": (0.05, 15.0), "leaf_peak": (0.0, 1.0), "leaf_angle_deg": (2.0, 85.0), "leaf_droop": (0.0, 1.0),
    "leaf_twist_deg": (0.0, 400.0), "leaf_fold": (0.0, 1.0), "margin_wave": (0.0, 1.0),
    "sheath_fraction": (0.1, 1.0), "ligule_mm": (0.0, 30.0), "auricle_mm": (0.0, 15.0),
    "root_primordia": (0.0, 1.0), "leaf_loss": (0.0, 0.9), "tip_dryness": (0.0, 1.0), "head_length_cm": (0.5, 80.0),
    "head_width_cm": (0.2, 40.0), "peduncle_cm": (0.0, 80.0), "spikelets": (1, 800), "spikelet_mm": (1.0, 40.0),
    "awn_cm": (0.0, 25.0), "branches": (0, 60), "branch_angle_deg": (5.0, 85.0), "nod": (0.0, 1.0),
    "ears": (0, 3), "ear_node": (0.1, 0.9), "ear_length_cm": (3.0, 40.0), "ear_diameter_cm": (1.0, 10.0),
    "kernel_rows": (4, 30), "husk": (0.0, 1.0), "silk_cm": (0.0, 30.0), "crown_roots": (0, 60),
    "root_length_cm": (1.0, 200.0), "brace_roots": (0, 30), "fine_roots": (0.0, 3.0), "ripeness": (0.0, 1.0),
}


def grass_items() -> list[tuple[str, str, str]]:
    return [(k, f"{v.scientific_name} ({v.common_name})", f"{v.profile.head.value}. {v.notes}")
            for k, v in sorted(GRASS_CATALOG.items(), key=lambda kv: kv[1].scientific_name)]
