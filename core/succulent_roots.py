"""
Root systems for succulents, built with the coarse-root engine (core.roots).

Most cacti are shallow-rooted: laterals rarely deeper than 15-30 cm but extending far from the
stem (saguaro: up to 9.7 m from a 6.8 m plant, with a stout taproot to ~0.77 m; Cannon 1911).
Opuntia concentrates roots in the top 100-150 mm and spreads 1.6-1.7 m in one season, with
ephemeral rain roots (Snyman 2005). Agaves and other rosette succulents have shallow fibrous
roots whose placement near the surface is critical for water uptake (Franco & Nobel 1990).
Some globose cacti (Lophophora, Ariocarpus) form a napiform storage taproot.

The root collar is the stem's vascular cylinder, not the whole succulent stem: water-storage
parenchyma does not carry the pipe-model flow, so `core_ratio` x stem radius feeds the roots.
The depth below which only 10% of roots lie sets the Jackson et al. (1996) coefficient:
beta = 0.1 ** (1 / depth_cm).

Fine laterals (core.root_architecture): cactus roots grow determinately (Dubrovsky 1997; Shishkova et al.
2013), and rain roots emerge from the laterals after wetting (Nobel 1988; Snyman 2005), so cactus
laterals are short, near-horizontal and close together ("rain roots"); rosettes carry longer fibrous
laterals. Schenk & Jackson (2002): stem succulents spread ~4.5 times as far as they root deep.
"""

import numpy as np

from .roots import RootSystemEngine, RootProfile, RootSystemType
from .root_architecture import RootBranching
from .mesh_engine import BotanicalMeshEngine, MeshConfig, MeshData


def succulent_roots(system: RootSystemType, stem_radius: float, spread_m: float, depth_m: float,
                    count: int, core_ratio: float = 0.25, taproot_share: float = 0.0, taproot_depth_m: float = 0.5,
                    root_radius_mm: float = 2.0, tuber_length_m: float = 0.12, tuber_radius_m: float = 0.03,
                    display_depth_m: float = 3.0, seed: int = 0, fine_roots: float = 1.0,
                    rain_roots: bool = True) -> MeshData:
    depth_cm = max(1.0, depth_m * 100.0)
    beta = float(0.1 ** (1.0 / depth_cm))
    core = max(0.004, core_ratio * stem_radius)
    prof = RootProfile(
        system=system, lateral_count=max(1, int(count)), spread_crown_ratio=1.0,
        max_depth_m=max(depth_m, taproot_depth_m), beta=beta, taproot_share=taproot_share,
        zrt_dbh_ratio=2.2, sinker_spacing_m=max(1.5, spread_m * 0.4), branch_spacing_m=max(0.15, spread_m * 0.08),
        surface_exposure=0.0, plank=0.0, buttress_height_dbh=0.0, tortuosity=0.35,
        fibrous_count=max(1, int(count)), fibrous_radius_m=root_radius_mm * 0.001, fibrous_spread_m=spread_m,
        tuber_length_m=tuber_length_m, tuber_radius_m=tuber_radius_m)
    root_eng = RootSystemEngine(prof)
    graph = root_eng.generate(2 * core, spread_m, core, 2.3, seed=seed, display_depth_m=display_depth_m)
    eng = BotanicalMeshEngine(MeshConfig(radial_resolution=8, twig_resolution=4))
    coarse = eng.build_wood_mesh(graph, 1.0, None, trunk_index=-1)
    if fine_roots <= 0:
        return coarse
    if rain_roots:      # Determinate, short, near-horizontal laterals, densely spaced
        br = RootBranching(orders=1, interbranch_cm=2.5 / fine_roots ** 0.5, basal_zone=0.15, apical_zone_cm=3.0,
                           insertion_deg=80.0, length_ratio=0.3, max_length_cm=12.0, radius_ratio=0.45,
                           gravitropism=0.03, tortuosity=0.4, determinate_cm=7.0, min_radius_mm=0.12,
                           budget=int(700 * fine_roots))
    else:               # Fibrous laterals of rosette succulents
        br = RootBranching(orders=2, interbranch_cm=4.0 / fine_roots ** 0.5, basal_zone=0.1, apical_zone_cm=3.0,
                           insertion_deg=70.0, length_ratio=0.35, max_length_cm=25.0, radius_ratio=0.45,
                           gravitropism=0.08, tortuosity=0.45, min_radius_mm=0.12, budget=int(700 * fine_roots))
    fine = root_eng.fine_roots(graph, seed=seed, display_depth_m=display_depth_m, branching=br)
    if not fine.axes:
        return coarse
    fine_mesh = BotanicalMeshEngine(MeshConfig(radial_resolution=4, twig_resolution=4)).build_wood_mesh(
        fine, 1.0, None, trunk_index=-1)
    return MeshData.concatenate([coarse, fine_mesh])
