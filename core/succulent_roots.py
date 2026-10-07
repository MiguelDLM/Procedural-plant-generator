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
"""

import numpy as np

from .roots import RootSystemEngine, RootProfile, RootSystemType
from .mesh_engine import BotanicalMeshEngine, MeshConfig, MeshData


def succulent_roots(system: RootSystemType, stem_radius: float, spread_m: float, depth_m: float,
                    count: int, core_ratio: float = 0.25, taproot_share: float = 0.0, taproot_depth_m: float = 0.5,
                    root_radius_mm: float = 2.0, tuber_length_m: float = 0.12, tuber_radius_m: float = 0.03,
                    display_depth_m: float = 3.0, seed: int = 0) -> MeshData:
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
    graph = RootSystemEngine(prof).generate(2 * core, spread_m, core, 2.3, seed=seed,
                                            display_depth_m=display_depth_m)
    eng = BotanicalMeshEngine(MeshConfig(radial_resolution=8, twig_resolution=4))
    return eng.build_wood_mesh(graph, 1.0, None, trunk_index=-1)
