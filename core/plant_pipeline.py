"""
Full procedural plant generation pipeline.

allometry (DBH -> height, crown radius, crown depth)
  -> architecture skeleton (with biomechanical bending during growth)
  -> foliage placement (phyllotaxis, light-facing laminae)
  -> leaf card geometry (outline/venation live in the leaf texture).
"""

from dataclasses import dataclass, replace
import numpy as np

from .allometry import AllometricEngine
from .architecture import ArchitectureEngine, BranchingGraph, HalleOldemanModel, PhyllotaxisType
from .leaf_morphology import LeafMorphologyEngine
from .biomechanics import BiomechanicalEngine
from .foliage import FoliageInstances, place_foliage
from .species_preset import BotanicalSpeciesPreset
from .roots import RootSystemEngine, RootSystemType


@dataclass
class BotanicalPlantResult:
    """Complete computed geometric and botanical data for a plant model."""
    preset: BotanicalSpeciesPreset
    dbh_m: float
    total_height_m: float
    crown_radius_m: float
    crown_depth_m: float
    base_radius_m: float
    skeleton_graph: BranchingGraph
    leaf_mesh_data: dict
    foliage: FoliageInstances
    leaf_engine: LeafMorphologyEngine
    root_graph: BranchingGraph | None = None
    flute_azimuth: float | None = None   # Azimuth of the first stem flute (aligned with a main root)

    @property
    def leaf_count(self) -> int:
        return len(self.foliage)


class BotanicalPlantPipeline:
    """Top-level generation controller."""

    def __init__(self, preset: BotanicalSpeciesPreset):
        self.preset = preset
        self.allometry = AllometricEngine(preset.allometry)
        self.biomechanics = BiomechanicalEngine(preset.biomechanics)
        self.architecture = ArchitectureEngine(preset.architecture,
                                               droop_fn=self.biomechanics.calculate_branch_tip_droop)
        self.leaf_morphology = LeafMorphologyEngine(preset.leaf_morphology)

    def supports_shoot_cards(self) -> bool:
        """Apical rosettes (palms) place whole fronds; every other habit can use shoot cards."""
        return self.preset.architecture.model != HalleOldemanModel.CORNER

    def dimensions(self, dbh_m: float) -> tuple[float, float, float]:
        """Allometric (height, crown radius, crown depth) for a stem diameter."""
        h = self.allometry.calculate_height(dbh_m)
        return h, self.allometry.calculate_crown_radius(dbh_m), self.allometry.calculate_crown_depth(h)

    def generate(
        self,
        dbh_m: float = None,
        leaf_density: float = 1.0,
        seed: int = 42,
        height_m: float = None,
        crown_radius_m: float = None,
        crown_depth_m: float = None,
        leaf_budget: int = 60000,
        card_grid: tuple[int, int] = (3, 6),
        shoot_leaves: int = 0,
        roots: bool = True,
        root_display_depth_m: float = 3.0,
    ) -> BotanicalPlantResult:
        """
        shoot_leaves > 0 draws each foliage card as a leafy shoot with that many
        leaves (dense crowns at a fraction of the polygon cost); 0 = one leaf per card.
        """
        al = self.preset.allometry
        if dbh_m is None:
            dbh_m = al.dbh_default_m
        dbh_m = float(max(0.01, dbh_m))

        h, cr, cd = self.dimensions(dbh_m)
        h = float(height_m) if height_m else h
        cr = float(crown_radius_m) if crown_radius_m else cr
        cd = float(np.clip(crown_depth_m if crown_depth_m else h * al.crown_depth_ratio, 0.2, h * 0.97))
        base_radius = dbh_m * 0.5

        skeleton = self.architecture.generate_skeleton(
            total_height_m=h, crown_radius_m=cr, crown_depth_m=cd, base_radius_m=base_radius,
            pipe_delta=al.pipe_exponent_delta, seed=seed,
            # With roots modelled, the main laterals build most of the root-collar flare themselves
            flare_amplitude=al.buttress_amplitude * (0.4 if roots else 1.0), flare_decay=al.buttress_decay)

        # Root system: main laterals sit under the stem flutes so buttresses continue into roots
        rng = np.random.default_rng(seed + 31)
        flute_az = float(rng.uniform(0.0, 2.0 * np.pi))
        root_graph = None
        rp = self.preset.roots
        if roots:
            trunk = skeleton.axes[0]
            collar_r = float(np.interp(0.0, trunk.positions[:, 2], trunk.radii))
            azimuths = None
            if al.buttress_lobes > 0 and rp.system != RootSystemType.FIBROUS:
                m = int(al.buttress_lobes)
                azimuths = flute_az + 2.0 * np.pi * np.arange(m) / m
            root_graph = RootSystemEngine(rp).generate(
                dbh_m, cr, collar_r, al.pipe_exponent_delta, seed=seed,
                display_depth_m=root_display_depth_m, azimuths=azimuths)

        if shoot_leaves > 0 and self.supports_shoot_cards():
            self.leaf_morphology = LeafMorphologyEngine(
                self.preset.leaf_morphology, shoot_leaves=shoot_leaves,
                opposite=self.preset.architecture.phyllotaxis == PhyllotaxisType.DECUSSATE)
        shoot = self.leaf_morphology.shoot_leaves > 0
        card = self.leaf_morphology.generate_3d_leaf_mesh(*card_grid)
        bounds = card["bounds_blu"]
        unit_len_m = (bounds[3] - bounds[1]) * card["blade_length_m"]
        # Leaf area index -> number of foliar units over the crown projection
        mask, (x0, y0, x1, y1) = self.leaf_morphology.rasterize_mask(96)
        unit_area_m2 = mask.mean() * (x1 - x0) * (y1 - y0) * card["blade_length_m"] ** 2
        lai = self.preset.architecture.leaf_area_index
        target = None
        if unit_area_m2 > 1e-8 and self.preset.architecture.model != HalleOldemanModel.CORNER:
            target = int(lai * np.pi * cr * cr / unit_area_m2)
        foliage = place_foliage(
            skeleton, self.preset.architecture, unit_length_m=unit_len_m,
            petiole_angle_deg=self.preset.leaf_morphology.petiole_angle_deg,
            density=leaf_density, budget=leaf_budget, seed=seed, shoot_cards=shoot, target_units=target,
            mean_leaf_angle_deg=self.preset.leaf_morphology.mean_leaf_angle_deg)

        return BotanicalPlantResult(
            preset=self.preset, dbh_m=dbh_m, total_height_m=h, crown_radius_m=cr, crown_depth_m=cd,
            base_radius_m=base_radius, skeleton_graph=skeleton, leaf_mesh_data=card, foliage=foliage,
            leaf_engine=self.leaf_morphology, root_graph=root_graph,
            flute_azimuth=flute_az if al.buttress_lobes > 0 else None)
