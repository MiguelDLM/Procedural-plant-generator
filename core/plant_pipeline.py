"""
Full Procedural Botanical Plant Generation Pipeline.
Integrates allometric scaling, Hallé-Oldeman branching architectures,
leaf morphometrics, venation networks, Gielis superformula buttressing, and Plant Ontology.
"""

from dataclasses import dataclass
import numpy as np

from .allometry import AllometricEngine, AllometricProfile
from .architecture import ArchitectureEngine, ArchitectureProfile, BranchingGraph
from .leaf_morphology import LeafMorphologyEngine, LeafMorphologyProfile
from .leaf_venation import VenationEngine, VenationProfile, LeafVeinNetwork
from .biomechanics import BiomechanicalEngine, BiomechanicalProfile
from .space_colonization import SpaceColonizationEngine, SpaceColonizationConfig
from .gielis import GielisEngine, GielisProfile
from .species_preset import BotanicalSpeciesPreset
from .ontology import get_po_term


@dataclass
class BotanicalPlantResult:
    """Complete computed geometric and botanical data for a plant model."""
    preset: BotanicalSpeciesPreset
    dbh_m: float
    total_height_m: float
    crown_radius_m: float
    crown_depth_m: float
    base_radius_m: float

    # Skeletons and meshes
    skeleton_graph: BranchingGraph
    leaf_mesh_data: dict
    vein_network: LeafVeinNetwork

    # Foliage placement data
    leaf_transforms: list[dict]  # [{'pos': [x,y,z], 'dir': [dx,dy,dz], 'scale': float}]


class BotanicalPlantPipeline:
    """Top-level generation controller."""

    def __init__(self, preset: BotanicalSpeciesPreset):
        self.preset = preset
        self.allometry = AllometricEngine(preset.allometry)
        self.architecture = ArchitectureEngine(preset.architecture)
        self.leaf_morphology = LeafMorphologyEngine(preset.leaf_morphology)
        self.venation = VenationEngine(preset.venation)
        self.biomechanics = BiomechanicalEngine(preset.biomechanics)
        self.space_colonization = SpaceColonizationEngine()

    def generate(
        self,
        dbh_m: float = None,
        leaf_density: float = 1.0,
        seed: int = 42
    ) -> BotanicalPlantResult:
        """
        Executes end-to-end biological generation.
        """
        if dbh_m is None:
            dbh_m = self.preset.allometry.dbh_default_m

        dbh_m = float(np.clip(dbh_m, self.preset.allometry.dbh_min_m, self.preset.allometry.dbh_max_m))

        # 1. Compute empirical allometric dimensions (TALLO)
        height_m = self.allometry.calculate_height(dbh_m)
        crown_radius_m = self.allometry.calculate_crown_radius(dbh_m)
        crown_depth_m = self.allometry.calculate_crown_depth(height_m)
        base_radius_m = dbh_m * 0.5

        # 2. Generate 3D branching skeleton
        skeleton = self.architecture.generate_skeleton(
            total_height_m=height_m,
            crown_radius_m=crown_radius_m,
            crown_depth_m=crown_depth_m,
            base_radius_m=base_radius_m,
            pipe_delta=self.preset.allometry.pipe_exponent_delta,
            seed=seed
        )

        # 3. Apply biomechanical self-weight droop to branches
        for b in skeleton.branches:
            if len(b) > 2 and skeleton.nodes[b[0]].order >= 1:
                first_node = skeleton.nodes[b[0]]
                last_node = skeleton.nodes[b[-1]]
                b_len = last_node.distance_along_stem - first_node.distance_along_stem
                droop = self.biomechanics.calculate_branch_tip_droop(b_len, first_node.radius)

                if droop > 0:
                    for i, n_idx in enumerate(b):
                        frac = (i / len(b)) ** 2
                        skeleton.nodes[n_idx].position[2] -= droop * frac

        # 4. Generate master leaf 3D mesh
        leaf_mesh = self.leaf_morphology.generate_3d_leaf_mesh(grid_x=12, grid_y=24)

        # 5. Generate empirical leaf venation graph (Duarte et al. 2025 & Runions et al. 2005)
        blade_len_m = self.preset.leaf_morphology.blade_length_cm * 0.01
        blade_width_m = blade_len_m / self.preset.leaf_morphology.aspect_ratio
        vein_net = self.venation.generate_network(blade_len_m, blade_width_m)

        # 6. Compute leaf instance locations
        leaf_transforms = []
        rng = np.random.default_rng(seed + 101)

        for node in skeleton.nodes:
            # Place leaves on twigs and terminal branch nodes
            if node.order >= max(1, self.preset.architecture.max_order - 1):
                prob = 0.45 * leaf_density
                if node.is_leaf_attachment or rng.random() < prob:
                    # Random rotation around branch axis
                    angle = rng.uniform(0.0, 2.0 * np.pi)
                    scale = rng.uniform(0.85, 1.15)

                    leaf_transforms.append({
                        "position": node.position.tolist(),
                        "direction": node.direction.tolist(),
                        "twist_rad": float(angle),
                        "scale": float(scale)
                    })

        return BotanicalPlantResult(
            preset=self.preset,
            dbh_m=dbh_m,
            total_height_m=height_m,
            crown_radius_m=crown_radius_m,
            crown_depth_m=crown_depth_m,
            base_radius_m=base_radius_m,
            skeleton_graph=skeleton,
            leaf_mesh_data=leaf_mesh,
            vein_network=vein_net,
            leaf_transforms=leaf_transforms
        )
