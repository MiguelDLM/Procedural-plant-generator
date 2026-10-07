"""
Unit tests verifying core botanical morphometric algorithms,
allometric power laws, and generation pipelines using standard library unittest.
"""

import math
import unittest
import numpy as np

from core.allometry import AllometricEngine, AllometricProfile
from core.architecture import ArchitectureEngine, ArchitectureProfile, HalleOldemanModel
from core.leaf_morphology import LeafMorphologyEngine, LeafMorphologyProfile, LeafArchetype, MarginType
from core.leaf_venation import VenationEngine, VenationProfile, VenationPattern
from core.biomechanics import BiomechanicalEngine, BiomechanicalProfile
from core.plant_pipeline import BotanicalPlantPipeline
from data.species_db import SPECIES_CATALOG, get_species_preset


class TestBotanicalPlantGenerator(unittest.TestCase):

    def test_allometry_tallo_scaling(self):
        """Verify height and crown radius scale monotonically and stay within physical limits."""
        profile = AllometricProfile(
            height_a=1.92,
            height_b=0.56,
            height_max_m=34.0,
            crown_radius_c=0.55,
            crown_radius_d=0.58
        )
        engine = AllometricEngine(profile)

        h_small = engine.calculate_height(0.10)
        h_medium = engine.calculate_height(0.40)
        h_large = engine.calculate_height(1.20)

        self.assertTrue(0.5 < h_small < h_medium < h_large <= 34.0)

        cr_small = engine.calculate_crown_radius(0.10)
        cr_large = engine.calculate_crown_radius(1.0)
        self.assertTrue(0.2 < cr_small < cr_large <= 18.0)

        # Leonardo da Vinci rule conservation
        parent_r = 0.10
        child_radii = engine.split_child_radii(parent_r, [0.6, 0.4])
        delta = profile.pipe_exponent_delta
        sum_child_power = sum(r ** delta for r in child_radii)
        parent_power = parent_r ** delta
        self.assertTrue(math.isclose(sum_child_power, parent_power, rel_tol=1e-5))

    def test_leaf_morphology_fourier_contours(self):
        """Verify Fourier boundary produces valid closed non-zero coordinates."""
        engine = LeafMorphologyEngine(LeafMorphologyProfile(
            archetype=LeafArchetype.OVATE,
            margin_type=MarginType.SERRATE,
            teeth_count=18
        ))
        pts = engine.generate_boundary_2d(num_points=64)
        self.assertEqual(pts.shape, (64, 2))
        self.assertTrue(-0.05 <= np.min(pts[:, 1]) <= 0.05)
        self.assertTrue(0.95 <= np.max(pts[:, 1]) <= 1.05)

        mesh = engine.generate_3d_leaf_mesh(grid_x=8, grid_y=16)
        self.assertTrue(len(mesh["vertices"]) > 0)
        self.assertTrue(len(mesh["faces"]) > 0)
        self.assertEqual(len(mesh["uvs"]), len(mesh["vertices"]))

    def test_leaf_venation_graph(self):
        """Verify primary and secondary veins are properly interconnected."""
        engine = VenationEngine(VenationProfile(
            pattern=VenationPattern.CRASPEDODROMOUS,
            secondary_vein_pairs=6
        ))
        net = engine.generate_network(blade_length_m=0.10, blade_width_m=0.06)
        self.assertTrue(len(net.nodes) > 30)
        self.assertTrue(len(net.segments) > 6)

        # Midrib is order 1
        self.assertEqual(net.nodes[0].order, 1)
        # Check that secondary veins exist
        self.assertTrue(any(n.order == 2 for n in net.nodes))

    def test_biomechanics_droop(self):
        """Verify branch droop increases with branch length and self-weight."""
        engine = BiomechanicalEngine(BiomechanicalProfile(wood_density_g_cm3=0.70))
        d_short = engine.calculate_branch_tip_droop(length_m=1.0, mean_radius_m=0.04)
        d_long = engine.calculate_branch_tip_droop(length_m=3.0, mean_radius_m=0.04)
        self.assertTrue(d_short >= 0.0)
        self.assertTrue(d_long > d_short)

    def test_plant_pipeline_all_presets(self):
        """Verify end-to-end pipeline execution across all empirical presets."""
        for species_id in SPECIES_CATALOG.keys():
            preset = get_species_preset(species_id)
            pipeline = BotanicalPlantPipeline(preset)
            result = pipeline.generate(dbh_m=preset.allometry.dbh_default_m, seed=123)

            self.assertTrue(result.total_height_m > 1.0)
            self.assertTrue(result.crown_radius_m > 0.5)
            self.assertTrue(len(result.skeleton_graph.nodes) > 10)
            self.assertTrue(len(result.leaf_mesh_data["vertices"]) > 0)
            self.assertTrue(len(result.vein_network.nodes) > 0)


if __name__ == "__main__":
    unittest.main()
