"""
Unit tests verifying core botanical morphometric algorithms,
allometric power laws, Gielis superformula, Runions space colonization,
and generation pipelines using standard library unittest.
"""

import math
import unittest
import numpy as np

from core.allometry import AllometricEngine, AllometricProfile
from core.architecture import ArchitectureEngine, ArchitectureProfile, HalleOldemanModel
from core.leaf_morphology import LeafMorphologyEngine, LeafMorphologyProfile, LeafArchetype, MarginType
from core.leaf_venation import VenationEngine, VenationProfile, VenationPattern
from core.biomechanics import BiomechanicalEngine, BiomechanicalProfile
from core.gielis import GielisEngine, GielisProfile, GIELIS_PRESETS
from core.space_colonization import SpaceColonizationEngine, SpaceColonizationConfig
from core.plant_pipeline import BotanicalPlantPipeline
from core.ontology import PLANT_ONTOLOGY_REGISTRY, get_po_term
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

    def test_gielis_superformula(self):
        """Verify Gielis superformula evaluates properly and preserves symmetry."""
        angles = np.linspace(0, 2 * np.pi, 36, endpoint=False)
        profile_5 = GIELIS_PRESETS["star_5"]
        r = GielisEngine.evaluate(angles, profile_5)
        self.assertEqual(len(r), 36)
        self.assertTrue(np.all(r > 0.0))

        # Check buttress cross section modulation at base (z=0) vs canopy (z=0.5)
        r_base = GielisEngine.compute_buttressed_cross_section(angles, 0.5, z_relative=0.0, gielis_profile=profile_5)
        r_top = GielisEngine.compute_buttressed_cross_section(angles, 0.5, z_relative=0.5, gielis_profile=profile_5)
        self.assertTrue(np.max(r_base) > np.max(r_top))
        self.assertTrue(math.isclose(np.mean(r_top), 0.5, rel_tol=1e-2))

    def test_runions_space_colonization(self):
        """Verify Runions et al. (2005) auxin space colonization generates hierarchical veins."""
        engine = SpaceColonizationEngine(SpaceColonizationConfig(
            num_auxin_sources=150,
            max_iterations=40
        ))
        nodes = engine.generate_venation(blade_length_m=0.10, blade_width_m=0.06, seed=42)
        self.assertTrue(len(nodes) > 15)
        # Check that nodes have Murray flux and valid radii
        self.assertTrue(all(n.radius_m > 0.0 for n in nodes))
        # Root node must have maximum flux
        self.assertTrue(nodes[0].flux >= nodes[-1].flux)

    def test_plant_ontology_registry(self):
        """Verify Plant Ontology terms are properly defined."""
        term_stem = get_po_term("PO:0009046")
        self.assertIsNotNone(term_stem)
        self.assertEqual(term_stem.name, "stem")

        term_vein = get_po_term("PO:0005022")
        self.assertIsNotNone(term_vein)
        self.assertEqual(term_vein.name, "leaf vein")

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
        self.assertEqual(net.nodes[0].order, 1)
        self.assertTrue(any(n.order == 2 for n in net.nodes))

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
