"""Tests for root system architecture (core.roots, core.root_architecture) across growth forms."""

import copy
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.roots import RootSystemEngine, RootProfile, RootSystemType as T      # noqa: E402
from core.root_architecture import RootBranching, lateral_roots, grow_root    # noqa: E402


def _graph(system, **kw):
    eng = RootSystemEngine(RootProfile(system=system, **kw))
    g = eng.generate(0.5, 4.0, 0.3, seed=2, display_depth_m=4.0)
    return eng, g


def _z(g):
    return np.concatenate([a.positions[:, 2] for a in g.axes])


class TestRootSystems(unittest.TestCase):

    def test_every_type_generates_below_ground(self):
        for s in T:
            eng, g = _graph(s)
            self.assertGreater(len(g.axes), 3, s)
            z = _z(g)
            self.assertTrue(np.isfinite(z).all(), s)
            self.assertLess(z.min(), -0.1, s)
            fine = eng.fine_roots(g, seed=2, display_depth_m=4.0)
            if fine.axes:
                zf = np.concatenate([a.positions[1:, 2] for a in fine.axes])
                self.assertLess(zf.max(), 0.0, s)                      # Laterals never grow in the air

    def test_heart_is_a_deep_cage(self):
        """Heart systems: several oblique roots reaching deep near the stem (Koestler; Danjon et al. 2005);
        plate systems stay shallow."""
        _, heart = _graph(T.HEART, max_depth_m=3.0, taproot_share=0.3, heart_roots=8, deep_roots=0.7)
        _, plate = _graph(T.PLATE, max_depth_m=1.0, taproot_share=0.0)
        self.assertGreater(-_z(heart).min(), 1.2)
        self.assertGreater(-_z(heart).min(), -_z(plate).min())
        obliques = [a for a in heart.axes if a.order == 1 and a.positions[-1, 2] < -1.0]
        self.assertGreaterEqual(len(obliques), 6)

    def test_deep_roots_controls_cage_depth(self):
        depths = []
        for d in (0.2, 0.9):
            _, g = _graph(T.TAPROOT, max_depth_m=4.0, deep_roots=d)
            near = [a for a in g.axes if np.hypot(*a.positions[0, :2]) < 1.5 and a.order == 3]
            depths.append(max(-a.positions[:, 2].min() for a in near) if near else 0.0)
        self.assertGreater(depths[1], depths[0])

    def test_stilt_roots_arch_from_the_stem(self):
        _, g = _graph(T.STILT, lateral_count=8, stilt_height_dbh=3.0)
        props = [a for a in g.axes if a.order == 1 and a.positions[0, 2] > 0.2]   # Leave the stem in the air
        self.assertGreaterEqual(len(props), 8)                         # (plus ordinary underground laterals)
        for a in props:
            self.assertLess(a.positions[-1, 2], 0.0)                    # and enter the soil

    def test_tuberous_cluster(self):
        _, g = _graph(T.TUBER_CLUSTER, tuber_count=6, tuber_radius_m=0.04, tuber_length_m=0.3)
        swollen = [a for a in g.axes if a.radii.max() > 0.02]
        self.assertEqual(len(swollen), 6)
        for a in swollen:                                               # Neck thinner than the body
            self.assertLess(a.radii[1], 0.6 * a.radii.max())

    def test_fine_roots_density(self):
        eng, g = _graph(T.HEART)
        n1 = len(eng.fine_roots(g, seed=1).axes)
        eng.profile.fine_roots = 0.0
        n0 = len(eng.fine_roots(g, seed=1).axes)
        self.assertEqual(n0, 0)
        self.assertGreater(n1, 100)


class TestLateralRules(unittest.TestCase):

    def test_laterals_thinner_than_mother_and_spaced(self):
        rng = np.random.default_rng(0)
        P, R = grow_root(np.zeros(3), np.array([1.0, 0, -0.2]), 1.0, 0.01, rng, 0.0, 0.0)
        br = RootBranching(orders=1, interbranch_cm=10, basal_zone=0.1, apical_zone_cm=10, radius_ratio=0.4,
                           budget=100)
        lat = lateral_roots([(P, R)], br, np.random.default_rng(1))
        self.assertTrue(5 <= len(lat) <= 12)                           # ~ (1 - 0.1 - 0.1) m / 10 cm
        for Pl, Rl, order, mid, i in lat:
            self.assertEqual(order, 1)
            self.assertLess(Rl[0], R[i])

    def test_determinate_laterals_stay_short(self):
        rng = np.random.default_rng(0)
        P, R = grow_root(np.zeros(3), np.array([1.0, 0, -0.1]), 2.0, 0.005, rng, 0.0, 0.1)
        br = RootBranching(orders=1, interbranch_cm=3, determinate_cm=5, length_ratio=0.5, budget=400)
        for Pl, *_ in lateral_roots([(P, R)], br, np.random.default_rng(1)):
            self.assertLess(np.linalg.norm(np.diff(Pl, axis=0), axis=1).sum(), 0.065)


class TestGrowthFormShapes(unittest.TestCase):
    """Schenk & Jackson (2002): lateral spread / depth ~4.5 in stem succulents, 0.3-0.35 in herbs."""

    def test_cactus_wide_and_shallow(self):
        from core.succulent_db import CATALOGS, GrowthForm
        from core.cactus import CactusEngine
        p = CATALOGS[GrowthForm.CACTUS]["opuntia_ficus_indica"].profile
        r = CactusEngine(p).generate(seed=3, detail=0.4, spine_budget=100, with_roots=True)
        V = r.roots.vertices
        spread = np.hypot(V[:, 0], V[:, 1]).max()
        self.assertGreater(spread / -V[:, 2].min(), 2.5)

    def test_grass_narrow_and_deep(self):
        from core.grass import GrassEngine
        from core.grass_db import GRASS_CATALOG
        r = GrassEngine(GRASS_CATALOG["triticum_aestivum"].profile).generate(seed=3, detail=0.6)
        V = r.roots.vertices
        spread = np.hypot(V[:, 0], V[:, 1]).max()
        self.assertLess(spread / -V[:, 2].min(), 1.0)
        self.assertGreater(-V[:, 2].min(), 0.5)

    def test_tuberous_vegetables(self):
        from core.vegetable import VegetableEngine
        from core.vegetable_db import VEGETABLE_CATALOG
        from core.mesh_engine import MeshData
        from tests.geometry_checks import floating_islands
        for key in ("ipomoea_batatas", "manihot_esculenta"):
            sp = copy.deepcopy(VEGETABLE_CATALOG[key])
            r = VegetableEngine(sp.profile, sp.leaf, sp.venation).generate(seed=1, detail=0.5)
            body = r.root.point_attributes["veg_part"] < 0.5
            self.assertGreater(body.sum(), 100, key)
            self.assertLess(r.root.vertices[:, 2].max(), 0.02, key)
            parts = [m for m in (r.root, r.stems) if len(m.vertices)]
            for m in parts:
                m.point_attributes = {}
            self.assertEqual(floating_islands(MeshData.concatenate(parts), 0.006), [], key)


if __name__ == "__main__":
    unittest.main()
