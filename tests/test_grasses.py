"""Tests for grasses and cereals (core.grass)."""

import copy
import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.grass import GrassEngine, patch_points          # noqa: E402
from core.grass_db import GRASS_CATALOG                    # noqa: E402
from core.mesh_engine import MeshData                      # noqa: E402


def _gen(key, **kw):
    sp = GRASS_CATALOG[key]
    return sp, GrassEngine(sp.profile).generate(seed=3, detail=0.6, **kw)


class TestGrasses(unittest.TestCase):

    def test_every_preset_generates_valid_meshes(self):
        for key, sp in GRASS_CATALOG.items():
            _, r = _gen(key)
            for m in (r.leaves, r.culms, r.heads, r.ears, r.roots):
                self.assertTrue(np.isfinite(m.vertices).all(), key)
                if len(m.loop_vertex):
                    self.assertLess(int(m.loop_vertex.max()), len(m.vertices), key)
            self.assertGreater(len(r.leaves.vertices), 0, key)
            if sp.profile.head.value != "None":
                self.assertGreater(len(r.heads.vertices), 0, key)

    def test_maize(self):
        sp, r = _gen("zea_mays")
        p = sp.profile
        top = float(max(r.culms.vertices[:, 2].max(), r.heads.vertices[:, 2].max()))
        self.assertGreater(top, p.culm_height_m)                                    # Tassel above the culm
        self.assertLess(top, p.culm_height_m + (p.peduncle_cm + p.head_length_cm) * 0.01 + 0.1)
        kernels = r.ears.point_attributes["grass_part"] == 4
        self.assertTrue(kernels.any())
        ez = float(r.ears.vertices[kernels, 2].mean())
        self.assertTrue(0.3 * p.culm_height_m < ez < 0.85 * p.culm_height_m)       # Ear at mid-height
        self.assertEqual(r.stats["leaves"], p.nodes)

    def test_distichous_leaves(self):
        """Successive leaves of a culm leave on opposite sides (about 180 degrees apart)."""
        eng = GrassEngine(copy.deepcopy(GRASS_CATALOG["zea_mays"].profile))
        lv, _, _, _, _ = eng.shoot(np.zeros(3), np.array([0, 0, 1.0]), 2.0, np.random.default_rng(0), 0.5)
        dirs = []
        for m in lv:
            V = m.vertices.astype(float)
            d = V[len(V) // 2:].mean(0)[:2]
            dirs.append(math.atan2(d[1], d[0]))
        steps = [abs(math.degrees((b - a + math.pi) % (2 * math.pi) - math.pi)) for a, b in zip(dirs, dirs[1:])]
        self.assertGreater(np.median(steps), 120)

    def test_flowering_tillers(self):
        prof = copy.deepcopy(GRASS_CATALOG["pennisetum_setaceum"].profile)
        heads = []
        for f in (0.2, 1.0):
            prof.flowering_tillers = f
            heads.append(len(GrassEngine(prof).generate(seed=1, detail=0.4).heads.vertices))
        self.assertLess(heads[0], 0.5 * heads[1])

    def test_patch_points(self):
        P = patch_points(2.0, 100.0, seed=1, min_dist=0.05)
        self.assertGreater(len(P), 200)
        d = np.sqrt(((P[:, None, :2] - P[None, :, :2]) ** 2).sum(-1)) + np.eye(len(P))
        self.assertGreaterEqual(float(d.min()), 0.05 - 1e-9)
        self.assertTrue((np.abs(P[:, :2]) <= 1.0).all())

    def test_connected(self):
        from tests.geometry_checks import floating_islands
        for key in ("zea_mays", "lolium_perenne", "triticum_aestivum"):    # Nodes, sheaths, collars, ears
            _, r = _gen(key)
            whole = MeshData.concatenate([m for m in (r.leaves, r.culms, r.heads, r.ears) if len(m.vertices)])
            self.assertEqual(floating_islands(whole, 0.006), [], key)

    def test_inactive_grass_fields_do_not_change_geometry(self):
        from tests.relevance_check import check
        for key in ("zea_mays", "triticum_aestivum"):
            self.assertEqual(check("Grass", key, report=lambda m: None), [], key)


if __name__ == "__main__":
    unittest.main()
