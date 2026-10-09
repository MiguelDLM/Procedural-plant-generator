"""Tests for orchids (core.orchid)."""

import copy
import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.orchid import OrchidEngine, single_flower, LIP, COLUMN, SEPAL, PETAL, CALLUS   # noqa: E402
from core.orchid_db import ORCHID_CATALOG                                                 # noqa: E402
from core.mesh_engine import MeshData                                                     # noqa: E402


def _flower(key, **changes):
    prof = copy.deepcopy(ORCHID_CATALOG[key].profile)
    for k, v in changes.items():
        setattr(prof, k, v)
    return prof, single_flower(prof, seed=1, detail=0.8)


def _centroid(m, part):
    sel = np.abs(m.point_attributes["orc_part"] - part) < 0.5
    return m.vertices[sel].astype(float).mean(0)


class TestOrchids(unittest.TestCase):

    def test_every_preset_generates_valid_meshes(self):
        for key, sp in ORCHID_CATALOG.items():
            r = OrchidEngine(sp.profile).generate(seed=3, detail=0.5)
            for m in (r.leaves, r.stems, r.roots, r.flowers, r.spikes, r.support):
                self.assertTrue(np.isfinite(m.vertices).all(), key)
                if len(m.loop_vertex):
                    self.assertLess(int(m.loop_vertex.max()), len(m.vertices), key)
            self.assertGreater(len(r.flowers.vertices), 0, key)
            self.assertGreater(r.stats["leaves"], 0, key)

    def test_flower_has_every_organ(self):
        _, m = _flower("phalaenopsis_hybrid")
        parts = set(np.round(m.point_attributes["orc_part"]).astype(int))
        for k in (SEPAL, PETAL, LIP, COLUMN, CALLUS):
            self.assertIn(k, parts)

    def test_resupination_puts_the_lip_down(self):
        """Resupinate flowers (twisted ovary) carry the lip below the column; non-resupinate ones above it."""
        for key, r in (("phalaenopsis_hybrid", 1.0), ("prosthechea_cochleata", 0.0)):
            for res in (0.0, 1.0):
                _, m = _flower(key, resupination=res)
                dz = _centroid(m, LIP)[2] - _centroid(m, COLUMN)[2]
                if res == 1.0:
                    self.assertLess(dz, 0, key)
                else:
                    self.assertGreater(dz, 0, key)

    def test_ovary_twist_on_erect_raceme(self):
        """On an erect raceme the lip starts uppermost (adaxial), so resupination twists the ovary ~180 deg."""
        p = copy.deepcopy(ORCHID_CATALOG["cymbidium_hybrid"].profile)
        p.spike_flex = 0.0
        eng = OrchidEngine(p)
        _, _, st = eng.inflorescence(np.zeros(3), np.array([1.0, 0, 0]), np.random.default_rng(0), 0.4)
        self.assertGreater(np.degrees(np.median(st["twists"])), 140)
        p.resupination = 0.0
        _, _, st = OrchidEngine(p).inflorescence(np.zeros(3), np.array([1.0, 0, 0]), np.random.default_rng(0), 0.4)
        self.assertLess(np.degrees(np.max(st["twists"])), 1)

    def test_phalaenopsis_flower_size(self):
        """'Northstar' (USPTO PP25,332): flower about 92 mm wide and 80 mm high."""
        _, m = _flower("phalaenopsis_hybrid")
        ext = m.vertices.max(0) - m.vertices.min(0)
        self.assertTrue(0.075 < ext[0] < 0.115, ext)
        self.assertTrue(0.065 < ext[2] < 0.105, ext)

    def test_rolled_lip_keeps_its_apparent_width(self):
        """A lip rolled into a tube (or pouch) is as wide as given, not narrower: the lamina is wider."""
        _, m = _flower("paphiopedilum_insigne")
        lip = m.vertices[np.abs(m.point_attributes["orc_part"] - LIP) < 0.5].astype(float)
        w = lip[:, 0].max() - lip[:, 0].min()
        self.assertGreater(w, 0.6 * ORCHID_CATALOG["paphiopedilum_insigne"].profile.lip_width_cm * 0.01)

    def test_spike_flex_arches_the_axis(self):
        p = copy.deepcopy(ORCHID_CATALOG["phalaenopsis_hybrid"].profile)
        tips = []
        for flex in (0.0, 0.6, 1.0):
            p.spike_flex = flex
            P, _, _ = OrchidEngine(p)._axis(np.zeros(3), np.array([1.0, 0, 0]), 0.6, 9, np.random.default_rng(0))
            d = P[-1] - P[-2]
            tips.append(math.degrees(math.atan2(np.hypot(d[0], d[1]), d[2])))
        self.assertLess(tips[0], 25)
        self.assertGreater(tips[1], 60)
        self.assertGreater(tips[2], tips[1])

    def test_monopodial_spikes_from_lower_axils(self):
        """Phalaenopsis spikes start from the axils of the 3rd-4th leaf below the apex, not from the top."""
        p = ORCHID_CATALOG["phalaenopsis_hybrid"].profile
        r = OrchidEngine(p).generate(seed=2, detail=0.4)
        stem_top = r.stems.vertices[:, 2].max()
        spike_low = r.spikes.vertices[:, 2].min()
        self.assertLess(spike_low, stem_top)
        self.assertGreater(r.spikes.vertices[:, 2].max(), stem_top + 0.2)

    def test_connected(self):
        from tests.geometry_checks import floating_islands
        for key in ("paphiopedilum_insigne", "cattleya_labiata"):
            r = OrchidEngine(ORCHID_CATALOG[key].profile).generate(seed=3, detail=0.4)
            whole = MeshData.concatenate([m for m in (r.leaves, r.stems, r.roots, r.flowers, r.spikes, r.support)
                                          if len(m.vertices)])
            self.assertEqual(floating_islands(whole, 0.006), [], key)
        _, m = _flower("phalaenopsis_hybrid")
        self.assertEqual(floating_islands(m, 0.003), [])

    def test_inactive_orchid_fields_do_not_change_geometry(self):
        from tests.relevance_check import check
        for key in ("phalaenopsis_hybrid", "paphiopedilum_insigne"):
            self.assertEqual(check("Orchid", key, report=lambda m: None), [], key)


if __name__ == "__main__":
    unittest.main()
