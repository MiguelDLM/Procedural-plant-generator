"""Tests for wind, terrain patches, grass collars, orchid mounts and organs, mangroves, bulbs, parasites and
epiphytes."""

import copy
import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.mesh_engine import MeshData                                  # noqa: E402
from tests.geometry_checks import floating_islands                     # noqa: E402


def _whole(*ms):
    ms = [copy.copy(m) for m in ms if m is not None and len(m.vertices)]
    for m in ms:
        m.point_attributes = {}
    return MeshData.concatenate(ms)


class TestWind(unittest.TestCase):
    def test_frequency_and_amplitude_scale(self):
        from core.wind import wind_frequency, wind_amplitude
        self.assertTrue(0.4 < wind_frequency(20.0) < 0.8)              # Trees ~0.6 Hz (de Langre 2008)
        self.assertGreater(wind_frequency(1.0), 2.0)                   # Cereals 2-3 Hz
        self.assertGreater(wind_amplitude(15, 10), wind_amplitude(5, 10))
        ratio = wind_amplitude(20, 10) / wind_amplitude(10, 10)
        self.assertAlmostEqual(ratio, 2 ** 1.3, places=3)              # Reconfiguration: v^1.3, not v^2


class TestGrassDetails(unittest.TestCase):
    def test_surface_points_follow_terrain(self):
        from core.grass import surface_points
        x, y = np.meshgrid(np.linspace(-1, 1, 11), np.linspace(-1, 1, 11))
        z = 0.3 * x
        V = np.stack([x, y, z], -1)
        tris = []
        for i in range(10):
            for j in range(10):
                a, b, c, d = V[i, j], V[i, j + 1], V[i + 1, j + 1], V[i + 1, j]
                tris += [[a, b, c], [a, c, d]]
        P, N = surface_points(np.array(tris), 200, seed=1, min_dist=0.03)
        self.assertGreater(len(P), 300)
        self.assertLess(np.abs(P[:, 2] - 0.3 * P[:, 0]).max(), 1e-6)
        d = np.sqrt(((P[:, None] - P[None]) ** 2).sum(-1)) + np.eye(len(P))
        self.assertGreaterEqual(d.min(), 0.03 - 1e-9)
        self.assertTrue((N[:, 2] > 0.9).all())
        P2, _ = surface_points(np.array(tris), 200, seed=1, max_slope_deg=10.0)   # 16.7 deg slope excluded
        self.assertEqual(len(P2), 0)

    def test_collars(self):
        from core.grass import GrassEngine
        from core.grass_db import GRASS_CATALOG
        for key, aur in (("hordeum_vulgare", True), ("avena_sativa", False)):
            r = GrassEngine(GRASS_CATALOG[key].profile).generate(seed=2, detail=0.8, with_roots=False)
            part = np.round(r.culms.point_attributes["grass_part"])
            self.assertTrue((part == 8).any(), key)                    # Ligule
            self.assertEqual(bool((part == 9).any()), aur, key)       # Auricles: barley yes, oats no

    def test_node_to_node_variation(self):
        from core.grass import GrassEngine
        from core.grass_db import GRASS_CATALOG
        r = GrassEngine(GRASS_CATALOG["saccharum_officinarum"].profile).generate(seed=2, detail=0.6,
                                                                               with_roots=False)
        self.assertGreater(len(np.unique(np.round(r.culms.point_attributes["grass_nid"], 3))), 5)


class TestOrchidAdditions(unittest.TestCase):
    def _prof(self, key, **kw):
        from core.orchid_db import ORCHID_CATALOG
        p = copy.deepcopy(ORCHID_CATALOG[key].profile)
        for k, v in kw.items():
            setattr(p, k, v)
        return p

    def test_spur_length(self):
        from core.orchid import single_flower, LIP
        m0 = single_flower(self._prof("angraecum_sesquipedale", spur_cm=0.0), detail=0.6)
        m1 = single_flower(self._prof("angraecum_sesquipedale"), detail=0.6)
        ext = lambda m: float((m.vertices.max(0) - m.vertices.min(0))[2])
        self.assertGreater(ext(m1) - ext(m0), 0.15)                     # A spur of ~33 cm hangs below

    def test_umbel_from_the_tip(self):
        from core.orchid import OrchidEngine
        p = self._prof("bulbophyllum_rothschildianum")
        eng = OrchidEngine(p)
        sp, fl, _ = eng.inflorescence(np.zeros(3), np.array([1.0, 0, 0]), np.random.default_rng(0), 0.5)
        tops = [f.vertices.astype(float).mean(0) for f in fl if f is not None]
        self.assertEqual(len(tops), p.flowers)
        z = np.array([t[2] for t in tops])
        self.assertLess(z.max() - z.min(), 0.12)                         # All at the top, in a fan

    def test_pot_and_branch_mounts(self):
        from core.orchid import OrchidEngine, OrchidMount
        for mount in (OrchidMount.POT, OrchidMount.BRANCH):
            p = self._prof("phalaenopsis_hybrid", mount=mount, aerial_share=0.3)
            r = OrchidEngine(p).generate(seed=4, detail=0.5)
            V = r.roots.vertices.astype(float)
            self.assertGreater(len(r.support.vertices), 0)
            if mount == OrchidMount.POT:
                Ro = p.pot_diameter_cm * 0.005
                inside = V[:, 2] < -0.005
                rh = np.hypot(V[inside, 0], V[inside, 1])
                # Below the substrate surface roots are inside the pot or hanging outside its wall
                self.assertTrue(((rh < Ro * 0.95) | (rh > Ro * 0.78)).all())
                self.assertGreater(V[:, 2].min(), -Ro * 1.7 - 0.02)
            else:
                Rb = p.branch_diameter_cm * 0.005
                dist = np.hypot(V[:, 1], V[:, 2] + Rb)
                self.assertGreater(dist.min(), Rb * 0.95)               # Never inside the bough

    def test_new_presets_connected(self):
        from core.orchid import OrchidEngine
        for key in ("stanhopea_tigrina", "bulbophyllum_rothschildianum", "oncidium_sphacelatum"):
            r = OrchidEngine(self._prof(key)).generate(seed=3, detail=0.4)
            self.assertEqual(floating_islands(_whole(r.leaves, r.stems, r.roots, r.flowers, r.spikes), 0.006),
                             [], key)


class TestMangroves(unittest.TestCase):
    def test_stilts_drop_roots_and_pneumatophores(self):
        from core.plant_pipeline import BotanicalPlantPipeline
        from core.species_db import SPECIES_CATALOG
        r = BotanicalPlantPipeline(SPECIES_CATALOG["rhizophora_mangle"]).generate(
            seed=3, leaf_budget=100, roots=True, shoot_leaves=0)
        h = r.total_height_m
        stilts = [a for a in r.root_graph.axes if a.order == 1 and a.positions[0, 2] > 0.2]
        self.assertGreaterEqual(len(stilts), 8)
        self.assertTrue(0.08 * h < max(a.positions[0, 2] for a in stilts) < 0.4 * h)   # 10-33 % of height
        drops = [a for a in r.fine_root_graph.axes if a.positions[0, 2] > 0.25 * h]
        self.assertGreater(len(drops), 2)
        for a in drops:
            self.assertLess(a.positions[-1, 2], 0.0)
        r2 = BotanicalPlantPipeline(SPECIES_CATALOG["avicennia_germinans"]).generate(
            seed=3, leaf_budget=100, roots=True, shoot_leaves=0)
        pn = [a for a in r2.root_graph.axes if a.positions[-1, 2] > 0.05 and a.positions[0, 2] < 0.0]
        self.assertGreater(len(pn), 50)
        self.assertLess(max(a.positions[-1, 2] for a in pn), 0.35)


class TestBulbsAndParasites(unittest.TestCase):
    def test_bulbs(self):
        from core.vegetable import VegetableEngine
        from core.vegetable_db import VEGETABLE_CATALOG
        for key in ("allium_cepa", "allium_sativum", "crocus_sativus"):
            sp = VEGETABLE_CATALOG[key]
            r = VegetableEngine(sp.profile, sp.leaf, sp.venation).generate(seed=1, detail=0.5)
            self.assertEqual(floating_islands(_whole(r.root, r.stems, r.leaves), 0.006), [], key)
        sp = VEGETABLE_CATALOG["crocus_sativus"]
        r = VegetableEngine(sp.profile, sp.leaf, sp.venation).generate(seed=1, detail=0.5)
        self.assertGreater(float(r.root.point_attributes["veg_ring"].max()), 0.1)   # Wrinkled contractile roots

    def test_cuscuta(self):
        from core.vine import VineEngine, guide_shape
        from core.vine_db import VINE_CATALOG
        sp = VINE_CATALOG["cuscuta_campestris"]
        r = VineEngine(sp.profile, sp.leaf, sp.venation).generate(guide_shape("Pole", 0.5, 0.5), seed=1, detail=0.6)
        self.assertEqual(len(r.leaves.vertices), 0)
        p0 = copy.deepcopy(sp.profile)
        p0.haustoria = 0.0
        r0 = VineEngine(p0, sp.leaf, sp.venation).generate(guide_shape("Pole", 0.5, 0.5), seed=1, detail=0.6)
        self.assertGreater(len(r.stem.vertices), len(r0.stem.vertices))

    def test_guests_on_branches(self):
        from core.plant_pipeline import BotanicalPlantPipeline
        from core.species_db import SPECIES_CATALOG
        from core.guests import branch_sites, mistletoes, epiphytic_orchids
        from core.mistletoe import MISTLETOE_CATALOG
        from core.orchid_db import ORCHID_CATALOG
        r = BotanicalPlantPipeline(SPECIES_CATALOG["quercus_robur"]).generate(seed=3, leaf_budget=100,
                                                                             roots=False, shoot_leaves=0)
        sites = branch_sites(r.skeleton_graph, 10, zones=(3, 4), min_radius=0.025, seed=1)
        self.assertGreater(len(sites), 3)
        for surf, axis_pt, t, rad, up in sites:
            self.assertGreater(up[2], 0.3)                               # Upper side of the branch
            self.assertAlmostEqual(float(np.linalg.norm(surf - axis_pt)), rad, places=6)
        m = mistletoes(MISTLETOE_CATALOG["viscum_album"].profile, r.skeleton_graph, 4, seed=1)
        self.assertGreater(len(m.vertices), 1000)
        e = epiphytic_orchids(ORCHID_CATALOG["cattleya_labiata"].profile, r.skeleton_graph, 2, seed=1, detail=0.4)
        self.assertGreater(len(e.vertices), 1000)


if __name__ == "__main__":
    unittest.main()
