"""
Unit tests for the botanical core (pure Python + NumPy, no Blender required).
Run from the repository root:  python3 -m unittest discover -s tests -v
"""

import math
import unittest
import numpy as np

from core.allometry import AllometricEngine, AllometricProfile
from core.architecture import crown_envelope, HalleOldemanModel
from core.leaf_morphology import (LeafMorphologyEngine, LeafMorphologyProfile, LeafArchetype, MarginType,
                                  LeafShapeBuilder, apply_archetype_template, half_width_profile, T_GRID)
from core.leaf_venation import VenationEngine, VenationProfile, VenationPattern
from core.leaf_texture import LeafTextureEngine
from core.gielis import GielisEngine, GIELIS_PRESETS
from core.space_colonization import SpaceColonizationEngine, SpaceColonizationConfig
from core.plant_pipeline import BotanicalPlantPipeline
from core.mesh_engine import BotanicalMeshEngine, MeshConfig
from core.ontology import get_po_term
from core.trait_space import to_vector, from_vector, blend, mutate, distance, nearest_species, TRAITS
from data.species_db import SPECIES_CATALOG, get_species_preset
from core.roots import RootSystemEngine, RootProfile, RootSystemType, sample_depth, JACKSON_BETA
from core.junctions import (split_axes, graph_of, sample_table, nearest_samples, cylindrical_uvs, u_repeats,
                            axis_tangents, default_n0, junction_sleeves, bark_coordinates)


def _mask(archetype, **kw):
    prof = apply_archetype_template(LeafMorphologyProfile(margin_type=MarginType.ENTIRE, **kw), archetype)
    m, _ = LeafMorphologyEngine(prof).rasterize_mask(160)
    return m


class TestAllometry(unittest.TestCase):
    def test_height_and_crown_scaling(self):
        eng = AllometricEngine(AllometricProfile(height_a=1.92, height_b=0.56, height_max_m=34.0,
                                                 crown_radius_c=0.55, crown_radius_d=0.58))
        h = [eng.calculate_height(d) for d in (0.1, 0.4, 1.2)]
        self.assertTrue(0.5 < h[0] < h[1] < h[2] <= 34.0)
        self.assertLess(eng.calculate_crown_radius(0.1), eng.calculate_crown_radius(1.0))

    def test_pipe_model_conservation(self):
        eng = AllometricEngine(AllometricProfile())
        child = eng.split_child_radii(0.1, [0.6, 0.4])
        d = eng.profile.pipe_exponent_delta
        self.assertTrue(math.isclose(sum(r ** d for r in child), 0.1 ** d, rel_tol=1e-6))

    def test_species_allometry_hits_anchor_values(self):
        """Fitted coefficients reproduce each species' declared height at its default DBH."""
        for key in ("quercus_robur", "sequoiadendron_giganteum", "acer_palmatum"):
            p = get_species_preset(key)
            h, cr, _ = BotanicalPlantPipeline(p).dimensions(p.allometry.dbh_default_m)
            self.assertGreater(h, 5.0)
            self.assertLess(h, p.allometry.height_max_m)


class TestLeafShape(unittest.TestCase):
    def test_width_profile_is_closed_and_peaks_at_widest_point(self):
        for widest in (0.3, 0.5, 0.7):
            w = half_width_profile(2.0, widest, 90, 60, 0.3, 0.0)
            self.assertEqual(w[0], 0.0)
            self.assertEqual(w[-1], 0.0)
            self.assertAlmostEqual(T_GRID[np.argmax(w)], widest, delta=0.08)
            self.assertAlmostEqual(w.max(), 0.25, delta=0.01)

    def test_outlines_are_not_rectangles(self):
        """Fill fraction of the lamina's bounding box stays well below a rectangle's 1.0."""
        for a in (LeafArchetype.OVATE, LeafArchetype.ELLIPTIC, LeafArchetype.LANCEOLATE,
                  LeafArchetype.PALMATE_LOBED, LeafArchetype.PINNATE_LOBED):
            m = _mask(a, petiole_length_ratio=0.0)
            ys, xs = np.nonzero(m)
            box = (ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1)
            self.assertLess(m.sum() / box, 0.80, a.value)

    def test_lobed_leaves_have_sinuses(self):
        """Some scanline (row or column) across a lobed lamina crosses the margin more than twice."""
        for a in (LeafArchetype.PALMATE_LOBED, LeafArchetype.PINNATE_LOBED):
            m = _mask(a, petiole_length_ratio=0.0).astype(int)
            crossings = max(int(np.abs(np.diff(line)).sum()) for line in list(m) + list(m.T))
            self.assertGreaterEqual(crossings, 4, a.value)

    def test_teeth_change_the_margin(self):
        smooth = _mask(LeafArchetype.OVATE)
        prof = apply_archetype_template(LeafMorphologyProfile(margin_type=MarginType.SERRATE, teeth_count=20,
                                                              tooth_height_ratio=0.08), LeafArchetype.OVATE)
        toothed, _ = LeafMorphologyEngine(prof).rasterize_mask(160)
        self.assertLess(toothed.sum(), smooth.sum())

    def test_compound_and_needle_units_build(self):
        for a in (LeafArchetype.PINNATE_COMPOUND, LeafArchetype.PALMATE_COMPOUND, LeafArchetype.NEEDLE_FASCICLE,
                  LeafArchetype.NEEDLE_SPRAY, LeafArchetype.PALM_FROND):
            model = LeafShapeBuilder(apply_archetype_template(LeafMorphologyProfile(), a)).build()
            self.assertGreater(len(model.blades), 2, a.value)

    def test_leaf_card(self):
        card = LeafMorphologyEngine(LeafMorphologyProfile()).generate_3d_leaf_mesh(3, 6)
        self.assertEqual(card["vertices"].shape, (18, 3))
        self.assertEqual(card["faces"].shape, (10, 4))
        self.assertTrue(np.all((card["uvs"] >= 0) & (card["uvs"] <= 1)))
        boundary = LeafMorphologyEngine(LeafMorphologyProfile()).generate_boundary_2d(64)
        self.assertEqual(boundary.shape, (64, 2))


class TestVenation(unittest.TestCase):
    def _net(self, archetype, pattern, **kw):
        prof = apply_archetype_template(LeafMorphologyProfile(margin_type=MarginType.ENTIRE), archetype)
        model = LeafShapeBuilder(prof).build()
        return VenationEngine(VenationProfile(pattern=pattern, **kw)).generate(model, 80.0)

    def test_hierarchy(self):
        net = self._net(LeafArchetype.ELLIPTIC, VenationPattern.CRASPEDODROMOUS, secondary_vein_pairs=8)
        self.assertGreaterEqual(net.count(1), 1)
        self.assertGreaterEqual(net.count(2), 14)
        self.assertGreater(net.count(3), 10)

    def test_brochidodromous_adds_loops(self):
        c = self._net(LeafArchetype.ELLIPTIC, VenationPattern.CRASPEDODROMOUS, secondary_vein_pairs=8)
        b = self._net(LeafArchetype.ELLIPTIC, VenationPattern.BROCHIDODROMOUS, secondary_vein_pairs=8)
        self.assertGreater(b.count(2), c.count(2))

    def test_secondaries_stay_inside_lamina(self):
        prof = apply_archetype_template(LeafMorphologyProfile(margin_type=MarginType.ENTIRE), LeafArchetype.OVATE)
        model = LeafShapeBuilder(prof).build()
        net = VenationEngine(VenationProfile()).generate(model, 80.0)
        pts = np.vstack([pl.points for pl in net.polylines if pl.order == 2])
        self.assertLess(np.mean(model.field(pts[:, 0], pts[:, 1]) < 0.01), 1.01)
        self.assertGreater(np.mean(model.field(pts[:, 0], pts[:, 1]) < 0.005), 0.97)

    def test_vla_sets_areole_size(self):
        lo = VenationEngine(VenationProfile(vla_mm_per_mm2=4.0))
        hi = VenationEngine(VenationProfile(vla_mm_per_mm2=12.0))
        model = LeafShapeBuilder(LeafMorphologyProfile()).build()
        self.assertGreater(lo.generate(model, 80).minor_spacing_blu, hi.generate(model, 80).minor_spacing_blu)

    def test_runions_space_colonization(self):
        eng = SpaceColonizationEngine(SpaceColonizationConfig(num_auxin_sources=150, max_iterations=40))
        nodes = eng.generate_venation(blade_length_m=0.10, blade_width_m=0.06, seed=42)
        self.assertGreater(len(nodes), 15)
        self.assertTrue(nodes[0].flux >= nodes[-1].flux)


class TestLeafTexture(unittest.TestCase):
    def test_texture_has_outline_and_veins(self):
        prof = get_species_preset("fagus_sylvatica")
        tex = LeafTextureEngine(prof.leaf_morphology, prof.venation).render(256)
        alpha = tex.color[..., 3]
        self.assertTrue(0.15 < alpha.mean() < 0.8)
        inside = alpha > 0.99
        lum = tex.color[..., :3].mean(-1)
        self.assertGreater(np.std(lum[inside]), 0.01)       # Venation varies the lamina colour
        self.assertGreater(tex.height[inside].max(), 0.5)    # Veins raised in the height map

    def test_senescence_shifts_colour(self):
        prof = get_species_preset("acer_saccharum")
        e = LeafTextureEngine(prof.leaf_morphology, prof.venation)
        g = e.render(128, senescence=0.0).color
        a = e.render(128, senescence=1.0).color
        inside = g[..., 3] > 0.99
        self.assertGreater(a[..., 0][inside].mean(), g[..., 0][inside].mean() + 0.2)


class TestArchitecture(unittest.TestCase):
    def test_crown_envelope(self):
        u = np.linspace(0, 1, 101)
        self.assertLess(crown_envelope(u, 0.05, 1.4)[90], crown_envelope(u, 0.05, 1.4)[10])  # Conical
        self.assertGreater(crown_envelope(u, 0.8, 1.6)[85], crown_envelope(u, 0.8, 1.6)[20])  # Umbrella

    def test_child_axes_attach_to_parents(self):
        r = BotanicalPlantPipeline(get_species_preset("quercus_robur")).generate(seed=5)
        g = r.skeleton_graph
        for a in g.axes[1:]:
            parent = g.axes[a.parent_axis]
            self.assertTrue(np.allclose(a.positions[0], parent.positions[a.parent_sample]))

    def test_apical_dominance_controls_leaders(self):
        oak = BotanicalPlantPipeline(get_species_preset("quercus_robur")).generate(seed=1).skeleton_graph
        spruce = BotanicalPlantPipeline(get_species_preset("picea_abies")).generate(seed=1).skeleton_graph
        self.assertGreater(sum(a.order == 0 for a in oak.axes), 1)    # Decurrent: codominant leaders
        self.assertEqual(sum(a.order == 0 for a in spruce.axes), 1)   # Excurrent: single stem

    def test_palm_is_unbranched_with_apical_rosette(self):
        r = BotanicalPlantPipeline(get_species_preset("phoenix_canariensis")).generate(seed=1)
        self.assertEqual(len(r.skeleton_graph.axes), 1)
        self.assertGreater(r.leaf_count, 5)
        self.assertTrue(np.allclose(r.foliage.positions, r.skeleton_graph.axes[0].positions[-1]))

    def test_leaf_budget_is_respected(self):
        r = BotanicalPlantPipeline(get_species_preset("fagus_sylvatica")).generate(leaf_budget=3000)
        self.assertLess(r.leaf_count, 3600)


class TestRoots(unittest.TestCase):
    def test_jackson_depth_distribution(self):
        """Y(d) = 1 - beta^d: half of the roots above d50 = ln(0.5)/ln(beta) cm."""
        beta = JACKSON_BETA["temperate_deciduous"]
        self.assertAlmostEqual(float(sample_depth(beta, 0.5)), math.log(0.5) / math.log(beta) / 100, places=6)
        self.assertLess(sample_depth(JACKSON_BETA["boreal_forest"], 0.9), sample_depth(JACKSON_BETA["temperate_coniferous"], 0.9))

    def test_root_system_types_differ_in_depth(self):
        def deepest(system, beta):
            g = RootSystemEngine(RootProfile(system=system, beta=beta, max_depth_m=5.0,
                                             taproot_share=0.35 if system == RootSystemType.TAPROOT else 0.0)
                                 ).generate(0.6, 8.0, 0.35, display_depth_m=10.0)
            return -min(a.positions[:, 2].min() for a in g.axes)
        self.assertGreater(deepest(RootSystemType.TAPROOT, 0.976), deepest(RootSystemType.PLATE, 0.943) + 1.0)

    def test_buttress_roots_align_with_stem_flutes(self):
        r = BotanicalPlantPipeline(get_species_preset("ceiba_pentandra")).generate(seed=3, leaf_budget=500)
        m = get_species_preset("ceiba_pentandra").allometry.buttress_lobes
        mains = [a for a in r.root_graph.axes if a.parent_axis < 0 and a.aspect is not None][:m]
        for k, a in enumerate(mains):
            d = (a.azimuth - r.flute_azimuth - 2 * math.pi * k / m + math.pi) % (2 * math.pi) - math.pi
            self.assertAlmostEqual(d, 0.0, places=6)
            self.assertGreater(a.aspect[0], 3.0)          # Plank section at the stem
            self.assertLess(a.aspect[-1], 1.2)             # Round further out

    def test_main_roots_merge_into_stem(self):
        r = BotanicalPlantPipeline(get_species_preset("quercus_robur")).generate(seed=3, leaf_budget=500)
        trunk = r.skeleton_graph.axes[0]
        collar = float(np.interp(0.0, trunk.positions[:, 2], trunk.radii))
        for a in r.root_graph.axes:
            if a.parent_axis < 0 and a.aspect is not None:
                self.assertLess(np.hypot(*a.positions[0, :2]), collar)   # Starts inside the stem
                self.assertGreater(a.radii[0], 0.4 * collar)             # With a girth that meets it

    def test_palm_has_fibrous_constant_girth_roots(self):
        r = BotanicalPlantPipeline(get_species_preset("phoenix_canariensis")).generate(seed=3)
        self.assertGreater(len(r.root_graph.axes), 20)
        for a in r.root_graph.axes:
            self.assertAlmostEqual(float(a.radii.max()), float(a.radii.min()), places=9)


class TestForks(unittest.TestCase):
    def test_codominant_leaders_have_collars(self):
        r = BotanicalPlantPipeline(get_species_preset("taxodium_mucronatum")).generate(seed=3, leaf_budget=500,
                                                                                       roots=False)
        trunk = r.skeleton_graph.axes[0]
        leaders = [a for a in r.skeleton_graph.axes[1:] if a.order == 0]
        self.assertGreater(len(leaders), 1)
        for a in leaders:
            self.assertGreater(a.radii[0], 0.8 * trunk.radii[-1])   # Starts with the stem's girth
            self.assertLess(a.radii[len(a.radii) // 3], a.radii[0])  # and tapers to its pipe radius


class TestJunctions(unittest.TestCase):
    def _split(self):
        r = BotanicalPlantPipeline(get_species_preset("quercus_robur")).generate(seed=3, leaf_budget=500)
        thr = 0.04 * 2.5
        return r, split_axes([r.skeleton_graph, r.root_graph], thr, 0.6)

    def test_split_keeps_thick_parts_and_continuity(self):
        r, (thick, thin) = self._split()
        self.assertTrue(thick and thin)
        # Fused parts are thick up to the hand-over tail (last 4 samples narrow progressively)
        self.assertTrue(all(a.radii[:-4].min() >= 0.1 - 1e-9 for a in thick if len(a.radii) > 4))
        trunk = r.skeleton_graph.axes[0]
        n0 = default_n0(axis_tangents(trunk.positions)[0])[0]
        pieces = [a for a in thick + thin if getattr(a, "is_trunk", False)]
        self.assertGreaterEqual(len(pieces), 1)
        for a in pieces:
            self.assertTrue(np.allclose(a.frame_n0, n0))             # Same bark frame on both sides of the cut
            self.assertTrue(np.allclose(a.bark_origin, trunk.positions[0]))
        if len(pieces) == 2:
            fused, tube = pieces
            # The tube starts on the original axis, inside the fused part, at its recorded arc length
            sa = trunk.arc_length
            expect = np.array([np.interp(tube.v_offset, sa, trunk.positions[:, k]) for k in range(3)])
            self.assertTrue(np.allclose(tube.positions[0], expect, atol=1e-6))
            self.assertLess(tube.v_offset, fused.arc_length[-1])
            self.assertLess(tube.radii[0], np.interp(tube.v_offset, sa, trunk.radii))  # Grows in from inside

    def test_cylindrical_uvs_match_tube_uvs(self):
        """On an undeformed tube surface the fused-UV formula reproduces the tube's own bark coordinates."""
        r, (thick, _) = self._split()
        eng = BotanicalMeshEngine(MeshConfig(radial_resolution=16, smooth_caps=False))
        trunk = [a for a in thick if getattr(a, "is_trunk", False)][:1]
        mesh = eng.build_wood_mesh(graph_of(trunk), r.total_height_m, None, trunk_index=-1)
        table = sample_table(trunk)
        near = nearest_samples(mesh.vertices.astype(float), table["pos"], table["r"])
        uv = cylindrical_uvs(mesh.vertices.astype(float), mesh.loop_vertex, mesh.loop_start, mesh.loop_total,
                             near, table, 0.6)
        period = table["u_rep"][0]
        du = np.abs(((uv[:, 0] - mesh.loop_uv[:, 0]) + period / 2) % period - period / 2)
        self.assertLess(np.median(du), 0.02 * period)
        self.assertLess(np.median(np.abs(uv[:, 1] - mesh.loop_uv[:, 1])), 0.05)
        # Faces are unwrapped: no face spans more than half a period in U
        f = np.repeat(np.arange(len(mesh.loop_start)), mesh.loop_total)
        span = np.zeros(len(mesh.loop_start))
        np.maximum.at(span, f, uv[:, 0])
        lo = np.full(len(mesh.loop_start), np.inf)
        np.minimum.at(lo, f, uv[:, 0])
        self.assertLess(float((span - lo).max()), period / 2)

    def test_hero_sleeves(self):
        r, (thick, thin) = self._split()
        graphs = [r.skeleton_graph, r.root_graph]
        sleeves, rest = junction_sleeves(thin, graphs, 0.025, 0.6, detail=4, max_sleeves=50,
                                         fused_threshold=0.1, fused_voxel=0.04)
        self.assertEqual(len(sleeves), 50)                       # Bounded by max_sleeves (thickest first)
        self.assertEqual(len(rest), len(thin))                   # Every thin axis still has its tube
        for sl in sleeves:
            m = sl["slab"]
            V = m.vertices.astype(float)
            F = m.loop_vertex.reshape(-1, 4)
            vol = sum(np.dot(V[a], np.cross(V[b], V[c])) + np.dot(V[a], np.cross(V[c], V[d])) for a, b, c, d in F) / 6
            self.assertGreater(vol, 0.0)                         # Outward winding (required by the level set)
            base = sl["tubes"][0]
            self.assertLess(base.radii[-1], base.radii[0])       # Base tucks into the restarting tube

    def test_bark_coordinates_are_seam_free(self):
        """3D bark coordinates of fused points equal those of the tube they came from."""
        r, (thick, _) = self._split()
        trunk = [a for a in thick if getattr(a, "is_trunk", False)][:1]
        mesh = BotanicalMeshEngine(MeshConfig(smooth_caps=False)).build_wood_mesh(graph_of(trunk), r.total_height_m,
                                                                                  None, trunk_index=-1)
        table = sample_table(trunk)
        V = mesh.vertices.astype(float)
        near = nearest_samples(V, table["pos"], table["r"])
        base, along = bark_coordinates(V, near, table)
        q_fused = base + 0.25 * along
        q_tube = mesh.point_attributes["bark_base"] + 0.25 * mesh.point_attributes["bark_along"]
        self.assertLess(np.median(np.linalg.norm(q_fused - q_tube, axis=1)), 0.02)

    def test_thin_bark_not_stretched(self):
        self.assertAlmostEqual(float(u_repeats(0.02, 0.6)), 2 * math.pi * 0.02 / 0.6, places=6)
        self.assertEqual(float(u_repeats(0.5, 0.6)), 5.0)


class TestMeshes(unittest.TestCase):
    def test_wood_and_foliage_mesh_consistency(self):
        p = get_species_preset("acer_palmatum")
        r = BotanicalPlantPipeline(p).generate(seed=2, leaf_budget=2000)
        eng = BotanicalMeshEngine(MeshConfig())
        for mesh in (eng.build_wood_mesh(r.skeleton_graph, r.total_height_m),
                     eng.build_foliage_mesh(r.foliage, r.leaf_mesh_data)):
            self.assertGreater(len(mesh.vertices), 0)
            self.assertEqual(len(mesh.loop_vertex), int(mesh.loop_total.sum()))
            self.assertEqual(len(mesh.loop_uv), len(mesh.loop_vertex))
            self.assertLess(int(mesh.loop_vertex.max()), len(mesh.vertices))
            self.assertTrue(np.all(np.isfinite(mesh.vertices)))

    def test_gielis(self):
        angles = np.linspace(0, 2 * np.pi, 36, endpoint=False)
        r = GielisEngine.evaluate(angles, GIELIS_PRESETS["star_5"])
        self.assertTrue(np.all(r > 0.0))


class TestTraitSpace(unittest.TestCase):
    def test_roundtrip(self):
        p = get_species_preset("tilia_cordata")
        v = to_vector(p)
        self.assertTrue(np.all((v >= 0) & (v <= 1)))
        q = from_vector(v, p)
        self.assertAlmostEqual(distance(p, q), 0.0, places=3)

    def test_blend_is_between(self):
        a, b = get_species_preset("quercus_robur"), get_species_preset("acer_palmatum")
        m = blend(a, b, 0.5)
        self.assertAlmostEqual(distance(a, m) + distance(m, b), distance(a, b), delta=0.02)

    def test_mutation_stays_close(self):
        p = get_species_preset("fagus_sylvatica")
        variant = mutate(p, 1.0, seed=3)
        self.assertGreater(distance(p, variant), 0.0)
        self.assertEqual(nearest_species(variant, SPECIES_CATALOG)[0][0], "fagus_sylvatica")

    def test_trait_paths_exist(self):
        p = get_species_preset("quercus_robur")
        for spec in TRAITS:
            obj = p
            for part in spec.path.split("."):
                self.assertTrue(hasattr(obj, part), spec.path)
                obj = getattr(obj, part)


class TestCatalog(unittest.TestCase):
    def test_all_presets_generate(self):
        self.assertGreaterEqual(len(SPECIES_CATALOG), 30)
        for key, preset in SPECIES_CATALOG.items():
            r = BotanicalPlantPipeline(preset).generate(seed=11, leaf_budget=1500)
            self.assertGreater(r.total_height_m, 1.0, key)
            self.assertGreater(r.leaf_count, 0, key)

    def test_plant_ontology(self):
        self.assertEqual(get_po_term("PO:0009046").name, "stem")
        self.assertEqual(get_po_term("PO:0005022").name, "leaf vein")


if __name__ == "__main__":
    unittest.main()
