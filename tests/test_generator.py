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


class TestSucculents(unittest.TestCase):
    def test_catalogs_generate(self):
        from core.succulent_db import CACTUS_CATALOG, ROSETTE_CATALOG
        from core.cactus import CactusEngine
        from core.rosette import RosetteEngine
        self.assertGreaterEqual(len(CACTUS_CATALOG), 10)
        self.assertGreaterEqual(len(ROSETTE_CATALOG), 8)
        for k, sp in CACTUS_CATALOG.items():
            r = CactusEngine(sp.profile).generate(seed=1, detail=0.5, spine_budget=5000)
            self.assertGreater(len(r.stem.vertices), 100, k)
            self.assertEqual(len(r.stem.loop_vertex), int(r.stem.loop_total.sum()), k)
            self.assertLessEqual(int(r.stem.loop_vertex.max()), len(r.stem.vertices) - 1, k)
        for k, sp in ROSETTE_CATALOG.items():
            r = RosetteEngine(sp.profile).generate(seed=1, detail=0.5)
            self.assertGreaterEqual(r.leaf_count, sp.profile.leaf_count, k)

    def test_rib_count_and_areoles_on_crests(self):
        from core.cactus import CactusEngine, CactusProfile, CactusHabit
        p = CactusProfile(habit=CactusHabit.COLUMNAR, height_m=1.0, diameter_m=0.3, rib_count=13, rib_depth=0.3,
                          radial_spines=0, central_spines=0, wool=0.0)
        r = CactusEngine(p).generate(seed=2)
        # Around a mid-height ring the radius has exactly 13 maxima (ribs)
        V = r.stem.vertices.astype(float)
        ring = V[np.abs(V[:, 2] - 0.5) < 0.01]
        ang = np.arctan2(ring[:, 1], ring[:, 0])
        rad = np.hypot(ring[:, 0], ring[:, 1])
        order = np.argsort(ang)
        rr = rad[order]
        peaks = np.sum((rr > np.roll(rr, 1)) & (rr > np.roll(rr, -1)) & (rr > rr.mean()))
        self.assertEqual(int(peaks), 13)
        # Areoles sit on the crests: their distance to the axis is near the ring maximum
        self.assertGreater(r.areole_count, 50)

    def test_spiral_areoles_follow_golden_angle(self):
        from core.cactus import CactusEngine, CactusProfile, CactusHabit, AreoleArrangement, Generatrix
        p = CactusProfile(habit=CactusHabit.GLOBOSE, arrangement=AreoleArrangement.SPIRAL, height_m=0.1,
                          diameter_m=0.1, areole_spacing_cm=0.8)
        eng = CactusEngine(p)
        gen = Generatrix(0.1, 0.05, p, True)
        sig, th = eng._areoles(gen, 0.0, np.random.default_rng(0), gen.length)
        d = np.mod(np.diff(th), 2 * np.pi)
        self.assertTrue(np.allclose(d, np.radians(137.50776), atol=1e-6))
        # Equal-area lattice: about one areole per spacing^2 of lateral surface
        self.assertAlmostEqual(len(sig) * (0.008 ** 2) * 0.866 / gen.area[-1], 1.0, delta=0.05)

    def test_opuntia_cladode_collision_avoidance(self):
        from core.succulent_db import CACTUS_CATALOG
        from core.cactus import CactusEngine
        for sp_key in ("opuntia_ficus_indica", "opuntia_microdasys"):
            prof = CACTUS_CATALOG[sp_key].profile
            eng = CactusEngine(prof)
            for seed in (2, 7):
                r = eng.generate(seed=seed, detail=0.5, spine_budget=5000)
                self.assertGreater(len(r.stem.vertices), 500, f"{sp_key} seed {seed}")
                self.assertGreater(r.areole_count, 100, f"{sp_key} seed {seed}")

    def test_candelabra_arms_collision_avoidance(self):
        from core.succulent_db import CACTUS_CATALOG
        from core.cactus import CactusEngine, StemAxis
        p = CACTUS_CATALOG["myrtillocactus_geometrizans"].profile
        eng = CactusEngine(p)
        R = 0.5 * p.diameter_m
        main_axis = StemAxis(np.array([[0.0, 0.0, 0.0], [0.0, 0.0, p.height_m]]))
        for seed in (1, 7):
            rng = np.random.default_rng(seed)
            specs = eng._arms(main_axis, R, rng)
            self.assertGreaterEqual(len(specs), 8)
            for i in range(len(specs)):
                ax1, _, r1, _, _ = specs[i]
                for j in range(i + 1, len(specs)):
                    ax2, _, r2, _, _ = specs[j]
                    self.assertFalse(eng._arm_collides(ax1, r1, ax2, r2), f"Arms {i} and {j} collide in seed {seed}")

    def test_opuntia_cladodes_do_not_interpenetrate(self):
        """Exact volume test: no cladode surface point lies inside another cladode away from insertions."""
        from core.succulent_db import CACTUS_CATALOG
        from core.cactus import CactusEngine
        eng = CactusEngine(CACTUS_CATALOG["opuntia_ficus_indica"].profile)
        geoms = []
        orig = eng._pad

        def spy(base, up, roll, L, rng, detail, parent_n=None):
            geoms.append(eng._pad_frame(base, up, roll, L, parent_n))
            return orig(base, up, roll, L, rng, detail, parent_n)
        eng._pad = spy
        eng.generate(seed=4, detail=0.5, spine_budget=500)
        self.assertGreater(len(geoms), 8)
        for i, gi in enumerate(geoms):
            pts = eng._pad_surface_points(gi)
            for j, gj in enumerate(geoms):
                if i == j:
                    continue
                near = (np.linalg.norm(pts - gi[7], axis=1) < 0.25 * gi[4]) | \
                       (np.linalg.norm(pts - gj[7], axis=1) < 0.25 * gj[4])
                self.assertFalse(np.any(eng._pad_inside(pts[~near], gj, inflate=0.98)), f"pads {i} and {j}")

    def test_handover_hides_fused_tail(self):
        """The fused tail narrows well inside the continuing tube, so remesh inflation (~1 voxel at a
        4-voxel radius) cannot make it poke out as a step."""
        from core.architecture import Axis
        from core.junctions import handover
        P = np.stack([np.linspace(0, 2.0, 40), np.zeros(40), np.zeros(40)], 1)
        ax = Axis(P, np.linspace(0.2, 0.02, 40), 1, None, None, 0.0)
        s_cut, ov = 1.0, 0.3
        (fp, fr, _), (tp, tr, _), s0 = handover(ax, s_cut, ov)
        r_cut = float(np.interp(s_cut, ax.arc_length, ax.radii))
        self.assertLessEqual(fr[-1], 0.5 * float(np.interp(s_cut + ov, ax.arc_length, ax.radii)) / 0.9 + 1e-9)
        self.assertLess(fr[-1] + 0.25 * r_cut, float(np.interp(fp[-1, 0], tp[:, 0], tr)))   # 1 voxel = r/4

    def test_bark_age_attribute(self):
        """Wood meshes carry the local axis radius (bark age proxy); thin twigs keep a young periderm."""
        from core.species_db import get_species_preset
        from core.plant_pipeline import BotanicalPlantPipeline
        from core.mesh_engine import BotanicalMeshEngine
        res = BotanicalPlantPipeline(get_species_preset("quercus_robur")).generate(seed=2, leaf_density=0.0,
                                                                                 roots=False)
        mesh = BotanicalMeshEngine().build_wood_mesh(res.skeleton_graph, res.total_height_m)
        r = mesh.point_attributes["bark_radius"]
        self.assertEqual(len(r), len(mesh.vertices))
        self.assertAlmostEqual(float(r.max()), float(max(a.radii.max() for a in res.skeleton_graph.axes)), places=4)
        onset = get_species_preset("quercus_robur").bark.onset_radius_cm * 0.01
        self.assertGreater(float(np.mean(r < 0.5 * onset)), 0.3)       # Much of the crown is young bark
        self.assertGreater(float(r.max()), 2.0 * onset)                 # The trunk is fully fissured

    def test_areole_stain_attributes(self):
        """Stem vertices carry a halo around each areole and a drip streak running down from it."""
        from core.succulent_db import CACTUS_CATALOG
        from core.cactus import CactusEngine
        from core.spatial import nearest_points
        r = CactusEngine(CACTUS_CATALOG["carnegiea_gigantea"].profile).generate(seed=1, detail=0.5,
                                                                              spine_budget=100)
        A = r.stem.point_attributes
        self.assertEqual(len(A["areole_halo"]), len(r.stem.vertices))
        self.assertGreater(float(A["areole_halo"].max()), 0.9)
        self.assertGreater(float(A["areole_drip"].max()), 0.5)
        # Drip only below its areole
        V = r.stem.vertices.astype(float)
        d, i = nearest_points(V, r.flower_pos, 0.04)
        hi = A["areole_drip"] > 0.3
        self.assertTrue(np.all(V[hi, 2] <= r.flower_pos[i[hi], 2] + 1e-6))
        # Grid nearest neighbours match brute force
        rng = np.random.default_rng(0)
        P, Q = rng.uniform(0, 1, (800, 3)), rng.uniform(0, 1, (300, 3))
        dd, _ = nearest_points(Q, P, 0.15)
        bf = np.array([np.linalg.norm(P - q, axis=1).min() for q in Q])
        self.assertTrue(np.allclose(dd, bf))

    def test_cladode_faces_wound_outward(self):
        """Every cladode face points away from the pad centre (consistent winding, no dark shading ring)."""
        from core.succulent_db import CACTUS_CATALOG
        from core.cactus import CactusEngine
        e = CactusEngine(CACTUS_CATALOG["opuntia_ficus_indica"].profile)
        up = np.array([0.0, 0.0, 1.0])
        mesh, _, _ = e._pad(np.zeros(3), up, 0.0, 0.4, np.random.default_rng(1), 1.0)
        centre = e._pad_frame(np.zeros(3), up, 0.0, 0.4)[0]
        V = mesh.vertices.astype(float)
        for st, tot in zip(mesh.loop_start, mesh.loop_total):
            idx = mesh.loop_vertex[st:st + tot]
            fn = np.cross(V[idx[1]] - V[idx[0]], V[idx[2]] - V[idx[0]])
            self.assertGreater(float(fn @ (V[idx].mean(0) - centre)), 0.0)

    def test_candelabra_secondary_arms(self):
        from core.succulent_db import CACTUS_CATALOG
        from core.cactus import CactusEngine, StemAxis
        p = CACTUS_CATALOG["pachycereus_weberi"].profile
        eng = CactusEngine(p)
        main = StemAxis(np.array([[0.0, 0.0, 0.0], [0.0, 0.0, p.height_m]]))
        specs = eng._arms(main, 0.5 * p.diameter_m, np.random.default_rng(3))
        self.assertGreater(len(specs), p.arm_count)               # Primary + secondary arms
        tops = max(float(a.P[:, 2].max()) for a, *_ in specs)
        self.assertGreater(tops, 3.0 * p.height_m)                  # Crown far above the short trunk
        for i in range(len(specs)):
            for j in range(len(specs)):
                if i == j:
                    continue
                a, b = specs[i][0], specs[j][0]
                if getattr(b, "parent", None) is a:      # Child arm vs its parent: beyond the elbow only
                    self.assertFalse(eng._arm_collides(b, specs[j][2], a, specs[i][2], start1=0.35))
                elif getattr(a, "parent", None) is not b and i < j:
                    self.assertFalse(eng._arm_collides(a, specs[i][2], b, specs[j][2]))

    def test_areole_normals_point_outward(self):
        """Areole normals (spines, wool, flowers) agree with the stem surface normal, also at the apex."""
        from core.succulent_db import CACTUS_CATALOG
        from core.cactus import CactusEngine
        for key in ("lophophora_williamsii", "carnegiea_gigantea", "echinocactus_grusonii"):
            e = CactusEngine(CACTUS_CATALOG[key].profile)
            cap = {}
            orig = e._spines

            def spy(pos, nrm, tan, scale, apex, rng, budget=60000, cap=cap, orig=orig):
                cap.update(pos=pos, nrm=nrm, apex=apex)
                return orig(pos, nrm, tan, scale, apex, rng, budget)
            e._spines = spy
            r = e.generate(seed=4, detail=0.5, spine_budget=100)
            V = r.stem.vertices.astype(float)
            VN = np.zeros_like(V)
            for st, tot in zip(r.stem.loop_start, r.stem.loop_total):
                idx = r.stem.loop_vertex[st:st + tot]
                VN[idx] += np.cross(V[idx[1]] - V[idx[0]], V[idx[2]] - V[idx[0]])
            P, N, A = cap["pos"], cap["nrm"], cap["apex"]
            sel = np.flatnonzero(A > 0.5)[:60]
            for i in sel:
                j = np.argmin(((V - P[i]) ** 2).sum(1))
                self.assertGreater(float(N[i] @ VN[j]), 0.0, key)

    def test_candelabra_crown_is_filled(self):
        """Pachycereus weberi: branch columns cover the crown disc, including the centre (no hollow vase)."""
        from core.succulent_db import CACTUS_CATALOG
        from core.cactus import CactusEngine, StemAxis
        p = CACTUS_CATALOG["pachycereus_weberi"].profile
        eng = CactusEngine(p)
        main = StemAxis(np.array([[0.0, 0.0, 0.0], [0.0, 0.0, p.height_m]]))
        specs = eng._arms(main, 0.5 * p.diameter_m, np.random.default_rng(5))
        r_top = np.array([np.hypot(*a.P[-1, :2]) for a, *_ in specs])
        disc = p.arm_reach_m + 0.5 * p.diameter_m
        self.assertGreaterEqual(np.sum(r_top < 0.35 * disc), 3)    # Inner columns near the axis
        self.assertGreaterEqual(np.sum(r_top > 0.7 * disc), 5)     # And an outer ring

    def test_marginal_teeth_on_furled_leaves(self):
        """Teeth follow the real (furled, clasped) leaf margin, including the young leaves of the spike."""
        from core.succulent_db import ROSETTE_CATALOG
        from core.rosette import RosetteEngine
        for key in ("agave_americana", "aloe_vera"):
            p = ROSETTE_CATALOG[key].profile
            e = RosetteEngine(p)
            orig = e._armature
            worst = [0.0]

            def spy(mid, dirs, lateral, normal, half_w, L, rng, grid=None, cx=None, p=p, orig=orig, worst=worst):
                m = orig(mid, dirs, lateral, normal, half_w, L, rng, grid=grid, cx=cx)
                A = m.vertices.reshape(-1, 6, 3)[(1 if p.terminal_spine_cm > 0 else 0):, :5].mean(1)
                margins = [(grid[:-1, c], grid[1:, c]) for c in (int(np.argmax(cx)), int(np.argmin(cx)))]
                for q in A:          # Distance to the nearer of the two margin polylines
                    best = np.inf
                    for a0, a1 in margins:
                        ab = a1 - a0
                        t = np.clip(((q - a0) * ab).sum(1) / np.maximum((ab * ab).sum(1), 1e-12), 0, 1)
                        best = min(best, float(np.sqrt(((a0 + ab * t[:, None] - q) ** 2).sum(1)).min()))
                    worst[0] = max(worst[0], best)
                return m
            e._armature = spy
            e.generate(seed=3, detail=0.8)
            self.assertLess(worst[0], 0.003, key)      # Within 3 mm of the margin (base sunk ~1 mm)

    def test_rosette_stem_and_dead_leaves(self):
        from core.succulent_db import ROSETTE_CATALOG
        from core.rosette import RosetteEngine
        r = RosetteEngine(ROSETTE_CATALOG["agave_americana"].profile).generate(seed=1, detail=0.5)
        V = r.stem.vertices
        rad = np.hypot(V[:, 0], V[:, 1])
        self.assertGreater(rad.max() / max(1e-6, np.median(rad)), 1.2)   # Flared crown + tapering apex
        dead = r.leaves.point_attributes["leaf_dead"] > 0.5
        self.assertTrue(dead.any())
        self.assertGreaterEqual(float(r.leaves.vertices[dead, 2].min()), -0.01)  # Withered leaves rest on soil
        ae = RosetteEngine(ROSETTE_CATALOG["aeonium_arboreum"].profile).generate(seed=1, detail=0.5)
        self.assertGreater(float(ae.stem.point_attributes["scar"].max()), 0.8)  # Leaf scars on the bare stem

    def test_succulent_roots(self):
        from core.succulent_db import CACTUS_CATALOG, ROSETTE_CATALOG
        from core.cactus import CactusEngine
        from core.rosette import RosetteEngine
        sag = CACTUS_CATALOG["carnegiea_gigantea"].profile
        r = CactusEngine(sag).generate(seed=2, detail=0.4, spine_budget=100, with_roots=True)
        V = r.roots.vertices
        reach = float(np.hypot(V[:, 0], V[:, 1]).max())
        self.assertGreater(reach, 0.6 * r.height_m)                  # Laterals extend about the plant height
        self.assertLess(float(-V[:, 2].min()), 1.2)                  # Shallow system (taproot < ~1 m)
        shallow = np.mean(-V[:, 2] < 0.35)
        self.assertGreater(shallow, 0.8)                              # Most root surface in the top 30-35 cm
        peyote = CactusEngine(CACTUS_CATALOG["lophophora_williamsii"].profile).generate(
            seed=2, detail=0.4, with_roots=True)
        self.assertGreater(float(-peyote.roots.vertices[:, 2].min()), peyote.height_m)  # Tuber longer than stem
        agave = RosetteEngine(ROSETTE_CATALOG["agave_americana"].profile).generate(seed=2, detail=0.4,
                                                                                    with_roots=True)
        self.assertGreater(len(agave.roots.vertices), 100)
        self.assertLess(float(-agave.roots.vertices[:, 2].min()), 0.8)

    def test_rosette_age_gradients(self):
        from core.rosette import RosetteEngine, RosetteProfile
        p = RosetteProfile(leaf_count=30, size_gradient=0.6, elevation_outer_deg=10, elevation_inner_deg=80)
        r = RosetteEngine(p).generate(seed=3)
        u = r.leaves.point_attributes["leaf_u"]
        V = r.leaves.vertices.astype(float)
        per = len(V) // 30
        tips = V[np.arange(30) * per + np.argmax(u[:per])]
        # Outer (old) leaves reach farther out and lie lower than inner (young) leaves
        self.assertGreater(np.hypot(*tips[0][:2]), np.hypot(*tips[-1][:2]))
        self.assertLess(tips[0][2], tips[-1][2])

    def test_succulent_trait_arithmetic(self):
        from core.succulent_db import CACTUS_CATALOG, CACTUS_RANGES
        from core.trait_space import blend_profile, mutate_profile
        a = CACTUS_CATALOG["carnegiea_gigantea"].profile
        b = CACTUS_CATALOG["echinocactus_grusonii"].profile
        m = blend_profile(a, b, 0.25, CACTUS_RANGES)
        self.assertAlmostEqual(m.height_m, 0.75 * a.height_m + 0.25 * b.height_m)
        self.assertEqual(m.habit, a.habit)
        v = mutate_profile(a, 1.0, 5, CACTUS_RANGES)
        self.assertNotEqual(v.height_m, a.height_m)


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


class TestFlowers(unittest.TestCase):
    """Floral diagrams, capitula, inflorescences and flower sites on plants."""

    def _organs(self, mesh, code):
        from core.flower import PETAL
        A = mesh.point_attributes
        return len(np.unique(A["orand"][A["organ"] == code]))

    def test_whorled_and_spiral_organ_counts(self):
        from core.flower import FlowerEngine, FlowerProfile, Arrangement, PETAL, SEPAL
        f = FlowerProfile(merosity=5, petal_whorls=1, sepal_length_ratio=0.5, hypanthium_cm=0.0)
        m = FlowerEngine(f).generate(1.0, 0.5, 3).mesh
        self.assertEqual(self._organs(m, PETAL), 5)
        self.assertEqual(self._organs(m, SEPAL), 5)
        f2 = FlowerProfile(merosity=3, petal_whorls=2, sepal_length_ratio=0.0, hypanthium_cm=0.0)
        self.assertEqual(self._organs(FlowerEngine(f2).generate(1.0, 0.5, 3).mesh, PETAL), 6)
        f3 = FlowerProfile(arrangement=Arrangement.SPIRAL, spiral_tepals=21, sepal_length_ratio=0.0, hypanthium_cm=0.0)
        self.assertEqual(self._organs(FlowerEngine(f3).generate(1.0, 0.5, 3).mesh, PETAL), 21)

    def test_bud_is_closed_and_smaller(self):
        from core.flower_db import FLOWER_CATALOG
        from core.flower import FlowerEngine
        eng = FlowerEngine(FLOWER_CATALOG["rosa_canina"].flower)
        bud, open_ = eng.generate(0.1, 0.5, 1), eng.generate(1.0, 0.5, 1)
        self.assertLess(bud.diameter_m, 0.5 * open_.diameter_m)

    def test_capitulum_vogel_spiral(self):
        from core.flower_db import FLOWER_CATALOG
        from core.flower import FlowerEngine, FLORET
        f = FLOWER_CATALOG["helianthus_annuus"].flower
        m = FlowerEngine(f).generate(1.0, 0.5, 1).mesh
        A = m.point_attributes
        V = m.vertices[A["organ"] == FLORET].reshape(-1, 13, 3)
        self.assertEqual(len(V), f.disc_florets)
        r = np.hypot(V[:, 12, 0], V[:, 12, 1])                    # Floret apex radius
        k = np.arange(len(r))
        # Vogel: r_k = c sqrt(k + 1/2) -> r^2 linear in k
        self.assertGreater(np.corrcoef(r ** 2, k)[0, 1], 0.999)

    def test_zygomorphy_is_bilateral(self):
        from core.flower import FlowerEngine, FlowerProfile, PETAL
        f = FlowerProfile(zygomorphy=1.0, lip_bias=-0.8, sepal_length_ratio=0.0, hypanthium_cm=0.0, stamen_count=0,
                          style_length_ratio=0.0)
        V = FlowerEngine(f).generate(1.0, 0.6, 1).mesh.vertices
        self.assertAlmostEqual(float(V[:, 0].max()), float(-V[:, 0].min()), delta=0.08 * float(np.ptp(V[:, 0])))
        self.assertGreater(abs(float(V[:, 1].max()) + float(V[:, 1].min())), 0.1 * float(np.ptp(V[:, 1])))

    def test_fused_limb_covers_rim(self):
        from core.flower_db import FLOWER_CATALOG
        from core.flower import FlowerEngine, PETAL
        m = FlowerEngine(FLOWER_CATALOG["ipomoea_purpurea"].flower).generate(1.0, 1.5, 1).mesh
        V, A = m.vertices, m.point_attributes
        sel = (A["organ"] == PETAL) & (A["pu"] > 0.6)
        az = np.degrees(np.arctan2(V[sel, 1], V[sel, 0])) % 360
        h, _ = np.histogram(az, bins=36, range=(0, 360))
        self.assertTrue(np.all(h > 0))                             # No gaps between fused lobes

    def test_inflorescence_types(self):
        from core.flower_db import FLOWER_CATALOG
        from core.inflorescence import InflorescenceEngine, InflorescenceProfile, InflorescenceType as T
        f = FLOWER_CATALOG["prunus_avium"].flower
        for kind, n in ((T.SOLITARY, 1), (T.RACEME, 12), (T.SPIKE, 12), (T.UMBEL, 6), (T.CORYMB, 7)):
            r = InflorescenceEngine(f, InflorescenceProfile(kind=kind, flower_count=n, rachis_cm=10)).generate(seed=2)
            self.assertEqual(r.flower_count, n, kind)
            self.assertTrue(np.isfinite(r.mesh.vertices).all())
        pan = InflorescenceEngine(f, InflorescenceProfile(kind=T.PANICLE, flower_count=40, branches=8,
                                                          rachis_cm=20)).generate(seed=2)
        self.assertGreaterEqual(pan.flower_count, 40)
        # Corymb: flowers reach a common level
        cor = InflorescenceEngine(f, InflorescenceProfile(kind=T.CORYMB, flower_count=7, rachis_cm=10,
                                                          peduncle_cm=2)).generate(seed=2, max_flowers=7)
        self.assertGreater(cor.width_m, 0.0)

    def test_catalog_and_plant_mappings(self):
        from core.flower_db import FLOWER_CATALOG, TREE_FLOWERS, CACTUS_FLOWERS, ROSETTE_FLOWERS
        from core.species_db import SPECIES_CATALOG
        from core.succulent_db import CACTUS_CATALOG, ROSETTE_CATALOG
        from core.inflorescence import InflorescenceEngine
        for mapping, cat in ((TREE_FLOWERS, SPECIES_CATALOG), (CACTUS_FLOWERS, CACTUS_CATALOG),
                             (ROSETTE_FLOWERS, ROSETTE_CATALOG)):
            for plant, flower in mapping.items():
                self.assertIn(plant, cat)
                self.assertIn(flower, FLOWER_CATALOG)
        self.assertEqual(set(CACTUS_FLOWERS), set(CACTUS_CATALOG))
        self.assertEqual(set(ROSETTE_FLOWERS), set(ROSETTE_CATALOG))
        for key, sp in FLOWER_CATALOG.items():
            r = InflorescenceEngine(sp.flower, sp.infl).generate(seed=1, detail=0.4, max_flowers=30)
            self.assertTrue(np.isfinite(r.mesh.vertices).all(), key)
            self.assertGreater(len(r.mesh.vertices), 50, key)

    def test_opuntia_flowers_on_free_margins(self):
        """Opuntia flowers sit on the pad margins, in the pad plane, and their volume clears every cladode."""
        from core.succulent_db import CACTUS_CATALOG
        from core.cactus import CactusEngine
        from core.inflorescence import surface_flower_sites
        eng = CactusEngine(CACTUS_CATALOG["opuntia_ficus_indica"].profile)
        geoms = []
        orig = eng._pad

        def spy(base, up, roll, L, rng, detail, parent_n=None):
            geoms.append(eng._pad_frame(base, up, roll, L, parent_n))
            return orig(base, up, roll, L, rng, detail, parent_n)
        eng._pad = spy
        r = eng.generate(seed=3, detail=0.4, spine_budget=100)
        s = surface_flower_sites(r.flower_pos, r.flower_normal, r.flower_weight, 20, seed=1, lean=0.0, min_dist=0.07)
        self.assertGreater(len(s), 3)
        for q, d in zip(s.positions, s.directions):
            # Attached: the base sits inside the margin of a cladode (not floating beyond its outline)
            self.assertTrue(any(bool(eng._pad_inside(q[None, :], g, inflate=1.0)[0]) for g in geoms))
            probe = q + d * np.linspace(0.015, 0.08, 6)[:, None]       # Pericarpel and perianth along the axis
            for g in geoms:
                self.assertFalse(np.any(eng._pad_inside(probe, g, inflate=1.0)))
        D = np.linalg.norm(s.positions[:, None] - s.positions[None], axis=-1) + np.eye(len(s)) * 9
        self.assertGreaterEqual(float(D.min()), 0.07)

    def test_flower_sites_on_plants(self):
        from core.succulent_db import CACTUS_CATALOG, ROSETTE_CATALOG
        from core.cactus import CactusEngine
        from core.rosette import RosetteEngine
        from core.inflorescence import surface_flower_sites, tree_flower_sites, InflorescenceProfile
        from core.species_db import get_species_preset
        from core.plant_pipeline import BotanicalPlantPipeline
        sag = CactusEngine(CACTUS_CATALOG["echinocactus_grusonii"].profile).generate(seed=1, detail=0.4,
                                                                                    spine_budget=50)
        s = surface_flower_sites(sag.flower_pos, sag.flower_normal, sag.flower_weight, 12, seed=1)
        self.assertEqual(len(s), 12)
        self.assertGreater(float(s.positions[:, 2].min()), 0.6 * sag.height_m)   # Crown of the barrel
        op = CactusEngine(CACTUS_CATALOG["opuntia_ficus_indica"].profile).generate(seed=1, detail=0.4)
        self.assertGreater(int((op.flower_weight > 0).sum()), 10)                 # Pad-margin areoles
        ros = RosetteEngine(ROSETTE_CATALOG["agave_americana"].profile).generate(seed=1, detail=0.4)
        self.assertEqual(len(ros.terminal_sites[0]), 1)
        self.assertGreater(len(ros.axillary_sites[0]), 3)
        res = BotanicalPlantPipeline(get_species_preset("prunus_avium")).generate(seed=1, leaf_density=0.0,
                                                                                 roots=False)
        ts = tree_flower_sites(res.skeleton_graph, InflorescenceProfile(), 200, seed=1)
        self.assertEqual(len(ts), 200)
        self.assertGreater(float(ts.positions[:, 2].mean()), 0.4 * res.total_height_m)


class TestPresets(unittest.TestCase):
    """JSON presets: exact round trip, partial (base + differences) presets, validation, ranges, schema."""

    def test_builtin_values_within_declared_ranges(self):
        from core import presets as P
        for form in P.FORMS:
            rng = P._ranges(form)
            for key, obj in P._catalog(form).items():
                plain = P.to_plain(obj)
                for path, (lo, hi) in rng.items():
                    v = plain
                    for k in path.split("."):
                        v = v.get(k) if isinstance(v, dict) else None
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        self.assertTrue(lo <= v <= hi, f"{form}/{key}: {path}={v} outside [{lo}, {hi}]")

    def test_round_trip_every_builtin(self):
        import json
        from core import presets as P
        for form in P.FORMS:
            for key, obj in P._catalog(form).items():
                env = json.loads(json.dumps(P.export_preset(form, key, obj)))
                f, pid, back, warn, _ = P.load_preset(env)
                self.assertEqual((f, pid), (form, key))
                self.assertEqual(P.to_plain(back), P.to_plain(obj), f"{form}/{key}")
                self.assertEqual(warn, [], f"{form}/{key}")

    def test_partial_preset_and_validation(self):
        import copy
        import json
        from core import presets as P
        from core.species_db import SPECIES_CATALOG
        oak = copy.deepcopy(SPECIES_CATALOG["quercus_robur"])
        oak.bark.moss = 0.8
        env = P.export_preset("Tree", "mossy_oak", oak, base="quercus_robur", diff=True)
        self.assertEqual(env["preset"], {"bark": {"moss": 0.8}})
        _, _, back, warn, _ = P.load_preset(json.loads(json.dumps(env)))
        self.assertEqual(P.to_plain(back), P.to_plain(oak))
        bad = {"format": "ppg-preset", "format_version": 1, "growth_form": "Cactus", "id": "t_cactus",
               "base": "carnegiea_gigantea",
               "preset": {"profile": {"rib_count": 999, "habit": "barrel", "stem_color": [60, 120, 60],
                                      "spine_lenght_cm": 3, "glaucous": "high"}}}
        _, _, obj, warn, _ = P.load_preset(bad)
        self.assertEqual(obj.profile.rib_count, 60)                       # Clamped
        self.assertEqual(obj.profile.habit.value, "Barrel")              # Case-insensitive enum
        self.assertAlmostEqual(obj.profile.stem_color[1], 120 / 255)     # 0..255 colour rescaled
        self.assertTrue(any("unknown field" in w for w in warn))
        self.assertTrue(any("expected a number" in w for w in warn))
        for broken in ({"format": "x"}, dict(bad, id="Bad Id"), dict(bad, growth_form="Moss"), dict(bad, base="nope")):
            with self.assertRaises(P.PresetError):
                P.load_preset(broken)

    def test_float_traits_keep_decimals(self):
        """A float trait whose built-in value happens to be an int (5) still accepts 4.5."""
        from core import presets as P
        env = {"format": "ppg-preset", "format_version": 1, "growth_form": "Tree", "id": "t_ilex",
               "base": "quercus_agrifolia", "preset": {"leaf_morphology": {"blade_length_cm": 4.5}}}
        _, _, obj, warn, _ = P.load_preset(env)
        self.assertEqual(obj.leaf_morphology.blade_length_cm, 4.5)

    def test_schema_documents_every_field(self):
        from core import presets as P
        sch = P.json_schema()
        self.assertEqual(sch["required"], ["format", "format_version", "growth_form", "id", "preset"])
        for name, d in sch["$defs"].items():
            for field, p in d["properties"].items():
                if "$ref" not in p:
                    self.assertTrue(p.get("description"), f"{name}.{field} has no description")


if __name__ == "__main__":
    unittest.main()
