"""
Blender mesh and object construction.

Writes core.mesh_engine.MeshData straight into Blender meshes with foreach_set
(no per-vertex Python lists) and updates the plant objects in place.
"""

import numpy as np

try:
    import bpy
    BLENDER_AVAILABLE = True
except ImportError:
    BLENDER_AVAILABLE = False

try:
    from ..core.mesh_engine import BotanicalMeshEngine, MeshConfig, MeshData, collar_bark_age
    from ..core.gielis import GielisProfile
except (ImportError, ValueError):
    from core.mesh_engine import BotanicalMeshEngine, MeshConfig, MeshData, collar_bark_age
    from core.gielis import GielisProfile


def populate_mesh(mesh: "bpy.types.Mesh", data: MeshData, smooth: bool = True):
    """Replaces a mesh's geometry with MeshData using bulk buffer writes."""
    mesh.clear_geometry()
    n_v, n_l, n_f = len(data.vertices), len(data.loop_vertex), len(data.loop_start)
    if n_v == 0:
        mesh.update()
        return
    mesh.vertices.add(n_v)
    mesh.vertices.foreach_set("co", data.vertices.astype(np.float32).ravel())
    mesh.loops.add(n_l)
    mesh.loops.foreach_set("vertex_index", data.loop_vertex.astype(np.int32))
    mesh.polygons.add(n_f)
    mesh.polygons.foreach_set("loop_start", data.loop_start.astype(np.int32))
    try:
        mesh.polygons.foreach_set("loop_total", data.loop_total.astype(np.int32))
    except (AttributeError, TypeError, RuntimeError):
        pass  # Derived from loop_start in Blender 4.0+
    mesh.update(calc_edges=True)

    uv_layer = mesh.uv_layers.get("UVMap") or mesh.uv_layers.new(name="UVMap")
    uv_layer.data.foreach_set("uv", data.loop_uv.astype(np.float32).ravel())

    for name, values in data.point_attributes.items():
        values = np.asarray(values)
        if values.ndim == 2:
            kind, field = 'FLOAT_VECTOR', "vector"
        elif values.dtype.kind in "iu":
            kind, field = 'INT', "value"
        else:
            kind, field = 'FLOAT', "value"
        attr = mesh.attributes.get(name)
        if attr is None or attr.data_type != kind or attr.domain != 'POINT':
            if attr is not None:
                mesh.attributes.remove(attr)
            attr = mesh.attributes.new(name=name, type=kind, domain='POINT')
        attr.data.foreach_set(field, values.astype(np.int32 if kind == 'INT' else np.float32).ravel())

    if smooth:
        mesh.polygons.foreach_set("use_smooth", np.ones(n_f, dtype=bool))
    mesh.update()


class BlenderMeshBuilder:
    """Builds and updates Blender objects from procedural botanical data."""

    def __init__(self, plant_result, radial_resolution: int = 12, buttress_profile: GielisProfile = None,
                 twig_resolution: int = None):
        self.result = plant_result
        self.preset = plant_result.preset
        al = self.preset.allometry
        self.config = MeshConfig(
            radial_resolution=radial_resolution,
            twig_resolution=twig_resolution or max(4, radial_resolution // 2),
            collar_flare_factor=1.0,
            buttress_profile=buttress_profile,
            flare_amplitude=al.buttress_amplitude,
            flare_decay=al.buttress_decay,
        )
        self.mesh_engine = BotanicalMeshEngine(self.config)

    def _find_root(self, context, root_name):
        active_root_name = context.scene.get("ppg_active_root_name")
        root_obj = bpy.data.objects.get(active_root_name) if active_root_name else None
        if root_obj is None or root_obj.name not in context.scene.objects:
            active = getattr(context, "active_object", None)
            if active is not None and active.name.startswith("PPG_"):
                while active.parent and active.parent.name.startswith("PPG_"):
                    active = active.parent
                root_obj = active
            else:
                root_obj = None
        if root_obj is None or root_obj.name not in context.scene.objects:
            root_obj = bpy.data.objects.new(root_name, None)
            root_obj.empty_display_type = 'PLAIN_AXES'
            context.scene.collection.objects.link(root_obj)
            cursor = context.scene.cursor.location
            root_obj.location = cursor
        elif not root_obj.name.startswith(root_name):
            root_obj.name = root_name
        context.scene["ppg_active_root_name"] = root_obj.name
        return root_obj

    @staticmethod
    def _child(context, root_obj, suffix, name):
        for child in root_obj.children:
            if child.name.endswith(suffix) or suffix in child.name:
                if child.name != name:
                    child.name = name
                    child.data.name = name
                return child
        mesh = bpy.data.meshes.new(name)
        obj = bpy.data.objects.new(name, mesh)
        context.scene.collection.objects.link(obj)
        obj.parent = root_obj
        return obj

    def build_or_update_plant(self, context, existing_root=None, leaf_density: float = 1.0,
                              leaf_scale: float = 1.0, show_leaves: bool = True, use_subsurf: bool = False,
                              show_roots: bool = True, junction_quality: str = 'FUSED',
                              fuse_detail: float = 10.0, fuse_smoothing: int = 6, hero_min_radius: float = 0.025,
                              sleeve_detail: float = 4.0, max_sleeves: int = 400, foliage_instanced: bool = False) -> dict:
        if not BLENDER_AVAILABLE:
            raise RuntimeError("Blender (bpy) is not available.")
        root_name = f"PPG_{self.preset.scientific_name.split(' (')[0].replace(' ', '_').replace(chr(39), '')}"
        root_obj = existing_root or self._find_root(context, root_name)
        for c in root_obj.children:   # Objects of the succulent growth forms on a reused root
            if c.name.endswith(("_Stem", "_Spines", "_Leaves", "_Armature", "_RosetteBase", "_SuccRoots")):
                c.hide_viewport = c.hide_render = True

        # Wood (stem flutes aligned with the main roots)
        self.config.flute_azimuth = getattr(self.result, "flute_azimuth", None)
        root_graph = getattr(self.result, "root_graph", None) if show_roots else None
        fuse_junctions = junction_quality in ('FUSED', 'HERO')
        if fuse_junctions:
            # Stem, limbs and roots fused into one continuous surface at their junctions
            from .junctions import build_fused_wood
            wood_data = build_fused_wood(self.mesh_engine, [self.result.skeleton_graph, root_graph],
                                         self.result.total_height_m, self.config.buttress_profile,
                                         detail=fuse_detail, smooth_iterations=fuse_smoothing,
                                         quality=junction_quality, hero_min_radius=hero_min_radius,
                                         sleeve_detail=sleeve_detail, max_sleeves=max_sleeves)
        else:
            wood_data = self.mesh_engine.build_wood_mesh(self.result.skeleton_graph, self.result.total_height_m,
                                                         self.config.buttress_profile)
        # Bark at the root collar: roots and stem base share the age of the collar
        trunk = self.result.skeleton_graph.axes[0]
        z = trunk.positions[:, 2]
        collar = np.array([np.interp(0.0, z, trunk.positions[:, k]) for k in range(3)])
        r_collar = float(np.interp(0.0, z, trunk.radii))
        zone = getattr(getattr(self.result.preset, "roots", None), "zrt_dbh_ratio", 2.2) * self.result.dbh_m
        wood_data = collar_bark_age(wood_data, collar, r_collar, zone)
        wood_obj = self._child(context, root_obj, "_Wood", f"{root_obj.name}_Wood")
        populate_mesh(wood_obj.data, wood_data)
        wood_obj.hide_viewport = wood_obj.hide_render = False
        self._assign_vertex_groups(wood_obj, wood_data.point_attributes.get("branch_order"))
        mod = wood_obj.modifiers.get("PPG_Subsurf")
        if use_subsurf and mod is None:
            mod = wood_obj.modifiers.new(name="PPG_Subsurf", type='SUBSURF')
            mod.levels = 1
            mod.render_levels = 2
        elif not use_subsurf and mod is not None:
            wood_obj.modifiers.remove(mod)

        # Roots
        roots_obj = self._child(context, root_obj, "_Roots", f"{root_obj.name}_Roots")
        if not fuse_junctions and root_graph is not None and root_graph.axes:
            populate_mesh(roots_obj.data, collar_bark_age(self.mesh_engine.build_wood_mesh(
                root_graph, self.result.total_height_m, None, trunk_index=-1), collar, r_collar, zone))
            roots_obj.hide_viewport = False
            roots_obj.hide_render = False
        else:
            roots_obj.data.clear_geometry()
            roots_obj.hide_viewport = True
            roots_obj.hide_render = True
        # Fine lateral roots (thin tubes, never fused)
        fine_obj = self._child(context, root_obj, "_FineRoots", f"{root_obj.name}_FineRoots")
        fine_graph = getattr(self.result, "fine_root_graph", None) if show_roots else None
        if fine_graph is not None and fine_graph.axes:
            fine_eng = BotanicalMeshEngine(MeshConfig(radial_resolution=5, twig_resolution=4))
            populate_mesh(fine_obj.data, collar_bark_age(fine_eng.build_wood_mesh(
                fine_graph, self.result.total_height_m, None, trunk_index=-1), collar, r_collar, zone))
            fine_obj.hide_viewport = fine_obj.hide_render = False
        else:
            fine_obj.data.clear_geometry()
            fine_obj.hide_viewport = fine_obj.hide_render = True

        # Foliage
        foliage_obj = self._child(context, root_obj, "_Foliage", f"{root_obj.name}_Foliage")
        card_obj = self._child(context, root_obj, "_LeafCard", f"{root_obj.name}_LeafCard")
        card_obj.hide_viewport = card_obj.hide_render = True
        from .instancing import attach_instancer, remove_instancer, frames_to_euler
        if show_leaves and leaf_density > 0.0 and foliage_instanced:
            # One leaf (or leafy shoot) card instanced on every foliage point: a fraction of the memory
            # and write time of realized cards, and the same render
            from ..core.foliage import FoliageInstances
            one = FoliageInstances(np.zeros((1, 3)), np.array([[0.0, 1.0, 0.0]]), np.array([[0.0, 0.0, 1.0]]),
                                   np.ones(1), np.zeros(1))
            populate_mesh(card_obj.data, self.mesh_engine.build_foliage_mesh(one, self.result.leaf_mesh_data, 1.0),
                          smooth=True)
            f = self.result.foliage
            pts = MeshData(f.positions.astype(np.float32), np.zeros(0, np.int32), np.zeros(0, np.int32),
                           np.zeros(0, np.int32), np.zeros((0, 2), np.float32),
                           {"frot": frames_to_euler(f.axis_x, f.axis_y, f.axis_z).astype(np.float32),
                            "fscale": (f.scales * leaf_scale).astype(np.float32),
                            "leaf_random": f.randoms.astype(np.float32)})
            populate_mesh(foliage_obj.data, pts)
            attach_instancer(foliage_obj, card_obj)
            foliage_obj.hide_viewport = False
            foliage_obj.hide_render = False
        elif show_leaves and leaf_density > 0.0:
            remove_instancer(foliage_obj)
            card_obj.data.clear_geometry()
            foliage = self.mesh_engine.build_foliage_mesh(self.result.foliage, self.result.leaf_mesh_data, leaf_scale)
            populate_mesh(foliage_obj.data, foliage, smooth=True)
            foliage_obj.hide_viewport = False
            foliage_obj.hide_render = False
        else:
            remove_instancer(foliage_obj)
            foliage_obj.data.clear_geometry()
            foliage_obj.hide_viewport = True
            foliage_obj.hide_render = True

        r = self.result
        root_obj["scientific_name"] = self.preset.scientific_name
        root_obj["common_name"] = self.preset.common_name
        root_obj["family"] = self.preset.family
        root_obj["dbh_m"] = r.dbh_m
        root_obj["total_height_m"] = r.total_height_m
        root_obj["crown_radius_m"] = r.crown_radius_m
        root_obj["leaf_count"] = r.leaf_count
        return {"root": root_obj, "wood": wood_obj, "foliage": foliage_obj, "roots": roots_obj, "leaf_card": card_obj,
                "fine_roots": fine_obj}

    @staticmethod
    def _assign_vertex_groups(obj, orders):
        if orders is None or len(orders) == 0:
            return
        orders = np.asarray(orders)
        for name, sel in (("Trunk", orders == 0), ("Scaffolds", orders == 1), ("Twigs", orders >= 2)):
            vg = obj.vertex_groups.get(name) or obj.vertex_groups.new(name=name)
            idx = np.nonzero(sel)[0].tolist()
            if idx:
                vg.add(idx, 1.0, 'REPLACE')
