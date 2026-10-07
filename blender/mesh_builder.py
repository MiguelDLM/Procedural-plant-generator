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
    from ..core.mesh_engine import BotanicalMeshEngine, MeshConfig, MeshData
    from ..core.gielis import GielisProfile
except (ImportError, ValueError):
    from core.mesh_engine import BotanicalMeshEngine, MeshConfig, MeshData
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
        kind = 'INT' if values.dtype.kind in "iu" else 'FLOAT'
        attr = mesh.attributes.get(name)
        if attr is None or attr.data_type != kind or attr.domain != 'POINT':
            if attr is not None:
                mesh.attributes.remove(attr)
            attr = mesh.attributes.new(name=name, type=kind, domain='POINT')
        attr.data.foreach_set("value", values.astype(np.int32 if kind == 'INT' else np.float32))

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
                              show_roots: bool = True, fuse_junctions: bool = True, fuse_detail: float = 10.0,
                              fuse_smoothing: int = 6) -> dict:
        if not BLENDER_AVAILABLE:
            raise RuntimeError("Blender (bpy) is not available.")
        root_name = f"PPG_{self.preset.scientific_name.split(' (')[0].replace(' ', '_').replace(chr(39), '')}"
        root_obj = existing_root or self._find_root(context, root_name)

        # Wood (stem flutes aligned with the main roots)
        self.config.flute_azimuth = getattr(self.result, "flute_azimuth", None)
        root_graph = getattr(self.result, "root_graph", None) if show_roots else None
        if fuse_junctions:
            # Stem, limbs and roots fused into one continuous surface at their junctions
            from .junctions import build_fused_wood
            wood_data = build_fused_wood(self.mesh_engine, [self.result.skeleton_graph, root_graph],
                                         self.result.total_height_m, self.config.buttress_profile,
                                         detail=fuse_detail, smooth_iterations=fuse_smoothing)
        else:
            wood_data = self.mesh_engine.build_wood_mesh(self.result.skeleton_graph, self.result.total_height_m,
                                                         self.config.buttress_profile)
        wood_obj = self._child(context, root_obj, "_Wood", f"{root_obj.name}_Wood")
        populate_mesh(wood_obj.data, wood_data)
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
            populate_mesh(roots_obj.data, self.mesh_engine.build_wood_mesh(
                root_graph, self.result.total_height_m, None, trunk_index=-1))
            roots_obj.hide_viewport = False
            roots_obj.hide_render = False
        else:
            roots_obj.data.clear_geometry()
            roots_obj.hide_viewport = True
            roots_obj.hide_render = True

        # Foliage
        foliage_obj = self._child(context, root_obj, "_Foliage", f"{root_obj.name}_Foliage")
        if show_leaves and leaf_density > 0.0:
            foliage = self.mesh_engine.build_foliage_mesh(self.result.foliage, self.result.leaf_mesh_data, leaf_scale)
            populate_mesh(foliage_obj.data, foliage, smooth=True)
            foliage_obj.hide_viewport = False
            foliage_obj.hide_render = False
        else:
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
        return {"root": root_obj, "wood": wood_obj, "foliage": foliage_obj, "roots": roots_obj}

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
