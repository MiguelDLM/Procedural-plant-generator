"""
Blender Mesh and Object Construction Engine.
Translates botanical skeletons into continuous quad meshes with Gielis buttressing,
branch collars, UV coordinates, and in-place real-time geometry updating.
"""

import math
import numpy as np

try:
    import bpy
    import bmesh
    from mathutils import Vector, Matrix
    BLENDER_AVAILABLE = True
except ImportError:
    BLENDER_AVAILABLE = False

try:
    from ..core.mesh_engine import BotanicalMeshEngine, MeshConfig
    from ..core.gielis import GielisProfile
except (ImportError, ValueError):
    from core.mesh_engine import BotanicalMeshEngine, MeshConfig
    from core.gielis import GielisProfile


class BlenderMeshBuilder:
    """Builds and updates Blender objects from procedural botanical data."""

    def __init__(self, plant_result, radial_resolution: int = 12, buttress_profile: GielisProfile = None):
        self.result = plant_result
        self.preset = plant_result.preset
        self.config = MeshConfig(
            radial_resolution=radial_resolution,
            twig_resolution=max(6, radial_resolution // 2),
            collar_flare_factor=1.35,
            buttress_profile=buttress_profile
        )
        self.mesh_engine = BotanicalMeshEngine(self.config)

    def build_or_update_plant(
        self,
        context,
        existing_root=None,
        leaf_density: float = 1.0,
        leaf_scale: float = 1.0,
        show_leaves: bool = True,
        use_subsurf: bool = False
    ) -> dict:
        """
        Builds a new plant or updates an existing one in-place in real time.
        """
        if not BLENDER_AVAILABLE:
            raise RuntimeError("Blender (bpy) is not available.")

        collection = context.scene.collection

        root_name = f"PPG_{self.preset.scientific_name.replace(' ', '_')}"
        wood_name = f"{root_name}_Wood"
        foliage_name = f"{root_name}_Foliage"

        # Find active PPG plant in scene to update in-place without duplicating
        active_root_name = context.scene.get("ppg_active_root_name")
        root_obj = existing_root
        if not root_obj and active_root_name:
            root_obj = bpy.data.objects.get(active_root_name)

        if not root_obj or root_obj.name not in context.scene.objects:
            if context.active_object and context.active_object.name.startswith("PPG_"):
                cand = context.active_object
                while cand.parent and cand.parent.name.startswith("PPG_"):
                    cand = cand.parent
                root_obj = cand
            else:
                root_obj = bpy.data.objects.get(root_name)

        if not root_obj or root_obj.name not in context.scene.objects:
            root_obj = bpy.data.objects.new(root_name, None)
            root_obj.empty_display_type = 'PLAIN_AXES'
            collection.objects.link(root_obj)
        else:
            root_obj.name = root_name

        context.scene["ppg_active_root_name"] = root_name

        # -------------------------------------------------------------
        # 1. Build or Update Continuous Quad Wood Mesh
        # -------------------------------------------------------------
        wood_data = self.mesh_engine.build_wood_mesh(
            skeleton=self.result.skeleton_graph,
            total_height_m=self.result.total_height_m,
            gielis_profile=self.config.buttress_profile
        )

        wood_obj = None
        for child in root_obj.children:
            if "_Wood" in child.name:
                wood_obj = child
                break

        if not wood_obj or wood_obj.name not in context.scene.objects:
            mesh = bpy.data.meshes.new(wood_name)
            wood_obj = bpy.data.objects.new(wood_name, mesh)
            collection.objects.link(wood_obj)
            wood_obj.parent = root_obj
        else:
            wood_obj.name = wood_name
            mesh = wood_obj.data

        # Populate mesh in-place
        self._populate_mesh(mesh, wood_data["vertices"], wood_data["faces"], wood_data["uvs"])

        # Create Vertex Groups for Trunk vs Branches
        self._assign_vertex_groups(wood_obj, wood_data["vertex_orders"])

        # Optional Subsurf modifier
        if use_subsurf:
            if "PPG_Subsurf" not in wood_obj.modifiers:
                mod = wood_obj.modifiers.new(name="PPG_Subsurf", type='SUBSURF')
                mod.levels = 1
                mod.render_levels = 2
        else:
            if "PPG_Subsurf" in wood_obj.modifiers:
                wood_obj.modifiers.remove(wood_obj.modifiers["PPG_Subsurf"])

        # -------------------------------------------------------------
        # 2. Build or Update Anchored Foliage Mesh
        # -------------------------------------------------------------
        foliage_obj = None
        for child in root_obj.children:
            if "_Foliage" in child.name:
                foliage_obj = child
                break

        if show_leaves and leaf_density > 0.0:
            foliage_data = self.mesh_engine.build_anchored_leaves(
                skeleton=self.result.skeleton_graph,
                master_leaf_data=self.result.leaf_mesh_data,
                leaf_density=leaf_density,
                leaf_scale=leaf_scale,
                seed=42
            )

            if not foliage_obj or foliage_obj.name not in context.scene.objects:
                f_mesh = bpy.data.meshes.new(foliage_name)
                foliage_obj = bpy.data.objects.new(foliage_name, f_mesh)
                collection.objects.link(foliage_obj)
                foliage_obj.parent = root_obj
            else:
                foliage_obj.name = foliage_name
                f_mesh = foliage_obj.data
                f_mesh.name = foliage_name

            self._populate_mesh(f_mesh, foliage_data["vertices"], foliage_data["faces"], foliage_data["uvs"])
            foliage_obj.hide_viewport = False
            foliage_obj.hide_render = False
        else:
            if foliage_obj:
                foliage_obj.hide_viewport = True
                foliage_obj.hide_render = True

        # Store botanical custom properties on root
        root_obj["scientific_name"] = self.preset.scientific_name
        root_obj["common_name"] = self.preset.common_name
        root_obj["family"] = self.preset.family
        root_obj["dbh_m"] = self.result.dbh_m
        root_obj["total_height_m"] = self.result.total_height_m
        root_obj["crown_radius_m"] = self.result.crown_radius_m

        return {
            "root": root_obj,
            "wood": wood_obj,
            "foliage": foliage_obj
        }

    def _populate_mesh(self, mesh: "bpy.types.Mesh", verts: list, faces: list, uvs: list):
        """Efficiently populates or updates a Blender Mesh using fast C-buffers."""
        mesh.clear_geometry()
        mesh.from_pydata(verts, [], faces)
        mesh.update()

        # Vectorized fast UV mapping
        if uvs and len(uvs) == len(verts) and len(mesh.loops) > 0:
            uv_layer = mesh.uv_layers.get("UVMap") or mesh.uv_layers.new(name="UVMap")
            loop_vert_indices = np.empty(len(mesh.loops), dtype=np.int32)
            mesh.loops.foreach_get('vertex_index', loop_vert_indices)
            uv_arr = np.array(uvs, dtype=np.float32)
            loop_uvs = uv_arr[loop_vert_indices].ravel()
            uv_layer.data.foreach_set('uv', loop_uvs)

        # Vectorized smooth shading
        if len(mesh.polygons) > 0:
            mesh.polygons.foreach_set('use_smooth', [True] * len(mesh.polygons))

    def _assign_vertex_groups(self, obj: "bpy.types.Object", orders: list):
        """Creates vertex groups for Trunk (order 0), Scaffolds (order 1), and Twigs efficiently."""
        if not orders:
            return

        for name in ["Trunk", "Scaffolds", "Twigs"]:
            if name not in obj.vertex_groups:
                obj.vertex_groups.new(name=name)

        vg_trunk = obj.vertex_groups["Trunk"]
        vg_scaffold = obj.vertex_groups["Scaffolds"]
        vg_twigs = obj.vertex_groups["Twigs"]

        orders_arr = np.array(orders)
        trunk_idx = np.where(orders_arr == 0)[0].tolist()
        scaffold_idx = np.where(orders_arr == 1)[0].tolist()
        twigs_idx = np.where(orders_arr >= 2)[0].tolist()

        if trunk_idx:
            vg_trunk.add(trunk_idx, 1.0, 'REPLACE')
        if scaffold_idx:
            vg_scaffold.add(scaffold_idx, 1.0, 'REPLACE')
        if twigs_idx:
            vg_twigs.add(twigs_idx, 1.0, 'REPLACE')
