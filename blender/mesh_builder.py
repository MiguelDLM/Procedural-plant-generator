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

        root_obj = existing_root or bpy.data.objects.get(root_name)
        if not root_obj or root_obj.name not in context.scene.objects:
            root_obj = bpy.data.objects.new(root_name, None)
            root_obj.empty_display_type = 'PLAIN_AXES'
            collection.objects.link(root_obj)

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
            if child.name.endswith("_Wood"):
                wood_obj = child
                break

        if not wood_obj or wood_obj.name not in context.scene.objects:
            mesh = bpy.data.meshes.new(wood_name)
            wood_obj = bpy.data.objects.new(wood_name, mesh)
            collection.objects.link(wood_obj)
            wood_obj.parent = root_obj
        else:
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
            if child.name.endswith("_Foliage"):
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
                f_mesh = foliage_obj.data

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
        """Efficiently populates or updates a Blender Mesh from Python lists."""
        mesh.clear_geometry()
        mesh.from_pydata(verts, [], faces)
        mesh.update()

        # Add UV coordinates
        if uvs and len(uvs) == len(verts):
            uv_layer = mesh.uv_layers.get("UVMap") or mesh.uv_layers.new(name="UVMap")
            for poly in mesh.polygons:
                for loop_idx in poly.loop_indices:
                    v_idx = mesh.loops[loop_idx].vertex_index
                    uv_layer.data[loop_idx].uv = Vector(uvs[v_idx])

        # Enable smooth shading
        for poly in mesh.polygons:
            poly.use_smooth = True

    def _assign_vertex_groups(self, obj: "bpy.types.Object", orders: list):
        """Creates vertex groups for Trunk (order 0), Scaffolds (order 1), and Twigs."""
        for name in ["Trunk", "Scaffolds", "Twigs"]:
            if name not in obj.vertex_groups:
                obj.vertex_groups.new(name=name)

        vg_trunk = obj.vertex_groups["Trunk"]
        vg_scaffold = obj.vertex_groups["Scaffolds"]
        vg_twigs = obj.vertex_groups["Twigs"]

        for v_idx, order in enumerate(orders):
            if order == 0:
                vg_trunk.add([v_idx], 1.0, 'REPLACE')
            elif order == 1:
                vg_scaffold.add([v_idx], 1.0, 'REPLACE')
            else:
                vg_twigs.add([v_idx], 1.0, 'REPLACE')
