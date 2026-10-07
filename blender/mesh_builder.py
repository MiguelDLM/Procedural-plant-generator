"""
Blender Mesh and Curve Construction Engine.
Translates computed botanical graphs and leaf morphometrics into
optimized 3D Blender geometry, UV coordinates, and vertex attributes.
"""

import math
import numpy as np

try:
    import bpy
    import bmesh
    from mathutils import Vector, Matrix, Euler
    BLENDER_AVAILABLE = True
except ImportError:
    BLENDER_AVAILABLE = False


class BlenderMeshBuilder:
    """Constructs Blender objects from BotanicalPlantResult data."""

    def __init__(self, plant_result):
        self.result = plant_result
        self.preset = plant_result.preset

    def build_full_plant(self, context, collection=None, use_geometry_nodes=True) -> dict:
        """
        Creates all plant objects in Blender:
        - Trunk and Branch wood object
        - Master leaf prototype
        - Foliage instancer or direct leaf mesh
        - Optional 3D leaf vein mesh
        """
        if not BLENDER_AVAILABLE:
            raise RuntimeError("Blender (bpy) is not available in the current Python environment.")

        if collection is None:
            collection = context.scene.collection

        # Root empty container
        plant_empty = bpy.data.objects.new(f"Plant_{self.preset.scientific_name.replace(' ', '_')}", None)
        plant_empty.empty_display_type = 'PLAIN_AXES'
        collection.objects.link(plant_empty)

        # 1. Build Wood Skeleton (Curve or Mesh)
        wood_obj = self.build_branch_curve_object(plant_empty.name + "_Wood")
        collection.objects.link(wood_obj)
        wood_obj.parent = plant_empty

        # 2. Build Master Leaf Prototype
        leaf_prototype = self.build_leaf_mesh_object(plant_empty.name + "_Leaf_Master")
        collection.objects.link(leaf_prototype)
        leaf_prototype.parent = plant_empty
        # Hide prototype from final render if instancing
        leaf_prototype.hide_render = False

        # 3. Instance Leaves on twigs
        leaves_obj = self.build_leaf_instances(plant_empty.name + "_Foliage", leaf_prototype)
        if leaves_obj:
            collection.objects.link(leaves_obj)
            leaves_obj.parent = plant_empty

        # Store botanical metadata in custom properties
        plant_empty["scientific_name"] = self.preset.scientific_name
        plant_empty["family"] = self.preset.family
        plant_empty["biome"] = self.preset.biome
        plant_empty["dbh_m"] = self.result.dbh_m
        plant_empty["total_height_m"] = self.result.total_height_m
        plant_empty["crown_radius_m"] = self.result.crown_radius_m
        plant_empty["vla_mm_per_mm2"] = self.preset.venation.vla_mm_per_mm2
        plant_empty["wood_density_g_cm3"] = self.preset.biomechanics.wood_density_g_cm3

        return {
            "root": plant_empty,
            "wood": wood_obj,
            "leaf_master": leaf_prototype,
            "foliage": leaves_obj
        }

    def build_branch_curve_object(self, name: str) -> "bpy.types.Object":
        """
        Creates a 3D bevel curve object representing trunk and branches
        with exact per-node taper radii.
        """
        curve_data = bpy.data.curves.new(name=name, type='CURVE')
        curve_data.dimensions = '3D'
        curve_data.bevel_depth = 1.0  # Radii will scale this
        curve_data.bevel_resolution = 4
        curve_data.fill_mode = 'FULL'

        graph = self.result.skeleton_graph

        for b_indices in graph.branches:
            if len(b_indices) < 2:
                continue

            spline = curve_data.splines.new('BEZIER')
            spline.bezier_points.add(len(b_indices) - 1)

            for i, node_idx in enumerate(b_indices):
                node = graph.nodes[node_idx]
                bp = spline.bezier_points[i]
                bp.co = Vector(node.position)
                bp.radius = node.radius
                bp.handle_left_type = 'AUTO'
                bp.handle_right_type = 'AUTO'

        obj = bpy.data.objects.new(name, curve_data)
        return obj

    def build_leaf_mesh_object(self, name: str) -> "bpy.types.Object":
        """
        Builds the 3D leaf blade and petiole mesh with UVs.
        """
        leaf_data = self.result.leaf_mesh_data
        verts = leaf_data["vertices"]
        faces = leaf_data["faces"]
        uv_coords = leaf_data["uvs"]

        mesh = bpy.data.meshes.new(name)
        mesh.from_pydata(verts, [], faces)
        mesh.update()

        # Generate UV layer
        if uv_coords and len(uv_coords) == len(verts):
            uv_layer = mesh.uv_layers.new(name="UVMap")
            for poly in mesh.polygons:
                for loop_idx in poly.loop_indices:
                    v_idx = mesh.loops[loop_idx].vertex_index
                    uv_layer.data[loop_idx].uv = Vector(uv_coords[v_idx])

        # Smooth shading
        for poly in mesh.polygons:
            poly.use_smooth = True

        obj = bpy.data.objects.new(name, mesh)
        return obj

    def build_leaf_instances(self, name: str, master_leaf_obj: "bpy.types.Object") -> "bpy.types.Object":
        """
        Creates combined foliage mesh joining leaf instances at terminal twig positions.
        """
        transforms = self.result.leaf_transforms
        if not transforms or not master_leaf_obj:
            return None

        bm = bmesh.new()
        master_bm = bmesh.new()
        master_bm.from_mesh(master_leaf_obj.data)

        for t in transforms:
            pos = Vector(t["position"])
            dir_vec = Vector(t["direction"]).normalized()
            twist = t["twist_rad"]
            scale = t["scale"]

            # Compute rotation matrix aligning leaf +Y axis with branch direction
            rot_quat = Vector((0.0, 1.0, 0.0)).rotation_difference(dir_vec)
            twist_rot = Matrix.Rotation(twist, 4, dir_vec)
            mat = Matrix.Translation(pos) @ twist_rot @ rot_quat.to_matrix().to_4x4() @ Matrix.Scale(scale, 4)

            # Copy vertices
            v_map = {}
            for v in master_bm.verts:
                world_co = mat @ v.co
                new_v = bm.verts.new(world_co)
                v_map[v] = new_v

            for f in master_bm.faces:
                try:
                    bm.faces.new([v_map[v] for v in f.verts])
                except ValueError:
                    pass

        master_bm.free()

        mesh = bpy.data.meshes.new(name)
        bm.to_mesh(mesh)
        bm.free()
        mesh.update()

        for poly in mesh.polygons:
            poly.use_smooth = True

        obj = bpy.data.objects.new(name, mesh)
        return obj

    def build_vein_curve_object(self, name: str) -> "bpy.types.Object":
        """
        Creates a high-resolution 3D spline network of the primary and secondary veins.
        """
        curve_data = bpy.data.curves.new(name=name, type='CURVE')
        curve_data.dimensions = '3D'
        curve_data.bevel_depth = 1.0
        curve_data.bevel_resolution = 3
        curve_data.fill_mode = 'FULL'

        net = self.result.vein_network

        for seg in net.segments:
            if len(seg) < 2:
                continue

            spline = curve_data.splines.new('BEZIER')
            spline.bezier_points.add(len(seg) - 1)

            for i, n_idx in enumerate(seg):
                node = net.nodes[n_idx]
                bp = spline.bezier_points[i]
                bp.co = Vector(node.position)
                bp.radius = node.radius_m
                bp.handle_left_type = 'AUTO'
                bp.handle_right_type = 'AUTO'

        obj = bpy.data.objects.new(name, curve_data)
        return obj
