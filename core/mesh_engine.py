"""
Continuous Botanical Mesh Engine.
Constructs coherent, continuous quad geometry for trunks and branches with:
- Gielis Superformula fluted buttressing and elliptical reaction wood
- Parallel transport frames (Bishop frames) to prevent twisting
- Branch bark collars and flush surface attachments (no floating disconnected branches)
- Petiole-anchored leaf placement (no floating leaves)
"""

from dataclasses import dataclass
import math
import numpy as np

from .architecture import BranchingGraph, BranchNode
from .gielis import GielisEngine, GielisProfile


@dataclass
class MeshConfig:
    """Topology and meshing resolution parameters."""
    radial_resolution: int = 12       # Number of circumferential vertices per ring (8, 12, 16)
    twig_resolution: int = 6          # Radial vertices for thin twigs
    collar_flare_factor: float = 1.35 # Branch collar swelling ratio at junction
    smooth_caps: bool = True          # Close branch tips with rounded cap
    buttress_profile: GielisProfile = None


class BotanicalMeshEngine:
    """Generates continuous quad meshes from branching graphs."""

    def __init__(self, config: MeshConfig = None):
        self.config = config or MeshConfig()

    def build_wood_mesh(
        self,
        skeleton: BranchingGraph,
        total_height_m: float,
        gielis_profile: GielisProfile = None
    ) -> dict:
        """
        Converts the entire branching skeleton into a coherent polygonal mesh
        with quad topology, Gielis fluted buttresses, and branch collars.
        
        Returns:
            dict with 'vertices': list of [x, y, z],
                      'faces': list of 4-tuples [i0, i1, i2, i3],
                      'uvs': list of [u, v],
                      'vertex_orders': list of int (order 0=trunk, 1, 2, 3)
        """
        all_verts = []
        all_faces = []
        all_uvs = []
        vert_orders = []

        g_prof = gielis_profile or self.config.buttress_profile or GielisProfile(m=0.0)

        for branch_idx, b_indices in enumerate(skeleton.branches):
            if len(b_indices) < 2:
                continue

            order = skeleton.nodes[b_indices[0]].order
            n_sides = self.config.radial_resolution if order <= 1 else self.config.twig_resolution
            angles = np.linspace(0.0, 2.0 * math.pi, n_sides, endpoint=False)

            # ---------------------------------------------------------
            # 1. Compute Parallel Transport Frames along the branch
            # ---------------------------------------------------------
            tangents = []
            normals = []
            binormals = []

            for i in range(len(b_indices)):
                curr_node = skeleton.nodes[b_indices[i]]
                if i == 0:
                    next_node = skeleton.nodes[b_indices[1]]
                    t = next_node.position - curr_node.position
                elif i == len(b_indices) - 1:
                    prev_node = skeleton.nodes[b_indices[i - 1]]
                    t = curr_node.position - prev_node.position
                else:
                    prev_node = skeleton.nodes[b_indices[i - 1]]
                    next_node = skeleton.nodes[b_indices[i + 1]]
                    t = next_node.position - prev_node.position

                norm_t = np.linalg.norm(t)
                t = t / max(1e-6, norm_t)
                tangents.append(t)

                if i == 0:
                    # Initial normal vector perpendicular to tangent
                    ref = np.array([0.0, 1.0, 0.0]) if abs(t[2]) > 0.9 else np.array([0.0, 0.0, 1.0])
                    n = np.cross(t, ref)
                    n = n / np.linalg.norm(n)
                    b = np.cross(t, n)
                    normals.append(n)
                    binormals.append(b)
                else:
                    # Parallel transport the previous normal
                    prev_n = normals[-1]
                    n = prev_n - np.dot(prev_n, t) * t
                    norm_n = np.linalg.norm(n)
                    if norm_n > 1e-6:
                        n = n / norm_n
                    else:
                        n = prev_n
                    b = np.cross(t, n)
                    normals.append(n)
                    binormals.append(b)

            # ---------------------------------------------------------
            # 2. Build Circumferential Vertex Rings
            # ---------------------------------------------------------
            branch_vert_start = len(all_verts)
            ring_start_indices = []

            cumulative_dist = 0.0

            for i, node_idx in enumerate(b_indices):
                node = skeleton.nodes[node_idx]
                p = node.position.copy()
                r = node.radius
                t = tangents[i]
                n = normals[i]
                b = binormals[i]

                z_rel = float(np.clip(p[2] / max(1.0, total_height_m), 0.0, 1.0))

                # For order > 0, branch collar swelling at insertion junction
                if order > 0 and i == 0:
                    r = r * self.config.collar_flare_factor

                # Radius modulation: Gielis buttress near ground for trunk (order 0)
                if order == 0:
                    ring_radii = GielisEngine.compute_buttressed_cross_section(
                        angles=angles,
                        base_radius=r,
                        z_relative=z_rel,
                        gielis_profile=g_prof
                    )
                else:
                    ring_radii = np.full(n_sides, r, dtype=float)

                ring_idx = len(all_verts)
                ring_start_indices.append(ring_idx)

                # Distance along branch for U texture coordinate
                if i > 0:
                    step_len = np.linalg.norm(p - skeleton.nodes[b_indices[i - 1]].position)
                    cumulative_dist += step_len

                v_coord = cumulative_dist * 2.0  # texture tiling

                for a_idx, ang in enumerate(angles):
                    rad = ring_radii[a_idx]
                    vert_pos = p + rad * (math.cos(ang) * n + math.sin(ang) * b)
                    u_coord = a_idx / n_sides

                    all_verts.append(vert_pos.tolist())
                    all_uvs.append([u_coord, v_coord])
                    vert_orders.append(order)

            # ---------------------------------------------------------
            # 3. Connect Rings with Quads
            # ---------------------------------------------------------
            for r_idx in range(len(b_indices) - 1):
                ring0 = ring_start_indices[r_idx]
                ring1 = ring_start_indices[r_idx + 1]

                for s in range(n_sides):
                    s_next = (s + 1) % n_sides
                    v0 = ring0 + s
                    v1 = ring0 + s_next
                    v2 = ring1 + s_next
                    v3 = ring1 + s
                    all_faces.append([v0, v1, v2, v3])

            # ---------------------------------------------------------
            # 4. Cap Branch Tip with a Dome
            # ---------------------------------------------------------
            if self.config.smooth_caps and len(b_indices) >= 2:
                last_node = skeleton.nodes[b_indices[-1]]
                last_ring = ring_start_indices[-1]
                tip_pos = last_node.position + tangents[-1] * (last_node.radius * 0.8)

                tip_vert_idx = len(all_verts)
                all_verts.append(tip_pos.tolist())
                all_uvs.append([0.5, cumulative_dist * 2.0])
                vert_orders.append(order)

                for s in range(n_sides):
                    s_next = (s + 1) % n_sides
                    all_faces.append([last_ring + s, last_ring + s_next, tip_vert_idx])

        return {
            "vertices": all_verts,
            "faces": all_faces,
            "uvs": all_uvs,
            "vertex_orders": vert_orders
        }

    def build_anchored_leaves(
        self,
        skeleton: BranchingGraph,
        master_leaf_data: dict,
        leaf_density: float = 1.0,
        leaf_scale: float = 1.0,
        seed: int = 42
    ) -> dict:
        """
        Builds a single combined foliage mesh where each leaf is anchored
        directly onto the bark surface of twig branches (zero floating leaves).
        """
        rng = np.random.default_rng(seed)

        master_verts = np.array(master_leaf_data["vertices"], dtype=float)
        master_faces = master_leaf_data["faces"]
        master_uvs = master_leaf_data["uvs"]

        foliage_verts = []
        foliage_faces = []
        foliage_uvs = []

        max_order = max(n.order for n in skeleton.nodes)

        for node_idx, node in enumerate(skeleton.nodes):
            # Only attach leaves to twigs and upper scaffolds
            if node.order < max(1, max_order - 1):
                continue

            # Deterministic probability based on leaf density
            prob = 0.55 * leaf_density
            if not node.is_leaf_attachment and rng.random() > prob:
                continue

            # Direction along branch
            dir_vec = node.direction.copy()
            norm = np.linalg.norm(dir_vec)
            if norm < 1e-6:
                dir_vec = np.array([0.0, 0.0, 1.0])
            else:
                dir_vec = dir_vec / norm

            # Radial bark surface anchor point
            azimuth = node.azimuth_rad + rng.uniform(-0.5, 0.5)
            ref_up = np.array([0.0, 0.0, 1.0])
            perp1 = np.cross(dir_vec, ref_up)
            if np.linalg.norm(perp1) < 1e-4:
                perp1 = np.cross(dir_vec, np.array([0.0, 1.0, 0.0]))
            perp1 = perp1 / np.linalg.norm(perp1)
            perp2 = np.cross(dir_vec, perp1)

            surface_offset = node.radius * (math.cos(azimuth) * perp1 + math.sin(azimuth) * perp2)
            anchor_pos = node.position + surface_offset

            # Build rotation matrix aligning leaf +Y axis with shoot orientation
            # Leaf points outward and upward towards sunlight
            outward_vec = (surface_offset / max(1e-4, node.radius)) * 0.7 + dir_vec * 0.5 + np.array([0.0, 0.0, 0.3])
            outward_vec = outward_vec / np.linalg.norm(outward_vec)

            # Local coordinate frame
            leaf_y = outward_vec
            leaf_x = np.cross(leaf_y, np.array([0.0, 0.0, 1.0]))
            if np.linalg.norm(leaf_x) < 1e-4:
                leaf_x = np.cross(leaf_y, np.array([1.0, 0.0, 0.0]))
            leaf_x = leaf_x / np.linalg.norm(leaf_x)
            leaf_z = np.cross(leaf_x, leaf_y)

            # Random scale and slight tilt
            rand_scale = leaf_scale * rng.uniform(0.85, 1.20)
            rot_mat = np.stack([leaf_x, leaf_y, leaf_z], axis=1) * rand_scale

            base_vert_idx = len(foliage_verts)

            # Transform master leaf vertices
            transformed_verts = (master_verts @ rot_mat.T) + anchor_pos
            foliage_verts.extend(transformed_verts.tolist())
            foliage_uvs.extend(master_uvs)

            for face in master_faces:
                foliage_faces.append([base_vert_idx + vi for vi in face])

        return {
            "vertices": foliage_verts,
            "faces": foliage_faces,
            "uvs": foliage_uvs
        }
