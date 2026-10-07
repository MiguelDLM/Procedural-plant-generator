"""
Botanical Branching Architecture Models.
Implements Hallé-Oldeman architectural models (Rauh, Massart, Troll, Attims, Corner)
coupled with phyllotaxis, apical dominance, gravitropism, and Leonardo pipe tapering.
"""

from dataclasses import dataclass, field
from enum import Enum
import math
import numpy as np


class HalleOldemanModel(str, Enum):
    RAUH = "Rauh"        # Monopodial, rhythmic orthotropic branches (Quercus, Pinus, Hevea)
    MASSART = "Massart"  # Monopodial, plagiotropic horizontal tiers (Abies, Araucaria, Taxus)
    TROLL = "Troll"      # Sympodial, plagiotropic shoots bending upright (Fagus, Ulmus, Celtis)
    ATTIMS = "Attims"    # Monopodial, continuous diffuse branching (Eucalyptus, Alnus)
    CORNER = "Corner"    # Unbranched or basal monocot/fern/palm architecture


class PhyllotaxisType(str, Enum):
    SPIRAL = "Spiral"          # Golden angle: 137.507764 deg
    DECUSSATE = "Decussate"    # Opposite pairs rotated by 90 deg
    DISTICHOUS = "Distichous"  # Alternating planes: 180 deg


@dataclass
class ArchitectureProfile:
    """Parameters governing the 3D branching skeleton."""
    model: HalleOldemanModel = HalleOldemanModel.RAUH
    phyllotaxis: PhyllotaxisType = PhyllotaxisType.SPIRAL

    # Maximum branching order (0 = trunk, 1 = primary scaffolds, 2 = secondary, 3 = twigs)
    max_order: int = 3

    # Primary branching insertion angle (degrees relative to parent axis)
    branch_angle_mean_deg: float = 48.0
    branch_angle_std_deg: float = 6.5

    # Secondary/tertiary branching angle
    twig_angle_mean_deg: float = 52.0

    # Phyllotactic divergence angle (computed or overridden)
    divergence_angle_deg: float = 137.508

    # Apical dominance: 0.0 (weak/shrubby) to 1.0 (strong single-leader excurrent conifer)
    apical_dominance: float = 0.65

    # Gravitropism: negative (bends up to +Z, typical), 0 (neutral), positive (weeping)
    gravitropism: float = -0.22

    # Phototropism / radial spreading bias
    phototropism: float = 0.35

    # Internode length base (meters) and decay factor per order
    internode_length_base_m: float = 0.45
    internode_decay_per_order: float = 0.68

    # Branch density: number of branch whorls or nodes per meter
    branch_frequency_per_meter: float = 4.0

    # Branch tortuosity / natural winding variance
    crookedness: float = 0.12


@dataclass
class BranchNode:
    """Single 3D point along a botanical axis."""
    position: np.ndarray        # [x, y, z] in meters
    direction: np.ndarray       # Unit vector [dx, dy, dz]
    radius: float               # Radius in meters
    order: int                  # 0=trunk, 1=branch, 2=sub-branch, 3=twig
    distance_along_stem: float  # Cumulative length along this branch in meters
    relative_height: float      # Height relative to total tree (0.0 to 1.0)
    parent_idx: int = -1
    children_indices: list[int] = field(default_factory=list)
    is_leaf_attachment: bool = False
    azimuth_rad: float = 0.0


class BranchingGraph:
    """Container for the entire tree skeleton topological network."""

    def __init__(self):
        self.nodes: list[BranchNode] = []
        self.branches: list[list[int]] = []  # Lists of node indices per contiguous branch

    def add_node(self, node: BranchNode) -> int:
        idx = len(self.nodes)
        self.nodes.append(node)
        if node.parent_idx >= 0 and node.parent_idx < len(self.nodes):
            self.nodes[node.parent_idx].children_indices.append(idx)
        return idx


class ArchitectureEngine:
    """Builds a complete botanical tree graph based on empirical architecture traits."""

    def __init__(self, profile: ArchitectureProfile = None):
        self.profile = profile or ArchitectureProfile()

    def generate_skeleton(
        self,
        total_height_m: float,
        crown_radius_m: float,
        crown_depth_m: float,
        base_radius_m: float,
        pipe_delta: float = 2.25,
        seed: int = 42
    ) -> BranchingGraph:
        """
        Generates the 3D branching skeleton adhering to the architectural model.
        """
        rng = np.random.default_rng(seed)
        graph = BranchingGraph()

        crown_base_height = max(0.2, total_height_m - crown_depth_m)
        trunk_segments = max(6, int(total_height_m * 4.0))

        # -------------------------------------------------------------
        # 1. Build the Main Trunk (Order 0)
        # -------------------------------------------------------------
        trunk_indices = []
        curr_pos = np.array([0.0, 0.0, 0.0], dtype=float)
        curr_dir = np.array([0.0, 0.0, 1.0], dtype=float)

        dz = total_height_m / trunk_segments

        for i in range(trunk_segments + 1):
            z_rel = (i * dz) / total_height_m

            # Metzger / Leonardo taper
            taper_factor = (1.0 - z_rel * 0.95) ** (1.0 / pipe_delta)
            buttress = 1.0 + 0.45 * math.exp(-12.0 * z_rel)
            r = base_radius_m * taper_factor * buttress

            # Slight natural trunk lean / deviation
            if i > 0 and self.profile.crookedness > 0:
                perturb = rng.normal(0.0, self.profile.crookedness * 0.02, size=3)
                perturb[2] = 0.0
                curr_dir = curr_dir + perturb
                curr_dir = curr_dir / np.linalg.norm(curr_dir)

            node = BranchNode(
                position=curr_pos.copy(),
                direction=curr_dir.copy(),
                radius=max(0.005, float(r)),
                order=0,
                distance_along_stem=float(i * dz),
                relative_height=float(z_rel),
                parent_idx=trunk_indices[-1] if trunk_indices else -1
            )
            idx = graph.add_node(node)
            trunk_indices.append(idx)

            curr_pos += curr_dir * dz

        graph.branches.append(trunk_indices)

        # -------------------------------------------------------------
        # 2. Build Scaffold Lateral Branches (Order 1)
        # -------------------------------------------------------------
        if self.profile.max_order < 1 or self.profile.model == HalleOldemanModel.CORNER:
            return graph

        azimuth_deg = 0.0
        golden_angle = self.profile.divergence_angle_deg

        # Find trunk nodes within the crown zone
        crown_nodes = [
            idx for idx in trunk_indices
            if graph.nodes[idx].position[2] >= crown_base_height and idx < trunk_indices[-1]
        ]

        for p_idx in crown_nodes:
            parent_node = graph.nodes[p_idx]
            z_pos = parent_node.position[2]
            crown_rel = (z_pos - crown_base_height) / max(0.1, crown_depth_m)

            # Crown envelope shape: parabolic or ellipsoidal max radius at crown_rel ~ 0.4
            envelope_factor = math.sin(math.pi * np.clip(crown_rel, 0.05, 0.95))
            target_branch_len = crown_radius_m * envelope_factor * rng.uniform(0.75, 1.15)

            if target_branch_len < 0.2:
                continue

            # Phyllotactic azimuth rotation
            if self.profile.phyllotaxis == PhyllotaxisType.SPIRAL:
                azimuth_deg = (azimuth_deg + golden_angle) % 360.0
            elif self.profile.phyllotaxis == PhyllotaxisType.DECUSSATE:
                azimuth_deg = (azimuth_deg + 90.0) % 360.0
            else:
                azimuth_deg = (azimuth_deg + 180.0) % 360.0

            az_rad = math.radians(azimuth_deg + rng.normal(0.0, 5.0))

            # Branch insertion angle relative to vertical trunk
            angle_mean = self.profile.branch_angle_mean_deg
            if self.profile.model == HalleOldemanModel.MASSART:
                # Flat tiers in Massart's model
                angle_mean = 78.0
            elif self.profile.model == HalleOldemanModel.TROLL:
                angle_mean = 62.0

            angle_deg = np.clip(
                rng.normal(angle_mean, self.profile.branch_angle_std_deg),
                25.0, 85.0
            )
            angle_rad = math.radians(angle_deg)

            # Compute initial branch direction
            bx = math.sin(angle_rad) * math.cos(az_rad)
            by = math.sin(angle_rad) * math.sin(az_rad)
            bz = math.cos(angle_rad)
            branch_dir = np.array([bx, by, bz], dtype=float)
            branch_dir /= np.linalg.norm(branch_dir)

            # Child radius by Leonardo pipe model: primary branches typically 0.25 - 0.45 of trunk radius
            child_radius = parent_node.radius * (0.35 ** (1.0 / pipe_delta))

            # Grow the branch line
            b_indices = self._grow_branch(
                graph=graph,
                parent_idx=p_idx,
                start_pos=parent_node.position.copy(),
                initial_dir=branch_dir,
                base_radius=child_radius,
                target_length=target_branch_len,
                order=1,
                pipe_delta=pipe_delta,
                rng=rng,
                azimuth_rad=az_rad
            )
            if b_indices:
                graph.branches.append(b_indices)

        # -------------------------------------------------------------
        # 3. Build Sub-branches (Order 2) and Twigs (Order 3)
        # -------------------------------------------------------------
        for current_order in range(2, self.profile.max_order + 1):
            parent_branches = [b for b in graph.branches if graph.nodes[b[0]].order == current_order - 1]

            for branch in parent_branches:
                # Sub-branches spawn along upper 60% of the parent branch
                spawn_candidates = branch[int(len(branch) * 0.35):-1]
                step = max(1, len(spawn_candidates) // 4)

                for node_idx in spawn_candidates[::step]:
                    parent_node = graph.nodes[node_idx]
                    if len(parent_node.children_indices) >= 2:
                        continue

                    # Sub-branch length scales down
                    sub_len = parent_node.distance_along_stem * self.profile.internode_decay_per_order * rng.uniform(0.6, 1.0)
                    if sub_len < 0.12:
                        continue

                    # Alternate side divergence
                    side = 1.0 if rng.random() > 0.5 else -1.0
                    sub_azimuth = parent_node.azimuth_rad + side * math.radians(rng.uniform(40.0, 75.0))

                    sub_angle_deg = rng.normal(self.profile.twig_angle_mean_deg, 6.0)
                    sub_angle_rad = math.radians(sub_angle_deg)

                    # Radial direction outward
                    rx = math.cos(sub_azimuth) * math.sin(sub_angle_rad)
                    ry = math.sin(sub_azimuth) * math.sin(sub_angle_rad)
                    rz = math.cos(sub_angle_rad) + self.profile.gravitropism
                    s_dir = np.array([rx, ry, rz], dtype=float)
                    s_dir /= np.linalg.norm(s_dir)

                    sub_radius = parent_node.radius * (0.45 ** (1.0 / pipe_delta))

                    sub_indices = self._grow_branch(
                        graph=graph,
                        parent_idx=node_idx,
                        start_pos=parent_node.position.copy(),
                        initial_dir=s_dir,
                        base_radius=sub_radius,
                        target_length=sub_len,
                        order=current_order,
                        pipe_delta=pipe_delta,
                        rng=rng,
                        azimuth_rad=sub_azimuth
                    )
                    if sub_indices:
                        graph.branches.append(sub_indices)

        # Mark terminal nodes for foliage / leaf attachment
        for node in graph.nodes:
            if len(node.children_indices) == 0 and node.order >= 1:
                node.is_leaf_attachment = True

        return graph

    def _grow_branch(
        self,
        graph: BranchingGraph,
        parent_idx: int,
        start_pos: np.ndarray,
        initial_dir: np.ndarray,
        base_radius: float,
        target_length: float,
        order: int,
        pipe_delta: float,
        rng: np.random.Generator,
        azimuth_rad: float
    ) -> list[int]:
        """Grows a single multi-segment branch curve."""
        segment_len = max(0.08, self.profile.internode_length_base_m * (self.profile.internode_decay_per_order ** order))
        num_segments = max(3, int(target_length / segment_len))
        actual_seg_len = target_length / num_segments

        branch_indices = []
        curr_pos = start_pos.copy()
        curr_dir = initial_dir.copy()
        curr_parent = parent_idx
        dist = 0.0

        for s in range(num_segments):
            s_rel = s / max(1, num_segments - 1)
            dist += actual_seg_len
            curr_pos += curr_dir * actual_seg_len

            # Taper along branch
            r = base_radius * ((1.0 - s_rel * 0.90) ** (1.0 / pipe_delta))
            r = max(0.002, float(r))

            # Apply tropisms: Gravitropism (negative bends up, positive bends down)
            grav_vector = np.array([0.0, 0.0, -self.profile.gravitropism], dtype=float)
            curr_dir += grav_vector * 0.15

            # Crookedness / natural curl
            if self.profile.crookedness > 0:
                curl = rng.normal(0.0, self.profile.crookedness * 0.06, size=3)
                curr_dir += curl

            norm = np.linalg.norm(curr_dir)
            if norm > 1e-6:
                curr_dir /= norm

            node = BranchNode(
                position=curr_pos.copy(),
                direction=curr_dir.copy(),
                radius=r,
                order=order,
                distance_along_stem=dist,
                relative_height=float(curr_pos[2]),
                parent_idx=curr_parent,
                azimuth_rad=azimuth_rad
            )
            idx = graph.add_node(node)
            branch_indices.append(idx)
            curr_parent = idx

        return branch_indices
