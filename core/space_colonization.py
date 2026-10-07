"""
Space Colonization Algorithm for Leaf Venation Patterns (Runions et al., 2005).
Reference:
Adam Runions, Martin Fuhrer, Brendan Lane, Pavol Federl, Anne-Gaëlle Rolland-Lagan,
and Przemyslaw Prusinkiewicz.
"Modeling and visualization of leaf venation patterns."
ACM Transactions on Graphics (SIGGRAPH 2005) 24(3), pp. 702–711.
"""

from dataclasses import dataclass, field
import math
import numpy as np


@dataclass
class AuxinSource:
    """Hormone/auxin attractor site within the developing leaf lamina."""
    position: np.ndarray  # [x, y] in meters
    is_active: bool = True


@dataclass
class SpaceColonizationNode:
    """Node in the growing venation vascular network."""
    position: np.ndarray      # [x, y] in meters
    parent_idx: int = -1
    children_indices: list[int] = field(default_factory=list)
    flux: float = 1.0         # Cumulative hydraulic flux for Murray's law scaling
    radius_m: float = 0.0001
    order: int = 1


@dataclass
class SpaceColonizationConfig:
    """Parameters for the Runions et al. (2005) auxin space colonization algorithm."""
    # Distance of influence (d_I): max radius an auxin source can attract a vein
    distance_of_influence_m: float = 0.015

    # Kill distance (d_K): threshold where vein absorbs auxin source
    kill_distance_m: float = 0.0035

    # Step length (delta): segment growth increment
    step_size_m: float = 0.0025

    # Murray / Leonardo flux exponent (gamma): r = r_base * flux^(1 / gamma)
    # Murray's hydraulic optimum = 3.0; Leonardo structural rule = 2.0 - 2.5
    murray_exponent_gamma: float = 2.6

    # Base vein radius for terminal minor veins
    minor_vein_radius_base_m: float = 0.00006

    # Maximum simulation iterations
    max_iterations: int = 120

    # Total auxin attractors to distribute across the lamina
    num_auxin_sources: int = 350


class SpaceColonizationEngine:
    """Simulates biological auxin-driven vascular leaf venation according to Runions et al. (2005)."""

    def __init__(self, config: SpaceColonizationConfig = None):
        self.config = config or SpaceColonizationConfig()

    def generate_venation(
        self,
        blade_length_m: float,
        blade_width_m: float,
        boundary_pts_2d: np.ndarray = None,
        seed: int = 42
    ) -> list[SpaceColonizationNode]:
        """
        Executes the Runions et al. auxin colonization simulation inside the leaf blade.
        """
        rng = np.random.default_rng(seed)

        # -------------------------------------------------------------
        # 1. Distribute Auxin Sources Inside Leaf Boundary
        # -------------------------------------------------------------
        auxin_sources: list[AuxinSource] = []
        target_sources = self.config.num_auxin_sources

        # Generate candidates within the leaf envelope
        attempts = 0
        while len(auxin_sources) < target_sources and attempts < target_sources * 6:
            attempts += 1
            y = rng.uniform(0.08 * blade_length_m, 0.95 * blade_length_m)
            v_rel = y / blade_length_m

            # Leaf half-width envelope
            envelope = math.sin(math.pi * (v_rel ** 0.65))
            half_w = (blade_width_m * 0.5) * envelope

            if half_w <= 0.001:
                continue

            x = rng.uniform(-half_w, half_w)
            auxin_sources.append(AuxinSource(position=np.array([x, y], dtype=float)))

        # -------------------------------------------------------------
        # 2. Initialize Root Vein Nodes (Primary Midrib)
        # -------------------------------------------------------------
        nodes: list[SpaceColonizationNode] = []

        # Primary midrib trunk starting at base (petiole insertion)
        midrib_segs = max(8, int(blade_length_m / self.config.step_size_m * 0.7))
        dy = (blade_length_m * 0.7) / midrib_segs

        curr_parent = -1
        for i in range(midrib_segs):
            y_pos = i * dy
            node = SpaceColonizationNode(
                position=np.array([0.0, y_pos], dtype=float),
                parent_idx=curr_parent,
                order=1
            )
            idx = len(nodes)
            nodes.append(node)
            if curr_parent >= 0:
                nodes[curr_parent].children_indices.append(idx)
            curr_parent = idx

        # -------------------------------------------------------------
        # 3. Space Colonization Growth Iterations
        # -------------------------------------------------------------
        d_I = self.config.distance_of_influence_m
        d_K = self.config.kill_distance_m
        delta = self.config.step_size_m

        for iteration in range(self.config.max_iterations):
            active_sources = [s for s in auxin_sources if s.is_active]
            if not active_sources:
                break

            # Find closest node for each active source
            source_attractions: dict[int, list[np.ndarray]] = {}

            node_positions = np.array([n.position for n in nodes])

            for source in active_sources:
                diffs = node_positions - source.position
                dists = np.linalg.norm(diffs, axis=1)

                min_idx = int(np.argmin(dists))
                min_dist = dists[min_idx]

                if min_dist <= d_I:
                    if min_idx not in source_attractions:
                        source_attractions[min_idx] = []
                    # Vector from node to source
                    vec = (source.position - nodes[min_idx].position) / max(1e-6, min_dist)
                    source_attractions[min_idx].append(vec)

            if not source_attractions:
                break

            # Grow new vein segments towards average auxin vector
            new_nodes_added = False
            for node_idx, dir_vectors in source_attractions.items():
                avg_dir = np.mean(dir_vectors, axis=0)
                norm = np.linalg.norm(avg_dir)
                if norm < 1e-6:
                    continue
                avg_dir = avg_dir / norm

                parent_pos = nodes[node_idx].position
                new_pos = parent_pos + avg_dir * delta

                # Check leaf boundaries
                v_rel = new_pos[1] / blade_length_m
                if 0.0 <= v_rel <= 1.0:
                    envelope = math.sin(math.pi * (v_rel ** 0.65))
                    half_w = (blade_width_m * 0.5) * envelope
                    if abs(new_pos[0]) <= half_w:
                        new_idx = len(nodes)
                        order = 2 if nodes[node_idx].order == 1 else 3
                        new_node = SpaceColonizationNode(
                            position=new_pos,
                            parent_idx=node_idx,
                            order=order
                        )
                        nodes.append(new_node)
                        nodes[node_idx].children_indices.append(new_idx)
                        new_nodes_added = True

            # Absorb/kill auxin sources close to any vein node
            node_positions = np.array([n.position for n in nodes])
            for source in active_sources:
                dists = np.linalg.norm(node_positions - source.position, axis=1)
                if np.min(dists) <= d_K:
                    source.is_active = False

            if not new_nodes_added:
                break

        # -------------------------------------------------------------
        # 4. Murray's Law Flux and Radius Calculation
        # -------------------------------------------------------------
        # Post-order traversal (leaves to root) to compute cumulative fluxes
        for i in reversed(range(len(nodes))):
            flux_sum = 1.0
            for child_idx in nodes[i].children_indices:
                flux_sum += nodes[child_idx].flux
            nodes[i].flux = flux_sum

        # r = r_base * flux^(1 / gamma)
        gamma = self.config.murray_exponent_gamma
        r_base = self.config.minor_vein_radius_base_m

        for node in nodes:
            node.radius_m = float(r_base * (node.flux ** (1.0 / gamma)))

        return nodes
