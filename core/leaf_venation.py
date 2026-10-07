"""
Leaf Venation Architecture and Network Modeling.
Directly grounded in empirical research:
- "Investigating the Functional and Architectural Diversity of Leaf Venation Networks"
  (Matos, Boakye, Antonio, Carlos, Chu, Duarte et al., 2025; Dryad: 10.5061/dryad.1g1jwsv36)
- Leaf Vein Dataset (LVD2021) & HALVS hierarchical venation segmentation
- Murray's hydraulic scaling law & Leonardo pipe scaling for vein tapering.
"""

from dataclasses import dataclass, field
from enum import Enum
import math
import numpy as np


class VenationPattern(str, Enum):
    CRASPEDODROMOUS = "Craspedodromous"  # Secondaries terminate directly at margin (Betula, Ulmus)
    BROCHIDODROMOUS = "Brochidodromous"  # Secondaries loop and join into marginal arches (Ficus, Citrus)
    ACTINODROMOUS = "Actinodromous"      # Multiple primaries radiate from base (Acer, Platanus)
    PARALLELODROMOUS = "Parallelodromous"# Parallel longitudinal veins (Monocots, Poaceae)
    FLABELLATE = "Flabellate"            # Equal dichotomous branching fan (Ginkgo)


@dataclass
class VenationProfile:
    """Empirical leaf venation traits from botanical garden measurements."""
    pattern: VenationPattern = VenationPattern.CRASPEDODROMOUS

    # Quantitative traits from Dryad UCBG 122spp dataset
    vla_mm_per_mm2: float = 6.8         # Vein Length per Unit Area (typically 3.0 to 16.0 mm/mm^2)
    secondary_vein_pairs: int = 9       # Number of secondary vein pairs along the midrib
    divergence_angle_deg: float = 46.0  # Insertion angle between secondary and primary vein
    divergence_angle_std: float = 4.2   # Variation across leaf length

    # Vein thickness hierarchy (Murray/Leonardo hydraulic scaling)
    primary_radius_mm: float = 0.85     # Base midrib radius in mm
    secondary_radius_mm: float = 0.28   # Secondary vein radius in mm
    tertiary_radius_mm: float = 0.08    # Tertiary intercostal vein radius in mm

    # Reticulation and areolation
    reticulation_density: float = 0.65  # Loopiness / anastomosing connectivity (0=tree, 1=closed mesh)
    areole_diameter_mm: float = 0.75    # Average diameter of minor vein enclosed regions

    # Curvature of secondary veins towards apex (0=straight, 1=strongly arcuate)
    secondary_curvature: float = 0.45


@dataclass
class VeinNode:
    """Single coordinate point in a leaf venation graph."""
    position: np.ndarray    # [x, y, z] in meters
    radius_m: float         # Vein thickness in meters
    order: int              # 1=primary, 2=secondary, 3=tertiary, 4=minor
    parent_idx: int = -1
    children_indices: list[int] = field(default_factory=list)


class LeafVeinNetwork:
    """Graph of interconnected veins across the leaf blade."""

    def __init__(self):
        self.nodes: list[VeinNode] = []
        self.segments: list[list[int]] = []  # Vein path node indices

    def add_node(self, node: VeinNode) -> int:
        idx = len(self.nodes)
        self.nodes.append(node)
        if node.parent_idx >= 0 and node.parent_idx < len(self.nodes):
            self.nodes[node.parent_idx].children_indices.append(idx)
        return idx


class VenationEngine:
    """Constructs empirical leaf venation networks matching botanical architectures."""

    def __init__(self, profile: VenationProfile = None):
        self.profile = profile or VenationProfile()

    def generate_network(
        self,
        blade_length_m: float,
        blade_width_m: float,
        boundary_pts: np.ndarray = None
    ) -> LeafVeinNetwork:
        """
        Generates the hierarchical vein graph for a leaf blade.
        """
        network = LeafVeinNetwork()
        rng = np.random.default_rng(42)

        # -------------------------------------------------------------
        # 1. Primary Midrib Vein (Order 1)
        # -------------------------------------------------------------
        primary_segs = max(12, int(blade_length_m * 400.0))
        r1_base = self.profile.primary_radius_mm * 0.001

        primary_indices = []
        for i in range(primary_segs + 1):
            v_rel = i / primary_segs
            y = v_rel * blade_length_m
            x = 0.0
            # Midrib tapers from base to tip
            r = r1_base * ((1.0 - v_rel * 0.85) ** 0.5)

            node = VeinNode(
                position=np.array([x, y, 0.0], dtype=float),
                radius_m=max(0.00005, float(r)),
                order=1,
                parent_idx=primary_indices[-1] if primary_indices else -1
            )
            idx = network.add_node(node)
            primary_indices.append(idx)

        network.segments.append(primary_indices)

        # -------------------------------------------------------------
        # 2. Secondary Veins (Order 2)
        # -------------------------------------------------------------
        n_pairs = self.profile.secondary_vein_pairs
        r2_base = self.profile.secondary_radius_mm * 0.001
        div_rad = math.radians(self.profile.divergence_angle_deg)

        # Place secondary veins along the midrib between 10% and 90% length
        pair_positions = np.linspace(0.12, 0.88, n_pairs)

        left_secondaries = []
        right_secondaries = []

        for p_rel in pair_positions:
            midrib_idx = int(p_rel * primary_segs)
            origin_node = network.nodes[primary_indices[midrib_idx]]

            # Leaf half-width at this relative height (assumes smooth envelope)
            envelope = math.sin(math.pi * (p_rel ** 0.7))
            half_w = (blade_width_m * 0.5) * envelope

            if half_w < 0.002:
                continue

            for side in [-1.0, 1.0]:  # -1 for left, +1 for right
                sec_path = []
                parent_idx = primary_indices[midrib_idx]

                # Secondary vein length
                target_len = half_w / math.sin(div_rad) * 0.95
                n_sec_segs = max(5, int(target_len * 500.0))
                dl = target_len / n_sec_segs

                curr_pos = origin_node.position.copy()
                # Angle curves upward as it approaches margin
                for s in range(1, n_sec_segs + 1):
                    s_rel = s / n_sec_segs
                    ang = div_rad - (self.profile.secondary_curvature * s_rel * 0.4)

                    dx = side * math.sin(ang) * dl
                    dy = math.cos(ang) * dl

                    # Brochidodromous: loops back upward near margin
                    if self.profile.pattern == VenationPattern.BROCHIDODROMOUS and s_rel > 0.75:
                        dy += dl * 0.5
                        dx *= 0.7

                    curr_pos = curr_pos + np.array([dx, dy, 0.0])
                    r_sec = r2_base * (1.0 - s_rel * 0.6)

                    sec_node = VeinNode(
                        position=curr_pos.copy(),
                        radius_m=max(0.00003, float(r_sec)),
                        order=2,
                        parent_idx=parent_idx
                    )
                    s_idx = network.add_node(sec_node)
                    sec_path.append(s_idx)
                    parent_idx = s_idx

                network.segments.append(sec_path)
                if side < 0:
                    left_secondaries.append(sec_path)
                else:
                    right_secondaries.append(sec_path)

        # -------------------------------------------------------------
        # 3. Tertiary Intercostal Veins (Order 3)
        # -------------------------------------------------------------
        # Connect consecutive secondary veins (ladder-like or reticulate)
        r3_base = self.profile.tertiary_radius_mm * 0.001
        for side_list in [left_secondaries, right_secondaries]:
            for p in range(len(side_list) - 1):
                sec1 = side_list[p]
                sec2 = side_list[p + 1]

                # Bridge veins between mid-points of secondary pairs
                n_bridges = max(2, int(len(sec1) * 0.45))
                for b in range(1, n_bridges):
                    idx1 = sec1[min(len(sec1) - 1, b * 2)]
                    idx2 = sec2[min(len(sec2) - 1, b * 2)]

                    pos1 = network.nodes[idx1].position
                    pos2 = network.nodes[idx2].position

                    # Create direct or slightly curved bridge
                    mid_pos = (pos1 + pos2) * 0.5
                    bridge_node = VeinNode(
                        position=mid_pos,
                        radius_m=r3_base,
                        order=3,
                        parent_idx=idx1
                    )
                    br_idx = network.add_node(bridge_node)
                    network.segments.append([idx1, br_idx, idx2])

        return network
