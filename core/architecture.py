"""
Botanical Branching Architecture.

Builds a tree skeleton from a compact set of architectural traits:

- Hallé-Oldeman model (Rauh, Massart, Troll, Attims, Corner) selects the growth
  logic (orthotropic vs plagiotropic laterals, tiers, unbranched monocaulous).
- Apical dominance separates excurrent crowns (one persistent leader: conifers,
  Liriodendron) from decurrent crowns (the leader splits into codominant stems:
  Quercus, Acer, Malus).
- The crown silhouette is a beta envelope E(u) = u^a (1-u)^b / max, with
  a = k p and b = k (1 - p), where p is the relative height of the widest point
  and k the crown "fullness" (low k -> columnar, high k -> peaked). Conical,
  ovoid, spherical, columnar and umbrella crowns are points of this 2-D space
  (cf. Horn 1971, The Adaptive Geometry of Trees).
- Phyllotaxis (divergence angle, whorl size) sets lateral azimuths; branch and
  leaf arrangement share it since laterals arise from axillary buds.
- Tropisms (gravitropism, phototropism), tortuosity and self-weight bending are
  integrated while each axis grows, so child axes always stay attached.
- Radii follow the pipe model: r_parent^Delta = sum r_child^Delta.
"""

from dataclasses import dataclass, field
from enum import Enum
import math
import numpy as np


class HalleOldemanModel(str, Enum):
    RAUH = "Rauh"        # Monopodial, rhythmic orthotropic branches (Quercus, Pinus, Hevea)
    MASSART = "Massart"  # Monopodial, plagiotropic horizontal tiers (Abies, Araucaria)
    TROLL = "Troll"      # Plagiotropic axes that become erect distally (Fagus, Ulmus)
    ATTIMS = "Attims"    # Continuous orthotropic branching (Eucalyptus, Alnus)
    CORNER = "Corner"    # Unbranched, leaves crowded at the apex (palms, cycads)


class PhyllotaxisType(str, Enum):
    SPIRAL = "Spiral"          # Golden angle 137.5 deg
    DECUSSATE = "Decussate"    # Opposite pairs rotated 90 deg
    DISTICHOUS = "Distichous"  # Alternate, two ranks (180 deg)
    WHORLED = "Whorled"        # 3+ per node


class CrownShape(str, Enum):
    CONICAL = "Conical"
    OVOID = "Ovoid"
    SPHERICAL = "Spherical"
    COLUMNAR = "Columnar"
    UMBRELLA = "Umbrella"
    WEEPING = "Weeping"


# (widest position p, fullness k)
CROWN_SHAPE_PARAMS: dict[CrownShape, tuple[float, float]] = {
    CrownShape.CONICAL: (0.04, 1.4),
    CrownShape.OVOID: (0.38, 1.8),
    CrownShape.SPHERICAL: (0.48, 1.5),
    CrownShape.COLUMNAR: (0.45, 0.5),
    CrownShape.UMBRELLA: (0.80, 1.6),
    CrownShape.WEEPING: (0.62, 1.2),
}


def crown_envelope(u: np.ndarray | float, widest: float, fullness: float) -> np.ndarray:
    """Beta crown envelope normalised to 1 at the widest point (u: 0 = crown base, 1 = apex)."""
    u = np.clip(np.asarray(u, dtype=float), 1e-4, 1.0 - 1e-4)
    p = float(np.clip(widest, 0.02, 0.98))
    k = max(0.1, float(fullness))
    a, b = k * p, k * (1.0 - p)
    e = (u ** a) * ((1.0 - u) ** b) / ((p ** a) * ((1.0 - p) ** b))
    return np.clip(e, 0.0, 1.0)


@dataclass
class ArchitectureProfile:
    """Parameters governing the 3D branching skeleton."""
    model: HalleOldemanModel = HalleOldemanModel.RAUH
    phyllotaxis: PhyllotaxisType = PhyllotaxisType.SPIRAL

    max_order: int = 3                    # 0 = trunk only, 1 = scaffolds, 2 = branches, 3 = twigs, 4 = shoots
    branch_angle_mean_deg: float = 48.0   # Scaffold insertion angle from the parent axis
    branch_angle_std_deg: float = 6.5
    twig_angle_mean_deg: float = 52.0     # Higher-order insertion angle
    divergence_angle_deg: float = 137.508
    whorl_size: int = 1                   # Laterals per node

    apical_dominance: float = 0.65        # 0 = shrubby/decurrent, 1 = single excurrent leader
    leader_count: int = 0                 # Codominant leaders (0 = from apical dominance)
    gravitropism: float = -0.22           # < 0 bends up, > 0 weeping
    phototropism: float = 0.35            # Outward (light-seeking) bias of laterals
    plagiotropy: float = 0.2              # 0 = laterals spiral in 3D, 1 = flattened horizontal sprays

    internode_length_base_m: float = 0.45
    internode_decay_per_order: float = 0.68  # Child / parent length ratio
    branch_frequency_per_meter: float = 3.0  # Scaffold nodes per meter of crown axis
    crookedness: float = 0.12

    crown_widest_position: float = 0.40
    crown_fullness: float = 1.6
    leaf_area_index: float = 4.5          # One-sided leaf area per unit crown projection area


@dataclass
class BranchNode:
    """Single 3D point along a botanical axis (compatibility view of an Axis sample)."""
    position: np.ndarray
    direction: np.ndarray
    radius: float
    order: int
    distance_along_stem: float
    relative_height: float
    parent_idx: int = -1
    children_indices: list[int] = field(default_factory=list)
    is_leaf_attachment: bool = False
    azimuth_rad: float = 0.0


@dataclass
class Axis:
    """A contiguous botanical axis (trunk, leader, branch, twig) stored as arrays."""
    positions: np.ndarray      # (k, 3)
    radii: np.ndarray          # (k,)
    order: int
    parent_axis: int = -1
    parent_sample: int = -1    # Index along the parent axis where this axis is inserted
    azimuth: float = 0.0
    n_children: int = 0
    aspect: np.ndarray | None = None   # Per-sample vertical/horizontal cross-section ratio (root I-beams)
    frame_up: bool = False             # Orient ring frames to world up (needed for vertical plank sections)

    @property
    def tangents(self) -> np.ndarray:
        t = np.empty_like(self.positions)
        t[1:-1] = self.positions[2:] - self.positions[:-2]
        t[0] = self.positions[1] - self.positions[0]
        t[-1] = self.positions[-1] - self.positions[-2]
        return t / np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)

    @property
    def arc_length(self) -> np.ndarray:
        return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(self.positions, axis=0), axis=1))])


class BranchingGraph:
    """Tree skeleton as a list of axes; `nodes`/`branches` give a flattened node view."""

    def __init__(self):
        self.axes: list[Axis] = []
        self._nodes_cache = None

    def add_axis(self, axis: Axis) -> int:
        self.axes.append(axis)
        self._nodes_cache = None
        if axis.parent_axis >= 0:
            self.axes[axis.parent_axis].n_children += 1
        return len(self.axes) - 1

    @property
    def branch_orders(self) -> list[int]:
        return [a.order for a in self.axes]

    @property
    def node_count(self) -> int:
        return int(sum(len(a.radii) for a in self.axes))

    def _flatten(self):
        nodes, branches, offsets = [], [], []
        off = 0
        for a in self.axes:
            offsets.append(off)
            off += len(a.radii)
        for ai, a in enumerate(self.axes):
            tan = a.tangents
            s = a.arc_length
            idx = []
            for i in range(len(a.radii)):
                if i > 0:
                    parent = offsets[ai] + i - 1
                elif a.parent_axis >= 0:
                    parent = offsets[a.parent_axis] + a.parent_sample
                else:
                    parent = -1
                nodes.append(BranchNode(a.positions[i], tan[i], float(a.radii[i]), a.order, float(s[i]),
                                        float(a.positions[i][2]), parent, azimuth_rad=a.azimuth))
                idx.append(offsets[ai] + i)
            branches.append(idx)
        for i, n in enumerate(nodes):
            if n.parent_idx >= 0:
                nodes[n.parent_idx].children_indices.append(i)
        for n in nodes:
            n.is_leaf_attachment = len(n.children_indices) == 0 and n.order >= 1
        self._nodes_cache = (nodes, branches)

    @property
    def nodes(self) -> list[BranchNode]:
        if self._nodes_cache is None:
            self._flatten()
        return self._nodes_cache[0]

    @property
    def branches(self) -> list[list[int]]:
        if self._nodes_cache is None:
            self._flatten()
        return self._nodes_cache[1]


UP = np.array([0.0, 0.0, 1.0])


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def _cross(a, b) -> np.ndarray:
    return np.array([a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]])


def _normalize(v: np.ndarray) -> np.ndarray:
    n = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    return v / n if n > 1e-9 else v


def _perp_frame(d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ref = (1.0, 0.0, 0.0) if abs(d[2]) > 0.9 else (0.0, 0.0, 1.0)
    u = _normalize(_cross(d, ref))
    return u, _cross(d, u)


def refine_base(pts: np.ndarray, radii: np.ndarray, extent: float, n: int = 4):
    """Inserts samples near the start of an axis so rapid girth changes (collars, flares) are resolved."""
    if len(pts) < 2 or extent <= 0.0:
        return pts, radii
    seg = np.linalg.norm(pts[1] - pts[0])
    targets = np.array([extent * f for f in (0.08, 0.2, 0.4, 0.7)[:n]])
    targets = targets[targets < seg * 0.85]
    if len(targets) == 0:
        return pts, radii
    u = (targets / seg)[:, None]
    new_pts = pts[0] + u * (pts[1] - pts[0])
    new_r = radii[0] + u[:, 0] * (radii[1] - radii[0])
    return np.vstack([pts[:1], new_pts, pts[1:]]), np.concatenate([radii[:1], new_r, radii[1:]])


class ArchitectureEngine:
    """Builds a complete botanical tree graph based on architectural traits."""

    MIN_RADIUS = 0.0025
    MAX_SEGMENTS = {0: 60, 1: 14, 2: 7, 3: 4, 4: 3}

    def __init__(self, profile: ArchitectureProfile = None, droop_fn=None, max_axes: int = 3500):
        self.profile = profile or ArchitectureProfile()
        # droop_fn(length_m, base_radius_m) -> tip deflection in meters
        self.droop_fn = droop_fn
        self.max_axes = max_axes

    # ------------------------------------------------------------------
    def generate_skeleton(
        self,
        total_height_m: float,
        crown_radius_m: float,
        crown_depth_m: float,
        base_radius_m: float,
        pipe_delta: float = 2.25,
        seed: int = 42,
        flare_amplitude: float = 0.0,
        flare_decay: float = 12.0,
    ) -> BranchingGraph:
        prof = self.profile
        rng = np.random.default_rng(seed)
        graph = BranchingGraph()
        H = float(total_height_m)
        crown_depth = float(np.clip(crown_depth_m, 0.2, H * 0.97))
        crown_base = H - crown_depth
        self._crown = (crown_base, crown_depth, float(crown_radius_m), H)
        self._delta = pipe_delta

        ad = float(np.clip(prof.apical_dominance, 0.0, 1.0))
        excurrent = prof.model in (HalleOldemanModel.MASSART, HalleOldemanModel.CORNER) or ad >= 0.6
        n_leaders = prof.leader_count if prof.leader_count > 0 else (
            1 if excurrent else int(np.clip(round(2 + (0.6 - ad) * 7), 2, 5)))
        if n_leaders <= 1:
            excurrent = True

        # 1. Trunk
        trunk_top = H if excurrent else crown_base + crown_depth * (0.08 + 0.35 * ad)
        end_r = None if excurrent else base_radius_m * (1.0 - trunk_top / H * 0.8) ** (1.0 / pipe_delta)
        if prof.model == HalleOldemanModel.CORNER:
            end_r = base_radius_m * 0.8  # Columnar monocot stem: no secondary thickening taper
        # The stem starts slightly below the soil surface so the collar never shows a flat bottom edge
        z0 = -0.6 * base_radius_m
        trunk = self._grow_axis(np.array([0.0, 0.0, z0]), UP.copy(), trunk_top - z0, base_radius_m, end_r, 0, rng,
                                seg_len=float(np.clip(H / 45.0, 0.12, 0.7)), tropism_up=0.25, weep=0.0,
                                crook=prof.crookedness * 0.35, droop=False, base_refine=True)
        if flare_amplitude > 0.0:
            # Root-collar flare is part of the stem radius, so axes inserted higher up see the real girth
            z = np.maximum(trunk.positions[:, 2], 0.0)
            trunk.radii = trunk.radii * (1.0 + flare_amplitude * np.exp(-flare_decay * z / H))
        trunk_id = graph.add_axis(trunk)
        self.trunk_id = trunk_id
        axes = [trunk_id]

        # 2. Codominant leaders (decurrent crowns)
        if not excurrent:
            split_pos = trunk.positions[-1]
            r_leader = trunk.radii[-1] * n_leaders ** (-1.0 / pipe_delta)
            az0 = rng.uniform(0, 2 * math.pi)
            for k in range(n_leaders):
                az = az0 + 2 * math.pi * k / n_leaders + rng.normal(0, 0.25)
                tilt = math.radians(np.clip(rng.normal(22 + 30 * (1 - ad), 5), 8, 60))
                d = np.array([math.sin(tilt) * math.cos(az), math.sin(tilt) * math.sin(az), math.cos(tilt)])
                length = (H - split_pos[2]) / max(0.4, math.cos(tilt)) * rng.uniform(0.82, 1.0)
                # Collar: each leader starts with nearly the stem's girth and tapers quickly to its pipe
                # radius, so the fork reads as one swelling crotch instead of tubes on a capped cylinder
                leader = self._grow_axis(split_pos.copy(), d, length, r_leader, None, 0, rng,
                                         seg_len=float(np.clip(H / 45.0, 0.12, 0.6)), tropism_up=0.10,
                                         weep=0.0, crook=prof.crookedness * 0.6, droop=False,
                                         collar_r=float(trunk.radii[-1]) * 0.9)
                leader.parent_axis, leader.parent_sample, leader.azimuth = trunk_id, len(trunk.radii) - 1, az
                axes.append(graph.add_axis(leader))

        if prof.model == HalleOldemanModel.CORNER or prof.max_order < 1:
            return graph

        # 3. Scaffold laterals (frequency shared among codominant leaders)
        scaffolds = []
        for i, ai in enumerate(axes):
            scaffolds += self._spawn_scaffolds(graph, ai, leader_axis=(i > 0), n_axes=len(axes), rng=rng)

        # 4. Higher orders under a global axis budget
        parents = scaffolds
        for order in range(2, prof.max_order + 1):
            wanted = [self._child_count(graph.axes[b], order) for b in parents]
            remaining = max(0, self.max_axes - len(graph.axes))
            total = sum(wanted)
            if total > remaining and total > 0:
                f = remaining / total
                wanted = [int(w * f + rng.random()) for w in wanted]
            children = []
            for b, w in zip(parents, wanted):
                if w > 0:
                    children += self._spawn_children(graph, b, order, w, rng)
            parents = children
        return graph

    # ------------------------------------------------------------------
    def _phyllotaxis_azimuths(self, node_count: int, rng) -> list[list[float]]:
        prof = self.profile
        whorl = max(1, int(prof.whorl_size))
        if prof.phyllotaxis == PhyllotaxisType.DECUSSATE:
            whorl, step = max(2, whorl), math.pi / 2
        elif prof.phyllotaxis == PhyllotaxisType.DISTICHOUS:
            step = math.pi
        elif prof.phyllotaxis == PhyllotaxisType.WHORLED:
            whorl = max(3, whorl)
            step = math.pi / whorl
        else:
            step = math.radians(prof.divergence_angle_deg)
        az = rng.uniform(0, 2 * math.pi)
        out = []
        for _ in range(node_count):
            out.append([az + 2 * math.pi * w / whorl + rng.normal(0, 0.08) for w in range(whorl)])
            az += step
        return out

    def _spawn_scaffolds(self, graph, axis_id, leader_axis: bool, n_axes: int, rng) -> list[int]:
        prof = self.profile
        crown_base, crown_depth, crown_r, H = self._crown
        axis = graph.axes[axis_id]
        pos = axis.positions
        tan = axis.tangents
        s_cum = axis.arc_length
        spacing = n_axes / max(0.2, prof.branch_frequency_per_meter)
        if prof.model == HalleOldemanModel.MASSART:
            spacing *= 1.6  # Tiers: fewer, fuller nodes
        above = pos[:, 2] >= crown_base
        start = s_cum[np.argmax(above)] if np.any(above) else s_cum[-1]
        stations = np.arange(start + spacing * rng.uniform(0.2, 0.8), s_cum[-1] - spacing * 0.3, spacing)
        if len(stations) == 0:
            return []
        azimuths = self._phyllotaxis_azimuths(len(stations), rng)
        specs = []
        for st, az_list in zip(stations, azimuths):
            ni = int(_clamp(int(np.searchsorted(s_cum, st)), 1, len(pos) - 1))
            node_pos, t = pos[ni], tan[ni]
            u = (node_pos[2] - crown_base) / max(0.1, crown_depth)
            if u < 0.0:
                continue
            env = float(crown_envelope(u, prof.crown_widest_position, prof.crown_fullness))
            for az in az_list:
                outward = np.array([math.cos(az), math.sin(az), 0.0])
                length = crown_r * env * rng.uniform(0.8, 1.15)
                if leader_axis and n_axes > 1:
                    # Laterals of codominant leaders: long outward, short toward the crown centre
                    radial = np.array([node_pos[0], node_pos[1], 0.0])
                    rn = np.linalg.norm(radial)
                    facing = float(np.dot(outward, radial / rn)) if rn > 1e-6 else 0.0
                    length = (crown_r - rn * facing) * env * rng.uniform(0.75, 1.05)
                    length *= 0.55 + 0.45 * max(0.0, facing)
                if length < 0.25:
                    continue
                angle_mean = prof.branch_angle_mean_deg
                if prof.model == HalleOldemanModel.MASSART:
                    angle_mean = max(angle_mean, 70.0)
                angle_mean *= 0.8 + 0.25 * (1.0 - u)  # Upper laterals more erect
                ang = math.radians(_clamp(rng.normal(angle_mean, prof.branch_angle_std_deg), 10.0, 100.0))
                lateral = _normalize(outward - (outward[0] * t[0] + outward[1] * t[1]) * t)
                d = _normalize(math.cos(ang) * t + math.sin(ang) * lateral)
                # The envelope is a horizontal radius: lengthen inclined laterals so their reach matches it
                length /= max(0.45, math.sqrt(d[0] * d[0] + d[1] * d[1]))
                specs.append((ni, az, d, length))
        if not specs:
            return []

        # Pipe model: the axis cross-section at the crown base is shared among its laterals in
        # proportion to the foliage they carry (~ length^2): r_i = R (L_i^2 / sum L^2)^(1/Delta)
        i_base = int(np.argmax(above)) if np.any(above) else 0
        r_base = float(axis.radii[i_base])
        weights = np.array([sp[3] ** 2 for sp in specs])
        shares = weights / weights.sum()
        created = []
        for (ni, az, d, length), share in zip(specs, shares):
            node_r = float(axis.radii[ni])
            r0 = max(self.MIN_RADIUS, min(node_r * 0.7, r_base * share ** (1.0 / self._delta)))
            child = self._grow_axis(pos[ni].copy(), d, length, r0, None, 1, rng,
                                    seg_len=max(0.12, prof.internode_length_base_m * 0.6),
                                    tropism_up=-prof.gravitropism, weep=max(0.0, prof.gravitropism),
                                    crook=prof.crookedness, droop=True, collar_r=min(0.85 * node_r, 1.7 * r0))
            child.parent_axis, child.parent_sample, child.azimuth = axis_id, ni, az
            created.append(graph.add_axis(child))
        return created

    def _child_count(self, axis: Axis, order: int) -> int:
        freq = self.profile.branch_frequency_per_meter * (1.7 ** (order - 1))
        max_children = {2: 12, 3: 8, 4: 5}.get(order, 4)
        return int(np.clip(round(axis.arc_length[-1] * freq * 0.6), 1, max_children))

    def _spawn_children(self, graph, axis_id: int, order: int, n: int, rng) -> list[int]:
        prof = self.profile
        axis = graph.axes[axis_id]
        if len(axis.radii) < 2:
            return []
        pos, tan, s_cum = axis.positions, axis.tangents, axis.arc_length
        parent_len = s_cum[-1]
        per_node = 2 if self.profile.phyllotaxis in (PhyllotaxisType.DECUSSATE, PhyllotaxisType.WHORLED) else 1
        n_nodes = max(1, int(math.ceil(n / per_node)))
        stations = np.sort(rng.uniform(0.18, 0.95, n_nodes)) * parent_len
        azimuths = self._phyllotaxis_azimuths(n_nodes, rng)
        created = []
        decay = prof.internode_decay_per_order
        for st, az_list in zip(stations, azimuths):
            ni = int(_clamp(int(np.searchsorted(s_cum, st)), 1, len(pos) - 1))
            node_pos, node_r, t = pos[ni], float(axis.radii[ni]), tan[ni]
            remaining = parent_len - s_cum[ni]
            u_vec, v_vec = _perp_frame(t)
            side = _cross(t, UP)
            side_ok = abs(side[0]) + abs(side[1]) > 1e-6 and prof.plagiotropy > 0.0
            if side_ok:
                side = _normalize(side)
            for az in az_list[:per_node]:
                length = min(parent_len * decay, remaining * 1.15 + 0.1) * rng.uniform(0.7, 1.1)
                if length < 0.08:
                    continue
                spiral = math.cos(az) * u_vec + math.sin(az) * v_vec
                if side_ok:
                    flat = side * (1.0 if math.cos(az) >= 0 else -1.0)
                    lateral = _normalize((1 - prof.plagiotropy) * spiral + prof.plagiotropy * flat)
                else:
                    lateral = spiral
                ang = math.radians(_clamp(rng.normal(prof.twig_angle_mean_deg, 7.0), 10.0, 95.0))
                d = _normalize(math.cos(ang) * t + math.sin(ang) * lateral)
                ratio = _clamp(length / max(1e-3, parent_len), 0.1, 0.8)
                r0 = max(self.MIN_RADIUS, min(node_r * 0.85, node_r * ratio ** (1.0 / self._delta)))
                child = self._grow_axis(node_pos.copy(), d, length, r0, None, order, rng,
                                        seg_len=max(0.06, prof.internode_length_base_m * decay ** order),
                                        tropism_up=-prof.gravitropism * 0.8,
                                        weep=max(0.0, prof.gravitropism) * 1.4, crook=prof.crookedness,
                                        droop=True, collar_r=min(0.85 * node_r, 1.6 * r0))
                child.parent_axis, child.parent_sample, child.azimuth = axis_id, ni, az
                created.append(graph.add_axis(child))
        return created

    def _grow_axis(self, start, direction, length, r0, end_radius, order, rng,
                   seg_len, tropism_up, weep, crook, droop, collar_r=None, base_refine=False) -> Axis:
        """Integrates one axis with tropisms, tortuosity and self-weight bending (scalar math for speed)."""
        prof = self.profile
        n_seg = int(_clamp(round(length / max(1e-3, seg_len)), 2, self.MAX_SEGMENTS.get(order, 3)))
        ds = length / n_seg
        deflection = 0.0
        if droop and self.droop_fn is not None and order >= 1:
            deflection = float(self.droop_fn(length, r0))
        ox, oy = float(direction[0]), float(direction[1])
        on = math.hypot(ox, oy)
        ox, oy = (ox / on, oy / on) if on > 1e-6 else (0.0, 0.0)
        photo = prof.phototropism * 0.04 if order >= 1 else 0.0
        noise = rng.normal(0.0, crook * 0.08, (n_seg, 3)) if crook > 0 else np.zeros((n_seg, 3))

        dx, dy, dz = (float(c) for c in direction)
        px, py, pz = (float(c) for c in start)
        pts = np.empty((n_seg + 1, 3))
        pts[0] = (px, py, pz)
        prev_drop = 0.0
        for s in range(1, n_seg + 1):
            s_rel = s / n_seg
            dz += tropism_up * 0.12 - weep * 0.10 * s_rel
            dx += ox * photo + noise[s - 1, 0]
            dy += oy * photo + noise[s - 1, 1]
            dz += noise[s - 1, 2]
            n = math.sqrt(dx * dx + dy * dy + dz * dz) or 1.0
            dx, dy, dz = dx / n, dy / n, dz / n
            px += dx * ds
            py += dy * ds
            pz += dz * ds
            # Self-weight cantilever deflection, applied while growing so children stay attached
            drop = deflection * s_rel * s_rel
            pz -= drop - prev_drop
            prev_drop = drop
            pts[s] = (px, py, pz)

        s_rel = np.linspace(0.0, 1.0, n_seg + 1)
        if end_radius is not None:
            radii = r0 + (end_radius - r0) * s_rel
        else:
            # Near-conical taper to a fine tip (radius falls as laterals take over the pipe flow)
            radii = r0 * np.maximum(0.0, 1.0 - 0.94 * s_rel) ** 0.85
        radii = np.maximum(radii, self.MIN_RADIUS * (0.6 if order >= 2 else 1.0))
        radii[0] = r0
        if (collar_r is not None and collar_r > r0 * 1.02) or base_refine:
            pts, radii = refine_base(pts, radii, (collar_r or r0) * 3.0)
        if collar_r is not None and collar_r > r0 * 1.02:
            # Branch collar / zone of rapid taper: girth falls exponentially from the insertion
            dist = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
            radii = radii + (collar_r - r0) * np.exp(-dist / (0.6 * collar_r + 0.8 * r0))
        return Axis(pts, radii, order)
