"""
Foliage placement on the branching skeleton (vectorized).

Leaves are placed at nodes along the youngest shoots following the plant's
phyllotaxis (spiral, decussate, distichous or whorled), inserted at the petiole
angle, and oriented so the lamina faces the light (+Z) as far as the petiole
direction allows. Distichous leaves on horizontal shoots therefore form flat
sprays (Fagus, Ulmus), and Corner-model plants (palms) get an apical rosette.
"""

from dataclasses import dataclass
import math
import numpy as np

from .architecture import BranchingGraph, ArchitectureProfile, HalleOldemanModel, PhyllotaxisType

UP = np.array([0.0, 0.0, 1.0])


@dataclass
class FoliageInstances:
    positions: np.ndarray  # (N, 3) petiole base
    axis_y: np.ndarray     # (N, 3) leaf direction (petiole -> apex)
    axis_z: np.ndarray     # (N, 3) adaxial normal
    scales: np.ndarray     # (N,)
    randoms: np.ndarray    # (N,) per-leaf random value in [0, 1)

    def __len__(self):
        return len(self.positions)

    @property
    def axis_x(self) -> np.ndarray:
        return np.cross(self.axis_y, self.axis_z)


def _normalize_rows(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.maximum(n, 1e-9)


def _orient(directions: np.ndarray, roll: np.ndarray) -> np.ndarray:
    """Adaxial normals facing up as much as possible, rotated by `roll` radians about the leaf axis."""
    x = np.cross(directions, UP)
    bad = np.linalg.norm(x, axis=1) < 1e-4
    x[bad] = np.array([1.0, 0.0, 0.0])
    x = _normalize_rows(x)
    z = np.cross(x, directions)
    flip = z[:, 2] < 0
    z[flip] *= -1.0
    z = _normalize_rows(z)
    # Roll about the leaf axis
    xr = np.cross(directions, z)
    c, s = np.cos(roll)[:, None], np.sin(roll)[:, None]
    return _normalize_rows(c * z + s * xr)


def place_foliage(
    graph: BranchingGraph,
    arch: ArchitectureProfile,
    unit_length_m: float,
    petiole_angle_deg: float = 50.0,
    density: float = 1.0,
    budget: int = 20000,
    seed: int = 42,
    spacing_factor: float = 0.35,
    shoot_cards: bool = False,
    target_units: int | None = None,
    mean_leaf_angle_deg: float = 40.0,
) -> FoliageInstances:
    """
    Places foliar units along the leaf-bearing shoots. With `shoot_cards` each
    unit is a leafy shoot (a card carrying several leaves) that continues the
    twig at a narrow angle, and every twig tip carries a terminal shoot.

    `target_units` (e.g. from leaf area index x crown area / unit leaf area) sets
    how many units to place; it is scaled by `density` and capped by `budget`.
    Without it, spacing follows the unit length.
    """
    rng = np.random.default_rng(seed + 101)
    if density <= 0.0 or not graph.axes:
        z = np.zeros((0, 3))
        return FoliageInstances(z, z, z, np.zeros(0), np.zeros(0))

    if arch.model == HalleOldemanModel.CORNER:
        return _apical_rosette(graph, density, rng)

    max_order = max(a.order for a in graph.axes) if graph.axes else 0
    whorl = 2 if arch.phyllotaxis == PhyllotaxisType.DECUSSATE else (
        max(3, arch.whorl_size) if arch.phyllotaxis == PhyllotaxisType.WHORLED else 1)
    if arch.phyllotaxis == PhyllotaxisType.DECUSSATE:
        step = math.pi / 2
    elif arch.phyllotaxis == PhyllotaxisType.DISTICHOUS:
        step = math.pi
    elif arch.phyllotaxis == PhyllotaxisType.WHORLED:
        step = math.pi / whorl
    else:
        step = math.radians(arch.divergence_angle_deg)

    # Leaf-bearing shoots: youngest order entirely, previous order on its distal half
    segments = []
    total_len = 0.0
    for axis in graph.axes:
        order = axis.order
        if order == 0 and max_order > 0:
            continue
        if order == max_order:
            start_frac = 0.08
        elif order == max_order - 1:
            start_frac = 0.3
        else:
            continue
        pos = axis.positions
        s = axis.arc_length
        if s[-1] < 1e-3:
            continue
        segments.append((pos, s, start_frac))
        total_len += s[-1] * (1.0 - start_frac)
    if not segments:
        z = np.zeros((0, 3))
        return FoliageInstances(z, z, z, np.zeros(0), np.zeros(0))

    if shoot_cards:
        spacing_factor = 0.55
    n_tips = len(segments) if shoot_cards else 0
    if target_units is not None:
        n_units = int(min(budget, max(1.0, target_units * density)))
    else:
        n_units = int(min(budget, total_len / max(1e-3, unit_length_m * spacing_factor) * density * whorl))
    per_unit = total_len / max(1, n_units - n_tips)
    if shoot_cards:
        # Shoots arise all around the bearing axis: several units per station when crowded
        whorl = int(np.clip(math.ceil(unit_length_m * spacing_factor * 0.6 / max(1e-4, per_unit)), 1, 4))
    spacing = max(0.004, per_unit * whorl)

    P, T, AZ = [], [], []
    tips_p, tips_t = [], []
    for pos, s, start_frac in segments:
        end = s[-1] - (unit_length_m * 0.3 if shoot_cards else 0.0)
        stations = np.arange(s[-1] * start_frac + spacing * rng.uniform(0.1, 0.9), end, spacing)
        if shoot_cards:
            tips_p.append(pos[-1])
            tips_t.append(pos[-1] - pos[-2])
        if len(stations) == 0:
            if shoot_cards:
                continue
            stations = np.array([s[-1] * 0.95])
        px = np.stack([np.interp(stations, s, pos[:, k]) for k in range(3)], axis=1)
        seg_i = np.clip(np.searchsorted(s, stations) - 1, 0, len(pos) - 2)
        tan = _normalize_rows(pos[seg_i + 1] - pos[seg_i])
        az0 = rng.uniform(0, 2 * math.pi)
        node_az = az0 + np.arange(len(stations)) * (step if not shoot_cards else math.radians(137.5))
        for w in range(whorl):
            P.append(px)
            T.append(tan)
            AZ.append(node_az + 2 * math.pi * w / whorl + rng.normal(0.0, 0.15, len(stations)))
    P = np.concatenate(P)
    T = np.concatenate(T)
    AZ = np.concatenate(AZ)

    # Frame around each shoot with u horizontal, so distichous leaves spread sideways
    u = np.cross(T, UP)
    bad = np.linalg.norm(u, axis=1) < 1e-4
    u[bad] = np.array([1.0, 0.0, 0.0])
    u = _normalize_rows(u)
    v = np.cross(T, u)
    radial = np.cos(AZ)[:, None] * u + np.sin(AZ)[:, None] * v
    if shoot_cards:
        phi = np.radians(rng.uniform(15.0, 45.0, len(P)))
    else:
        phi = np.radians(np.clip(petiole_angle_deg + rng.normal(0.0, 8.0, len(P)), 5.0, 120.0))
    D = _normalize_rows(np.cos(phi)[:, None] * T + np.sin(phi)[:, None] * radial)
    if tips_p:
        P = np.vstack([P, np.array(tips_p)])
        D = np.vstack([D, _normalize_rows(np.array(tips_t))])
    # Phototropic petiole reorientation lifts the blade slightly toward the light
    D = _normalize_rows(D + UP * 0.25 * max(0.0, arch.phototropism))
    # Lamina inclination: roll each unit about its axis so normals follow the species' mean leaf angle
    mla = math.radians(np.clip(mean_leaf_angle_deg, 0.0, 89.0))
    roll = np.abs(rng.normal(mla, 0.35 * mla + 0.05, len(P))) * rng.choice([-1.0, 1.0], len(P))
    Z = _orient(D, roll)
    scales = rng.uniform(0.8, 1.15, len(P))
    return FoliageInstances(P, D, Z, scales, rng.random(len(P)))


def _apical_rosette(graph: BranchingGraph, density: float, rng) -> FoliageInstances:
    top = graph.axes[0].positions[-1]
    n = int(np.clip(round(60 * density), 4, 160))
    i = np.arange(n)
    az = i * math.radians(137.508)
    # Young leaves (end of list) erect, old leaves arching down
    theta = np.radians(12.0 + 100.0 * (1.0 - i / max(1, n - 1)) ** 0.9)
    D = np.stack([np.sin(theta) * np.cos(az), np.sin(theta) * np.sin(az), np.cos(theta)], axis=1)
    P = np.repeat(top[None, :], n, axis=0)
    Z = _orient(D, rng.normal(0.0, 0.15, n))
    scales = rng.uniform(0.85, 1.1, n) * (0.7 + 0.3 * (1.0 - i / max(1, n - 1)))
    return FoliageInstances(P, D, Z, scales, rng.random(n))
