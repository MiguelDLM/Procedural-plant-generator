"""
Blender side of fruits: properties generated from core.fruit, one material driven by the fruit attributes,
fruits (or bunches) instanced on trees and shown on their own (growth form 'Fruit').
"""

import copy
import hashlib
import math
from dataclasses import fields

import numpy as np

try:
    import bpy
    BLENDER_AVAILABLE = True
except ImportError:
    BLENDER_AVAILABLE = False

try:
    from ..core.fruit import FruitProfile, hanging_fruit
    from ..core.fruit_db import FRUIT_CATALOG, FRUIT_RANGES, TREE_FRUITS, VINE_FRUITS
    from ..core.mesh_engine import MeshData
    from .flowers import _props_for, _child, _assign, _instancer_nodes, _euler_xyz
    from .mesh_builder import populate_mesh
except (ImportError, ValueError):
    from core.fruit import FruitProfile, hanging_fruit
    from core.fruit_db import FRUIT_CATALOG, FRUIT_RANGES, TREE_FRUITS, VINE_FRUITS
    from core.mesh_engine import MeshData
    from blender.flowers import _props_for, _child, _assign, _instancer_nodes, _euler_xyz
    from blender.mesh_builder import populate_mesh

FRT = "frt_"
LAYOUT = [
    ("Fruit Shape", ["kind", "length_cm", "diameter_cm", "widest_position", "bluntness", "distal_point", "neck",
                     "stalk_cavity", "calyx_basin", "ribs", "rib_depth", "lopsided"]),
    ("Calyx & Stalk", ["crown_lobes", "crown_length_cm", "crown_flare_deg", "crown_position", "crown_color",
                       "stalk_length_cm", "stalk_radius_mm", "stalk_color"]),
    ("Bunch", ["cluster_berries", "cluster_length_cm", "cluster_width_cm", "shoulders", "compactness",
               "pedicel_cm", "peduncle_cm"]),
    ("Fruit Colour", ["color", "blush_color", "blush", "stripes", "stripe_count", "stripe_width", "stripe_color",
                      "dots", "dot_color", "russet",
                      "bloom", "gloss"]),
]
PARTS = ("_FruitProto", "_Fruits", "_FruitBody")


def fruit_properties(update) -> dict:
    if not BLENDER_AVAILABLE:
        return {}
    return _props_for(FruitProfile, FRT, FRUIT_RANGES, update)


def write_fruit_to_props(props, key):
    sp = FRUIT_CATALOG.get(key)
    if sp is None:
        return
    for f in fields(sp.fruit):
        name = FRT + f.name
        if hasattr(props, name):
            v = getattr(sp.fruit, f.name)
            setattr(props, name, v.value if hasattr(v, "value") else v)


def fruit_from_props(props) -> FruitProfile:
    sp = FRUIT_CATALOG.get(props.fruit_species) or next(iter(FRUIT_CATALOG.values()))
    obj = copy.deepcopy(sp.fruit)
    for f in fields(obj):
        name = FRT + f.name
        if not hasattr(props, name):
            continue
        cur, val = getattr(obj, f.name), getattr(props, name)
        if hasattr(cur, "value"):
            val = type(cur)(val)
        elif isinstance(cur, tuple):
            val = tuple(float(x) for x in val)
        elif isinstance(cur, bool):
            val = bool(val)
        elif isinstance(cur, int):
            val = int(val)
        else:
            val = float(val)
        setattr(obj, f.name, val)
    return obj


def fruit_values(props) -> dict:
    return {f.name: getattr(props, FRT + f.name) for f in fields(FruitProfile) if hasattr(props, FRT + f.name)}


def default_fruit_for(props):
    if props.growth_form == 'Tree':
        return TREE_FRUITS.get(props.species_enum)
    if props.growth_form == 'Vine':
        return VINE_FRUITS.get(props.vine_species)
    return None


def sync_fruits_to_plant(props):
    """Load the default fruit of the selected plant preset into the fruit sliders; fruits are shown when the
    preset bears any (no update)."""
    key = default_fruit_for(props)
    if props.growth_form in ('Tree', 'Vine'):
        props.show_fruits = key is not None
    if key and key in FRUIT_CATALOG:
        props.fruit_species = key
        write_fruit_to_props(props, key)


# -----------------------------------------------------------------------------
# Material
# -----------------------------------------------------------------------------
def _key(*parts) -> str:
    return hashlib.sha1(repr(parts).encode()).hexdigest()[:12]


def fruit_material(name, p: FruitProfile):
    from .succulents import _new_mat, _attr, _mix, _noise, _math
    from .materials import _srgb_to_linear, _set
    mat, nt, b = _new_mat(name)

    def rgb(c, loc):
        n = nt.nodes.new('ShaderNodeRGB')
        n.location = loc
        n.outputs[0].default_value = _srgb_to_linear(c)
        return n.outputs[0]

    u = _attr(nt, "fruit_u", (-1700, 200)).outputs['Fac']
    rib = _attr(nt, "fruit_rib", (-1700, 0)).outputs['Fac']
    crown = _attr(nt, "fruit_crown", (-1700, -200)).outputs['Fac']
    stalk = _attr(nt, "fruit_stalk", (-1700, -400)).outputs['Fac']
    rnd = _attr(nt, "fruit_random", (-1700, -600)).outputs['Fac']
    info = nt.nodes.new('ShaderNodeObjectInfo')                   # Per-instance variation
    info.location = (-1700, -800)
    w = _math(nt, 'MULTIPLY_ADD', info.outputs['Random'], 20.0, (-1500, -700), c=rnd)
    tc = nt.nodes.new('ShaderNodeTexCoord')
    tc.location = (-1900, -1000)

    def noise4(scale, detail, loc):
        n = _noise(nt, tc.outputs['Object'], scale, detail, loc)
        n.noise_dimensions = '4D'
        nt.links.new(w, n.inputs['W'])
        return n

    col = rgb(p.color, (-1100, 600))
    # Sun-side blush: one side of the fruit (angle u), broken by noise, varying from fruit to fruit
    side = _math(nt, 'MULTIPLY', u, 2 * math.pi, (-1400, 400))
    side = _math(nt, 'COSINE', side, None, (-1250, 400))
    side = _math(nt, 'MULTIPLY_ADD', side, 0.5, (-1100, 400), c=0.5)
    bn = noise4(4.0, 3.0, (-1250, 250))
    side = _math(nt, 'MULTIPLY_ADD', bn.outputs['Fac'], 0.6, (-950, 350), c=side)
    side = _math(nt, 'SUBTRACT', side, 0.45, (-800, 350))
    side = _math(nt, 'MULTIPLY', side, 2.0 * p.blush, (-650, 350))
    side = _math(nt, 'MINIMUM', side, 1.0, (-500, 350))
    col = _mix(nt, _math(nt, 'MAXIMUM', side, 0.0, (-400, 350)), col, rgb(p.blush_color, (-600, 550)), (-250, 500))
    # Streaks / stripes along the fruit with a noisy edge
    a = _math(nt, 'MULTIPLY', u, 2 * math.pi * p.stripe_count, (-1400, 0))
    wn = noise4(6.0, 3.0, (-1400, -150))
    a = _math(nt, 'MULTIPLY_ADD', wn.outputs['Fac'], 2.5, (-1250, -50), c=a)
    s = _math(nt, 'COSINE', a, None, (-1100, -50))
    s = _math(nt, 'GREATER_THAN', s, math.cos(math.pi * p.stripe_width), (-950, -50))
    s = _math(nt, 'MULTIPLY', s, p.stripes, (-800, -50))
    col = _mix(nt, s, col, rgb(p.stripe_color, (-800, 150)), (-100, 400))
    # Lenticels / oil glands: small Voronoi cells
    vor = nt.nodes.new('ShaderNodeTexVoronoi')
    vor.location = (-1250, -350)
    vor.voronoi_dimensions = '4D'
    nt.links.new(tc.outputs['Object'], vor.inputs['Vector'])
    nt.links.new(w, vor.inputs['W'])
    _set(vor, 220.0 / max(p.diameter_cm, 0.3), "Scale")
    d = _math(nt, 'LESS_THAN', vor.outputs['Distance'], 0.18, (-1050, -350))
    d = _math(nt, 'MULTIPLY', d, p.dots, (-900, -350))
    col = _mix(nt, d, col, rgb(p.dot_color, (-900, -200)), (50, 300))
    # Russet: corky brown patches
    rn = noise4(3.0, 5.0, (-1250, -600))
    r = _math(nt, 'SUBTRACT', rn.outputs['Fac'], 0.62 - 0.25 * p.russet, (-1050, -600))
    r = _math(nt, 'MULTIPLY', r, 6.0 * p.russet, (-900, -600))
    r = _math(nt, 'MINIMUM', _math(nt, 'MAXIMUM', r, 0.0, (-750, -600)), 1.0, (-600, -600))
    col = _mix(nt, r, col, rgb((0.50, 0.36, 0.20), (-600, -450)), (200, 250))
    # Ribs: grooves slightly darker
    g = _math(nt, 'MULTIPLY', _math(nt, 'SUBTRACT', 1.0, rib, (-650, -800)), 0.3, (-500, -800))
    col = _mix(nt, g, col, (0.0, 0.0, 0.0, 1.0), (300, 200))
    # Waxy bloom
    # Waxy bloom: a thin bluish-white film, strongest where the noise lets it gather
    bl = noise4(25.0, 2.0, (0, 500))
    bl = _math(nt, 'MULTIPLY', bl.outputs['Fac'], 0.5 * p.bloom, (150, 500))
    col = _mix(nt, bl, col, rgb((0.55, 0.60, 0.72), (200, 650)), (400, 200))
    # Calyx and stalk
    col = _mix(nt, crown, col, rgb(p.crown_color, (350, 450)), (500, 150))
    col = _mix(nt, stalk, col, rgb(p.stalk_color, (450, 450)), (600, 100))
    nt.links.new(col, b.inputs['Base Color'])
    rough = _math(nt, 'MULTIPLY_ADD', stalk, 0.4, (500, -150), c=min(0.95, 0.8 - 0.7 * p.gloss + 0.35 * p.bloom))
    nt.links.new(rough, b.inputs['Roughness'])
    if p.kind.value in ("Berry", "Drupe"):
        _set(b, 0.15, "Subsurface Weight", "Subsurface")
    b.location = (900, 0)
    mat.node_tree.nodes["Material Output"].location = (1150, 0) if "Material Output" in mat.node_tree.nodes else (0, 0)
    return mat


def ensure_fruit_material(root, props, p: FruitProfile):
    if not props.assign_materials:
        return None
    k = _key(p)
    mat = bpy.data.materials.get(root.name + "_FruitMat")
    if mat is None or mat.get("ppg_key") != k:
        mat = fruit_material(root.name + "_FruitMat", p)
        mat["ppg_key"] = k
    return mat


# -----------------------------------------------------------------------------
# Placement
# -----------------------------------------------------------------------------
def hide_fruits(root):
    for c in root.children:
        if c.name.endswith(PARTS):
            c.hide_viewport = c.hide_render = True


def update_fruits_on_plant(context, root, props, positions):
    """Instances the current fruit (or bunch) hanging from each position (plant-local)."""
    if not props.show_fruits or positions is None or len(positions) == 0 or props.fruit_species not in FRUIT_CATALOG:
        hide_fruits(root)
        return
    p = fruit_from_props(props)
    proto = _child(context, root, "_FruitProto")
    populate_mesh(proto.data, hanging_fruit(p, props.fruit_detail, props.seed))
    _assign(proto, ensure_fruit_material(root, props, p))
    proto.hide_viewport = proto.hide_render = True
    pts = _child(context, root, "_Fruits")
    rng = np.random.default_rng(props.seed + 13)
    n = len(positions)
    # Hanging: small sway away from the vertical, any azimuth
    tilt = rng.normal(0.0, 0.12, (n, 2))
    yaw = rng.uniform(0, 2 * math.pi, n)
    R = []
    for (tx, ty), z in zip(tilt, yaw):
        cx, sx, cy, sy, cz, sz = math.cos(tx), math.sin(tx), math.cos(ty), math.sin(ty), math.cos(z), math.sin(z)
        Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
        R.append(Rz @ Ry @ Rx)
    data = MeshData(np.asarray(positions, np.float32), np.zeros(0, np.int32), np.zeros(0, np.int32),
                    np.zeros(0, np.int32), np.zeros((0, 2), np.float32),
                    {"frot": _euler_xyz(np.array(R)).astype(np.float32),
                     "fscale": (props.fruit_scale * rng.uniform(0.88, 1.12, n)).astype(np.float32)})
    populate_mesh(pts.data, data)
    _instancer_nodes(pts, proto)
    pts.hide_viewport = pts.hide_render = False
    root["fruits"] = n


def fruit_site_count(props, available: int) -> int:
    if available <= 0 or props.fruit_density <= 0:
        return 0
    return int(min(props.fruit_max, max(1, round(available * props.fruit_density))))


def tree_fruit_positions(result, props):
    """Fruits where the tree flowers (terminal or axillary sites of the current inflorescence)."""
    from .flowers import flower_from_props
    try:
        from ..core.inflorescence import tree_flower_sites
    except (ImportError, ValueError):
        from core.inflorescence import tree_flower_sites
    _, ip = flower_from_props(props)
    every = tree_flower_sites(result.skeleton_graph, ip, 10 ** 9, seed=props.seed + 101)
    sites = tree_flower_sites(result.skeleton_graph, ip, fruit_site_count(props, len(every)), seed=props.seed + 101)
    return sites.positions


def update_fruit_geometry(context, props, find_root):
    """A single fruit or bunch on its own (growth form 'Fruit')."""
    sp = FRUIT_CATALOG.get(props.fruit_species)
    if sp is None:
        print("[PPG] No valid fruit preset selected")
        return None
    root = find_root(context, f"PPG_Fruit_{props.fruit_species}")
    for c in root.children:
        if not c.name.endswith("_FruitBody"):
            c.hide_viewport = c.hide_render = True
    p = fruit_from_props(props)
    obj = _child(context, root, "_FruitBody")
    populate_mesh(obj.data, hanging_fruit(p, props.fruit_detail, props.seed))
    _assign(obj, ensure_fruit_material(root, props, p))
    obj.scale = (props.fruit_scale,) * 3
    obj.hide_viewport = obj.hide_render = False
    root["scientific_name"] = sp.scientific_name
    root["common_name"] = sp.common_name
    root["family"] = sp.family
    root["fruit_kind"] = p.kind.value
    root["growth_form"] = "Fruit"
    return root
