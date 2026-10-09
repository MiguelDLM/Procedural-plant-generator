"""
Wind: one Geometry Nodes modifier animated by the scene time, added to every visible part of a plant (wood,
foliage instances, leaves, culms, flowers...; never to roots).

- Main bending (Sousa 2008, GPU Gems 3 ch. 16; Zioma 2008, ch. 6): the whole plant leans along the wind and
  sways around that lean. The displacement follows the static deflection of a cantilever under a uniform
  load (Euler-Bernoulli), y(zeta) = zeta^2 (6 - 4 zeta + zeta^2) / 3 with zeta = z / H, and points move down
  as they move sideways so the stem keeps its length.
- Frequency of the first mode: f ~ D / H^2 for a cantilever; with elastic similarity D ~ H^1.5 (McMahon
  1973) f ~ H^-0.5, about 0.6 Hz for a 20 m tree and 2-3 Hz for a cereal (de Langre 2008, Annu. Rev. Fluid
  Mech. 40: 141).
- Amplitude: drag grows slower than v^2 because leaves and crowns reconfigure (Vogel 1989, J. Exp. Bot. 40:
  941; Vogel number about -0.7), so the deflection grows as v^1.3.
- Gusts travel with the wind (noise sampled at x - v t); leaves and blades flutter at a higher frequency
  (detail bending), more toward the tips.
"""

import math

try:
    import bpy
    BLENDER_AVAILABLE = True
except ImportError:
    BLENDER_AVAILABLE = False

try:
    from ..core.wind import wind_frequency, wind_amplitude
except (ImportError, ValueError):
    from core.wind import wind_frequency, wind_amplitude

GROUP = "PPG_Wind"
VERSION = 2
SKIP = ("_Roots", "_FineRoots", "_VegRoot", "_GrassRoots", "_OrchidRoots", "_OrchidSupport", "_VineRoots",
        "_SuccRoots", "_LeafCard", "_FruitProto", "_GrassProto")


def wind_properties(update) -> dict:
    if not BLENDER_AVAILABLE:
        return {}
    from bpy.props import BoolProperty, FloatProperty
    return {
        "wind": BoolProperty(name="Wind", default=False, update=update,
                             description="Animate the plant in the wind (Geometry Nodes, scene time)"),
        "wind_speed": FloatProperty(name="Speed (m/s)", default=6.0, min=0.0, max=40.0, update=update,
                                    description="Mean wind speed (Beaufort 4 ~ 6 m/s, 7 ~ 15 m/s)"),
        "wind_direction": FloatProperty(name="Direction", default=0.0, min=-math.pi, max=math.pi,
                                        subtype='ANGLE', update=update,
                                        description="Direction the wind blows toward (0 = +X)"),
        "wind_gusts": FloatProperty(name="Gusts", default=0.5, min=0.0, max=1.0, update=update,
                                    description="Strength of the gusts travelling with the wind"),
        "wind_flutter": FloatProperty(name="Leaf Flutter", default=0.5, min=0.0, max=2.0, update=update,
                                      description="Fast detail motion of leaves, blades and petals"),
        "wind_stiffness": FloatProperty(name="Stiffness", default=1.0, min=0.2, max=5.0, update=update,
                                        description="Scales the natural frequency and reduces the deflection"),
    }


def _group():
    g = bpy.data.node_groups.get(GROUP)
    if g is not None and g.get("ppg_version") == VERSION:
        return g
    if g is None:
        g = bpy.data.node_groups.new(GROUP, 'GeometryNodeTree')
    g.nodes.clear()
    for s in list(g.interface.items_tree):
        g.interface.remove(s)
    g["ppg_version"] = VERSION
    sock = g.interface.new_socket
    sock('Geometry', in_out='INPUT', socket_type='NodeSocketGeometry')
    sock('Geometry', in_out='OUTPUT', socket_type='NodeSocketGeometry')
    for name, default in (("Height", 1.0), ("Frequency", 1.0), ("Amplitude", 0.05), ("Gusts", 0.5),
                          ("Flutter", 0.5), ("Speed", 6.0), ("Direction", 0.0)):
        s = sock(name, in_out='INPUT', socket_type='NodeSocketFloat')
        s.default_value = default
    n, L = g.nodes, g.links
    gi = n.new('NodeGroupInput')
    gi.location = (-1800, 0)
    go = n.new('NodeGroupOutput')
    go.location = (1200, 0)
    x = [-1500]

    def node(kind, **kw):
        nd = n.new(kind)
        nd.location = (x[0], kw.pop("y", 0))
        x[0] += 0
        for k, v in kw.items():
            setattr(nd, k, v)
        return nd

    def math_(op, a, b=None, y=0, xx=0):
        m = n.new('ShaderNodeMath')
        m.operation = op
        m.location = (xx, y)
        for i, v in enumerate((a, b)):
            if v is None:
                continue
            if isinstance(v, (int, float)):
                m.inputs[i].default_value = v
            else:
                L.new(v, m.inputs[i])
        return m.outputs[0]
    pos = node('GeometryNodeInputPosition', y=-200)
    sep = n.new('ShaderNodeSeparateXYZ')
    sep.location = (-1500, -200)
    L.new(pos.outputs[0], sep.inputs[0])
    t = n.new('GeometryNodeInputSceneTime')
    t.location = (-1500, 300)
    H = gi.outputs["Height"]
    # zeta and the cantilever deflection shape
    zeta = math_('DIVIDE', sep.outputs[2], H, -200, -1300)
    zeta = math_('MINIMUM', math_('MAXIMUM', zeta, 0.0, -200, -1150), 1.3, -200, -1000)
    z2 = math_('MULTIPLY', zeta, zeta, -350, -850)
    poly = math_('ADD', math_('SUBTRACT', 6.0, math_('MULTIPLY', zeta, 4.0, -500, -850), -500, -700),
                 z2, -500, -550)
    shape = math_('DIVIDE', math_('MULTIPLY', z2, poly, -350, -400), 3.0, -350, -250)
    # Wind direction
    cx = math_('COSINE', gi.outputs["Direction"], None, 400, -1300)
    sy = math_('SINE', gi.outputs["Direction"], None, 300, -1300)
    dvec = n.new('ShaderNodeCombineXYZ')
    dvec.location = (-1100, 450)
    L.new(cx, dvec.inputs[0])
    L.new(sy, dvec.inputs[1])
    # Gusts travelling with the wind: noise at x - v t
    adv = n.new('ShaderNodeVectorMath')
    adv.operation = 'SCALE'
    adv.location = (-900, 450)
    L.new(dvec.outputs[0], adv.inputs[0])
    L.new(math_('MULTIPLY', t.outputs['Seconds'], gi.outputs["Speed"], 600, -1100), adv.inputs['Scale'])
    rel = n.new('ShaderNodeVectorMath')
    rel.operation = 'SUBTRACT'
    rel.location = (-700, 450)
    L.new(pos.outputs[0], rel.inputs[0])
    L.new(adv.outputs[0], rel.inputs[1])
    gust = n.new('ShaderNodeTexNoise')
    gust.location = (-500, 450)
    gust.inputs['Scale'].default_value = 0.08
    gust.inputs['Detail'].default_value = 1.0
    L.new(rel.outputs[0], gust.inputs['Vector'])
    gfac = math_('ADD', 1.0, math_('MULTIPLY', math_('SUBTRACT', gust.outputs['Fac'], 0.5, 600, -300),
                                    math_('MULTIPLY', gi.outputs["Gusts"], 2.4, 600, -500), 750, -400),
                 900, -400)
    # Sway around the mean lean, phase varying slowly over the plant (crown parts lag each other)
    ph = n.new('ShaderNodeTexNoise')
    ph.location = (-500, 150)
    ph.inputs['Scale'].default_value = 0.25
    L.new(pos.outputs[0], ph.inputs['Vector'])
    omega = math_('MULTIPLY', math_('MULTIPLY', t.outputs['Seconds'], gi.outputs["Frequency"], 0, 300),
                  2 * math.pi, 150, 300)
    arg = math_('ADD', omega, math_('MULTIPLY', ph.outputs['Fac'], 4.0, 150, 150), 300, 250)
    sway = math_('ADD', 0.65, math_('MULTIPLY', math_('SINE', arg, None, 450, 250), 0.45, 600, 250), 750, 250)
    mag = math_('MULTIPLY', math_('MULTIPLY', math_('MULTIPLY', shape, gi.outputs["Amplitude"], 0, -100),
                                  sway, 900, 100), gfac, 1000, -50)
    lat = n.new('ShaderNodeVectorMath')
    lat.operation = 'SCALE'
    lat.location = (1050, 300)
    L.new(dvec.outputs[0], lat.inputs[0])
    L.new(mag, lat.inputs['Scale'])
    # Keep the length: points sink by d^2 / (2 z)
    drop = math_('DIVIDE', math_('MULTIPLY', math_('MULTIPLY', mag, mag, 1100, -300), -0.5, 1100, -450),
                 math_('MAXIMUM', sep.outputs[2], math_('MULTIPLY', H, 0.05, 900, -650), 1000, -600), 1150, -600)
    dz = n.new('ShaderNodeCombineXYZ')
    dz.location = (1250, -300)
    L.new(drop, dz.inputs[2])
    # Flutter: fast detail motion growing toward the tips
    fl = n.new('ShaderNodeTexNoise')
    fl.location = (300, -800)
    fl.inputs['Scale'].default_value = 4.0
    fl.noise_dimensions = '4D'
    L.new(pos.outputs[0], fl.inputs['Vector'])
    L.new(math_('MULTIPLY', omega, 0.9, 150, -900), fl.inputs['W'])
    flc = n.new('ShaderNodeVectorMath')
    flc.operation = 'SUBTRACT'
    flc.location = (500, -800)
    L.new(fl.outputs['Color'], flc.inputs[0])
    flc.inputs[1].default_value = (0.5, 0.5, 0.5)
    famp = math_('MULTIPLY', math_('MULTIPLY', gi.outputs["Flutter"], gi.outputs["Amplitude"], 500, -1000),
                 math_('MULTIPLY', zeta, 0.35, 500, -1150), 700, -1050)
    fls = n.new('ShaderNodeVectorMath')
    fls.operation = 'SCALE'
    fls.location = (750, -800)
    L.new(flc.outputs[0], fls.inputs[0])
    L.new(math_('MULTIPLY', famp, gfac, 850, -1050), fls.inputs['Scale'])
    s1 = n.new('ShaderNodeVectorMath')
    s1.operation = 'ADD'
    s1.location = (1350, 200)
    L.new(lat.outputs[0], s1.inputs[0])
    L.new(dz.outputs[0], s1.inputs[1])
    s2 = n.new('ShaderNodeVectorMath')
    s2.operation = 'ADD'
    s2.location = (1450, 0)
    L.new(s1.outputs[0], s2.inputs[0])
    L.new(fls.outputs[0], s2.inputs[1])
    setp = n.new('GeometryNodeSetPosition')
    setp.location = (1600, 0)
    L.new(gi.outputs[0], setp.inputs['Geometry'])
    L.new(s2.outputs[0], setp.inputs['Offset'])
    go.location = (1800, 0)
    L.new(setp.outputs[0], go.inputs[0])
    return g


def _plant_height(root) -> float:
    h = 0.0
    for c in root.children:
        if c.type != 'MESH' or c.hide_viewport or c.name.endswith(SKIP):
            continue
        h = max(h, max(v[2] for v in c.bound_box))
    return max(h, 0.05)


def apply_wind(root, props):
    """Adds, updates or removes the wind modifier on the visible parts of a plant."""
    if root is None:
        return
    on = bool(getattr(props, "wind", False))
    H = _plant_height(root) if on else 1.0
    stiff = getattr(props, "wind_stiffness", 1.0)
    for c in root.children:
        if c.type != 'MESH':
            continue
        mod = c.modifiers.get("PPG_Wind")
        if not on or c.name.endswith(SKIP):
            if mod is not None:
                c.modifiers.remove(mod)
            continue
        if mod is None:
            mod = c.modifiers.new("PPG_Wind", 'NODES')
        mod.node_group = _group()
        vals = {"Height": H, "Frequency": wind_frequency(H) * stiff ** 0.5,
                "Amplitude": wind_amplitude(props.wind_speed, H) / stiff, "Gusts": props.wind_gusts,
                "Flutter": props.wind_flutter, "Speed": props.wind_speed, "Direction": props.wind_direction}
        for item in mod.node_group.interface.items_tree:
            if getattr(item, "in_out", "") == 'INPUT' and item.name in vals:
                mod[item.identifier] = float(vals[item.name])
        c.update_tag()


def apply_wind_to_scene(context, props):
    """Wind is a scene setting: every generated plant root gets it."""
    for obj in context.scene.objects:
        if obj.parent is None and obj.name.startswith("PPG_") and obj.children:
            apply_wind(obj, props)
