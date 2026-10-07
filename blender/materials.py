"""
Botanical materials for Blender.

Leaf: image-based (outline alpha + venation colour + vein height map generated
by core.leaf_texture), with
- abaxial (underside) tint on back faces,
- translucency (light through the lamina, yellower along veins less so),
- per-leaf hue/value jitter from the `leaf_random` mesh attribute,
- gloss from cuticle wax.

Bark: procedural node graph per rhytidome pattern (fissured, plated, peeling,
lenticelled, fibrous, smooth, annulated) driven by the species BarkProfile, on
seam-free 3D bark coordinates so the pattern keeps a constant world scale and
stays continuous across forks, fused junctions and root flares.
"""

try:
    import bpy
    BLENDER_AVAILABLE = True
except ImportError:
    BLENDER_AVAILABLE = False

import numpy as np


def _srgb_to_linear(c):
    c = np.asarray(c, dtype=float)
    return tuple(np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4).tolist()) + (1.0,)


def _input(node, *names):
    for n in names:
        if n in node.inputs:
            return node.inputs[n]
    return None


def _set(node, value, *names):
    sock = _input(node, *names)
    if sock is not None:
        sock.default_value = value


def _configure_alpha(mat):
    if hasattr(mat, "surface_render_method"):
        mat.surface_render_method = 'DITHERED'
    if hasattr(mat, "blend_method"):
        try:
            mat.blend_method = 'HASHED'
        except TypeError:
            pass
    if hasattr(mat, "shadow_method"):
        try:
            mat.shadow_method = 'HASHED'
        except TypeError:
            pass
    mat.use_backface_culling = False


def leaf_images_from_texture(name: str, texture) -> tuple:
    """Creates (or refreshes) packed Blender images from a core LeafTexture."""
    w, h = texture.size
    col = bpy.data.images.get(name + "_Color")
    if col is None or tuple(col.size) != (w, h):
        if col is not None:
            bpy.data.images.remove(col)
        col = bpy.data.images.new(name + "_Color", w, h, alpha=True)
    col.pixels.foreach_set(texture.color.astype(np.float32).ravel())
    col.alpha_mode = 'STRAIGHT'
    col.pack()

    hgt = bpy.data.images.get(name + "_Height")
    if hgt is None or tuple(hgt.size) != (w, h):
        if hgt is not None:
            bpy.data.images.remove(hgt)
        hgt = bpy.data.images.new(name + "_Height", w, h, alpha=False)
    rgba = np.repeat(texture.height[..., None], 4, axis=-1)
    rgba[..., 3] = 1.0
    hgt.pixels.foreach_set(rgba.astype(np.float32).ravel())
    hgt.colorspace_settings.name = 'Non-Color'
    hgt.pack()
    return col, hgt


def create_leaf_material(name: str, color_img, height_img, morph) -> "bpy.types.Material":
    if not BLENDER_AVAILABLE:
        return None
    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nt = mat.node_tree
    nodes, links = nt.nodes, nt.links
    nodes.clear()

    out = nodes.new('ShaderNodeOutputMaterial')
    out.location = (1300, 0)

    tex = nodes.new('ShaderNodeTexImage')
    tex.location = (-900, 200)
    tex.image = color_img
    tex.interpolation = 'Linear'
    tex.extension = 'CLIP'

    htex = nodes.new('ShaderNodeTexImage')
    htex.location = (-900, -250)
    htex.image = height_img
    htex.extension = 'CLIP'

    # Per-leaf colour jitter
    attr = nodes.new('ShaderNodeAttribute')
    attr.location = (-900, 500)
    attr.attribute_name = "leaf_random"
    hue_map = nodes.new('ShaderNodeMapRange')
    hue_map.location = (-650, 600)
    _set(hue_map, 0.475, "To Min")
    _set(hue_map, 0.525, "To Max")
    val_map = nodes.new('ShaderNodeMapRange')
    val_map.location = (-650, 400)
    _set(val_map, 0.85, "To Min")
    _set(val_map, 1.12, "To Max")
    links.new(attr.outputs['Fac'], hue_map.inputs['Value'])
    links.new(attr.outputs['Fac'], val_map.inputs['Value'])
    hsv = nodes.new('ShaderNodeHueSaturation')
    hsv.location = (-400, 300)
    links.new(hue_map.outputs['Result'], hsv.inputs['Hue'])
    links.new(val_map.outputs['Result'], hsv.inputs['Value'])
    links.new(tex.outputs['Color'], hsv.inputs['Color'])

    # Abaxial (underside) tint on back faces
    geo = nodes.new('ShaderNodeNewGeometry')
    geo.location = (-400, 650)
    ab = morph.abaxial_color
    ad = morph.adaxial_color
    ratio = tuple(min(3.0, a / max(0.02, d)) for a, d in zip(ab, ad))
    tint = nodes.new('ShaderNodeMix')
    tint.data_type = 'RGBA'
    tint.blend_type = 'MULTIPLY'
    tint.location = (-150, 400)
    links.new(geo.outputs['Backfacing'], tint.inputs[0])
    links.new(hsv.outputs['Color'], tint.inputs[6])
    tint.inputs[7].default_value = _srgb_to_linear(ratio)

    bump = nodes.new('ShaderNodeBump')
    bump.location = (-150, -250)
    _set(bump, 0.35, "Strength")
    _set(bump, 0.0006, "Distance")
    links.new(htex.outputs['Color'], bump.inputs['Height'])

    bsdf = nodes.new('ShaderNodeBsdfPrincipled')
    bsdf.location = (200, 200)
    links.new(tint.outputs[2], bsdf.inputs['Base Color'])
    links.new(bump.outputs['Normal'], bsdf.inputs['Normal'])
    gloss = float(getattr(morph, "gloss", 0.35))
    _set(bsdf, 0.75 - 0.55 * gloss, "Roughness")
    _set(bsdf, 0.3 + 0.4 * gloss, "Specular IOR Level", "Specular")
    _set(bsdf, 0.15, "Subsurface Weight", "Subsurface")

    # Translucency: thin mesophyll transmits a yellowed green
    trans_col = nodes.new('ShaderNodeMix')
    trans_col.data_type = 'RGBA'
    trans_col.blend_type = 'MULTIPLY'
    trans_col.location = (200, -150)
    trans_col.inputs[0].default_value = 1.0
    links.new(tint.outputs[2], trans_col.inputs[6])
    trans_col.inputs[7].default_value = (1.6, 1.8, 0.7, 1.0)
    translucent = nodes.new('ShaderNodeBsdfTranslucent')
    translucent.location = (450, -150)
    links.new(trans_col.outputs[2], translucent.inputs['Color'])
    links.new(bump.outputs['Normal'], translucent.inputs['Normal'])

    thickness = float(getattr(morph, "thickness_mm", 0.22))
    mix_t = nodes.new('ShaderNodeMixShader')
    mix_t.location = (700, 100)
    mix_t.inputs['Fac'].default_value = float(np.clip(0.38 - thickness * 0.4, 0.08, 0.35))
    links.new(bsdf.outputs['BSDF'], mix_t.inputs[1])
    links.new(translucent.outputs['BSDF'], mix_t.inputs[2])

    # Alpha cut-out from the leaf outline
    transparent = nodes.new('ShaderNodeBsdfTransparent')
    transparent.location = (700, -150)
    mix_a = nodes.new('ShaderNodeMixShader')
    mix_a.location = (1000, 0)
    links.new(tex.outputs['Alpha'], mix_a.inputs['Fac'])
    links.new(transparent.outputs['BSDF'], mix_a.inputs[1])
    links.new(mix_t.outputs['Shader'], mix_a.inputs[2])
    links.new(mix_a.outputs['Shader'], out.inputs['Surface'])

    _configure_alpha(mat)
    return mat


def create_bark_material(name: str, bark=None) -> "bpy.types.Material":
    """Procedural bark shader driven by a core.bark.BarkProfile."""
    if not BLENDER_AVAILABLE:
        return None
    if bark is None:
        from ..core.bark import BarkProfile
        bark = BarkProfile()
    pattern = bark.pattern.value if hasattr(bark.pattern, "value") else str(bark.pattern)

    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nt = mat.node_tree
    nodes, links = nt.nodes, nt.links
    nodes.clear()

    out = nodes.new('ShaderNodeOutputMaterial')
    out.location = (1100, 0)
    bsdf = nodes.new('ShaderNodeBsdfPrincipled')
    bsdf.location = (800, 0)
    links.new(bsdf.outputs['BSDF'], out.inputs['Surface'])
    _set(bsdf, float(bark.roughness), "Roughness")

    # Seam-free 3D bark coordinates (world metres): Q = bark_base + k * bark_along, where bark_along runs
    # along the axis from its origin and bark_base holds the radial offset. k < 1 elongates features
    # along the axis (fissures, fibres), k > 1 elongates them around it (lenticels, leaf-scar rings).
    k = {"Fissured": 0.25, "Plated": 0.55, "Lenticelled": 3.0, "Fibrous": 0.08, "Annulated": 5.0,
         "Peeling": 0.7}.get(pattern, 0.5)
    a_base = nodes.new('ShaderNodeAttribute')
    a_base.location = (-1500, 100)
    a_base.attribute_name = "bark_base"
    a_along = nodes.new('ShaderNodeAttribute')
    a_along.location = (-1500, -100)
    a_along.attribute_name = "bark_along"
    scale_k = nodes.new('ShaderNodeVectorMath')
    scale_k.operation = 'SCALE'
    scale_k.location = (-1300, -100)
    links.new(a_along.outputs['Vector'], scale_k.inputs[0])
    scale_k.inputs['Scale'].default_value = k
    q = nodes.new('ShaderNodeVectorMath')
    q.operation = 'ADD'
    q.location = (-1150, 0)
    links.new(a_base.outputs['Vector'], q.inputs[0])
    links.new(scale_k.outputs['Vector'], q.inputs[1])
    mapping = nodes.new('ShaderNodeMapping')
    mapping.location = (-1000, 0)
    links.new(q.outputs['Vector'], mapping.inputs['Vector'])
    s = 1.0 / max(0.005, float(bark.feature_scale_m))  # Features per metre
    mapping.inputs['Scale'].default_value = (s, s, s)

    Qv = mapping.outputs['Vector']           # Bark-feature units (1 = one plate / fissure spacing)

    def mnode(op, a, b=None, loc=(0, 0), c=None, clamp=False):
        m = nodes.new('ShaderNodeMath')
        m.operation = op
        m.location = loc
        m.use_clamp = clamp
        for i, v in enumerate((a, b, c)):
            if v is None:
                continue
            if isinstance(v, (int, float)):
                m.inputs[i].default_value = v
            else:
                links.new(v, m.inputs[i])
        return m.outputs[0]

    def smooth(x, lo, hi, loc):
        m = nodes.new('ShaderNodeMapRange')
        m.interpolation_type = 'SMOOTHSTEP'
        m.location = loc
        for name, v in (("From Min", lo), ("From Max", hi)):
            if isinstance(v, (int, float)):
                m.inputs[name].default_value = v
            else:
                links.new(v, m.inputs[name])
        links.new(x, m.inputs['Value'])
        return m.outputs['Result']

    def mixc(fac, a, b, loc, blend='MIX'):
        m = nodes.new('ShaderNodeMix')
        m.data_type = 'RGBA'
        m.blend_type = blend
        m.location = loc
        for sock, v in ((m.inputs[0], fac), (m.inputs[6], a), (m.inputs[7], b)):
            if isinstance(v, (int, float, tuple)):
                sock.default_value = v
            else:
                links.new(v, sock)
        return m.outputs[2]

    def voronoi(vec, scale, feature, loc, rand=1.0):
        v = nodes.new('ShaderNodeTexVoronoi')
        v.location = loc
        v.voronoi_dimensions = '3D'
        v.feature = feature
        _set(v, scale, "Scale")
        _set(v, rand, "Randomness")
        links.new(vec, v.inputs['Vector'])
        return v

    def noise_tex(vec, scale, detail, loc, rough=0.55):
        n = nodes.new('ShaderNodeTexNoise')
        n.location = loc
        _set(n, scale, "Scale")
        _set(n, detail, "Detail")
        _set(n, rough, "Roughness")
        links.new(vec, n.inputs['Vector'])
        return n

    # --- Local bark age from the axis radius (pipe model: thicker = older) -----------------------------
    r_attr = nodes.new('ShaderNodeAttribute')
    r_attr.location = (-1500, -400)
    r_attr.attribute_name = "bark_radius"
    onset = 0.01 * max(0.05, float(getattr(bark, "onset_radius_cm", 4.0)))
    r = r_attr.outputs['Fac']
    age = smooth(r, 0.5 * onset, 2.0 * onset, (-1250, -400))            # Young periderm -> rhytidome
    old = smooth(r, 2.0 * onset, 9.0 * onset, (-1250, -550))            # Fissures keep widening

    # Domain warp: organic, non-cellular outlines
    warp_n = noise_tex(Qv, 1.6, 4.0, (-1000, 350))
    warp = nodes.new('ShaderNodeMix')
    warp.data_type = 'VECTOR'
    warp.location = (-800, 300)
    warp.inputs[0].default_value = 0.22
    links.new(Qv, warp.inputs[4])
    links.new(warp_n.outputs['Color'], warp.inputs[5])
    W = warp.outputs[1]
    fine = noise_tex(Qv, 14.0, 6.0, (-800, 600), 0.6)                    # Fibrous micro-texture
    detail_n = fine.outputs['Fac']

    # --- Mature rhytidome height field H (0 fissure bottom .. 1 ridge / plate top) ----------------------
    big_edge = voronoi(W, 1.0, 'DISTANCE_TO_EDGE', (-600, 200))
    cell = voronoi(W, 1.0, 'F1', (-600, 0))                              # Per-plate random colour/height
    small_edge = voronoi(W, 3.3, 'DISTANCE_TO_EDGE', (-600, -200))       # Secondary checking
    cell_v = mnode('MULTIPLY', cell.outputs['Color'], 1.0, (-400, 0))
    E1, E2 = big_edge.outputs['Distance'], small_edge.outputs['Distance']
    if pattern == "Fissured":
        width = mnode('MULTIPLY_ADD', old, 0.16, (-400, 400), 0.05)
        ridge = mnode('POWER', smooth(E1, 0.0, width, (-250, 300)), 0.55, (-100, 300))   # Rounded ridges
        chk = smooth(E2, 0.0, 0.05, (-250, 150))
        Hm = mnode('MULTIPLY', ridge, mnode('MULTIPLY_ADD', chk, 0.25, (-100, 150), 0.75), (50, 250))
        Hm = mnode('MULTIPLY', Hm, mnode('MULTIPLY_ADD', detail_n, 0.3, (50, 100), 0.7), (200, 250))
    elif pattern == "Fibrous":
        # Long, stringy strands (Sequoiadendron, Taxodium, Cupressus): axially stretched noise ridges with
        # a few deeper splits between broad fibre bundles
        strands = noise_tex(W, 5.0, 8.0, (-600, 450), 0.5)
        fib = smooth(strands.outputs['Fac'], 0.35, 0.65, (-250, 450))
        split = smooth(E1, 0.0, 0.12, (-250, 300))
        Hm = mnode('MULTIPLY', mnode('MULTIPLY_ADD', fib, 0.6, (-100, 400), 0.4), split, (50, 250))
    elif pattern == "Plated":
        width = mnode('MULTIPLY_ADD', old, 0.10, (-400, 400), 0.04)
        plate = smooth(E1, 0.0, width, (-250, 300))                       # Flat-topped scales
        tier = mnode('MULTIPLY_ADD', cell_v, 0.35, (-100, 100), 0.65)     # Plates at different heights
        Hm = mnode('MULTIPLY', plate, tier, (50, 250))
        Hm = mnode('MULTIPLY', Hm, mnode('MULTIPLY_ADD', detail_n, 0.2, (50, 100), 0.8), (200, 250))
    elif pattern == "Lenticelled":
        dash = mnode('SUBTRACT', 1.0, smooth(E1, 0.0, 0.05, (-250, 300)), (-100, 300))
        Hm = mnode('SUBTRACT', mnode('MULTIPLY_ADD', detail_n, 0.1, (50, 100), 0.9), mnode('MULTIPLY', dash, 0.5,
                   (50, 300)), (200, 250))
    elif pattern == "Annulated":
        Hm = mnode('POWER', smooth(E1, 0.0, 0.18, (-250, 300)), 0.6, (50, 250))
    elif pattern == "Peeling":
        patches = noise_tex(W, 0.9, 3.0, (-600, 400))
        layer = smooth(patches.outputs['Fac'], 0.46, 0.54, (-250, 400))
        lip = mnode('SUBTRACT', 1.0, mnode('ABSOLUTE', mnode('MULTIPLY_ADD', layer, 2.0, (-250, 250), -1.0),
                    None, (-100, 250)), (50, 250))
        # Freshly exposed underbark (low H) vs older outer patches (high H), raised lips at patch edges
        Hm = mnode('ADD', mnode('MULTIPLY_ADD', layer, 0.6, (50, 400), 0.25), mnode('MULTIPLY', lip, 0.15, (200, 300)),
                   (350, 300))
    else:  # Smooth
        Hm = mnode('MULTIPLY_ADD', detail_n, 0.15, (50, 250), 0.85)

    # --- Young periderm: smooth, with transverse lenticels -------------------------------------------
    len_v = voronoi(Qv, 2.5, 'F1', (-600, -450), 0.9)
    lent = mnode('SUBTRACT', 1.0, smooth(len_v.outputs['Distance'], 0.04, 0.09, (-400, -450)), (-250, -450))
    Hy = mnode('SUBTRACT', mnode('MULTIPLY_ADD', detail_n, 0.08, (-250, -600), 0.92),
               mnode('MULTIPLY', lent, 0.25, (-100, -450)), (50, -500))
    H = nodes.new('ShaderNodeMix')
    H.data_type = 'FLOAT'
    H.location = (400, -100)
    links.new(age, H.inputs[0])
    links.new(Hy, H.inputs[2])
    links.new(Hm, H.inputs[3])
    Hs = H.outputs[0]

    # --- Colour: depth-dependent (inner bark at the bottom, weathered grey ridge tops) -----------------
    base = _srgb_to_linear(bark.base_color)
    sec = _srgb_to_linear(bark.secondary_color)
    inner = _srgb_to_linear(getattr(bark, "inner_color", None) or bark.secondary_color)
    if pattern == "Peeling":       # Exfoliating barks expose pale underbark, not a dark furrow
        inner = sec
    young = _srgb_to_linear(getattr(bark, "young_color", None) or bark.base_color)
    fiss = mixc(smooth(Hs, 0.0, 0.3, (550, 450)), inner, sec, (700, 450))
    hsv = nodes.new('ShaderNodeHueSaturation')
    hsv.location = (550, 650)
    hsv.inputs['Color'].default_value = base
    links.new(mnode('MULTIPLY_ADD', cell_v, 0.35, (400, 700), 0.82), hsv.inputs['Value'])
    plate_c = hsv.outputs['Color']
    colm = mixc(smooth(Hs, 0.12, 0.55, (700, 300)), fiss, plate_c, (900, 400))
    weather = mnode('MULTIPLY', smooth(Hs, 0.65, 1.0, (700, 150)), 0.7 * float(getattr(bark, "weathering", 0.35)),
                    (850, 150))
    colm = mixc(weather, colm, _srgb_to_linear((0.56, 0.55, 0.52)), (1050, 300))
    coly = mixc(mnode('MULTIPLY', lent, 0.6, (700, -350)), young, mixc(0.5, young, sec, (700, -500)), (900, -400))
    col = mixc(age, coly, colm, (1200, 100))
    # Large-scale variation along the tree (avoid a uniform tint)
    tc = nodes.new('ShaderNodeTexCoord')
    tc.location = (700, 800)
    macro = noise_tex(tc.outputs['Object'], 0.6, 2.0, (900, 800))
    col = mixc(0.18, col, macro.outputs['Color'], (1350, 200), 'OVERLAY')
    # Cavity: crotches, collar folds and deep furrows darken (local ambient occlusion)
    ao = nodes.new('ShaderNodeAmbientOcclusion')
    ao.location = (1350, 450)
    ao.only_local = True
    ao.samples = 8
    _set(ao, max(0.05, 2.0 * float(bark.feature_scale_m)), "Distance")
    col = mixc(0.55, col, ao.outputs['Color'], (1500, 250), 'MULTIPLY')
    links.new(col, bsdf.inputs['Base Color'])
    out.location = (2100, 0)
    bsdf.location = (1800, 0)

    rough = nodes.new('ShaderNodeMapRange')
    rough.location = (1500, -200)
    rough.inputs['To Min'].default_value = min(1.0, float(bark.roughness) + 0.1)
    rough.inputs['To Max'].default_value = max(0.3, float(bark.roughness) - 0.15)
    links.new(Hs, rough.inputs['Value'])
    links.new(rough.outputs['Result'], bsdf.inputs['Roughness'])

    # Relief: bump from H (world-scaled), stronger on old bark, plus fibrous micro-bump
    bump = nodes.new('ShaderNodeBump')
    bump.location = (1500, -450)
    relief = float(bark.relief)
    links.new(mnode('MULTIPLY_ADD', age, 0.6 * relief, (1300, -550), 0.25 + 0.15 * relief), bump.inputs['Strength'])
    _set(bump, max(0.002, 0.35 * relief * float(bark.feature_scale_m)), "Distance")
    links.new(mnode('MULTIPLY_ADD', detail_n, 0.08, (1300, -450), Hs), bump.inputs['Height'])
    links.new(bump.outputs['Normal'], bsdf.inputs['Normal'])
    return mat
