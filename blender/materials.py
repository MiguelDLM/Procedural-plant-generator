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
    attr_g = nodes.new('ShaderNodeAttribute')
    attr_g.location = (-1100, 560)
    attr_g.attribute_name = "leaf_random"
    attr_i = nodes.new('ShaderNodeAttribute')         # Same value on instanced leaf cards
    attr_i.location = (-1100, 420)
    attr_i.attribute_name = "leaf_random"
    attr_i.attribute_type = 'INSTANCER'
    attr = nodes.new('ShaderNodeMath')
    attr.operation = 'ADD'
    attr.location = (-900, 500)
    links.new(attr_g.outputs['Fac'], attr.inputs[0])
    links.new(attr_i.outputs['Fac'], attr.inputs[1])
    hue_map = nodes.new('ShaderNodeMapRange')
    hue_map.location = (-650, 600)
    _set(hue_map, 0.475, "To Min")
    _set(hue_map, 0.525, "To Max")
    val_map = nodes.new('ShaderNodeMapRange')
    val_map.location = (-650, 400)
    _set(val_map, 0.85, "To Min")
    _set(val_map, 1.12, "To Max")
    links.new(attr.outputs[0], hue_map.inputs['Value'])
    links.new(attr.outputs[0], val_map.inputs['Value'])
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


def create_bark_material(name: str, bark=None, displacement: bool = False) -> "bpy.types.Material":
    """
    Procedural bark driven by a core.bark.BarkProfile.

    Height field (0 fissure bottom .. 1 ridge / plate top) built on seam-free 3D bark coordinates:
    - organic domain warp with high-detail noise whose roughness itself varies (Ryan King Art, 2022);
    - blocks from a Minkowski Voronoi (exponent 2 irregular .. high = rectangular, `blockiness`), with
      cracks as the F2 - F1 gap; transverse splits quantised along the axis cut ridges into segments, and
      each plate gets its own tilt and offset (per-segment variation as in Substance bark workflows);
    - rhytidome appears with girth (`bark_radius`) and fissures widen as it grows (Lefebvre & Neyret 2002;
      the bark-fissure index scales with stem diameter, MacFarlane & Luo 2009);
    - colour by depth (inner bark, dirt in cracks, weathered ridge tops), per-plate tint, macro variation,
      cavity AO; moss on upper, shaded and basal surfaces and crustose lichen patches;
    - stacked bumps (bark, micro detail, moss) and optional true displacement.
    """
    if not BLENDER_AVAILABLE:
        return None
    if bark is None:
        from ..core.bark import BarkProfile
        bark = BarkProfile()
    pattern = bark.pattern.value if hasattr(bark.pattern, "value") else str(bark.pattern)
    G = lambda k, d: float(getattr(bark, k, d))  # noqa: E731

    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nt = mat.node_tree
    nodes, links = nt.nodes, nt.links
    nodes.clear()
    out = nodes.new('ShaderNodeOutputMaterial')
    out.location = (2600, 0)
    bsdf = nodes.new('ShaderNodeBsdfPrincipled')
    bsdf.location = (2300, 0)
    links.new(bsdf.outputs['BSDF'], out.inputs['Surface'])

    def mnode(op, a, b=None, loc=(0, 0), c=None):
        m = nodes.new('ShaderNodeMath')
        m.operation = op
        m.location = loc
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
        for nm, v in (("From Min", lo), ("From Max", hi)):
            if isinstance(v, (int, float)):
                m.inputs[nm].default_value = v
            else:
                links.new(v, m.inputs[nm])
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

    def mixf(fac, a, b, loc):
        m = nodes.new('ShaderNodeMix')
        m.data_type = 'FLOAT'
        m.location = loc
        for sock, v in ((m.inputs[0], fac), (m.inputs[2], a), (m.inputs[3], b)):
            if isinstance(v, (int, float)):
                sock.default_value = v
            else:
                links.new(v, sock)
        return m.outputs[0]

    def noise_tex(vec, scale, detail, loc, rough=0.55, distortion=0.0):
        n = nodes.new('ShaderNodeTexNoise')
        n.location = loc
        _set(n, scale, "Scale")
        _set(n, detail, "Detail")
        if isinstance(rough, (int, float)):
            _set(n, rough, "Roughness")
        else:
            links.new(rough, n.inputs['Roughness'])
        _set(n, distortion, "Distortion")
        links.new(vec, n.inputs['Vector'])
        return n

    def voronoi(vec, scale, feature, loc, exponent=None, rand=1.0):
        v = nodes.new('ShaderNodeTexVoronoi')
        v.location = loc
        v.voronoi_dimensions = '3D'
        v.feature = feature
        if exponent is not None:
            v.distance = 'MINKOWSKI'
            _set(v, exponent, "Exponent")
        _set(v, scale, "Scale")
        _set(v, rand, "Randomness")
        links.new(vec, v.inputs['Vector'])
        return v

    def attr(name_, loc):
        a = nodes.new('ShaderNodeAttribute')
        a.location = loc
        a.attribute_name = name_
        return a

    def bump(height, strength, distance, normal, loc):
        b = nodes.new('ShaderNodeBump')
        b.location = loc
        if isinstance(strength, (int, float)):
            _set(b, strength, "Strength")
        else:
            links.new(strength, b.inputs['Strength'])
        _set(b, distance, "Distance")
        links.new(height, b.inputs['Height'])
        if normal is not None:
            links.new(normal, b.inputs['Normal'])
        return b.outputs['Normal']

    # --- Seam-free bark coordinates in feature units (anisotropy k per pattern) -------------------------
    k = {"Fissured": 0.16, "Plated": 0.5, "Lenticelled": 3.0, "Fibrous": 0.08, "Annulated": 5.0,
         "Peeling": 0.7}.get(pattern, 0.5)
    a_base, a_along = attr("bark_base", (-2600, 200)), attr("bark_along", (-2600, 0))
    scale_k = nodes.new('ShaderNodeVectorMath')
    scale_k.operation = 'SCALE'
    scale_k.location = (-2400, 0)
    links.new(a_along.outputs['Vector'], scale_k.inputs[0])
    scale_k.inputs['Scale'].default_value = k
    q = nodes.new('ShaderNodeVectorMath')
    q.operation = 'ADD'
    q.location = (-2250, 100)
    links.new(a_base.outputs['Vector'], q.inputs[0])
    links.new(scale_k.outputs['Vector'], q.inputs[1])
    fs = max(0.005, float(bark.feature_scale_m))
    mapping = nodes.new('ShaderNodeMapping')
    mapping.location = (-2100, 100)
    links.new(q.outputs['Vector'], mapping.inputs['Vector'])
    mapping.inputs['Scale'].default_value = (1.0 / fs,) * 3
    Qv = mapping.outputs['Vector']
    # Distance along the axis in feature units (for transverse splits)
    along_len = nodes.new('ShaderNodeVectorMath')
    along_len.operation = 'LENGTH'
    along_len.location = (-2400, -200)
    links.new(a_along.outputs['Vector'], along_len.inputs[0])
    S = mnode('DIVIDE', along_len.outputs['Value'], fs, (-2250, -200))

    # --- Bark age from girth -------------------------------------------------------------------------
    onset = 0.01 * max(0.05, G("onset_radius_cm", 4.0))
    r = attr("bark_radius", (-2600, -400)).outputs['Fac']
    age = smooth(r, 0.5 * onset, 2.0 * onset, (-2250, -400))
    old = smooth(r, 2.0 * onset, 9.0 * onset, (-2250, -550))

    # --- Organic warp: high-detail noise mixed in Linear Light; its roughness varies in space -----------
    rough_n = noise_tex(Qv, 0.6, 15.0, (-2100, 450))
    rough_r = nodes.new('ShaderNodeMapRange')
    rough_r.location = (-1900, 450)
    rough_r.inputs['To Min'].default_value = 0.45
    rough_r.inputs['To Max'].default_value = 0.8
    links.new(rough_n.outputs['Fac'], rough_r.inputs['Value'])
    warp_n = noise_tex(Qv, 2.0, 15.0, (-1700, 400), rough=rough_r.outputs['Result'])
    warp = nodes.new('ShaderNodeMix')
    warp.data_type = 'RGBA'
    warp.blend_type = 'LINEAR_LIGHT'
    warp.location = (-1500, 300)
    warp.inputs[0].default_value = 0.03 + 0.12 * G("warp", 0.4)
    links.new(Qv, warp.inputs[6])
    links.new(warp_n.outputs['Color'], warp.inputs[7])
    W = warp.outputs[2]
    micro = noise_tex(Qv, 9.0, 15.0, (-1500, 650), rough=rough_r.outputs['Result'])
    # Directional striations along the axis (anisotropic noise: the fibrous grain of ridge faces)
    stri_map = nodes.new('ShaderNodeMapping')
    stri_map.location = (-1700, 800)
    links.new(W, stri_map.inputs['Vector'])
    stri_map.inputs['Scale'].default_value = (12.0, 12.0, 12.0 * k * 0.6)
    stri = noise_tex(stri_map.outputs['Vector'], 1.0, 8.0, (-1500, 850), 0.6)
    detail_n = mnode('ADD', mnode('MULTIPLY', micro.outputs['Fac'], 0.5, (-1300, 700)),
                     mnode('MULTIPLY', stri.outputs['Fac'], 0.5, (-1300, 850)), (-1150, 750))

    # --- Blocks: Minkowski Voronoi F1/F2 (rectangular when blocky); cracks = F2 - F1 ---------------------
    expo = 2.0 + 10.0 * G("blockiness", 0.4) ** 1.5
    v1 = voronoi(W, 1.0, 'F1', (-1300, 200), exponent=expo)
    v2 = voronoi(W, 1.0, 'F2', (-1300, 0), exponent=expo)
    gap = mnode('SUBTRACT', v2.outputs['Distance'], v1.outputs['Distance'], (-1100, 100))
    cell_rand = mnode('MULTIPLY', v1.outputs['Color'], 1.0, (-1100, 300))
    # Per-plate tilt: height ramps across each plate along a random direction
    rel = nodes.new('ShaderNodeVectorMath')
    rel.operation = 'SUBTRACT'
    rel.location = (-1100, -150)
    links.new(W, rel.inputs[0])
    links.new(v1.outputs['Position'], rel.inputs[1])
    dirv = nodes.new('ShaderNodeVectorMath')
    dirv.operation = 'MULTIPLY_ADD'
    dirv.location = (-1100, -300)
    links.new(v1.outputs['Color'], dirv.inputs[0])
    dirv.inputs[1].default_value = (2.0, 2.0, 2.0)
    dirv.inputs[2].default_value = (-1.0, -1.0, -1.0)
    tdot = nodes.new('ShaderNodeVectorMath')
    tdot.operation = 'DOT_PRODUCT'
    tdot.location = (-900, -200)
    links.new(rel.outputs[0], tdot.inputs[0])
    links.new(dirv.outputs[0], tdot.inputs[1])
    tilt = mnode('MULTIPLY', tdot.outputs['Value'], 0.6 * G("plate_tilt", 0.3), (-750, -200))
    # Transverse splits along the axis, offset per ridge
    nseg = 0.6 + 2.4 * G("segments", 0.3)
    # Wavy, irregular splits: the axial coordinate is perturbed by the warp noise before quantising
    S_w = mnode('ADD', S, mnode('MULTIPLY', mnode('SUBTRACT', warp_n.outputs['Fac'], 0.5, (-1400, -700)),
                                 1.2 / max(1e-3, nseg * k), (-1250, -700)), (-1100, -650))
    sx = mnode('FRACT', mnode('MULTIPLY_ADD', S_w, nseg * k, (-1100, -500), mnode('MULTIPLY', cell_rand, 7.3,
                                                                                  (-1250, -600))), None, (-950, -500))
    seg_cut = mnode('MULTIPLY', smooth(sx, 0.0, 0.16, (-800, -450)),
                    smooth(mnode('SUBTRACT', 1.0, sx, (-800, -600)), 0.0, 0.16, (-650, -600)), (-500, -500))
    seg_cut = mixf(min(1.0, 1.5 * G("segments", 0.3)), 1.0, seg_cut, (-350, -500))

    # --- Mature height field by pattern ----------------------------------------------------------------
    # Dome of each cell: high along its centre, falling toward the cracks (F1 distance, as in the
    # Chebyshev-Voronoi bark of the reference tutorial)
    dome = mnode('SUBTRACT', 1.0, smooth(v1.outputs['Distance'], 0.0, 0.6, (-1100, 500)), (-950, 500))
    if pattern == "Fissured":
        # Anastomosing ridges: contours of an axially stretched noise are long furrows that split and merge
        # into the diamond network of furrowed bark (Fraxinus, Robinia, Quercus); blocky barks additionally
        # break the ridges with Voronoi blocks and transverse splits.
        rn = noise_tex(W, 1.1, 3.0, (-1100, 750), 0.5)
        contour = mnode('ABSOLUTE', mnode('MULTIPLY_ADD', rn.outputs['Fac'], 2.0, (-950, 750), -1.0), None, (-800, 750))
        width = mnode('MULTIPLY_ADD', old, 0.25, (-900, 450), 0.10)                       # Furrows widen
        ridge = mnode('POWER', smooth(contour, 0.0, width, (-650, 700)), 0.6, (-500, 700))   # Rounded ridges
        blocks = mnode('MULTIPLY', smooth(gap, 0.0, mnode('MULTIPLY', width, 0.7, (-750, 450)), (-650, 450)),
                       mnode('MULTIPLY_ADD', dome, 0.4, (-650, 550), 0.6), (-500, 450))
        ridge = mixf(0.85 * G("blockiness", 0.4), ridge, mnode('MINIMUM', ridge, blocks, (-400, 550)), (-300, 600))
        Hm = mnode('MULTIPLY', ridge, seg_cut, (-150, 500))
        Hm = mnode('ADD', Hm, mnode('MULTIPLY', tilt, ridge, (-150, 350)), (0, 450))
        Hm = mnode('MULTIPLY', Hm, mnode('MULTIPLY_ADD', detail_n, 0.3, (0, 300), 0.7), (150, 450))
    elif pattern in ("Plated", "Fibrous"):
        fib = pattern == "Fibrous"
        # Furrow network from the contours of a stretched noise (less stretched for plates); plates are
        # broken by Voronoi blocks and transverse splits, and their faces carry flaky terraces (scales
        # peeling in layers). Fibrous barks: long strands with deep splits between broad bundles.
        rn = noise_tex(W, 0.9 if not fib else 1.6, 3.0, (-1100, 750), 0.5)
        contour = mnode('ABSOLUTE', mnode('MULTIPLY_ADD', rn.outputs['Fac'], 2.0, (-950, 750), -1.0), None, (-800, 750))
        width = mnode('MULTIPLY_ADD', old, 0.2 if not fib else 0.12, (-900, 450), 0.08 if not fib else 0.05)
        furrow = smooth(contour, 0.0, width, (-650, 700))
        if fib:          # Strands are not broken into blocks
            plate = mnode('MULTIPLY', furrow, seg_cut, (-350, 600))
        else:
            blocks = smooth(gap, 0.0, mnode('MULTIPLY', width, 0.8, (-750, 450)), (-650, 450))
            plate = mnode('MULTIPLY', mnode('MINIMUM', furrow, blocks, (-500, 600)), seg_cut, (-350, 600))
        if fib:
            strands = noise_tex(W, 6.0, 12.0, (-900, 950), 0.5)
            face = mnode('MULTIPLY_ADD', smooth(strands.outputs['Fac'], 0.3, 0.7, (-700, 950)), 0.5, (-550, 950), 0.5)
        else:
            flake = noise_tex(W, 2.5, 4.0, (-900, 950), 0.5)
            terr = mnode('DIVIDE', mnode('FLOOR', mnode('MULTIPLY', flake.outputs['Fac'], 4.0, (-750, 950)), None,
                                         (-600, 950)), 4.0, (-450, 950))                  # Layered scales
            face = mnode('ADD', mnode('MULTIPLY_ADD', terr, 0.35, (-300, 950), 0.55),
                         mnode('MULTIPLY', cell_rand, 0.15, (-300, 800)), (-150, 900))
        Hm = mnode('MULTIPLY', plate, mnode('ADD', face, tilt, (-150, 750)), (0, 650))
        Hm = mnode('MULTIPLY', Hm, mnode('MULTIPLY_ADD', detail_n, 0.25, (0, 500), 0.75), (150, 600))
    elif pattern == "Lenticelled":
        e = voronoi(W, 1.0, 'DISTANCE_TO_EDGE', (-900, 400))
        dash = mnode('SUBTRACT', 1.0, smooth(e.outputs['Distance'], 0.0, 0.05, (-750, 400)), (-600, 400))
        Hm = mnode('SUBTRACT', mnode('MULTIPLY_ADD', detail_n, 0.1, (-600, 250), 0.9),
                   mnode('MULTIPLY', dash, 0.5, (-450, 400)), (-300, 350))
    elif pattern == "Annulated":
        Hm = mnode('ADD', mnode('POWER', smooth(gap, 0.0, 0.2, (-750, 400)), 0.6, (-600, 400)), tilt, (-450, 350))
    elif pattern == "Peeling":
        patches = noise_tex(W, 0.9, 4.0, (-900, 600))
        layer = smooth(patches.outputs['Fac'], 0.46, 0.54, (-750, 600))
        lip = mnode('SUBTRACT', 1.0, mnode('ABSOLUTE', mnode('MULTIPLY_ADD', layer, 2.0, (-750, 450), -1.0),
                    None, (-600, 450)), (-450, 450))
        Hm = mnode('ADD', mnode('MULTIPLY_ADD', layer, 0.6, (-600, 600), 0.25), mnode('MULTIPLY', lip, 0.15,
                   (-450, 600)), (-300, 500))
    else:  # Smooth
        Hm = mnode('MULTIPLY_ADD', detail_n, 0.15, (-450, 350), 0.85)

    # --- Young periderm with transverse lenticels --------------------------------------------------------
    len_v = voronoi(Qv, 2.5, 'F1', (-900, -800), rand=0.9)
    lent = mnode('SUBTRACT', 1.0, smooth(len_v.outputs['Distance'], 0.04, 0.09, (-750, -800)), (-600, -800))
    Hy = mnode('SUBTRACT', mnode('MULTIPLY_ADD', detail_n, 0.08, (-600, -950), 0.92),
               mnode('MULTIPLY', lent, 0.25, (-450, -800)), (-300, -850))
    Hs = mixf(age, Hy, Hm, (0, -100))

    # --- Colour ------------------------------------------------------------------------------------------
    base_c = _srgb_to_linear(bark.base_color)
    sec = _srgb_to_linear(bark.secondary_color)
    inner = sec if pattern == "Peeling" else _srgb_to_linear(getattr(bark, "inner_color", None) or bark.secondary_color)
    young = _srgb_to_linear(getattr(bark, "young_color", None) or bark.base_color)
    fiss = mixc(smooth(Hs, 0.0, 0.3, (150, 450)), inner, sec, (300, 450))
    dirt = mnode('MULTIPLY', mnode('SUBTRACT', 1.0, smooth(Hs, 0.0, 0.35, (150, 600)), (300, 600)), 0.45, (450, 600))
    fiss = mixc(dirt, fiss, _srgb_to_linear((0.05, 0.04, 0.03)), (500, 450))                 # Dirt in cracks
    hsv = nodes.new('ShaderNodeHueSaturation')
    hsv.location = (300, 750)
    hsv.inputs['Color'].default_value = base_c
    links.new(mnode('MULTIPLY_ADD', cell_rand, 0.18, (150, 800), 0.91), hsv.inputs['Value'])
    links.new(mnode('MULTIPLY_ADD', cell_rand, 0.015, (150, 950), 0.4925), hsv.inputs['Hue'])
    colm = mixc(smooth(Hs, 0.12, 0.55, (500, 300)), fiss, hsv.outputs['Color'], (700, 400))
    weather = mnode('MULTIPLY', smooth(Hs, 0.65, 1.0, (500, 150)), 0.7 * G("weathering", 0.35), (700, 150))
    colm = mixc(weather, colm, _srgb_to_linear((0.56, 0.55, 0.52)), (900, 300))
    coly = mixc(mnode('MULTIPLY', lent, 0.6, (500, -600)), young, mixc(0.5, young, sec, (500, -750)), (700, -650))
    col = mixc(age, coly, colm, (1100, 100))
    tc = nodes.new('ShaderNodeTexCoord')
    tc.location = (-2600, -1300)
    macro = noise_tex(tc.outputs['Object'], 0.6, 2.0, (900, 900))
    col = mixc(0.18, col, macro.outputs['Color'], (1250, 200), 'OVERLAY')

    # --- Epiphytes: moss (upper / shaded / basal surfaces, in crevices) and crustose lichen --------------
    geo = nodes.new('ShaderNodeNewGeometry')
    geo.location = (-2600, -1550)
    sepn = nodes.new('ShaderNodeSeparateXYZ')
    sepn.location = (-2400, -1550)
    links.new(geo.outputs['Normal'], sepn.inputs[0])
    sepo = nodes.new('ShaderNodeSeparateXYZ')
    sepo.location = (-2400, -1300)
    links.new(tc.outputs['Object'], sepo.inputs[0])
    up = smooth(sepn.outputs['Z'], 0.0, 0.8, (-2200, -1500))
    shade = smooth(sepn.outputs['Y'], -0.2, 0.9, (-2200, -1650))                     # +Y = shaded (polar) side
    basal = mnode('SUBTRACT', 1.0, smooth(sepo.outputs['Z'], 0.2, 2.0, (-2200, -1300)), (-2050, -1300))
    expo_m = mnode('MINIMUM', mnode('ADD', mnode('ADD', mnode('MULTIPLY', up, 0.55, (-1900, -1500)),
                                                 mnode('MULTIPLY', shade, 0.35, (-1900, -1650)), (-1750, -1550)),
                                    mnode('MULTIPLY', basal, 0.45, (-1900, -1300)), (-1600, -1450)), 1.0, (-1450, -1450))
    mossv = noise_tex(tc.outputs['Object'], 7.0, 15.0, (-1500, -1200), 0.865, 0.3)
    # Noise threshold calibrated on the noise distribution (~0.35-0.65): no moss at 0.64, dense cover ~0.42
    m_eff = mnode('MINIMUM', mnode('MULTIPLY', expo_m, 2.0 * G("moss", 0.15), (-1300, -1450)), 1.0, (-1200, -1450))
    m_thr = mnode('MULTIPLY_ADD', m_eff, -0.24, (-1100, -1450), 0.66)
    crevice = mnode('MULTIPLY_ADD', Hs, -0.5, (-1150, -1600), 1.0)
    m_mask = mnode('MULTIPLY', smooth(mossv.outputs['Fac'], m_thr, mnode('ADD', m_thr, 0.06, (-1000, -1350)),
                                      (-850, -1300)), crevice, (-700, -1350))
    m_mask = mnode('MULTIPLY', m_mask, mnode('MULTIPLY_ADD', age, 0.8, (-850, -1500), 0.2), (-550, -1400))
    moss_col = mixc(mossv.outputs['Fac'], _srgb_to_linear((0.20, 0.34, 0.10)), _srgb_to_linear((0.42, 0.45, 0.18)),
                    (-550, -1150))
    lic = voronoi(tc.outputs['Object'], 9.0, 'F1', (-1300, -1850))
    lic_edge = noise_tex(tc.outputs['Object'], 40.0, 6.0, (-1300, -2050))
    lic_r = mnode('ADD', lic.outputs['Distance'], mnode('MULTIPLY', lic_edge.outputs['Fac'], 0.25, (-1100, -2050)),
                  (-950, -1950))
    pick = mnode('LESS_THAN', mnode('MULTIPLY', lic.outputs['Color'], 1.0, (-1100, -1800)),
                 0.6 * G("lichen", 0.15), (-950, -1800))
    l_mask = mnode('MULTIPLY', mnode('SUBTRACT', 1.0, smooth(lic_r, 0.36, 0.44, (-800, -1950)), (-650, -1950)), pick,
                   (-500, -1900))
    l_mask = mnode('MULTIPLY', l_mask, smooth(Hs, 0.3, 0.6, (-650, -2100)), (-350, -1950))   # On ridge tops
    lic_col = mixc(lic.outputs['Color'], _srgb_to_linear((0.66, 0.70, 0.60)), _srgb_to_linear((0.78, 0.74, 0.46)),
                   (-350, -1750))
    col = mixc(l_mask, col, lic_col, (1450, 0))
    col = mixc(m_mask, col, moss_col, (1600, -100))

    # Cavity: crotches, collar folds and deep furrows darken
    ao = nodes.new('ShaderNodeAmbientOcclusion')
    ao.location = (1600, 300)
    ao.only_local = True
    ao.samples = 8
    _set(ao, max(0.05, 2.0 * fs), "Distance")
    col = mixc(0.5, col, ao.outputs['Color'], (1800, 100), 'MULTIPLY')
    links.new(col, bsdf.inputs['Base Color'])

    # Roughness: ridges and moss rough, fissure walls a bit smoother
    rough = nodes.new('ShaderNodeMapRange')
    rough.location = (1800, -250)
    rough.inputs['To Min'].default_value = min(1.0, float(bark.roughness) + 0.08)
    rough.inputs['To Max'].default_value = max(0.6, float(bark.roughness) - 0.12)
    links.new(Hs, rough.inputs['Value'])
    links.new(mixf(m_mask, rough.outputs['Result'], 0.95, (2000, -250)), bsdf.inputs['Roughness'])
    links.new(mnode('MULTIPLY', m_mask, 0.4, (2000, -400)), bsdf.inputs['Sheen Weight'])

    # Stacked bumps: bark relief (stronger on old bark), micro detail, moss cushions
    relief = float(bark.relief)
    b1 = bump(Hs, mnode('MULTIPLY_ADD', age, 0.6 * relief, (1700, -600), 0.25 + 0.15 * relief),
              max(0.002, 0.35 * relief * fs), None, (1900, -600))
    b2 = bump(detail_n, 0.35, 0.002, b1, (2050, -650))
    mfuzz = noise_tex(tc.outputs['Object'], 400.0, 4.0, (1700, -900))
    b3 = bump(mnode('MULTIPLY', m_mask, mnode('MULTIPLY_ADD', mfuzz.outputs['Fac'], 0.5, (1850, -900), 0.75),
                    (2000, -850)), 1.0, 0.004, b2, (2150, -750))
    links.new(b3, bsdf.inputs['Normal'])

    # Optional true displacement (Cycles; pair with adaptive subdivision on the wood object)
    if displacement:
        disp = nodes.new('ShaderNodeDisplacement')
        disp.location = (2300, -500)
        links.new(mnode('ADD', Hs, mnode('MULTIPLY', m_mask, 0.3, (2100, -950)), (2200, -900)), disp.inputs['Height'])
        _set(disp, 0.5, "Midlevel")
        _set(disp, 0.45 * relief * fs, "Scale")
        links.new(disp.outputs['Displacement'], out.inputs['Displacement'])
        mat.displacement_method = 'BOTH'
    else:
        mat.displacement_method = 'BUMP'
    return mat
