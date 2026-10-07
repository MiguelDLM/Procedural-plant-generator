"""
Botanical materials for Blender.

Leaf: image-based (outline alpha + venation colour + vein height map generated
by core.leaf_texture), with
- abaxial (underside) tint on back faces,
- translucency (light through the lamina, yellower along veins less so),
- per-leaf hue/value jitter from the `leaf_random` mesh attribute,
- gloss from cuticle wax.

Bark: procedural node graph per rhytidome pattern (fissured, plated, peeling,
lenticelled, fibrous, smooth, annulated) driven by the species BarkProfile.
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

    # UV: U wraps the circumference, V runs along the axis (1 UV unit = bark tile)
    uv = nodes.new('ShaderNodeTexCoord')
    uv.location = (-1200, 0)
    mapping = nodes.new('ShaderNodeMapping')
    mapping.location = (-1000, 0)
    links.new(uv.outputs['UV'], mapping.inputs['Vector'])
    s = 0.6 / max(0.005, float(bark.feature_scale_m))  # Features per tile

    noise = nodes.new('ShaderNodeTexNoise')
    noise.location = (-800, 300)
    _set(noise, 4.0, "Scale")
    _set(noise, 6.0, "Detail")
    links.new(mapping.outputs['Vector'], noise.inputs['Vector'])

    warp = nodes.new('ShaderNodeMix')
    warp.data_type = 'VECTOR'
    warp.location = (-600, 100)
    warp.inputs[0].default_value = 0.08
    links.new(mapping.outputs['Vector'], warp.inputs[4])
    links.new(noise.outputs['Color'], warp.inputs[5])

    fac_socket = None
    height_socket = None
    if pattern in ("Fissured", "Plated", "Lenticelled"):
        vor = nodes.new('ShaderNodeTexVoronoi')
        vor.location = (-400, 100)
        vor.feature = 'DISTANCE_TO_EDGE'
        links.new(warp.outputs[1], vor.inputs['Vector'])
        if pattern == "Fissured":
            mapping.inputs['Scale'].default_value = (s, s * 0.22, 1.0)   # Long vertical ridges
        elif pattern == "Plated":
            mapping.inputs['Scale'].default_value = (s, s * 0.55, 1.0)
        else:
            mapping.inputs['Scale'].default_value = (s * 0.25, s * 2.0, 1.0)  # Horizontal lenticel dashes
        ramp = nodes.new('ShaderNodeValToRGB')
        ramp.location = (-150, 100)
        if pattern == "Lenticelled":
            ramp.color_ramp.elements[0].position = 0.0
            ramp.color_ramp.elements[1].position = 0.06
        else:
            ramp.color_ramp.elements[0].position = 0.0
            ramp.color_ramp.elements[1].position = 0.12 if pattern == "Fissured" else 0.05
        links.new(vor.outputs['Distance'], ramp.inputs['Fac'])
        inv = nodes.new('ShaderNodeMath')
        inv.operation = 'SUBTRACT'
        inv.location = (100, 100)
        inv.inputs[0].default_value = 1.0
        links.new(ramp.outputs['Color'], inv.inputs[1])
        fac_socket = inv.outputs[0]
        height_socket = ramp.outputs['Color']
    elif pattern in ("Fibrous", "Annulated"):
        wave = nodes.new('ShaderNodeTexWave')
        wave.location = (-400, 100)
        wave.wave_type = 'BANDS'
        wave.bands_direction = 'X' if pattern == "Fibrous" else 'Y'
        _set(wave, 1.0, "Scale")
        _set(wave, 9.0 if pattern == "Fibrous" else 2.0, "Distortion")
        _set(wave, 4.0, "Detail")
        mapping.inputs['Scale'].default_value = (s, s, 1.0)
        links.new(warp.outputs[1], wave.inputs['Vector'])
        ramp = nodes.new('ShaderNodeValToRGB')
        ramp.location = (-150, 100)
        ramp.color_ramp.elements[0].position = 0.25
        ramp.color_ramp.elements[1].position = 0.7
        links.new(wave.outputs['Fac'], ramp.inputs['Fac'])
        inv = nodes.new('ShaderNodeMath')
        inv.operation = 'SUBTRACT'
        inv.location = (100, 100)
        inv.inputs[0].default_value = 1.0
        links.new(ramp.outputs['Color'], inv.inputs[1])
        fac_socket = inv.outputs[0]
        height_socket = ramp.outputs['Color']
    elif pattern == "Peeling":
        mapping.inputs['Scale'].default_value = (s * 0.6, s * 0.35, 1.0)
        patches = nodes.new('ShaderNodeTexNoise')
        patches.location = (-400, 100)
        _set(patches, 2.0, "Scale")
        _set(patches, 3.0, "Detail")
        links.new(warp.outputs[1], patches.inputs['Vector'])
        ramp = nodes.new('ShaderNodeValToRGB')
        ramp.location = (-150, 100)
        ramp.color_ramp.elements[0].position = 0.47
        ramp.color_ramp.elements[1].position = 0.53
        links.new(patches.outputs['Fac'], ramp.inputs['Fac'])
        fac_socket = ramp.outputs['Color']
        height_socket = ramp.outputs['Color']
    else:  # Smooth
        mapping.inputs['Scale'].default_value = (s * 0.3, s * 0.3, 1.0)
        ramp = nodes.new('ShaderNodeValToRGB')
        ramp.location = (-150, 100)
        ramp.color_ramp.elements[0].position = 0.35
        ramp.color_ramp.elements[1].position = 0.75
        links.new(noise.outputs['Fac'], ramp.inputs['Fac'])
        fac_socket = ramp.outputs['Color']
        height_socket = noise.outputs['Fac']

    # Colour: base vs secondary (furrows / lenticels / underbark) with fine mottling
    mix = nodes.new('ShaderNodeMix')
    mix.data_type = 'RGBA'
    mix.location = (350, 200)
    mix.inputs[6].default_value = _srgb_to_linear(bark.base_color)
    mix.inputs[7].default_value = _srgb_to_linear(bark.secondary_color)
    links.new(fac_socket, mix.inputs[0])

    mottle = nodes.new('ShaderNodeMix')
    mottle.data_type = 'RGBA'
    mottle.blend_type = 'OVERLAY'
    mottle.location = (560, 200)
    mottle.inputs[0].default_value = 0.35
    links.new(mix.outputs[2], mottle.inputs[6])
    links.new(noise.outputs['Color'], mottle.inputs[7])
    links.new(mottle.outputs[2], bsdf.inputs['Base Color'])

    bump = nodes.new('ShaderNodeBump')
    bump.location = (560, -200)
    _set(bump, 0.25 + 0.75 * float(bark.relief), "Strength")
    _set(bump, 0.004 + 0.02 * float(bark.relief), "Distance")
    links.new(height_socket, bump.inputs['Height'])
    links.new(bump.outputs['Normal'], bsdf.inputs['Normal'])
    return mat
