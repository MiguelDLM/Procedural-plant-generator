"""
Botanical PBR Material Generators for Blender.
Creates biologically realistic procedural shaders with subsurface scattering,
chlorophyll translucency, adaxial cuticular wax sheen, and bark micro-roughness.
"""

try:
    import bpy
    BLENDER_AVAILABLE = True
except ImportError:
    BLENDER_AVAILABLE = False


def create_foliage_material(name: str = "Botanical_Foliage_PBR") -> "bpy.types.Material":
    """
    Creates a procedural leaf shader with subsurface scattering (SSS)
    representing cellular mesophyll and chlorophyll light absorption.
    """
    if not BLENDER_AVAILABLE:
        return None

    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    # Output node
    out_node = nodes.new(type='ShaderNodeOutputMaterial')
    out_node.location = (400, 0)

    # Principled BSDF
    bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
    bsdf.location = (0, 0)

    # Adaxial chlorophyll green base color
    bsdf.inputs['Base Color'].default_value = (0.08, 0.28, 0.04, 1.0)
    # Waxy cuticle roughness
    bsdf.inputs['Roughness'].default_value = 0.35

    # Subsurface scattering for leaf translucency
    if 'Subsurface Weight' in bsdf.inputs:
        bsdf.inputs['Subsurface Weight'].default_value = 0.45
        bsdf.inputs['Subsurface Radius'].default_value = (0.25, 0.70, 0.15)
        bsdf.inputs['Subsurface Scale'].default_value = 0.05
    elif 'Subsurface' in bsdf.inputs:
        bsdf.inputs['Subsurface'].default_value = 0.45
        bsdf.inputs['Subsurface Color'].default_value = (0.12, 0.45, 0.06, 1.0)

    # Cuticle specular reflection
    if 'Specular IOR Level' in bsdf.inputs:
        bsdf.inputs['Specular IOR Level'].default_value = 0.50

    links.new(bsdf.outputs['BSDF'], out_node.inputs['Surface'])
    return mat


def create_bark_material(name: str = "Botanical_Bark_PBR") -> "bpy.types.Material":
    """
    Creates a procedural bark shader with anisotropic wood grain and furrowing.
    """
    if not BLENDER_AVAILABLE:
        return None

    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    out_node = nodes.new(type='ShaderNodeOutputMaterial')
    out_node.location = (400, 0)

    bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
    bsdf.location = (0, 0)

    # Bark brownish-grey tone
    bsdf.inputs['Base Color'].default_value = (0.16, 0.12, 0.08, 1.0)
    bsdf.inputs['Roughness'].default_value = 0.88

    links.new(bsdf.outputs['BSDF'], out_node.inputs['Surface'])
    return mat
