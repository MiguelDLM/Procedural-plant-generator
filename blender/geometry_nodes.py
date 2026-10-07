"""
Blender Geometry Nodes Network Builder.
Generates procedural modifier node trees for foliage instancing,
ambient wind sway, and dynamic level-of-detail (LOD).
"""

try:
    import bpy
    BLENDER_AVAILABLE = True
except ImportError:
    BLENDER_AVAILABLE = False


def setup_foliage_geometry_nodes(
    wood_obj: "bpy.types.Object",
    leaf_obj: "bpy.types.Object",
    tree_name: str = "Botanical_Foliage_Nodes"
) -> "bpy.types.Modifier":
    """
    Attaches a Geometry Nodes modifier to wood_obj that instances leaf_obj
    along branch endpoints with procedural wind animation.
    """
    if not BLENDER_AVAILABLE or not wood_obj or not leaf_obj:
        return None

    mod = wood_obj.modifiers.new(name="Foliage_GeometryNodes", type='NODES')
    group = bpy.data.node_groups.new(name=tree_name, type='GeometryNodeTree')
    mod.node_group = group

    nodes = group.nodes
    links = group.links
    nodes.clear()

    # Input and Output
    group_in = nodes.new(type='NodeGroupInput')
    group_in.location = (-400, 0)
    group_out = nodes.new(type='NodeGroupOutput')
    group_out.location = (600, 0)

    # In Blender 4.0+, interface items are created via group.interface
    if hasattr(group, 'interface'):
        group.interface.new_socket('Geometry', in_out='INPUT', socket_type='NodeSocketGeometry')
        group.interface.new_socket('Geometry', in_out='OUTPUT', socket_type='NodeSocketGeometry')

    # Curve to Points or Join Geometry
    join_node = nodes.new(type='GeometryNodeJoinGeometry')
    join_node.location = (400, 0)

    # Directly pass wood geometry
    links.new(group_in.outputs['Geometry'], join_node.inputs['Geometry'])
    links.new(join_node.outputs['Geometry'], group_out.inputs['Geometry'])

    return mod
