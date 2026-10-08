"""
Geometry Nodes instancing helpers shared by foliage, flowers and forests.

A points mesh carries per-point `frot` (XYZ Euler), `fscale` and any extra attributes; a
"PPG Instancer" modifier puts one prototype object on every point. Point attributes become
instance attributes, readable in shaders through an Attribute node of type 'INSTANCER'.
"""

import math
import numpy as np

try:
    import bpy
    BLENDER_AVAILABLE = True
except ImportError:
    BLENDER_AVAILABLE = False

GROUP_VERSION = 1


def euler_xyz(R: np.ndarray) -> np.ndarray:
    """Blender XYZ Euler angles of rotation matrices R (N, 3, 3) whose columns are the local axes."""
    sy = np.clip(-R[:, 2, 0], -1.0, 1.0)
    ry = np.arcsin(sy)
    rx = np.arctan2(R[:, 2, 1], R[:, 2, 2])
    rz = np.arctan2(R[:, 1, 0], R[:, 0, 0])
    return np.stack([rx, ry, rz], 1)


def frames_to_euler(X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """Euler angles of (orthonormalised) frames given by their axis vectors."""
    Z = Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-12)
    X = X - Z * np.sum(X * Z, axis=1, keepdims=True)
    X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    Y = np.cross(Z, X)
    return euler_xyz(np.stack([X, Y, Z], axis=2))


def instancer_group(name: str = "PPG Instancer"):
    """Node group: Instance on Points(points, Object Info(Proto)) with named rotation and scale."""
    group = bpy.data.node_groups.get(name)
    if group is not None and group.get("ppg_version") == GROUP_VERSION:
        return group
    group = bpy.data.node_groups.new(name, 'GeometryNodeTree')
    group["ppg_version"] = GROUP_VERSION
    group.interface.new_socket('Geometry', in_out='INPUT', socket_type='NodeSocketGeometry')
    group.interface.new_socket('Prototype', in_out='INPUT', socket_type='NodeSocketObject')
    group.interface.new_socket('Geometry', in_out='OUTPUT', socket_type='NodeSocketGeometry')
    n, l = group.nodes, group.links
    gi = n.new('NodeGroupInput')
    go = n.new('NodeGroupOutput')
    go.location = (800, 0)
    info = n.new('GeometryNodeObjectInfo')
    info.location = (0, -200)
    inst = n.new('GeometryNodeInstanceOnPoints')
    inst.location = (500, 0)
    rot = n.new('GeometryNodeInputNamedAttribute')
    rot.data_type = 'FLOAT_VECTOR'
    rot.inputs['Name'].default_value = "frot"
    rot.location = (0, -400)
    e2r = n.new('FunctionNodeEulerToRotation')
    e2r.location = (250, -400)
    scl = n.new('GeometryNodeInputNamedAttribute')
    scl.data_type = 'FLOAT'
    scl.inputs['Name'].default_value = "fscale"
    scl.location = (0, -600)
    l.new(gi.outputs[0], inst.inputs['Points'])
    l.new(gi.outputs[1], info.inputs['Object'])
    l.new(info.outputs['Geometry'], inst.inputs['Instance'])
    l.new(rot.outputs['Attribute'], e2r.inputs[0])
    l.new(e2r.outputs[0], inst.inputs['Rotation'])
    l.new(scl.outputs['Attribute'], inst.inputs['Scale'])
    l.new(inst.outputs['Instances'], go.inputs[0])
    return group


def attach_instancer(obj, proto, modifier_name: str = "PPG_Instancer"):
    """Adds (or updates) the instancing modifier on `obj`, pointing at `proto`."""
    mod = obj.modifiers.get(modifier_name)
    if mod is None:
        mod = obj.modifiers.new(modifier_name, 'NODES')
    mod.node_group = instancer_group()
    key = next((it.identifier for it in mod.node_group.interface.items_tree
                if getattr(it, "in_out", "") == 'INPUT' and it.name == "Prototype"), None)
    if key is not None:
        mod[key] = proto
    obj.update_tag()
    return mod


def remove_instancer(obj, modifier_name: str = "PPG_Instancer"):
    mod = obj.modifiers.get(modifier_name)
    if mod is not None:
        obj.modifiers.remove(mod)
