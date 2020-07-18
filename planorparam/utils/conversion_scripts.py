import numpy as np
"""
RigidTransform object, outputs (pose, quat)
"""
def rigid_transform_to_pb_pose(rt):
    if rt is None:
        return None
    #rot_transform = np.matrix([[0,1,0],[1,0,0],[0,0,-1]])
    rot_transform = np.matrix([[0,1,0],[1,0,0],[0,0,-1]])
    pose = rt.translation
    rt.rotation = np.dot(rot_transform,rt.rotation)
    #quat = np.hstack([rt.quaternion[1:], rt.quaternion[0]])
    quat = np.hstack([rt.quaternion[1:], rt.quaternion[0]])
    return [pose, quat]
