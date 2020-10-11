import numpy as np
"""
RigidTransform object, outputs (pose, quat)
"""
def rigid_transform_to_pb_pose(rt):
    new_rt = rt.copy()
    if rt is None:
        return None
    #rot_transform = np.matrix([[0,1,0],[1,0,0],[0,0,-1]])
    rot_transform = np.matrix([[0,1,0],[1,0,0],[0,0,-1]])
    pose = new_rt.translation
    new_rt.rotation = np.dot(rot_transform,new_rt.rotation)
    #quat = np.hstack([rt.quaternion[1:], rt.quaternion[0]])
    quat = np.hstack([new_rt.quaternion[1:], new_rt.quaternion[0]])
    return [pose, quat]
