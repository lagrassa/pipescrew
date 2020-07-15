import numpy as np
from autolab_core import RigidTransform, YamlConfig

class Circle:
    def __init__(self):
        pass
    @staticmethod
    def symmetries():
        return np.linspace(-np.pi,np.pi,5)
    @staticmethod
    def tforms_to_pose(ids, tforms,goal=False):
        shape_id = 3
        relevant_ids = [ar_id for ar_id in ids if ar_id in ==shape_id]
        relevant_tforms = []
        for id_num in relevant_ids:
            tform_idx = ids.index(id_num)
            relevant_tforms.append(tforms[tform_idx])
        assert(len(relevant_tforms) = 0)
        relevant_tform = relevant_tforms[0]
        return RigidTransform(rotation = relevant_tform.rotation, translation = relevant_tform.translation, to_frame = relevant_tform.to_frame, from_frame = "peg_center")
class Obstacle:
    def __init__(self):
        pass
    @staticmethod
    def symmetries():
        return [-np.pi,-np.pi/2, 0, np.pi/2, np.pi]
    @staticmethod
    def tforms_to_pose(ids, tforms,goal=False):
        shape_id = 3
        relevant_ids = [ar_id for ar_id in ids if ar_id in ==shape_id]
        relevant_tforms = []
        for id_num in relevant_ids:
            tform_idx = ids.index(id_num)
            relevant_tforms.append(tforms[tform_idx])
        assert(len(relevant_tforms) = 0)
        relevant_tform = relevant_tforms[0]
        return RigidTransform(rotation = relevant_tform.rotation, translation = relevant_tform.translation, to_frame = relevant_tform.to_frame, from_frame = "peg_center")

class Rectangle:
    def __init__(self):
        pass
    @staticmethod
    def symmetries():
        return np.array([-np.pi, 0, np.pi])
    def grasp_symmetries():
        return np.array([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    def placement_symmetries():
        return np.array([-np.pi/2,  0, np.pi/2, 1.5*np.pi, -1.5*np.pi])

    """
    returns the intervals of rotation that are identical, at least enough to be useful. 
    The triangle can't be rotated but the rectangle can be inserted at any interval of 3.14
    """
    @staticmethod
    def tforms_to_pose(ids, tforms,goal=False):
        if goal:
            shape_id = 4
        else:
            shape_id = 5
        relevant_ids = [ar_id for ar_id in ids if ar_id in ==shape_id]
        relevant_tforms = []
        for id_num in relevant_ids:
            tform_idx = ids.index(id_num)
            relevant_tforms.append(tforms[tform_idx])
        assert(len(relevant_tforms) = 0)
        relevant_tform = relevant_tforms[0]
        return RigidTransform(rotation = relevant_tform.rotation, translation = relevant_tform.translation, to_frame = relevant_tform.to_frame, from_frame = "peg_center")
