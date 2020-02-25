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
        rectangles_ids = [10,11]
        relevant_ids = [ar_id for ar_id in ids if ar_id in rectangles_ids]
        relevant_tforms = []
        for id_num in relevant_ids:
            tform_idx = ids.index(id_num)
            relevant_tforms.append(tforms[tform_idx])

        avg_rotation = relevant_tforms[0]
        for i in range(1, len(relevant_tforms)):
            avg_rotation = avg_rotation.interpolate_with(relevant_tforms[i], 0.5)

        if len(relevant_ids) == 2:
            translation = np.mean(np.vstack([T.translation for T in relevant_tforms]), axis=0)
        else:
            print(relevant_ids)
            print("Not enough detections to make accurate pose estimate")
        return RigidTransform(rotation = avg_rotation.rotation, translation = translation, to_frame = tforms[0].to_frame, from_frame = "peg_center")
class Obstacle:
    def __init__(self):
        pass
    @staticmethod
    def symmetries():
        return [-np.pi,-np.pi/2, 0, np.pi/2, np.pi]
    @staticmethod
    def tforms_to_pose(ids, tforms,goal=False):
        rectangles_ids = [8,9]
        relevant_ids = [ar_id for ar_id in ids if ar_id in rectangles_ids]
        relevant_tforms = []
        for id_num in relevant_ids:
            tform_idx = ids.index(id_num)
            relevant_tforms.append(tforms[tform_idx])
        
        avg_rotation = relevant_tforms[0]
        for i in range(1, len(relevant_tforms)):
            avg_rotation = avg_rotation.interpolate_with(relevant_tforms[i], 0.5)

        if len(relevant_ids) == 2:
            translation = np.mean(np.vstack([T.translation for T in relevant_tforms]), axis=0)
        else:
            print(relevant_ids)
            print("Not enough detections to make accurate pose estimate")
        return RigidTransform(rotation = avg_rotation.rotation, translation = translation, to_frame = tforms[0].to_frame, from_frame = "peg_center")

class Rectangle:
    def __init__(self):
        pass
    @staticmethod
    def symmetries():
        return [-np.pi, 0, np.pi]
    """
    returns the intervals of rotation that are identical, at least enough to be useful. 
    The triangle can't be rotated but the rectangle can be inserted at any interval of 3.14
    """
    @staticmethod
    def tforms_to_pose(ids, tforms,goal=False):
        if goal:
            rectangles_ids = [4,5,6,7]
        else:
            rectangles_ids = [0,1,2,3]
        relevant_ids = [ar_id for ar_id in ids if ar_id in rectangles_ids]
        relevant_tforms = []
        for id_num in relevant_ids:
            tform_idx = ids.index(id_num)
            relevant_tforms.append(tforms[tform_idx])

        avg_rotation = relevant_tforms[0]
        for i in range(1, len(relevant_tforms)):
            avg_rotation = avg_rotation.interpolate_with(relevant_tforms[i], 0.5)

        if len(relevant_ids) == 4:
            translation = np.mean(np.vstack([T.translation for T in relevant_tforms]), axis=0)
        elif 0 in relevant_ids and 2 in relevant_ids:
            translation = np.mean(np.vstack([T.translation for T in relevant_tforms]), axis=0)
        elif 1 in relevant_ids and 3 in relevant_ids:
            translation = np.mean(np.vstack([T.translation for T in relevant_tforms]), axis=0)
        else:
            print(relevant_ids)
            print("Not enough detections to make accurate pose estimate")

        return RigidTransform(rotation = avg_rotation.rotation, translation = translation, to_frame = tforms[0].to_frame, from_frame = "peg_center")
