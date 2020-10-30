import numpy as np
import os
from planning.transition_models import BlockPushSimpleTransitionModel
from pillar_state_py import State
from carbongym_utils.math_utils import rpy_to_quat, quat_to_rpy, quat_to_np
from pyquaternion import Quaternion


robot_pos_fqn = "frame:pose/position"
robot_orn_fqn = "frame:pose/quaternion"
block_pos_fqn = "frame:block:pose/position"
block_orn_fqn = "frame:block:pose/quaternion"
block_on_color_fqn = "frame:block:on_color"

dir_to_rpy = {0: [-np.pi / 2, np.pi / 2, 0],
                   1: [-np.pi / 2, np.pi / 2, np.pi / 2],
                   2: [-np.pi / 2, np.pi / 2, np.pi],
                   3: [-np.pi / 2, np.pi / 2, 1.5 * np.pi]}
class Operator:
    def __init__(self, *args, cfg=None):
        self.new_states = []
        self.new_actions = []
        self.new_expected_next_states = []
        self.new_actual_next_states = []
        self.cfg=cfg

    def monitor_execution(self, env, action_feature):
        current_pillar_state = env.get_pillar_state()[0]
        state = self.pillar_state_to_feature(current_pillar_state)
        self.new_states.append(state)
        self.new_actions.append(action_feature)
        self.execute_prim(env)
        actual_state = self.pillar_state_to_feature(env.get_pillar_state()[0])
        self.new_actual_next_states.append(actual_state)
        expected_next_state_pillar = self.transition_model(current_pillar_state, action_feature)
        expected_next_state = self.pillar_state_to_feature(expected_next_state_pillar)
        self.new_expected_next_states.append(expected_next_state)
        self.save_transitions(cfg=self.cfg)

    def execute_prim(self):
        raise NotImplementedError
    def cost(self):
        return 1
    def save_transitions(self, cfg=None):
        if cfg is None:
            cfg = {}
            cfg["data_path"] = "data/default/"
        fn = cfg["data_path"]+self.__class__.__name__+".npy"
        if not os.path.exists(cfg["data_path"]):
            os.mkdir(cfg["data_path"])
        if os.path.exists(fn):
            data = np.load(fn, allow_pickle=True).all()
        else:
            data = {"states":[], "actions":[], "expected_next_states":[], "actual_next_states":[]}
        if len(data["states"]) == 0:
            data["states"] = self.new_states
            data["actions"] = self.new_actions
            data["expected_next_states"] = self.new_expected_next_states
            data["actual_next_states"] = self.new_actual_next_states
        else:
            data["states"] = np.vstack([data["states"], self.new_states])
            data["actions"] = np.vstack([data["actions"], self.new_actions])
            data["expected_next_states"] = np.vstack([data["expected_next_states"], self.new_expected_next_states])
            data["actual_next_states"] = np.vstack([data["actual_next_states"], self.new_actual_next_states])
        np.save(fn, data)
        for list_name in [self.new_states, self.new_actions, self.new_expected_next_states, self.new_actual_next_states]:
            list_name.clear()

class GoToSide(Operator):
    deviations = []
    def __init__(self, sidenum, cfg=None):
        self.sidenum = sidenum
        super().__init__(cfg=cfg)
    def execute_prim(self, env):
        return env.goto_side(self.sidenum)
    def precond(self, state):
        return True
    def monitor_execution(self, env):
        action_feature = [self.sidenum]
        super().monitor_execution(env, action_feature)
    def pillar_state_to_feature(self,pillar_state):
        return []
    def transition_model(self, state, action):
        return state
    def cost(self):
        return 1

class PushInDir(Operator):
    def __init__(self, sidenum, amount, T, cfg=None):
        self.sidenum = sidenum
        self.amount = amount
        self.T = T
        self._transition_model = BlockPushSimpleTransitionModel()
        super().__init__(cfg=cfg)

    def execute_prim(self, env):
        env.push_in_dir(self.sidenum, self.amount, self.T)
    def cost(self):
        return self.amount
    def precond(self, state_str):
        state = State.create_from_serialized_string(state_str)
        robot_pos = state.get_values_as_vec([robot_pos_fqn])
        robot_orn = state.get_values_as_vec([robot_orn_fqn])
        block_pos = state.get_values_as_vec([block_pos_fqn])
        delta_side = state.get_values_as_vec(["constants/block_width"])/2
        #close to block side and also in right orientation
        robot_des_pos = np.array([block_pos[0] + delta_side * np.sin(dir_to_rpy[dir][2]),
                                 block_pos[1],
                                 block_pos[2] + delta_side * np.cos(dir_to_rpy[dir][2])])
        gripper_pos_close = np.linalg.norm(robot_des_pos-robot_pos < 0.03)
        des_quat = quat_to_np(rpy_to_quat(np.array(self.dir_to_rpy[dir])), format="wxyz")
        orn_dist = Quaternion.absolute_distance(des_quat, robot_orn)
        orn_close = orn_dist < 0.01
        return gripper_pos_close and orn_close


    def transition_model(self, state, action):
        return self._transition_model.predict(state, action)

    def monitor_execution(self, env):
        action_feature = [self.sidenum, self.amount, self.T]
        super().monitor_execution(env, action_feature)

    def pillar_state_to_feature(self, pillar_state_str):
        pillar_state = State.create_from_serialized_string(pillar_state_str)
        states = np.array(pillar_state.get_values_as_vec([block_pos_fqn, block_on_color_fqn]))
        return states.flatten()


