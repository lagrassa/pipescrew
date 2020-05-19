import numpy as np
from carbongym import gymapi
from carbongym_utils.assets import GymFranka, GymBoxAsset, GymURDFAsset
from carbongym_utils.math_utils import np_to_quat, np_to_vec3, transform_to_np_rpy, rpy_to_quat, transform_to_np
import argparse
from autolab_core import YamlConfig
from gym.spaces import Box, Discrete
from carbongym_utils.draw import draw_transforms
from pyquaternion import Quaternion
from carbongym_utils.rl.franka_vec_env import GymFrankaVecEnv


def custom_draws(scene):
    franka = scene.get_asset('franka0')
    for env_ptr in scene.env_ptrs:
        ee_transform = franka.get_ee_transform(env_ptr, 'franka0')
        draw_transforms(scene.gym, scene.viewer, [env_ptr], [ee_transform])

class GymFrankaBlockPushEnv(GymFrankaVecEnv):

    def __init__(self, cfg):
        render_func = lambda x, y, z: self.render(custom_draws=custom_draws)
        super().__init__(cfg, n_inter_steps=cfg["env"]["n_inter_steps"], inter_step_cb=render_func, auto_reset_after_done=False)
    """
    IMPORTANT ASSUMPTION:
    if env_idx = 0, then it's the "real world" and if env_idx = 1 then its the "planning world" for many of these methods.
    If i get around to it, I'll note which are which.

    Just ignore 1 and use 0
    """
    def _fill_scene(self, cfg):
        super()._fill_scene(cfg)
        self.real_env_idx = 0
        self._block = GymBoxAsset(self._scene.gym, self._scene.sim, **cfg['block']['dims'],
                            shape_props=cfg['block']['shape_props'],
                            rb_props=cfg['block']['rb_props'],
                            asset_options=cfg['block']['asset_options']
                            )
        self._block_name = 'block0'
        rough_coef = 0.17
        self.dt = cfg['scene']['gym']['dt']
        self._board = GymBoxAsset(self._scene.gym, self._scene.sim, **cfg['boardpiece']['dims'],
                                  shape_props=cfg['boardpiece']['shape_props'],
                                  rb_props=cfg['boardpiece']['rb_props'],
                                  asset_options=cfg['boardpiece']['asset_options']
                                  )
        default_board_shape_props = cfg['boardpiece']['shape_props'].copy()
        default_board_shape_props["friction"] = rough_coef
        default_color = cfg['boardpiece']['rb_props'].copy()
        default_color['color'] = (0.5,0.8,0)
        print(cfg['block']['dims'])
        #self._n_inter_steps = cfg["env"]["n_inter_steps"]
        self._board2 = GymBoxAsset(self._scene.gym, self._scene.sim, **cfg['boardpiece']['dims'],
                                  shape_props=default_board_shape_props,
                                  rb_props=default_color,
                                  asset_options=cfg['boardpiece']['asset_options']
                                  )
        visual_block_color = default_color.copy()
        visual_block_color['color'] = (1,0.1,0.7)
        self._visual_block = GymBoxAsset(self._scene.gym, self._scene.sim, **cfg['block']['dims'],
                                         shape_props=cfg['block']['shape_props'],
                                         rb_props=visual_block_color,
                                         asset_options=cfg['block']['asset_options']
                                         )

        #add 3 boards
        self._board_name = "board1"
        self._board2_name = "board2"
        self._scene.add_asset(self._board_name, self._board, gymapi.Transform())
        self._scene.add_asset(self._board2_name, self._board2, gymapi.Transform())
        self._scene.add_asset(self._block_name, self._block, gymapi.Transform())
        #default goal setting you need to
        default_x = 0.1
        self.get_delta_goal(default_x)

    def goto_start(self, teleport=False):
        if teleport:
            for env_index, env_ptr in enumerate(self._scene.env_ptrs):
                joint_angles = np.load("data/push_joint_angles.npy")
                ah = self._scene.ah_map[env_index][self._franka_name]
                self._franka.set_joints(env_ptr, ah, joint_angles)
                self._scene.render()

        else:
            from planning.blockpushpolicy import BlockPushPolicy
            policy = BlockPushPolicy()
            vec_env = self._scene
            policy.go_to_push_start(self)
            [(vec_env.step(), vec_env.render()) for i in range(100)]
            policy.go_to_block(self)
            [(vec_env.step(), vec_env.render()) for i in range(90)]
            for env_index, env_ptr in enumerate(self._scene.env_ptrs):
                ah = self._scene.ah_map[env_index][self._franka_name]
                if env_index == 0:
                    joint_angles = self._franka.get_joints(env_ptr, ah)
            np.save("data/push_joint_angles.npy", joint_angles)

    def get_states(self, env_idx =None):
        """
        :return 2D array of envs and states relevant to the planner
        """
        return self.get_block_poses(env_idx=env_idx)
    def get_vels(self, env_idx = None):
        return 0*self.get_block_poses(env_idx=env_idx) #TODO make this the actual block poses
    def dists_to_goal(self, goal):
        block_poses = self.get_block_poses()[:,:3]
        return np.linalg.norm(block_poses[0,1:3]-goal)

    def get_block_poses(self, env_idx = None):
        if env_idx is None:
            box_pose_obs = np.zeros((self.n_envs, 7))
        for env_index, env_ptr in enumerate(self._scene.env_ptrs):
            ah = self._scene.ah_map[env_index][self._block_name]
            block_transform = self._block.get_rb_transforms(env_ptr, ah)[0]
            block_transform_np = transform_to_np(block_transform, format="wxyz")
            if env_idx == env_index:
                return block_transform_np
            else:
                box_pose_obs[env_index, :] = block_transform_np
        return box_pose_obs

    def get_delta_goal(self, delta_x, visualize=False):
        """
        goal state of block that's a delta
        """
        box_pose_obs= self.get_block_poses()
        delta_goal = box_pose_obs.copy()
        delta_goal[:,2] += delta_x
        if visualize:
            self._visual_block_name = "visualblock0"
            self._scene.add_asset(self._visual_block_name, self._visual_block, gymapi.Transform() )
            i = 0
            for env_index, env_ptr in enumerate(self._scene.env_ptrs):
                ah = self._scene.ah_map[env_index][self._visual_block_name]
                goal_pose = gymapi.Transform(p=np_to_vec3(delta_goal[i,:]))
                self._visual_block.set_rb_transforms(env_ptr, ah, [goal_pose])
                i+=1
        world_idx = 0
        self.desired_goal = delta_goal[world_idx,1:3]
        return self.desired_goal.copy()

    def set_goal(self, goal):
        self.desired_goal = goal

    def get_dists_to_goal(self):
        raise NotImplementedError
    def _reset(self, env_idxs=None):
        self._pre_grasp_transforms = []
        self._grasp_transforms = []
        self._init_ee_transforms = []
        if env_idxs is None:
            env_idxs = self._scene.env_ptrs
        super()._reset(env_idxs)
        for env_idx in env_idxs:
            env_ptr = self._scene.env_ptrs[env_idx]
            block_ah = self._scene.ah_map[env_idx][self._block_name]
            board_ah = self._scene.ah_map[env_idx][self._board_name]
            board2_ah = self._scene.ah_map[env_idx][self._board2_name]
            eps = 0.001
            block_pose = gymapi.Transform(
                p=np_to_vec3(np.array([
                    0.5,
                    self._cfg['table']['dims']['height'] + self._cfg['block']['dims']['height'] / 2 + self._cfg['boardpiece']['dims']['height'] / 2+ 0.01,
                    0.25]))
                )
            board_pose = gymapi.Transform(
                p = np_to_vec3(np.array([
                    0.5,
                    self._cfg['table']['dims']['height'],
                    0.18
                ]))
            )
            board2_pose = transform_to_np(board_pose)
            board2_pose[2] -= self._cfg['boardpiece']['dims']['depth']+0.001
            board2_pose = gymapi.Transform(p=np_to_vec3(board2_pose))

            self._block.set_rb_transforms(env_ptr, block_ah, [block_pose])
            self._block.set_rb_transforms(env_ptr, board_ah, [board_pose])
            self._block.set_rb_transforms(env_ptr, board2_ah, [board2_pose])
        self._scene.render()
        self.goto_start(teleport=False)

    def _init_action_space(self, cfg):
        action_space = super()._init_action_space(cfg)
        self.num_discrete_actions = 4
        self.discrete_actions_list = [[-0.1,10],[-0.3,30],[-0.01,10]]
        self.discrete_actions = {}
        i = 0
        for discrete_action in self.discrete_actions_list:
            self.discrete_actions[tuple(discrete_action)] = i

        return Discrete(len(self.discrete_actions_list))

    def _init_obs_space(self, cfg):
        self.unwrapped = self
        self._seed = 17 #lambda x: np.random.seed(x)
        self.seed = 17
        self.reward_range = None
        self.metadata = None
        obs_space = super()._init_obs_space(cfg)

        # add pose of block to obs_space
        limits_low = np.concatenate([
            obs_space.low,
            [-10] * 3 + [0] * 4
        ])
        limits_high = np.concatenate([
            obs_space.high,
            [10] * 3 + [1] * 4
        ])
        new_obs_space = Box(limits_low, limits_high, dtype=np.float32)
        self.num_features = obs_space.shape[0]
        return new_obs_space
    def _apply_actions(self, action, planning_env = False):
        for env_index, env_ptr in enumerate(self._scene.env_ptrs):
            ee_pose = self._franka.get_ee_transform(env_ptr, self._franka_name)
            np_pose = transform_to_np(ee_pose)[0:3]
            np_pose[2] += action[0]
            transformed_pt = gymapi.Transform(p=np_to_vec3(np.array(np_pose)), r = ee_pose.r);
            stiffness = action[1]
            self._franka.set_attractor_props(env_index, env_ptr, self._franka_name,
                                                {
                                                    'stiffness': stiffness,
                                                    'damping': 6 * np.sqrt(stiffness)
                                                })
            self._franka.set_ee_transform(env_ptr, env_index, self._franka_name, transformed_pt)

    def _compute_obs(self, all_actions):
        all_obs = super()._compute_obs(all_actions)
        box_pose_obs = self.get_block_poses()
        all_obs = np.c_[all_obs, box_pose_obs]
        obj_pos = box_pose_obs[self.real_env_idx,1:3]
        self._scene.render(custom_draws=custom_draws)
        return {"observation":obj_pos, "desired_goal":self.desired_goal, 'qpos':self.get_states(), 'qvel':self.get_vels(), 'achieved_goal': self.is_success(obj_pos, self.desired_goal) }
    """
    :param all_obs obj_pos and goal_pos in planning space since this is a planner specific function
    
    """
    def is_success(self, obj_pos, goal_pos):
        return np.linalg.norm(obj_pos-goal_pos) < 0.02
    def extract_features(self, obs, goal):
        return np.linalg.norm(obs-goal[:2])
    def _compute_dones(self, all_obs, all_actions, all_rews):
        if self.is_success(all_obs["observation"], self.desired_goal):
            return True
        return False #maybe have it error if it goes too far but I dont think its important
    def _compute_rews(self, obs, action):
        act_cost = np.linalg.norm(action)
        pose_cost = np.linalg.norm(self.desired_goal[:2]-obs['observation'])
        return act_cost +pose_cost
    def _compute_infos(self, all_obs, all_actions, all_rews, all_dones):
        return {}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', '-c', type=str, default='cfg/run_franka_rl_vec_env.yaml')
    args = parser.parse_args()
    cfg = YamlConfig(args.cfg)

    vec_env = GymFrankaBlockPushEnv(cfg)


    def custom_draws(scene):
        franka = scene.get_asset('franka0')
        for env_ptr in scene.env_ptrs:
            ee_transform = franka.get_ee_transform(env_ptr, 'franka0')
            draw_transforms(scene.gym, scene.viewer, [env_ptr], [ee_transform])


    all_obs = vec_env.reset()
    t = 0
    while True:
        all_actions = np.array([vec_env.action_space.sample() for _ in range(vec_env.n_envs)])
        #all_obs, all_rews, all_dones, all_infos = vec_env.step(all_actions)
        vec_env.render(custom_draws=custom_draws)

        t += 1
        if t == 100:
            vec_env.reset()
            t = 0
