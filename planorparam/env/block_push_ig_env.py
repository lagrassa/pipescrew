import numpy as np
from carbongym import gymapi
from carbongym_utils.assets import GymFranka, GymBoxAsset, GymURDFAsset
from carbongym_utils.math_utils import np_to_quat, np_to_vec3, transform_to_np_rpy, rpy_to_quat, transform_to_np
import argparse
from autolab_core import YamlConfig
from gym.spaces import Box
from carbongym_utils.draw import draw_transforms
from pyquaternion import Quaternion
from carbongym_utils.rl.franka_vec_env import GymFrankaVecEnv
class GymFrankaBlockPushEnv(GymFrankaVecEnv):

    def _fill_scene(self, cfg):
        super()._fill_scene(cfg)
        self._block = GymBoxAsset(self._scene.gym, self._scene.sim, **cfg['block']['dims'],
                            shape_props=cfg['block']['shape_props'],
                            rb_props=cfg['block']['rb_props'],
                            asset_options=cfg['block']['asset_options']
                            )
        self._block_name = 'block0'
        slippery = 0.1
        default_board_shape_props = cfg['boardpiece']['shape_props']
        self._board = GymBoxAsset(self._scene.gym, self._scene.sim, **cfg['boardpiece']['dims'],
                                  shape_props=cfg['boardpiece']['shape_props'],
                                  rb_props=cfg['boardpiece']['rb_props'],
                                  asset_options=cfg['boardpiece']['asset_options']
                                  )
        default_board_shape_props["friction"] = slippery
        default_color = cfg['boardpiece']['rb_props'].copy()
        default_color['color'] = (0,0,0.7)
        print(cfg['block']['dims'])
        self._board2 = GymBoxAsset(self._scene.gym, self._scene.sim, **cfg['boardpiece']['dims'],
                                  shape_props=cfg['boardpiece']['shape_props'],
                                  rb_props=default_color,
                                  asset_options=cfg['boardpiece']['asset_options']
                                  )

        #add 3 boards
        self._board_name = "board1"
        self._board2_name = "board2"
        self._scene.add_asset(self._board_name, self._board, gymapi.Transform())
        self._scene.add_asset(self._board2_name, self._board2, gymapi.Transform())
        self._scene.add_asset(self._block_name, self._block, gymapi.Transform())

    def get_states(self):
        """
        :return 2D array of envs and states relevant to the planner
        """
        raise NotImplementedError
    def get_delta_goal(self, delta_x):
        """
        goal state of block that's a delta
        """
        raise NotImplementedError

    def get_dists_to_goal(self):
        raise NotImplementedError
    def _reset(self, env_idxs):
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
                    self._cfg['table']['dims']['height'] + self._cfg['block']['dims']['height'] / 2 + self._cfg['boardpiece']['dims']['height'] / 2+ 0.001,
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

    def _init_obs_space(self, cfg):
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

        return new_obs_space
    def _step(self, action):
        #todo get above from action, make a planner
        self._franka.set_delta_ee_transform(self, env_ptr, env_index, name, tf)

    def _compute_obs(self, all_actions):
        all_obs = super()._compute_obs(all_actions)
        box_pose_obs = np.zeros((self.n_envs, 7))

        for env_index, env_ptr in enumerate(self._scene.env_ptrs):
            ah = self._scene.ah_map[env_index][self._block_name]

            block_transform = self._block.get_rb_transforms(env_ptr, ah)[0]
            box_pose_obs[env_index, :] = transform_to_np(block_transform, format='wxyz')

        all_obs = np.c_[all_obs, box_pose_obs]
        return all_obs
    """
    :param all_obs list of +
    
    """
    def _compute_rews(self, all_obs, all_actions):
        rews = []
        des_pitch = np.pi/2
        des_roll = np.pi/2
        for (obs, act) in zip(all_obs, all_actions):
            act_cost = np.linalg.norm(act)
            yaw, pitch, roll = Quaternion(obs[-4:]).yaw_pitch_roll #difference from upright. We want roll OR pitch to be close to straight but we don't care about yaw
            pose_cost = min((pitch-des_pitch)**2,(des_roll- roll)**2)
            rews.append(-act_cost-pose_cost)
        return np.array(rews)

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
