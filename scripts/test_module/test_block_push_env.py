import argparse

import numpy as np
from autolab_core import YamlConfig
from carbongym_utils.draw import draw_transforms
from env.block_push_ig_env import GymFrankaBlockPushEnv
from planning.blockpushpolicy import BlockPushPolicy

def make_block_push_env():
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
    return vec_env, custom_draws


def test_delta_controller():
    """
    robot can get nudge block some delta
    """
    vec_env, custom_draws = make_block_push_env()
    policy = BlockPushPolicy()
    block_goal = vec_env.get_delta_goal(0.01)
    starts = vec_env.get_states()
    _, actions = policy.plan(starts, block_goal)
    vec_env.step(actions)
    vec_env.render(custom_draws=custom_draws)
    dists = vec_env.dists_to_goal(block_goal)
    tol = 0.01
    assert((np.abs(dists) < tol).all())

def test_short_goal():
    """
    robot can reach a goal that doesn't go into a slippery area
    """
    vec_env, custom_draws = make_block_push_env()
    block_goal = vec_env.get_delta_goal(0.1)
    policy = BlockPushPolicy()
    _, actions = policy.plan(block_goal)
    vec_env.step(actions)
    vec_env.render(custom_draws=custom_draws)
    dists = vec_env.dists_to_goal(block_goal)
    tol = 0.01
    assert ((np.abs(dists) < tol).all())


def test_no_anomaly():
    """
    robot detects no significant deviation
    """
    vec_env, custom_draws = make_block_push_env()
    block_goal = vec_env.get_delta_goal(0.1)
    policy = BlockPushPolicy()
    states, actions = policy.plan(block_goal)
    deviations = policy.monitored_execution(vec_env, states, actions)
    assert(len(deviations) == 0)

def test_anomaly():
    """
    robot detects significant deviations
    """
    vec_env, custom_draws = make_block_push_env()
    block_goal = vec_env.get_delta_goal(0.1)
    policy = BlockPushPolicy()
    states, actions = policy.plan(block_goal)
    deviations = policy.monitored_execution(vec_env, states, actions)
    assert(len(deviations) > 0)
    #plot points where there was an anomaly
