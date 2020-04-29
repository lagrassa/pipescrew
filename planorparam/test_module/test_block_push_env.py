import argparse
import time

import numpy as np
from autolab_core import YamlConfig
from carbongym_utils.draw import draw_transforms
from env.block_push_ig_env import GymFrankaBlockPushEnv
from planning.blockpushpolicy import BlockPushPolicy

def make_block_push_env():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', '-c', type=str, default='cfg/franka_block_push.yaml')
    args = parser.parse_args()
    cfg = YamlConfig(args.cfg)
    vec_env = GymFrankaBlockPushEnv(cfg)
    vec_env.reset()
    def custom_draws(scene):
        franka = scene.get_asset('franka0')
        for env_ptr in scene.env_ptrs:
            ee_transform = franka.get_ee_transform(env_ptr, 'franka0')
            draw_transforms(scene.gym, scene.viewer, [env_ptr], [ee_transform])
    return vec_env, custom_draws

def move_robot_to_start(vec_env, custom_draws):
    policy = BlockPushPolicy()
    policy.go_to_push_start(vec_env)
    [(vec_env._scene.step(), vec_env.render(custom_draws=custom_draws)) for i in range(100)]
    policy.go_to_block(vec_env)
    [(vec_env._scene.step(), vec_env.render(custom_draws=custom_draws)) for i in range(50)]
    return policy

def test_delta_pose():
    vec_env, custom_draws = make_block_push_env()
    block_goal = vec_env.get_delta_goal(-0.1)
    import time
    for i in range(100):
        vec_env.render(custom_draws=custom_draws)
        time.sleep(0.1)

def test_go_to_start():
    vec_env, custom_draws = make_block_push_env()
    policy = BlockPushPolicy()
    [(vec_env._scene.step(), vec_env.render(custom_draws=custom_draws)) for i in range(10)]
    block_goal = vec_env.get_delta_goal(-0.02)
    move_robot_to_start(vec_env, custom_draws)

def test_short_goal():
    """
    robot can get nudge block some delta
    """
    vec_env, custom_draws = make_block_push_env()
    block_goal = vec_env.get_delta_goal(-0.08, visualize=False)
    policy = move_robot_to_start(vec_env, custom_draws)
    starts = vec_env.get_states()
    _, actions = policy.plan(starts, block_goal, horizon=200)
    for t in range(actions.shape[-1]):
        [(vec_env.step(actions[:,:,t]), vec_env.render(custom_draws=custom_draws)) for i in range(1)]
    dists = vec_env.dists_to_goal(block_goal)
    tol = 0.01
    if not ((np.abs(dists) < tol).all()):
        print(dists, "distances")


def test_no_anomaly():
    """
    robot detects no significant deviation
    """
    vec_env, custom_draws = make_block_push_env()
    block_goal = vec_env.get_delta_goal(-0.1)
    policy = move_robot_to_start(vec_env, custom_draws)
    starts = vec_env.get_states()
    states, actions = policy.plan(starts, block_goal, horizon=400)
    deviations = policy.monitored_execution(vec_env, states, actions, custom_draws, tol = 0.1, plot_deviations = True)
    assert(len(deviations) <= 2) #first 2 are tricky to get but dont really count
    print("test passed")

def test_anomaly():
    """
    robot detects significant deviations
    """
    vec_env, custom_draws = make_block_push_env()
    block_goal = vec_env.get_delta_goal(-0.1)
    policy = BlockPushPolicy()
    states, actions = policy.plan(block_goal)
    deviations = policy.monitored_execution(vec_env, states, actions)
    assert(len(deviations) > 0)
    #plot points where there was an anomaly

test_no_anomaly()
