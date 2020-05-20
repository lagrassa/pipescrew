import argparse
from agent.model_selection import ModelSelector

import numpy as np
from autolab_core import YamlConfig
from carbongym_utils.draw import draw_transforms
from env.block_push_ig_env import GymFrankaBlockPushEnv
from planning.blockpushpolicy import BlockPushPolicy
from planning.transition_models import LearnedTransitionModel, BlockPushSimpleTransitionModel

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
    [(vec_env._scene.step(), vec_env.render(custom_draws=custom_draws)) for i in range(5)]
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
    #policy = move_robot_to_start(vec_env, custom_draws)
    policy = BlockPushPolicy()
    obs = vec_env._compute_obs(None)
    start_state = obs["observation"]
    _, actions, _ = policy.plan(start_state, block_goal, delta = 0.00005, horizon=40)
    for t in range(actions.shape[-1]):
        [(vec_env.step(actions[:,t]), vec_env.render(custom_draws=custom_draws)) for i in range(1)]
    dists = vec_env.dists_to_goal(block_goal)
    tol = 0.01
    print(dists, "distances")
    if not ((np.abs(dists) < tol).all()):
        print("Test failed")
    else:
        print("test passed")


def test_action_sampling():
    vec_env, custom_draws = make_block_push_env()
    block_goal = vec_env.get_delta_goal(-0.08, visualize=False)
    policy = move_robot_to_start(vec_env, custom_draws)
    delta = 0.05
    start_states = vec_env.get_states()
    actions = policy.sample_actions(start_states, block_goal, delta = delta)
    next_states = start_states.copy()
    next_states[:,:3] = next_states[:,:3] + actions[:,:3]
    #actions go toward goal
    dist_to_goal_before = np.linalg.norm(start_states-block_goal)
    dist_to_goal_after = np.linalg.norm(next_states-block_goal)
    assert(dist_to_goal_after < dist_to_goal_before)
    assert(np.allclose(np.linalg.norm(actions[:,:3]), delta))
    print("test passed")

def test_model_selection():
    model_selector = ModelSelector()
    tm = BlockPushSimpleTransitionModel()
    lm = LearnedTransitionModel()
    model_selector.add(tm)
    model_selector.add(lm, model_type="learned")
    N=5
    good_states = np.random.uniform(low=-0.1, high = 0.1, size=(N,7))
    bad_states = np.random.uniform(low=0.1, high = 0.2, size=(N,7))
    states = np.vstack([good_states, bad_states])
    errors = [0.02,0.01,0,0,0,1,3,1,5,1]
    model_selector.add_history(states, errors, tm)
    null_action = np.array([0,0,0,1,0,0,0,10])
    tol = 0.5
    low_error_model = model_selector.select_model(states[0], null_action, tol)
    assert(low_error_model == tm)
    high_error_model = model_selector.select_model(states[8], null_action, tol)
    assert(high_error_model == lm)

    #pick one where it should pick the manual one
    # and where is should pick the learned one

def test_learned_transition_model():
    N = 500
    stiffness = 100
    random_states = np.random.uniform(low = -0.1, high = 0.1, size=(N,7))
    random_actions = np.random.uniform(low = -0.0001, high = 0.0001, size=(N,1))
    actions_rest =np.multiply(np.ones((random_states.shape[0], 5)), np.array([1,0,0,0,stiffness]) )

    random_actions = np.hstack([np.zeros((N,2)),random_actions, actions_rest])
    tm = BlockPushSimpleTransitionModel()
    next_states = tm.predict(random_states, random_actions)
    lm = LearnedTransitionModel()
    lm.train(random_states, random_actions, next_states)
    predicted_next_states = lm.predict(random_states, random_actions)[0]
    mean_dist = np.mean(np.abs(predicted_next_states[:,:3] - next_states[:,:3]))
    print("mean distance", mean_dist)
    assert(mean_dist) < 0.01
    print("test passed")

def test_learned_transition_model_real_data():
    N = 40
    states = np.load("data/states.npy")
    actions = np.load("data/actions.npy")
    next_states = np.load("data/next_states.npy")
    lm = LearnedTransitionModel()
    lm.train(states, actions, next_states)
    predicted_next_states = lm.predict(states, actions)
    max_dist = np.max(np.abs(predicted_next_states - next_states))
    assert(max_dist) < 0.01
    print("test passed")

def test_no_anomaly():
    """
    robot detects no significant deviation
    """
    vec_env, custom_draws = make_block_push_env()
    block_goal = vec_env.get_delta_goal(-0.1)
    #policy = move_robot_to_start(vec_env, custom_draws)
    policy = BlockPushPolicy()
    obs = vec_env._compute_obs(None)
    start_state = obs["observation"]
    states, actions, model_per_t = policy.plan(start_state, block_goal, delta = 0.001, horizon=50)
    deviations = policy.monitored_execution(vec_env, states, actions, custom_draws, model_per_t = model_per_t, tol = 0.25, plot_deviations = True)
    assert(len(deviations) <= 2) #first 2 are tricky to get but dont really count
    print("test passed")

def test_anomaly():
    """
    robot detects significant deviations
    """
    vec_env, custom_draws = make_block_push_env()
    block_goal = vec_env.get_delta_goal(-0.4)
    policy = BlockPushPolicy(use_history=True)
    obs = vec_env._compute_obs(None)
    start_state = obs["observation"]
    states, actions, model_per_t = policy.plan(start_state, block_goal, horizon=900)
    deviations = policy.monitored_execution(vec_env, states, actions, custom_draws, tol = 0.25, use_history=True, model_per_t=model_per_t, plot_deviations = True)
    assert(len(deviations) > 5)
    #plot points where there was an anomaly
#test_go_to_start()
test_anomaly()
# test_learned_transition_model()
#test_learned_transition_model_real_data()
#test_short_goal()
