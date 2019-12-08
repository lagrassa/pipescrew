from nav_env import NavEnv
from screwpipe import PipeGraspEnv
from agent import Agent as NavAgent
from pipe_agent import PipeAgent
import numpy as np

def test_plan_and_dmp(n_tests, n_train, agent, env_class, goal, start_func, visualize=False, goal_threshold = 0.05):
    start = start_func()
    agent.collect_planning_history(N=n_train, goals = (goal,))
    agent.cluster_planning_history(k=2) #here the agent clusters trajectories to put them in the DMP
    dmp_successes = test_dmp(n_tests, n_train, agent, env_class, goal, start, visualize=visualize, goal_threshold = goal_threshold)
    plan_successes = test_plan(n_tests, n_train, agent, env_class, goal, visualize, goal_threshold, start, test_shift = test_shift)
    print("Results")
    print("########################################################")
    print("DMP metric mean", np.mean(dmp_successes))
    print("Planner metric mean", np.mean(plan_successes))

def test_dmp(n_tests, n_train, agent, env_class, goal, start, visualize=False, goal_threshold = 0.05):
    dmp_successes = []
    test_shift = 0.005
    for i in range(n_tests):
        print("DMP Test #", i)
        env = env_class(start=start, goal = goal, visualize=visualize, shift = test_shift)
        imitation_path = agent.dmp_plan(env.get_pos(), goal)
        if visualize:
            env.plot_path(imitation_path)
        agent.follow_path(env, imitation_path)
        dmp_successes.append(env.goal_condition_met())
        env.close()
    return dmp_successes
def test_plan(n_tests, n_train, agent, env_class, goal, visualize, goal_threshold, start, force=None, test_shift = 0):
    plan_successes = []
    for i in range(n_tests):
        print("Plan Test #", i)
        env = env_class(start=start, goal = goal, visualize=visualize, shift = test_shift)
        actual_path = agent.plan_path(env, env.get_pos(), goal)
        if actual_path is None:
            print("no path found")
            #plan_successes.append(False)
            continue

        if visualize:
            env.plot_path(np.vstack(actual_path))
        agent.follow_path(env, actual_path, force=force)
        plan_successes.append(env.goal_condition_met())
        env.close()
    return plan_successes

def test_rl_nav():
    n_tests = 2
    n_train = 10
    agent = NavAgent(show_training=False)
    goal = np.array((1.05,1.15))
    start_func=lambda : np.array((0.05, 0.06)) + 0.03*np.random.random(2,)
    print("Before")
    agent.collect_planning_history(N=n_train, goals = (goal,))
    agent.cluster_planning_history(k=2) #here the agent clusters trajectories to put them in the DMP
    test_dmp(n_tests, n_train, agent, NavEnv, goal, start_func(), visualize=False)
    print("After")
    import ipdb; ipdb.set_trace()
    agent.do_rl()
    test_dmp(n_tests, n_train, agent, NavEnv, goal, start_func(), visualize=False)


def test_cluster_pipe_many():
    goal = np.array([0,0,0.1])
    n_tests = 2
    n_train = 3
    agent = PipeAgent(show_training=False)
    start_func=lambda : 0.001*np.random.random()
    test_plan_and_dmp(n_tests, n_train, agent, PipeGraspEnv, goal, start_func, visualize=False)

def test_force_effect():
    env_class = PipeGraspEnv
    goal = np.array([0,0,0.1])
    n_tests = 1
    n_train = 3
    goal_threshold = 0.05 #largely irrelevant
    visualize =  False
    agent = PipeAgent(show_training=False)
    start_func=lambda : 0.001*np.random.random()
    forces = np.linspace(10,200,70)
    #forces = [30,150]
    shift_to_force_to_success = []
    shifts = np.linspace(0, 0.01, 40)
    for shift in shifts:
        force_to_success_mean = []
        force_to_success_std = []
        for force in forces:
            plan_successes = test_plan(n_tests, n_train, agent, env_class, goal, visualize, goal_threshold, start_func(), force=force, test_shift = shift)
            force_to_success_mean.append(np.mean(plan_successes))
            force_to_success_std.append(np.std(plan_successes))
        shift_to_force_to_success.append((force_to_success_mean, force_to_success_std))
        
    import ipdb; ipdb.set_trace()
    means = np.vstack([shift[0] for shift in shift_to_force_to_success])
    np.save("shift_to_force_to_success_2.npy", means)
    np.save("shifts.npy", shifts)
    np.save("forces.npy", forces)


def test_cluster_nav_many():
    n_tests = 5
    n_train = 4
    agent = NavAgent(show_training=False)
    goal = np.array((1.05,1.15))
    start_func=lambda : np.array((0.05, 0.06)) + 0.03*np.random.random(2,)
    test_plan_and_dmp(n_tests, n_train, agent, NavEnv, goal, start_func, visualize=False)

#test_force_effect()
test_rl_nav()
