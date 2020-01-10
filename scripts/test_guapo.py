from nav_env import NavEnv, line_world_obstacles
from scipy.stats import multivariate_normal as mvn
from agent import Agent
import numpy as np


def test_lineworld():
    goal = np.array((0.4, 0.5))
    ne = NavEnv(start=np.array((0.1, 0.1)), goal=goal, obstacles=line_world_obstacles(goal))
    ne.render()


def test_achieve_goal():
    agent = Agent(show_training=True)
    goal = np.array((0.4, 0.5))
    obstacles = line_world_obstacles(goal)
    ne = NavEnv(start=np.array((0.1, 0.1)), goal=goal, obstacles=obstacles)
    ne.render()
    cov = 0.001
    obs_center = [obstacles[0].origin[0] + obstacles[0].x / 2.0, obstacles[0].origin[1]]
    obs_prior = mvn(mean=obs_center, cov=cov)  # prior on the center of the line in xy space
    agent.achieve_goal(ne, goal, obs_prior, N=10)


def test_model_based_traj():
    agent = Agent(show_training=True)
    goal = np.array((0.4, 0.5))
    obstacles = line_world_obstacles(goal)
    cov = 0.001
    obs_center = [obstacles[0].origin[0] + obstacles[0].x / 2.0, obstacles[0].origin[1]]
    obs_prior = mvn(mean=obs_center, cov=cov)  # prior on the center of the line in xy space
    agent.belief.in_s_uncertain = obs_prior
    ne = NavEnv(start=np.array((0.1, 0.1)), goal=goal, obstacles=obstacles)
    nsteps = 40
    s0 = np.concatenate([ne.get_pos(), ne.get_vel()])
    mb_xs = agent.model_based_trajectory(s0, ne, nsteps=nsteps)
    ne.plot_path(mb_xs.T)


def test_model_based_policy():
    agent = Agent(show_training=True)
    goal = np.array((0.4, 0.5))
    obstacles = line_world_obstacles(goal)
    cov = 0.001
    obs_center = [obstacles[0].origin[0] + obstacles[0].x / 2.0, obstacles[0].origin[1]]
    obs_prior = mvn(mean=obs_center, cov=cov)  # prior on the center of the line in xy space
    agent.belief.in_s_uncertain = obs_prior
    ne = NavEnv(start=np.array((0.85, 0.1)), goal=goal, obstacles=obstacles)
    nsteps = 40
    s0 = np.concatenate([ne.get_pos(), ne.get_vel()])
    s = s0
    agent.model_based_policy(s, ne, nsteps=40, recompute_rmp=True)
    for i in range(nsteps):
        action = agent.model_based_policy(s, ne, recompute_rmp=False)
        print(action, "action")
        ne.step(action)
    ne.render(flip=True)
    resp = input("confirm looks good: \n")
    assert ("y" in resp)


def test_get_obs():
    goal = np.array((0.1, 0.3))
    obstacles = line_world_obstacles(goal)
    ne = NavEnv(start=np.array((0.1, 0.28)), goal=goal, obstacles=obstacles)
    ne.render()
    grid = ne.get_obs()
    from PIL import Image
    im = Image.fromarray(grid)
    im.resize((400, 400)).show()
    resp = input("confirm looks good: \n")
    assert ("y" in resp)


def test_model_free():
    goal = np.array((0.1, 0.3))
    obstacles = line_world_obstacles(goal)
    obs_center = [obstacles[0].origin[0] + obstacles[0].x / 2.0, obstacles[0].origin[1]]
    cov = 0.001
    obs_prior = mvn(mean=obs_center, cov=cov)  # prior on the center of the line in xy space
    agent = Agent(show_training=True)
    agent.belief.in_s_uncertain = obs_prior
    start = np.array((0.1, 0.25))
    nsteps = 40
    ne = NavEnv(start=start, goal=goal, obstacles=obstacles, gridsize=[5 * 50, 5 * 70], visualize=False)
    agent.model_free_policy(ne.get_obs(), ne, n_epochs=40, train=True)
    ne.reset()
    ne.setup_visuals()
    for i in range(nsteps):
        action = agent.model_free_policy(ne.get_obs(), ne, n_epochs=1, train=False)
        ne.step(action)
        if i % 10 == 0:
            print("action", action)
            print("goal distance", ne.goal_distance())
            print("position", ne.agent.position)
            ne.render_start_goal()
    print("goal distance", ne.goal_distance())
    assert (ne.goal_condition_met())
    ne.render()
    print("Test passed")


def test_mb_mf_switch():
    goal = np.array((0.1, 0.4))
    obstacles = line_world_obstacles(goal)
    obs_center = [obstacles[0].origin[0] + obstacles[0].x / 2.0, obstacles[0].origin[1]]
    cov = 0.001
    obs_prior = mvn(mean=obs_center, cov=cov)  # prior on the center of the line in xy space
    agent = Agent(show_training=True)
    agent.belief.in_s_uncertain = obs_prior
    start = np.array((0.1, 0.1))
    ne = NavEnv(start=start, goal=goal, obstacles=obstacles, gridsize=[10 * 50, 10 * 70], visualize=False)
    ne.reset()
    ne.setup_visuals()
    agent.achieve_goal(ne, goal, obs_prior, N = 200)
    print("goal distance", ne.goal_distance())
    assert (ne.goal_condition_met())
    print("Test passed")


#test_model_free()
test_mb_mf_switch()
