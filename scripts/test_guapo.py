from nav_env import NavEnv, line_world_obstacles
from scipy.stats import multivariate_normal as mvn
from screwpipe import PipeGraspEnv
from agent import Agent
from pipe_agent import PipeAgent
import numpy as np

def test_lineworld():
    agent = Agent(show_training = True)
    goal = np.array((0.4,0.5))
    ne = NavEnv(start=np.array((0.1, 0.1)), goal = goal, obstacles = line_world_obstacles(goal))
    ne.render()


def test_achieve_goal():
    agent = Agent(show_training = True)
    goal = np.array((0.4,0.5))
    obstacles = line_world_obstacles(goal)
    ne = NavEnv(start=np.array((0.1, 0.1)), goal = goal, obstacles = obstacles)
    ne.render()
    cov = 0.001
    mean = [obstacles[0].x, obstacles[0].y]
    obs_prior = mvn(mean=mean, cov= cov) # prior on the center of the line in xy space
    states, actions = agent.achieve_goal(ne,goal,obs_prior, N = 10)

test_achieve_goal()





