from nav_env import NavEnv
from agent import Agent
import numpy as np

GOAL_THRESHOLD = 0.08
def test_cluster_two_simple():
    agent = Agent()
    agent.collect_planning_history()
    #TODO after this synthesize planning history
    agent.cluster_planning_history() #here the agent clusters trajectories to put them in the DMP
    goal = np.array((1,0.67))
    ne = NavEnv(start=np.array((0.2, 0.12)), goal = goal)
    imitation_path = agent.dmp_plan(ne.get_pos(), goal)
    ne.plot_path(imitation_path)
    agent.follow_path(ne, imitation_path)
    assert(np.linalg.norm(ne.get_pos()-goal) < GOAL_THRESHOLD)

test_cluster_two_simple()

