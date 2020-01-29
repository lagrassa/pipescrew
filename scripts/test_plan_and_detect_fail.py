from belief import Belief
from concerrt import *
from belief import Wall
from agent import Agent
from nav_env import NavEnv

def test_trivial():
    backboard =  Wall((0.05, 0.12), (0.3, 0.12))
    goal = np.array((0.1,0.12))
    start = np.array((0.1, 0.1))
    b0 = Belief(mu=start, cov = 0.001, walls = [backboard])
    bg = Belief(mu=goal, cov = 0.001, walls = [backboard])
    bg.connected = True #because it's the goal
    policy = concerrt(b0, bg, gamma=0.02, p_bg=0.98)
    agent = Agent()
    ne = NavEnv(start=start, goal = goal)
    agent.follow_policy(policy, ne)

def test_execute_action():
    backboard =  Wall((0.05, 0.12), (0.3, 0.12))
    goal = np.array((0.1,0.12))
    start = np.array((0.1, 0.1))
    b0 = Belief(mu=start, cov = 0.001, walls = [backboard])
    bg = Belief(mu=goal, cov = 0.001, walls = [backboard])
    bg.connected = True #because it's the goal
    action = Connect(q_rand=goal, b_near = b0, b_old = b0, delta = 0.02)
    agent = Agent()
    ne = NavEnv(start=start, goal = goal)
    res = agent.execute_action(ne, action)
    assert(res)
    assert(ne.goal_condition_met())
    print("Test passed")


test_execute_action()


