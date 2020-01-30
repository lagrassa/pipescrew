from belief import Belief
from concerrt import *
from belief import Wall
from agent import Agent
from obstacle import line_world_obstacles
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
    agent.follow_policy(policy, ne, bg)

def test_long_distance_ground():
    goal = np.array((0.1,0.8))
    start = np.array((0.1, 0.1))
    obstacles = line_world_obstacles(goal)
    obst = obstacles[0]
    backboard =  Wall(obst.origin, obst.origin+np.array(obst.x))
    #make the backboard just the longest part of it, best is to make the one in the world real skinny
    ne = NavEnv(start=start, goal = goal, obstacles=obstacles )
    b0 = Belief(mu=start, cov = 0.0001, walls = [backboard])
    bg = Belief(mu=goal, cov = 0.01, walls = [backboard])
    bg.connected = True #because it's the goal
    policy = concerrt(b0, bg, gamma=0.5, p_bg=0.98, delta=0.1)
    agent = Agent()
    agent.follow_policy(policy, ne, bg)
    assert(ne.goal_condition_met())
    print("Test passed")

def test_long_distance_ice():
    pass


#runs Nx with model_based only
def test_success_method():
    data = test_success_ours()
    data = test_success_baseline()

def test_success_ours():
    #train ours
    pass
def test_success_baseline():
    #run baseline
    pass

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

def test_execute_guarded():
    backboard =  Wall((0.1, 0.12), (0.3, 0.12))
    goal = np.array((0.1,0.12))
    start = np.array((0.05, 0.05))
    b0 = Belief(mu=start, cov = 0.001, walls = [backboard])
    bg = Belief(mu=goal, cov = 0.001, walls = [backboard])
    bg.connected = True #because it's the goal
    action = Guarded(q_rand=goal, b_near = b0, b_old = b0, delta = 0.02)
    agent = Agent()
    ne = NavEnv(start=start, goal = goal)
    res = agent.execute_action(ne, action)
    assert(res)
    assert(ne.goal_condition_met())
    print("Test passed")

test_long_distance_ground()