from agent.belief import Belief
from planning.concerrt import *
from agent.belief import Wall


def test_concerrt_trivial_connect():
    backboard =  Wall((0.05, 0.12), (0.3, 0.12))
    b0 = Belief(mu=(0.1,0.1), cov = 0.001, walls = [backboard], init_only = True)
    bg = Belief(mu=(0.1,0.12), cov = 0.1, walls = [backboard], init_only = True)
    bg.connected = True #because it's the goal
    policy = concerrt(b0, bg, gamma=0.02, p_bg=0.98)
    action = policy(b0, bg)
    assert(action.b_near == b0)
    next_b = simulate(action) # a little weird to write out but the action has a beginning belief stored in it
    assert(mala_distance(next_b, bg) < 0.5)
    #should get closer to the goal!!! Even if not in terms of the tree
    print("Test passed")

def test_guarded_close():
    backboard =  Wall((0.05, 0.12), (0.3, 0.12))
    sideboard =  Wall((0.11, 0.12), (0.11, 0))
    #make the covariance vertical so hitting the wall *does* squish it.
    vert_cov_matrix = np.matrix([0.0000005, 0,
                         0, 0.00005]).reshape((2, 2))
    b0 = Belief(mu=(0.1,0.08), cov = vert_cov_matrix, walls = [backboard, sideboard],init_only = True)
    bg = Belief(mu=(0.1,0.1), cov = 0.0001, walls = [backboard, sideboard], init_only = True)
    bg.connected = True #because it's the goal
    policy = concerrt(b0, bg, p_bg = 0.98)
    curr_belief = b0
    for i in range(10):
        action = policy(curr_belief, bg)
        curr_belief.visualize()
        print(action)
        curr_belief = simulate(action) # a little weird to write out but the action has a beginning belief stored in it
    assert(mala_distance(b0,bg) > mala_distance(curr_belief, bg))
    print(mala_distance(curr_belief, bg), "final distance")
    assert(mala_distance(curr_belief, bg) < 2)
    #should get closer to the goal!!! Even if not in terms of the tree
    print("Test passed")

def test_guarded_far():
    backboard =  Wall((0.05, 0.12), (0.3, 0.12))
    sideboard =  Wall((0.11, 0.12), (0.11, 0))
    b0 = Belief(mu=(0.06,0.05), cov = 0.001, walls = [backboard, sideboard], init_only = True)
    bg = Belief(mu=(0.1,0.12), cov = 0.001, walls = [backboard, sideboard], init_only = True)
    bg.connected = True #because it's the goal
    policy = concerrt(b0, bg)
    curr_belief = b0
    for i in range(10):
        action = policy(curr_belief, bg)
        print(action)
        curr_belief = simulate(action) # a little weird to write out but the action has a beginning belief stored in it
    assert(mala_distance(b0,bg) > mala_distance(curr_belief, bg))
    #should get closer to the goal!!! Even if not in terms of the tree
    print("Test passed")

def test_need_slide():
    backboard =  Wall((0.05, 0.12), (0.3, 0.12))
    sideboard =  Wall((0.2, 0.12), (0.2, 0))
    b0 = Belief(mu=(0.04,0.05), cov = 0.0001, walls = [backboard, sideboard], init_only = True)
    bg = Belief(mu=(0.18,0.12), cov = 0.001, walls = [backboard, sideboard], init_only = True)
    #b0.visualize(goal=bg)
    bg.connected = True #because it's the goal
    policy = concerrt(b0, bg,gamma = 0.9999, delta = 0.02, p_bg=0.98)
    curr_belief = b0
    actions = policy.get_best_actions(curr_belief, bg)
    for i in range(15):
        action = actions[i]#policy(curr_belief, bg)
        print(action)
        curr_belief = simulate(action) # a little weird to write out but the action has a beginning belief stored in it
        curr_belief.high_prob_collision(old_belief=b0)
    assert(mala_distance(curr_belief,bg) < 0.3)



#test_concerrt_trivial_connect()
#test_guarded_close()
test_need_slide()


