from belief import Belief
from concerrt import *
from belief import Wall


def test_concerrt_trivial_connect():
    b0 = Belief(mu=(0.1,0.1), cov = 0.001)
    bg = Belief(mu=(0.1,0.12), cov = 0.1)
    bg.connected = True #because it's the goal
    policy = concerrt(b0, bg)
    action = policy(b0, bg)
    assert(action.b_near == b0)
    next_b = simulate(action) # a little weird to write out but the action has a beginning belief stored in it
    assert(mala_distance(b0,bg) > mala_distance(next_b, bg))
    assert(mala_distance(next_b, bg) < 2)
    #should get closer to the goal!!! Even if not in terms of the tree
    print("Test passed")

def test_guarded_close():
    backboard =  Wall((0.05, 0.1), (0.3, 0.1))
    sideboard =  Wall((0.11, 0.1), (0.11, 0))
    b0 = Belief(mu=(0.1,0.1), cov = 0.001, walls = [backboard, sideboard])
    bg = Belief(mu=(0.1,0.12), cov = 0.001, walls = [backboard, sideboard])
    bg.connected = True #because it's the goal
    policy = concerrt(b0, bg)
    curr_belief = b0
    for i in range(10):
        action = policy(curr_belief, bg)
        print(action)
        curr_belief = simulate(action) # a little weird to write out but the action has a beginning belief stored in it
    assert(mala_distance(b0,bg) > mala_distance(curr_belief, bg))
    print(mala_distance(curr_belief, bg), "final distance")
    assert(mala_distance(curr_belief, bg) < 2)
    #should get closer to the goal!!! Even if not in terms of the tree
    print("Test passed")

def test_guarded_far():
    backboard =  Wall((0.05, 0.1), (0.3, 0.1))
    sideboard =  Wall((0.11, 0.1), (0.11, 0))
    b0 = Belief(mu=(0.06,0.05), cov = 0.001, walls = [backboard, sideboard])
    bg = Belief(mu=(0.1,0.12), cov = 0.001, walls = [backboard, sideboard])
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
#test_concerrt_trivial_connect()
test_guarded_close()
#test_guarded_far()


