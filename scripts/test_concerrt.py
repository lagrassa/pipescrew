from belief import Belief
from concerrt import *
b0 = Belief([0.1,0.1],0.1)
bg = Belief([0.5,0.5], 0.1)
bg.connected = True #because it's the goal
def test_concerrt_trivial(b0, bg):
    b0 = Belief(mu=(0.1,0.1), cov = 0.001)
    bg = Belief(mu=(0.1,0.12), cov = 0.001)
    policy = concerrt(b0, bg)

def test_nearest_neighbor():
    q_rand = random_config()
    T = Tree(b0)
    nn = nearest_neighbor(q_rand, T)
    #ensure there isn't one that is closer.

test_concerrt_trivial(b0, bg)


