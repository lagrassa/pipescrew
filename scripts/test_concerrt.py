from belief import Belief
from concerrt import *
b0 = Belief([0.1,0.1],0.1)
bg = Belief([0.5,0.5], 0.1)
def test_concerrt(b0, bg):
    pass

def test_nearest_neighbor():
    q_rand = random_config()
    T = Tree(b0)
    nn = nearest_neighbor(q_rand, T)
    #ensure there isn't one that is closer.

