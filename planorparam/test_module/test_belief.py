from agent.belief import Belief
from agent.belief import Wall
import numpy as np
from planning.concerrt import Connect, Guarded, Slide, Tree

def test_belief_constructor():
    mu = np.array([1,0])
    belief = Belief(mu=mu, cov = 0.1, init_only = True)
    assert(np.allclose(mu, belief.mean()))
    assert(belief.cov().shape == (2,2))
    particles = belief.particles
    particles_mean = np.mean([particle.pose for particle in particles], axis=0)
    assert(np.allclose(particles_mean, belief.mean(), atol = 0.3))
    print("Test passed")

def test_collision_one_wall():
    wall = Wall((0.05, 0.1), (0.3, 0.1))
    far_belief = Belief(mu=(0.2, 0.0), cov = 0.001, walls = [wall], init_only = True)
    assert(len(far_belief.find_collisions().keys()) == 0)
    collide_belief = Belief(mu=(0.2, 0.09), cov = 0.1, walls = [wall])
    #import ipdb; ipdb.set_trace()
    assert(len(collide_belief.find_collisions().keys()) > 0)
    print("test passed")

def test_collision_two_walls():
    wall_1 = Wall((0.05, 0.1), (0.3, 0.1))
    wall_2 = Wall((0.3, 0.1), (0.3, 0.0))
    far_belief = Belief(mu=(0.2, 0.0), cov = 0.001, walls = [wall_1, wall_2], init_only = True)
    assert(len(far_belief.find_collisions().keys()) == 0)
    far_belief.visualize()
    collide_belief = Belief(mu=(0.29, 0.09), cov = 0.1, walls = [wall_1, wall_2], init_only = True)
    assert(len(collide_belief.find_collisions().keys()) > 0)
    print("test passed")

def test_wall_close():
    wall_1 = Wall((0.05, 0.1), (0.3, 0.1))
    far_belief = Belief(mu=(0.2, 0.0), cov = 0.001, walls = [wall_1], init_only = True)
    #import ipdb; ipdb.set_trace()
    assert(len(wall_1.get_particles_in_collision(far_belief)) == 0)
    close_belief = Belief(mu=(0.2, 0.09), cov = 0.1, walls = [wall_1], init_only = True)
    assert(len(wall_1.get_particles_in_collision(close_belief)) > 0)

def test_connect_motion_model():
    wall_1 = Wall((0.05, 0.1), (0.3, 0.1))
    close_belief = Belief(mu=(0.17, 0.05), cov = 0.00005, walls = [wall_1], init_only = True)
    new_q = (0.2, 0.09)
    close_belief.visualize(goal = new_q)
    u = Connect(new_q, close_belief, close_belief, delta = 0.03, sigma = 0.01)
    new_belief = u.motion_model()
    new_belief.visualize(goal=new_q)

def test_guarded_motion_model():
    wall_1 = Wall((0.05, 0.1), (0.3, 0.1))
    close_belief = Belief(mu=(0.17, 0.05), cov = 0.00005, walls = [wall_1], init_only = True)
    new_q = (0.2, 0.09)
    close_belief.visualize(goal = new_q)
    u = Guarded(new_q, close_belief, close_belief,delta = 0.07, sigma = 0.01)
    new_belief = u.motion_model()
    new_belief.visualize(goal=new_q)

def test_slide_motion_model():
    wall_1 = Wall((0.05, 0.1), (0.3, 0.1))
    close_belief = Belief(mu=(0.17, 0.05), cov = 0.00005, walls = [wall_1], init_only = True)
    new_q = (0.2, 0.09)
    u = Guarded(new_q, close_belief, close_belief,delta = 0.07, sigma = 0.01)
    new_belief = u.motion_model()
    side_q = (0.08,0.1)
    u = Slide(side_q, new_belief, close_belief,delta = 0.08)
    new_belief2 = u.motion_model()
    new_belief2.visualize(goal=side_q)


def test_visualize_belief():
    wall_1 = Wall((0.05, 0.1), (0.3, 0.1))
    close_belief = Belief(mu=(0.17, 0.05), cov = 0.00005, walls = [wall_1], init_only = True)
    b2 = Belief(mu=(0.14, 0.02), cov = 0.00005, walls = [wall_1], init_only = True)
    b3 = Belief(mu=(0.11, 0.03), cov = 0.00005, walls = [wall_1], init_only = True)
    b4 = Belief(mu=(0.12, 0.05), cov = 0.00005, walls = [wall_1], init_only = True)
    tree = Tree(close_belief)
    tree.add_belief(close_belief, b2)
    tree.display()
    resp = input("confirm looks good")
    assert('y' in resp)
    tree.add_belief(b2, b3)
    tree.display()
    resp = input("confirm looks good")
    assert('y' in resp)
    tree.add_belief(b2, b4)
    tree.display()
    resp = input("confirm looks good")
    assert('y' in resp)



#test_wall_close()
#test_collision_one_wall()
#test_slide_motion_model()
test_visualize_belief()





