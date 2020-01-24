from belief import Belief
from belief import Wall
import numpy as np
from concerrt import Connect, Guarded, Slide

def test_belief_constructor():
    mu = np.array([1,0])
    belief = Belief(mu=mu, cov = 0.1)
    assert(np.allclose(mu, belief.mean()))
    assert(belief.cov().shape == (2,2))
    particles = belief.particles
    particles_mean = np.mean([particle.pose for particle in particles], axis=0)
    assert(np.allclose(particles_mean, belief.mean(), atol = 0.3))
    print("Test passed")

def test_collision_one_wall():
    wall = Wall((0.05, 0.1), (0.3, 0.1))
    far_belief = Belief(mu=(0.2, 0.0), cov = 0.001, walls = [wall])
    assert(len(far_belief.find_collisions().keys()) == 0)
    collide_belief = Belief(mu=(0.2, 0.09), cov = 0.1, walls = [wall])
    #import ipdb; ipdb.set_trace()
    assert(len(collide_belief.find_collisions().keys()) > 0)
    print("test passed")

def test_collision_two_walls():
    wall_1 = Wall((0.05, 0.1), (0.3, 0.1))
    wall_2 = Wall((0.3, 0.1), (0.3, 0.0))
    far_belief = Belief(mu=(0.2, 0.0), cov = 0.001, walls = [wall_1, wall_2])
    assert(len(far_belief.find_collisions().keys()) == 0)
    far_belief.visualize()
    collide_belief = Belief(mu=(0.29, 0.09), cov = 0.1, walls = [wall_1, wall_2])
    assert(len(collide_belief.find_collisions().keys()) > 0)
    print("test passed")

def test_wall_close():
    wall_1 = Wall((0.05, 0.1), (0.3, 0.1))
    far_belief = Belief(mu=(0.2, 0.0), cov = 0.001, walls = [wall_1])
    #import ipdb; ipdb.set_trace()
    assert(len(wall_1.get_particles_in_collision(far_belief)) == 0)
    close_belief = Belief(mu=(0.2, 0.09), cov = 0.1, walls = [wall_1])
    assert(len(wall_1.get_particles_in_collision(close_belief)) > 0)

def test_connect_motion_model():
    wall_1 = Wall((0.05, 0.1), (0.3, 0.1))
    close_belief = Belief(mu=(0.17, 0.05), cov = 0.00005, walls = [wall_1])
    new_q = (0.2, 0.09)
    close_belief.visualize(goal = new_q)
    u = Connect(new_q, close_belief, close_belief, delta = 0.03, sigma = 0.01)
    new_belief = u.motion_model()
    new_belief.visualize(goal=new_q)

def test_guarded_motion_model():
    wall_1 = Wall((0.05, 0.1), (0.3, 0.1))
    close_belief = Belief(mu=(0.17, 0.05), cov = 0.00005, walls = [wall_1])
    new_q = (0.2, 0.09)
    close_belief.visualize(goal = new_q)
    u = Guarded(new_q, close_belief, close_belief,delta = 0.07, sigma = 0.01)
    new_belief = u.motion_model()
    new_belief.visualize(goal=new_q)

def test_slide_motion_model():
    wall_1 = Wall((0.05, 0.1), (0.3, 0.1))
    close_belief = Belief(mu=(0.17, 0.05), cov = 0.00005, walls = [wall_1])
    new_q = (0.2, 0.09)
    u = Guarded(new_q, close_belief, close_belief,delta = 0.07, sigma = 0.01)
    new_belief = u.motion_model()
    side_q = (0.08,0.1)
    u = Slide(side_q, new_belief, close_belief,delta = 0.08)
    new_belief2 = u.motion_model()
    new_belief2.visualize(goal=side_q)



#test_wall_close()
#test_collision_one_wall()
test_slide_motion_model()





