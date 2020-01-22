from belief import Belief
from belief import Wall
import numpy as np

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


#test_wall_close()
#test_collision_one_wall()





