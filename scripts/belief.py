import numpy as np
from scipy.stats import multivariate_normal
from sympy import *
from sympy.geometry import *


class Belief():
    def __init__(self, mu=None, cov=None, particles = [], walls = []):
        n_particles = 5
        if mu is None:
            assert(particles is not None)
            mu = np.mean(particles, axis = 0)
            cov = np.cov(particles)
        if len(particles) == 0:
            assert mu is not None
            mvn = multivariate_normal(mean=mu, cov=cov)
            self.particles = [Particle(mvn.rvs()) for n in range(n_particles)]
        self.mean = mu
        self.cov = cov
    def mean(self):
        return self.mean
    def cov(self):
        return self.cov
    """
    Check for inconsistencies in the belief state representation,
    - all q must be in free space OR in contact with the same pair of surfaces
    - if there are contacts, the contact must be in a link with a contact sensor (null for pt robot)
    """
    def is_valid(self):
        return self.any_collisions()
    def any_collisions(self):
        return bool(len(self.find_collisions()))
    def find_collisions(self):
        idxs = []
        for wall, i in zip(self.walls, range(len(self.walls))):
            parts_in_collision = wall.in_collision()
            for part in parts_in_collision:
                if wall.endpoints not in part.world_contact_surfaces():
                    idxs.append(i)
        return idxs


class Wall():
    """
    line going from e1 to e2
    """
    def __init__(self, e1, e2):
        self.line = Line(Point(*e1), Point(*e2))
        self.endpoints = (e1, e2)
    """
    if there is a 96% probability that the belief will 
    coincide with the line 
    """
    def in_collision(self, belief):
        #start integrating from the center out
        def func(pt):
            #inside_wall = #on opposite side of wall, best we have now is mean is on the
            #side inside. Does pt cross line?
            seg = Segment(Point(*belief.mean()), pt)
            inside_wall = bool(len(intersection(seg, self.line)))
            return int(inside_wall)
        parts_in_collision = [func(part) for part in belief.particles]
        return parts_in_collision

class Particle():
    def __init__(self, pose):
        self.pose = pose
        self.contacts = [] #tuples of (robot_surface, world_surface)
    def world_contact_surfaces(self):
        return [x[1] for x in self.contacts]

