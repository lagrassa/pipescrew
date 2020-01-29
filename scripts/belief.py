import numpy as np
from scipy.stats import multivariate_normal
from sympy import *
from sympy.geometry import *


class Belief():
    def __init__(self, mu=None, cov=None, particles = [], walls = [], action=None, siblings=[], parent=None, connected = False):
        n_particles = 5
        if mu is None:
            assert(particles is not None)
            assert(isinstance(particles[0], Particle))
            particle_poses = np.vstack([part.pose for part in particles])
            mu = np.mean(particle_poses, axis = 0)
            cov = np.cov(particle_poses.T)
            self.particles = particles
        if len(particles) == 0:
            assert mu is not None
            mvn = multivariate_normal(mean=mu, cov=cov)
            self.particles = [Particle(mvn.rvs()) for n in range(n_particles)]
            mu = mvn.mean
            cov = mvn.cov
        self.walls = walls
        if parent is None:
            parent = self
        self.parent = parent
        self._mean = mu
        self.connected = connected
        self._cov = cov
        self.action = action
        self.siblings = siblings
    def mean(self):
        return self._mean
    def siblings(self):
        return self.siblings()
    def cov(self):
        return self._cov
    def get_action(self):
        return self.action
    """
    Check for inconsistencies in the belief state representation,
    - all q must be in free space OR in contact with the same pair of surfaces
    - if there are contacts, the contact must be in a link with a contact sensor (null for pt robot)
    """
    def is_valid(self):
        return True #not useful rn
    def visualize(self, goal=None):
        import matplotlib.pyplot as plt
        for wall in self.walls:
            ee_xs = [ee[0] for ee in wall.endpoints]
            ee_ys = [ee[1] for ee in wall.endpoints]
            plt.plot(ee_xs, ee_ys)
        xs = [part.pose[0] for part in self.particles]
        ys = [part.pose[1] for part in self.particles]
        plt.xlim(0,0.3)
        plt.ylim(0,0.15)
        plt.scatter(xs, ys)
        if goal is not None:
            plt.scatter([goal[0]], [goal[1]], s= 500, color = 'g')
        plt.show()



    def high_prob_collision(self, old_belief, p = 0.96, wall = None):
        wall_i_to_colliding_parts = self.find_collisions(old_belief)
        p_collision = 0
        if len(self.walls) == 0:
            return False
        for part in self.particles:
            if wall is None:
                walls_in_contact = [self.walls[i] for i in wall_i_to_colliding_parts.keys()
                                if part in wall_i_to_colliding_parts[i]]
            else:
                walls_in_contact = [self.walls[i] for i in wall_i_to_colliding_parts.keys()
                                    if part in wall_i_to_colliding_parts[i] and self.walls[i] == wall]

            p_collision_given_n_walls = int(len(walls_in_contact) > 0 )
            if len(walls_in_contact) > 0:
                p_collision += 1./len(self.particles)*p_collision_given_n_walls
        return p_collision > p
    """
    walls that collide with some number of particles. 
    A collision is NOT the same as a contact. Contacts are expected and planned for
    dict mapping wall with particles
    """
    def find_collisions(self, old_belief):
        wall_i_to_colliding_parts = {}
        for wall, i in zip(self.walls, range(len(self.walls))):
            parts_in_collision = wall.get_particles_in_collision(self, old_belief)
            for part in parts_in_collision:
                if wall.endpoints not in part.world_contact_surfaces():
                    if i not in wall_i_to_colliding_parts.keys():
                        wall_i_to_colliding_parts[i] = [part]
                    else:
                        wall_i_to_colliding_parts[i].append(part)
        return wall_i_to_colliding_parts
    def collision_with_particle(self, old_belief, part):
        colliding_walls = []
        for wall in self.walls:
            if wall.is_particle_in_collision(part.pose, old_belief):
                if wall.endpoints not in part.world_contact_surfaces():
                    colliding_walls.append(wall)
        return colliding_walls



class Wall():
    """
    line going from e1 to e2
    """
    def __init__(self, e1, e2):
        self.line = Segment(Point(*e1), Point(*e2))
        self.endpoints = (e1, e2)
    """
    if there is a 96% probability that the belief will 
    coincide with the line 
    returns idxs
    """
    def get_particles_in_collision(self, belief, old_belief):
        #start integrating from the center out
        parts_in_collision = [part for part in belief.particles if self.is_particle_in_collision(part.pose, old_belief)]
        return parts_in_collision

    def is_particle_in_collision(self, pt, old_belief):
        seg = Segment(Point(*old_belief.mean()), pt)
        inside_wall = bool(len(intersection(seg, self.line)))
        return int(inside_wall)

    def dist_to(self, pt):
        return self.line.distance(Point2D(pt))
    def closest_pt(self, pt):
        perp_line = self.line.perpendicular_line(pt)
        intersecting_pts = intersection(perp_line, self.line)
        if len(intersecting_pts) == 0:
            return self.endpoints[np.argmin([Point2D(ep).distance(pt) for ep in self.endpoints])]
        else:
            return (float(intersecting_pts[0].x), float(intersecting_pts[0].y))



class Particle():
    def __init__(self, pose):
        self.pose = pose
        self.contacts = [] #tuples of (robot_surface, world_surface)
    def world_contact_surfaces(self):
        return [x[1] for x in self.contacts]

