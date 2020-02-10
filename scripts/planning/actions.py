import numpy as np
from agent.belief import mala_distance
from agent.belief import Particle, Belief

def closest_wall_point(belief, pt):
    dist_to_walls = [wall.dist_to(pt) for wall in belief.walls]
    return belief.walls[np.argmin(dist_to_walls)].closest_pt(pt)
"""
endpoints of most commonly interacted with surface
"""
def most_common_surface(belief):
    #select surface to move on
    wall_endpoints_to_parts = get_walls_to_endpoints_to_parts(belief)
    best_ee = max(wall_endpoints_to_parts.keys(), key=lambda x: len(wall_endpoints_to_parts[x]))
    return best_ee
"""
Moves a particle with an amount of Gaussian noise 
"""


"""
Find contacts and update them
"""
def update_particle_status(belief, old_belief):
    collision_walls_to_parts = belief.find_collisions(old_belief)
    for part in belief.particles:
        walls_in_contact = [belief.walls[i] for i in collision_walls_to_parts.keys()
                            if part in collision_walls_to_parts[i]]
        if len(walls_in_contact) > 0:
            for wall in walls_in_contact:
                part.contacts.append((None, wall))


def get_walls_to_endpoints_to_parts(belief):
    wall_endpoints_to_parts = {}
    for part in belief.particles:
        walls = part.world_contact_surfaces()
        for wall in walls:
            if wall.endpoints not in wall_endpoints_to_parts.keys():
                wall_endpoints_to_parts[wall] = [part]
            else:
                wall_endpoints_to_parts[wall].append(part)
    return wall_endpoints_to_parts


"""
Moves particle and updates if it's in contact with a wall. 
"""
def propagate_particle_to_q(part, shift=None, sigma=None, old_belief =None, belief=None):
    if np.linalg.norm(shift) == 0:
        return part #we're already there
    new_part = propagate_particle(part, shift, sigma=sigma,old_belief=old_belief, belief=belief)
    return new_part
"""
Propagates particle but only as far as the belief will go 
"""
def propagate_particle(part, shift, sigma=0, old_belief = None, belief=None):
    new_pose = part.pose + shift + np.random.normal(np.zeros(part.pose.shape),sigma)
    new_part = Particle(new_pose)
    walls_in_contact = belief.collision_with_particle(old_belief, new_part)
    if len(walls_in_contact) > 0:
        for wall in walls_in_contact:
            new_part.contacts.append((None, wall))
        stopped_pose = np.array(wall.closest_pt(part.pose))
        new_part.pose = stopped_pose
    return new_part

def propagate_particle_by_dir(part, dir, belief = None, old_belief = None, delta = 0.02):
    shift = (dir / np.linalg.norm(dir)) * delta
    new_part = propagate_particle(part, shift, belief=belief,old_belief = old_belief, sigma=0)
    return new_part

class Connect:
    def __init__(self, q_rand, b_near, b_old,  sigma = 0.0001, delta = 0.05):
        self.q_rand = q_rand
        self.b_near = b_near
        self.sigma = sigma
        self.delta = delta
        self.old_belief = b_old
    '''
    Moves by delta toward self.q_rand
    '''
    def get_control(self, state,dt, ext_force_mag):
        q = self.q_rand
        control, expected = get_control(q, state,dt,ext_force_mag, self.delta, self.sigma, self.old_belief)
        return control, expected


    def motion_model(self):
        current_b = self.b_near
        while(mala_distance(self.q_rand, current_b)) > 1:
            if current_b.high_prob_collision(self.old_belief):
                break
            diff = self.q_rand - current_b.mean()
            shift = diff / np.linalg.norm(diff) * self.delta
            moved_particles = [propagate_particle_to_q(part, shift =shift, sigma=self.sigma,  belief = current_b, old_belief = self.old_belief) for part in self.b_near.particles]
            self.old_belief = self.b_near
            current_b = Belief(particles = moved_particles, siblings = [], walls = self.b_near.walls, parent=self.b_near)
            #print(self.b_near.mean(), self.q_rand)
        return current_b

def get_control(q, state,dt,ext_force_mag, delta, sigma, belief):
    diff = q - state[0:2]
    if np.linalg.norm(diff) == 0:
        dir = np.array([0,0])
    else:
        dir = diff/np.linalg.norm(diff)
    ext_force = -dir*ext_force_mag #really only works for opposing force
    control = (dir*delta-state[2:]*dt)/(dt**2) + ext_force
    k = 1
    d = 0.1
    #control = k*(self.q_rand-state[:2])+d*(0-state[2:])-ext_force
    expected_next_mu = state[0:2]+(control-ext_force)*dt**2+state[2:]*dt
    colliding_walls = belief.collision_with_particle(belief.parent, expected_next_mu)
    if len(colliding_walls) == 0:
        expected_next = multivariate_normal(mean = expected_next_mu, cov = sigma**2)
    else:
        collide_pt = colliding_walls[0].closest_pt(state[0:2], dir=dir)
        expected_next = multivariate_normal(mean = collide_pt, cov = sigma**2)
    return control, expected_next

class Guarded:
    def __init__(self, q_rand, b_near, b_old, sigma = 0.0001, delta = 0.05):
        self.q_rand = q_rand
        self.b_near = b_near
        self.sigma = sigma
        self.old_belief = b_old
        self.delta = delta
    """
    Moves to wall closest to q_rand until achieves contact, should be identical to connect unless a collision is detected. 
    then it turns into a contact. 
    """

    def motion_model(self):
        closest_wall_pt = closest_wall_point(self.b_near, self.q_rand)
        current_b = self.b_near
        while(mala_distance(self.q_rand, current_b)) > 1 and not current_b.high_prob_collision(self.old_belief, p = 0.9):
            diff = closest_wall_pt - self.b_near.mean()
            shift = diff / np.linalg.norm(diff) * self.delta
            moved_particles = [propagate_particle_to_q(part, shift=shift, sigma=self.sigma, belief = current_b, old_belief = self.old_belief) for part in current_b.particles]
            current_b=  Belief(particles = moved_particles, siblings = [], walls = self.b_near.walls, parent=self.b_near)
        return current_b

    def get_control(self, state,dt, ext_force_mag):
        q = self.q_rand
        control, expected = get_control(q, state,dt,ext_force_mag, self.delta, self.sigma, self.old_belief)
        return control, expected


class Slide:
    def __init__(self, q_rand, b_near, b_old, sigma=0.01, delta = 0.02):
        self.q_rand = q_rand
        self.b_near = b_near
        self.sigma = sigma
        self.delta = delta #this should depend on the friction coefficient and the controller
        self.old_belief= b_old

    '''only valid along one contact surface really
     so pick the one the majority of particles are in'''
    def motion_model(self):
        best_ees = np.array(most_common_surface(self.b_near).endpoints)
        forward_dir = best_ees[0]-best_ees[1]
        wall_i_to_particles = self.b_near.find_collisions(self.old_belief) #wall majority of particles are in contact with
        old_wall = self.b_near.walls[max(wall_i_to_particles.keys(), key = lambda x: len(wall_i_to_particles[x]))]
        current_b = self.b_near
        #direction along best_ee to q
        while(mala_distance(self.q_rand, self.b_near)) > 2 or current_b.high_prob_collision(self.old_belief, wall = old_wall):
            forward_particles = [propagate_particle_by_dir(part, forward_dir, belief = current_b, old_belief = self.old_belief, delta = self.delta) for part in current_b.particles]
            backward_particles = [propagate_particle_by_dir(part, -forward_dir, belief =current_b, old_belief = self.old_belief, delta = self.delta) for part in current_b.particles]
            forward_belief = Belief(particles=forward_particles, siblings= [], walls = self.b_near.walls, parent = self.b_near)
            backward_belief = Belief(particles=backward_particles, siblings= [], walls = self.b_near.walls, parent=self.b_near)
            current_b =  min([forward_belief, backward_belief], key=lambda x: mala_distance(self.q_rand, x))
        return current_b

class NullAction:
    def __init__(self,q_rand, b_near):
        self.b_near = b_near
        self.sigma = 0.01
        self.delta = 0.05

    def get_control(self, state,dt, ext_force_mag):
        q = state[0:2]
        control, expected = get_control(q, state,dt,ext_force_mag, self.delta, self.sigma, self.b_near)
        return control, expected
    def motion_model(self):
        return self.b_near
