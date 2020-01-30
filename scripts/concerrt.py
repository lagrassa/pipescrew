import numpy as np
from scipy.stats import multivariate_normal
from belief import Belief, Particle
np.random.seed(17)

"""
@param b0 input belief
@param bg goal belief
"""


def concerrt(b0, bg, gamma = 0.8, p_bg = 0.5, delta=0.03):
    b_open = [b0]
    b_connected = [bg]
    tree = Tree(b0)
    policy = Policy(tree)
    while policy.prob(tree) < 0.1: #sketchy is fine rn
        tree, unconnected_partitions = tree.expand(b_connected, bg, gamma=gamma, p_bg=p_bg, delta = delta)
        tree.display()
        policy.update(tree)
        b_open = update(b_open, tree, return_connected=False)
        b_connected = update(b_connected, tree, return_connected=True)
        #print(len(b_connected))
        #print(len(b_open))
    return policy


'''
add any beliefs in the tree that are not already in belief set to the belief set
'''


def update(belief_set, tree, return_connected=None):
    belief_nodes = tree.traverse(return_connected=return_connected)
    for belief in belief_nodes:
        if belief not in belief_set:
            belief_set.append(belief)
    return belief_set


class Tree:
    def __init__(self, belief_seed, children = []):
        self.children = children
        self.data = belief_seed

    def add_belief(self,parent_belief, child_belief):
        if parent_belief == self.data:
            self.children.append(child_belief)
        else:
            for child in self.children:
                if isinstance(child, Tree):
                    child.add_belief(parent_belief, child_belief)
                else:
                    if parent_belief == child:
                        new_child = Tree(child, [child_belief])
                        child_idx = self.children.index(child)
                        self.children[child_idx] = new_child

    def display(self):
        import pygraphviz as pgv
        G = pgv.AGraph(directed=True)
        parent_name =make_node_name(self.data)
        G.add_node(parent_name, color='blue')
        construct_tree(self, G, parent_name)
        G.write("/home/lagrassa/git/pipescrew/scripts/search.dot")


    def goal_connect(self, b_connected):
        b_dprimes_that_worked = []
        b_dfailures = []
        for b_goal in b_connected:
            b_dprimes = self.try_to_connect_b_to_tree(b_goal)
            for b_dprime in b_dprimes:
                if in_goal_belief(b_dprime, b_goal):
                    b_dprimes_that_worked.append(b_dprime)
                    b_dprime.connected = True
                    #print("somethings in the goal belief")
        return b_dprimes_that_worked, b_dfailures

    """
    traverses through tree and returns all beliefs moved towards bg
    """

    def try_to_connect_b_to_tree(self, bg):
        """

        :rtype: list of Beliefs
        """
        candidate_beliefs = []
        for child in self.children:
            if isinstance(child, Tree):
                candidate_beliefs += child.try_to_connect_b_to_tree(bg)
            else:
                if in_goal_belief(child, bg):
                    candidate_beliefs.append(child)
        return candidate_beliefs

    def traverse(self, return_connected=None):
        nodes = []
        if isinstance(self.data, Belief):
            nodes.append(self.data)
        for child in self.children:
            if isinstance(child, Tree):
                nodes += child.traverse()
            else:
                if child.connected == return_connected:
                    nodes.append(child)
        return nodes

    def update_tree(self):
        for child in self.children:
            if isinstance(child, Tree):
                child.update_tree()
            else:
                if child.connected:
                    self.data.connected = True
        return self

    def expand(self, b_connected, b_g, gamma, p_bg=0.5,delta = 0.04):
        q_rand = random_config(b_g, p_bg=p_bg)
        b_near = nearest_neighbor(q_rand, self)[0]
        u = select_action(q_rand, b_near, b_near.parent,gamma, delta = delta)
        b_near.action = u
        b_prime = simulate(u)
        unconnected_partitions = []
        if is_valid(b_prime):
            b_contingencies = belief_partitioning(b_prime)
            for belief in b_contingencies:
                self.add_belief(b_near, belief)
                unconnected_partitions += self.goal_connect(b_connected)[1]
        new_tree = self.update_tree()
        return new_tree, unconnected_partitions


class Policy:
    def __init__(self, tree):
        self.tree = tree

    """
    Given a belief state returns an action
    estimates which state we are in and then returns the appropriate action
    """

    def __call__(self, belief, bg):
        if in_goal_belief(belief, bg):
            return NullAction(None, belief)
        _,most_likely_current_tree = self.highest_prob_belief(belief)
        print(most_likely_current_tree, "most likely node in tree")
        return most_likely_current_tree.get_action()

    """
    tree that represents the most likely belief state we are in
    that still has an action
    """

    def highest_prob_belief(self, belief):
        best_distance = mala_distance(self.tree.data, belief)
        best_child = self.tree.data
        for child in self.tree.children:
            if isinstance(child, Tree):
                return self.highest_prob_belief(child)
            else:
                distance = mala_distance(child, belief)
                if distance < best_distance and child.get_action() is not None:
                    best_distance = distance
                    best_child = child
        return best_distance, best_child

    """
    Probability that the policy works, it's the probability we're connected, all just total probability
    """

    def prob(self, subtree):
        total_p = 0
        for child in subtree.children:
            n = len(subtree.children)
            if isinstance(child, Tree):
                total_p += (1. / n) * self.prob(child)
            else:
                total_p += (1. / n) * int(child.connected)
        return total_p

    def update(self, tree):
        self.tree = tree

def construct_tree(subtree, G, parent_name):

    for child in subtree.children:
        if isinstance(child, Belief):
            color = "green" if child.connected else "red"
            G.add_node(make_node_name(child), color = color)
            G.add_edge(parent_name, make_node_name(child))
        else:
            color = "green" if child.data.connected else "red"
            G.add_node(make_node_name(child.data), color = color)
            G.add_edge(parent_name, make_node_name(child.data))
            construct_tree(child, G, make_node_name(child.data))

def make_node_name(belief):
    stdev = str(np.round(np.std(np.vstack(part.pose for part in belief.particles),axis=0), 3))
    return str(np.round(belief.mean(),3))+" "+stdev
"""
q is of either type Particle or Belief comprised of Particles
"""


def mala_distance(q, belief):
    if isinstance(q, Belief):
        distance = 0
        for particle in q.particles:
            distance += 1. / len(q.particles) * mala_distance(particle, belief)**2
        return np.sqrt(distance)
    elif isinstance(q, tuple):
        q = np.array(q)
    elif isinstance(q, Particle):
        q = q.pose
    diff = np.matrix(np.array(q)- belief.mean())
    if np.linalg.det(belief.cov()) < 1e-12:
        return np.linalg.norm(diff)
    return np.sqrt(diff * np.linalg.inv(belief.cov()) * diff.T).item()


def random_config(b_g, p_bg = 0.5):
    if np.random.random() < p_bg:
        return b_g.mean()
    #also consider adding the nearest wall....
    return np.random.uniform(0, 0.3, (2,))


"""
b nearest to T in the direction of q_rand
"""
def nearest_neighbor(q_rand, tree, gamma=0.5):
    if not isinstance(tree.data, Tree):
        min_score = gamma * (d_sigma(tree.data) + sum([d_sigma(b2) for b2 in sib(tree.data)])) + (1 - gamma) * d_mu(
            tree.data, q_rand)
        best_b = tree.data
    else:
        min_score = np.inf
        best_b = None
    for child in tree.children:
        if isinstance(child, Tree):
            cand_best_b, score = nearest_neighbor(q_rand, child)
        else:
            b = child
            try:
                score = gamma * (d_sigma(b) + sum([d_sigma(b2) for b2 in sib(b)])) + (1 - gamma) * d_mu(b,q_rand)
            except:
                import ipdb; ipdb.set_trace()
            cand_best_b = child
        if score < min_score:
            min_score = score
            best_b = cand_best_b
    return best_b, min_score


"""
Returns all siblings from the belief partition of b
partitions that were reached from the same action
"""


def sib(b):
    return b.siblings


def d_sigma(belief):
    return np.trace(belief.cov())


def d_mu(belief, q_rand):
    return np.linalg.norm(belief.mean() - q_rand)


"""
connect, guarded, or slide. 
"""


def select_action(q_rand, b_near,b_old, gamma, delta = 0.04):
    walls_endpoints_to_parts = get_walls_to_endpoints_to_parts(b_near)
    p_in_contact = 0
    if len(b_near.walls) > 0:
        for part in b_near.particles:
            p_on_wall = len([wall for wall in b_near.walls if wall in walls_endpoints_to_parts.keys() and part in walls_endpoints_to_parts[wall]])/len(b_near.walls)
            p_in_contact += (1./len(b_near.particles))*p_on_wall
    if p_in_contact > 0:
        print(p_in_contact, "p in contact")
    in_contact = p_in_contact >= 0.96
    if in_contact:
        return np.random.choice([Connect, Slide], p=[1-gamma, gamma])(q_rand, b_near, b_old, delta = delta)
    else:
        return np.random.choice([Connect, Guarded], p=[1-gamma, gamma])(q_rand, b_near, b_old, delta=delta)


"""
Definitely test in_collision
"""

def simulate(u):
    return u.motion_model()


def belief_partitioning(b_prime):
    contact_types = {}
    for particle in b_prime.particles:
        if frozenset(particle.contacts) not in contact_types:
            contact_types[frozenset(particle.contacts)] = [particle]
        else:
            contact_types[frozenset(particle.contacts)].append(particle)
    beliefs = []
    if len(list(contact_types.keys())) == 1 and len(list(contact_types.keys())[0]) == 0:
        beliefs.append(Belief(particles=list(contact_types.values())[0], action = b_prime.get_action(), siblings = [], walls = b_prime.walls, parent=b_prime.parent))
    else:
        for contact_set in contact_types.keys():
            beliefs.append(Belief(particles=contact_types[contact_set], action = b_prime.get_action(), siblings = [], walls = b_prime.walls, parent=b_prime.parent))
        for belief in beliefs:
            siblings = []
            for other_belief in beliefs:
                if belief != other_belief:
                    siblings.append(other_belief)
            belief.siblings = siblings
    return beliefs


'''
Ensures that the agent is not in collision with a wall
'''


def is_valid(belief):
    return True
    #return belief.is_valid()


"""
True if d_M(q) < \epsilon_m = 2 for all q \in b_dprime
"""


def in_goal_belief(b_dprime, b_goal):
    all_close = True
    for q in b_dprime.particles:
        distance = mala_distance(q, b_goal)
        if distance > 1:
            all_close = False
    return all_close
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
"""
endpoints of most commonly interacted with surface
"""
def most_common_surface(belief):
    #select surface to move on
    wall_endpoints_to_parts = get_walls_to_endpoints_to_parts(belief)
    best_ee = max(wall_endpoints_to_parts.keys(), key=lambda x: len(wall_endpoints_to_parts[x]))
    return best_ee

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


def closest_wall_point(belief, pt):
    dist_to_walls = [wall.dist_to(pt) for wall in belief.walls]
    return belief.walls[np.argmin(dist_to_walls)].closest_pt(pt)
"""
Moves particle and updates if it's in contact with a wall. 
"""
def propagate_particle_to_q(part, sigma=None, shift=None, old_belief =None, belief=None):
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
        while(mala_distance(self.q_rand, self.b_near)) > 1:
            if self.b_near.high_prob_collision(self.old_belief):
                break
            diff = self.q_rand - self.b_near.mean()
            shift = diff / np.linalg.norm(diff) * self.delta
            moved_particles = [propagate_particle_to_q(part, shift =shift, sigma=self.sigma,  belief = self.b_near, old_belief = self.old_belief) for part in self.b_near.particles]
            self.old_belief = self.b_near
            self.b_near = Belief(particles = moved_particles, siblings = [], walls = self.b_near.walls, parent=self.b_near)
            #print(self.b_near.mean(), self.q_rand)
        return self.b_near

def get_control(q, state,dt,ext_force_mag, delta, sigma, belief):
    diff = q - state[0:2]
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
        while(mala_distance(self.q_rand, self.b_near)) > 1 and not self.b_near.high_prob_collision(self.old_belief, p = 0.9):
            closest_wall_pt = closest_wall_point(self.b_near, self.q_rand)
            diff = closest_wall_pt - self.b_near.mean()
            shift = diff / np.linalg.norm(diff) * self.delta
            moved_particles = [propagate_particle_to_q(part, sigma=self.sigma, belief = self.b_near, old_belief = self.old_belief) for part in self.b_near.particles]
            self.b_near =  Belief(particles = moved_particles, siblings = [], walls = self.b_near.walls, parent=self.b_near)
        return self.b_near

    def get_control(self, state,dt, ext_force_mag):
        q = self.q_rand
        control, expected = get_control(q, state,dt,ext_force_mag, self.delta, self.sigma, self.old_belief)
        return control, expected


class Slide:
    def __init__(self, q_rand, b_near, b_old, sigma=0.01, delta = 0.02):
        self.q_rand = q_rand
        self.b_near = b_near
        self.sigma = sigma
        self.delta = delta
        self.old_belief= b_old

    '''only valid along one contact surface really
     so pick the one the majority of particles are in'''
    def motion_model(self):
        best_ees = np.array(most_common_surface(self.b_near).endpoints)
        forward_dir = best_ees[0]-best_ees[1]
        wall_i_to_particles = self.b_near.find_collisions(self.old_belief) #wall majority of particles are in contact with
        old_wall = self.b_near.walls[max(wall_i_to_particles.keys(), key = lambda x: len(wall_i_to_particles[x]))]
        #direction along best_ee to q
        while(mala_distance(self.q_rand, self.b_near)) > 2 or self.b_near.high_prob_collision(self.old_belief, wall = old_wall):
            forward_particles = [propagate_particle_by_dir(part, forward_dir, belief = self.b_near, old_belief = self.old_belief, delta = self.delta) for part in self.b_near.particles]
            backward_particles = [propagate_particle_by_dir(part, -forward_dir, belief = self.b_near, old_belief = self.old_belief, delta = self.delta) for part in self.b_near.particles]
            forward_belief = Belief(particles=forward_particles, siblings= [], walls = self.b_near.walls, parent = self.b_near)
            backward_belief = Belief(particles=backward_particles, siblings= [], walls = self.b_near.walls, parent=self.b_near)
            self.b_near =  min([forward_belief, backward_belief], key=lambda x: mala_distance(self.q_rand, x))
        return self.b_near

class NullAction:
    def __init__(self,q_rand, b_near):
        self.b_near = b_near
        self.sigma = 0.01

    def motion_model(self):
        return self.b_near

