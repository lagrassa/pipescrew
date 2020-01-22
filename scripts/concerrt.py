import numpy as np
from belief import Belief, Particle

"""
@param b0 input belief
@param bg goal belief
"""


def concerrt(b0, bg, gamma = 0.1):
    b_open = [b0]
    b_connected = [bg]
    tree = Tree(b0)
    policy = Policy(tree)
    while policy.prob(tree) < 0.4: #sketchy is fine rn
        tree, unconnected_partitions = tree.expand(b_connected, bg, gamma=gamma)
        tree.display()
        policy.update(tree)
        b_open = update(b_open, tree, return_connected=False)
        b_connected = update(b_connected, tree, return_connected=True)
        print(len(b_connected))
        print(len(b_open))
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
    def __init__(self, belief_seed):
        self.children = []
        self.data = belief_seed

    def add_belief(self, belief):
        self.children.append(belief)
    def display(self):
        import pygraphviz as pgv
        G = pgv.AGraph(directed=True)
        parent_name =self.data.mean()
        G.add_node(parent_name, color='blue')
        construct_tree(self, G, parent_name)
        return G.string()


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

    def expand(self, b_connected, b_g, gamma):
        q_rand = random_config(b_g)
        b_near = nearest_neighbor(q_rand, self)[0]
        u = select_action(q_rand, b_near, gamma)
        b_near.action = u
        b_prime = simulate(u)
        unconnected_partitions = []
        if is_valid(b_prime):
            b_contingencies = belief_partitioning(b_prime)
            for belief in b_contingencies:
                self.add_belief(belief)
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
                total_p += (1. / n) * self.prob(subtree)
            else:
                total_p += (1. / n) * int(child.connected)
        return total_p

    def update(self, tree):
        self.tree = tree

def construct_tree(subtree, G, parent_name):
    for child in subtree.children:
        if isinstance(child, Belief):
            G.add_node(child.mean())
            G.add_edge(parent_name, child.mean())
        else:
            construct_tree(child, G, child.mean())

"""
q is of either type Particle or Belief comprised of Particles
"""


def mala_distance(q, belief):
    if isinstance(q, Belief):
        distance = 0
        for particle in q.particles:
            distance += 1. / len(q.particles) * mala_distance(particle, belief)**2
        return np.sqrt(distance)
    else:
        diff = np.matrix(q.pose - belief.mean())
        try:
            return np.sqrt(diff * np.linalg.inv(belief.cov()) * diff.T).item()
        except:
            import ipdb; ipdb.set_trace()


def random_config(b_g):
    p_bg = 0.98  # make higher for more "exploration", basically necessary to get around obstacles
    if np.random.random() < p_bg:
        return b_g.mean()
    return np.random.uniform(-2, 2, (2,))


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
            score = gamma * (d_sigma(b) + sum([d_sigma(b2) for b2 in sib(b)])) + (1 - gamma) * d_mu(b,q_rand)
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


def select_action(q_rand, b_near, gamma):
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
        return np.random.choice([Connect, Slide], p=[1-gamma, gamma])(q_rand, b_near)
    else:
        return np.random.choice([Connect, Guarded], p=[1-gamma, gamma])(q_rand, b_near)


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
        beliefs.append(Belief(particles=list(contact_types.values())[0], action = b_prime.get_action(), siblings = [], walls = b_prime.walls))
    else:
        for contact_set in contact_types.keys():
            siblings = list(set(contact_types.keys())-contact_set) #in all others
            if len(siblings) == 0:
                siblings = [] #fix set weirdness
            beliefs.append(Belief(particles=contact_types[contact_set], action = b_prime.get_action(), siblings = siblings, walls = b_prime.walls))
    return beliefs


'''
Ensures that the agent is not in collision with a wall
'''


def is_valid(belief):
    return belief.is_valid()


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


def propagate_particle(part, q,sigma):
    delta = 0.02
    diff = q-part.pose
    shift = diff / np.linalg.norm(diff) * delta
    new_pose = part.pose + shift + np.random.normal(np.zeros(part.pose.shape),sigma)
    new_part = Particle(new_pose)
    return new_part

def propagate_particle_by_dir(part, dir):
    delta = 0.02
    shift = dir / np.linalg.norm(dir) * delta
    new_pose = part.pose + shift
    new_part = Particle(new_pose)
    return new_part

class Connect:
    def __init__(self, q_rand, b_near):
        self.q_rand = q_rand
        self.b_near = b_near
        self.sigma = 0.0001

    def motion_model(self):
        delta = 0.02
        diff = self.q_rand - self.b_near.mean()
        moved_particles = [propagate_particle(part, self.q_rand,  self.sigma)  for part in self.b_near.particles]
        return Belief(particles = moved_particles, siblings = [], walls = self.b_near.walls)
"""
Find contacts and update them
"""
def update_particle_status(belief):
    collision_walls_to_parts = belief.find_collisions()
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




class Guarded:
    def __init__(self, q_rand, b_near):
        self.q_rand = q_rand
        self.b_near = b_near
        self.sigma = 0.01

    """
    Moves until achieves contact, should be identical to connect unless a collision is detected. 
    then it turns into a contact. 
    """

    def motion_model(self):
        moved_particles = [propagate_particle(part, self.q_rand, self.sigma)  for part in self.b_near.particles]
        potential_b = Belief(particles = moved_particles, siblings = [], walls = self.b_near.walls)
        collision_walls_to_parts = potential_b.find_collisions()
        if len(collision_walls_to_parts.keys()) == 0:
            return potential_b
        else:
            new_particles = []
            for part, potential_part in zip(self.b_near.particles, potential_b.particles):
                walls_in_contact = [potential_b.walls[i] for i in collision_walls_to_parts.keys()
                                    if part in collision_walls_to_parts[i]]
                if len(walls_in_contact) > 0:
                    for wall in walls_in_contact:
                        part.contacts.append((None, wall))
                    new_particles.append(part)
                else:
                    new_particles.append(potential_part)
            return Belief(particles = new_particles, siblings = [], walls = self.b_near.walls)

class Slide:
    def __init__(self, q_rand, b_near):
        self.q_rand = q_rand
        self.b_near = b_near
        self.sigma = 0.01

    '''only valid along one contact surface really
     so pick the one the majority of particles are in'''
    def motion_model(self):
        #update contacts
        update_particle_status(self.b_near)
        best_ee = most_common_surface(self.b_near)
        #direction along best_ee to q
        forward_particles = [propagate_particle_by_dir(part, best_ee) for part in self.b_near.particles]
        backward_particles = [propagate_particle_by_dir(part, -best_ee) for part in self.b_near.particles]
        forward_belief = Belief(particles=forward_particles, siblings= [], walls = self.b_near.walls)
        backward_belief = Belief(particles=backward_particles, siblings= [], walls = self.b_near.walls)
        return max([forward_belief, backward_belief], key=lambda x: mala_distance(self.q_rand, x))

class NullAction:
    def __init__(self,q_rand, b_near):
        self.b_near = b_near
        self.sigma = 0.01

    def motion_model(self):
        return self.b_near

