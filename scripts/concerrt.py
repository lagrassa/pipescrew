import numpy as np
from belief import Belief

"""
@param b0 input belief
@param bg goal belief
"""


def concerrt(b0, bg):
    b_open = [b0]
    b_connected = [bg]
    tree = Tree(b0)
    policy = Policy(tree)
    while policy.prob(tree) < 1:
        print(policy.prob(tree), "P tree")
        tree, unconnected_partitions = tree.expand(b_connected, bg)
        policy.update(tree)
        b_open = update(b_open, tree, return_connected=False)
        b_connected = update(b_connected, tree, return_connected=True)
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

    def goal_connect(self, b_connected):
        b_dprimes_that_worked = []
        b_dfailures = []
        for b_goal in b_connected:
            b_dprimes = self.try_to_connect_b_to_tree(b_goal)
            for b_dprime in b_dprimes:
                if in_goal_belief(b_dprime, b_goal):
                    b_dprimes_that_worked.append(b_dprime)
                    b_dprime.connected = True
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
                if child.connected == 0:
                    self.data.connected = True
        return self

    def expand(self, b_connected, b_g):
        q_rand = random_config(b_g)
        b_near = nearest_neighbor(q_rand, self)[0]
        u = select_action(q_rand, b_near)
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

    def __call__(self, belief):
        _,most_likely_current_tree = self.highest_prob_belief(belief)
        return most_likely_current_tree.get_action()

    """
    tree that represents the most likely belief state we are in
    """

    def highest_prob_belief(self, belief):
        best_distance = mala_distance(self.tree.data, belief)
        best_child = self.tree.data
        import ipdb; ipdb.set_trace()
        for child in self.tree.children:
            if isinstance(child, Tree):
                return self.highest_prob_belief(child)
            else:
                distance = mala_distance(child, belief)
                if distance < best_distance:
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


"""
q is of either type Particle or Belief comprised of Particles
"""


def mala_distance(q, belief):
    if isinstance(q, Belief):
        distance = 0
        for particle in q.particles:
            distance += 1. / len(q.particles) * mala_distance(particle, belief)
        return distance
    else:
        diff = np.matrix(q.pose - belief.mean())
        return np.sqrt(diff * np.linalg.inv(belief.cov()) * diff.T).item()


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
            cand_best_b, score = gamma * (d_sigma(b) + sum([d_sigma(b2) for b2 in sib(b)])) + (1 - gamma) * d_mu(b,
                                                                                                                 q_rand)
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


def select_action(q_rand, b_near):
    return np.random.choice([Connect, Guarded])(q_rand, b_near)


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
    for contact_set in contact_types.keys():
        siblings = set(contact_types.keys())-contact_set #in all others
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
        if distance > 2:
            all_close = False
    return all_close


class Connect:
    def __init__(self, q_rand, b_near):
        self.q_rand = q_rand
        self.b_near = b_near

    def motion_model(self):
        delta = 0.02
        diff = self.q_rand - self.b_near.mean()
        mu_shift = diff / np.linalg.norm(diff) * delta
        new_mu = mu_shift + self.b_near.mean()
        new_cov = 1.05 * self.b_near.cov()
        return Belief(new_mu, new_cov, siblings=[], walls = self.b_near.walls)


class Guarded:
    def __init__(self, q_rand, b_near):
        self.q_rand = q_rand
        self.b_near = b_near

    """
    Moves until achieves contact
    """

    def motion_model(self):
        delta = 0.02
        diff = self.q_rand - self.b_near
        mu_shift = diff / np.linalg.norm(diff) * delta
        new_mu = mu_shift + self.b_near.mean()
        new_cov = 1.05 * self.b_near.cov()
        potential_b = Belief(new_mu, new_cov)
        collisions = potential_b.find_collisions()
        # find collision point and squash there (projection)


class Slide:
    def __init__(self, q_rand, b_near):
        self.q_rand = q_rand
        self.b_near = b_near

    def motion_model(self):
        pass
