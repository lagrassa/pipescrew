import numpy as np
from scipy.stats import multivariate_normal
from agent.belief import Belief, Particle
from planning.actions import *
from agent.belief import mala_distance
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
                    b_dprime.visualize(goal=b_goal)
                    b_dprime.connected = True
                    b_dprime.action = NullAction(b_goal.mean(), b_dprime)
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
    """
    Also does validation. parents with children need actions to get to them
    """
    def update_tree(self):
        assert(self.data.action is not None)
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
                try:
                    self.display()
                except:
                    import ipdb; ipdb.set_trace()
                unconnected_partitions += self.goal_connect(b_connected)[1]
        new_tree = self.update_tree()
        return new_tree, unconnected_partitions


class Policy:
    def __init__(self, tree):
        self.tree = tree
    """
    List of actions needed to get to the goal
    Assumes starts at the root
    """
    def get_best_actions(self, init_belief, bg):
        curr_node = self.tree #TODO assert these are close to each other
        actions = [curr_node.data.get_action()]
        should_contacts = [curr_node.data.high_prob_collision(curr_node.data)]
        def child_is_connected(node):
            return ((isinstance(node, Belief) and node.connected)
                                            or (isinstance(node, Tree) and node.data.connected))
        def n_particles(node):
            if isinstance(node, Belief):
                return len(node.particles)
            else:
                return len(node.data.particles)
        while True:
            most_likely_child = np.argmax([n_particles(node) for node in curr_node.children if child_is_connected(node) ])
            child = curr_node.children[most_likely_child]
            if isinstance(child, Belief) and in_goal_belief(child, bg): #we've reached a leaf node
                return actions
            actions.append(child.data.action)
            if child.data.high_prob_collision(curr_node.data):
                should_contacts.append(True)
            else:
                should_contacts.append(False)
            curr_node = child
    """
    Given a belief state returns an action
    estimates which state we are in and then returns the appropriate action
    """

    def __call__(self, belief, bg, check_contact= False):
        if in_goal_belief(belief, bg):
            return NullAction(None, belief)
        _,most_likely_current_tree = self.highest_prob_belief(belief)
        most_likely_action =  most_likely_current_tree.get_action()
        if most_likely_action is None:
            print("Action is Nonetype")
        if check_contact:
            contact = most_likely_current_tree.data.high_probability_collision(most_likely_current_tree.data.parent)
            return most_likely_action, contact
        else:
            return most_likely_action

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
            shape = "box" if child.high_prob_collision(child.parent) else "circle"
            G.add_node(make_node_name(child), color = color, shape=shape)
            #assert(make_node_name(child) != parent_name)
            G.add_edge(parent_name, make_node_name(child))
        else:
            color = "green" if child.data.connected else "red"
            G.add_node(make_node_name(child.data), color = color)
            G.add_edge(parent_name, make_node_name(child.data))
            #assert (make_node_name(child.data) != parent_name)
            construct_tree(child, G, make_node_name(child.data))

def make_node_name(belief):
    stdev = str(np.round(np.std(np.vstack(part.pose for part in belief.particles),axis=0), 3))
    return str(np.round(belief.mean(),3))+" "+stdev
"""
q is of either type Particle or Belief comprised of Particles
"""





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
    in_contact = b_near.high_prob_collision(b_old)
    print(in_contact, "contact")
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
        if distance > 0.2:
            all_close = False

    return all_close
