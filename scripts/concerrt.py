import numpy as np
from belief import Belief
"""
@param b0 input belief
@param bg goal belief
"""
def concerrt(b0, bg):
    b_open = [b0]
    policy = Policy()
    b_connected = [bg]
    tree = Tree(b0)
    while policy.P() < 1:
        for b in b_open:
            t_b = tree.expand(b_connected, bg)
            policy.update(tree)
            b_open = update(b_open, tree)
            b_connected = update(b_connected, tree)
    return policy


def update(belief_set, tree):


class Tree:
    def __init__(self, belief_seed):
        self.children = []
        self.data = belief_seed

    
    def add_belief(self, belief):
        self.children.append(belief)
    
    def goal_connect(self, belief, B_connected):
        pass

    def expand(self, b_connected, b_g):
        q_rand = random_config(b_g)
        b_near = nearest_neighbor(q_rand, self)
        u = select_action(q_rand, b_near)
        b_prime = simulate(q_rand, b_near, u)
        if is_valid(b_prime):
            b_contingencies = belief_partitioning(b_prime)
            for belief in b_contingencies:
                self.add_belief(belief)
                self.goal_connect(belief, b_connected)


class Policy:
    def __init__(self):
        pass

def random_config(b_g):
    p_bg = 0.9
    if np.random.random() < p_bg:
        return b_g.mean()
    return np.random.uniform(-2,2,(2,))

"""
b nearest to T in the direction of q_rand
"""
def nearest_neighbor(q_rand, T, gamma=0.5):
    best_idx =  np.argmin([gamma(d_sigma(b)+sum([d_sigma(b2) for b2 in sib(b)]))+(1-gamma)*d_mu(b, q_rand) for b in T.children])
    return T.children[best_idx]
"""
Returns all siblings from the belief partition of b
partitions that were reached from the same action
"""
def sib(b):
    return NotImplementedError

def d_sigma(belief):
    return np.trace(belief.cov())

def d_mu(belief, q_rand):
    return np.linalg.norm(belief.mean()-q_rand)

"""
connect, guarded, or slide. 
"""
def select_action(q_rand, b_near):
    return np.random.choice([Connect, Guarded, Slide])(q_rand, b_near)


def simulate( b_near, u):
    return u.motion_model(b_near)
def belief_partitioning(b_prime):
    return NotImplementedError
def is_valid(belief):
    return NotImplementedError



class Connect():
    def __init__(self, q_rand, b_near):
        self.q_rand = q_rand
        self.b_near = b_near
    def motion_model(self):
        delta = 0.1
        distance = np.linalg.norm(self.q_rand-self.b_near)
        diff = self.q_rand - self.b_near
        mu_shift = diff/np.linalg.norm(diff)*0.1
        new_mu = mu_shift+self.b_near.mean()
        new_cov = 1.05*self.b_near.cov()
        return Belief(new_mu, new_cov)



class Guarded():
    def __init__(self, q_rand, b_near):
        self.q_rand = q_rand
        self.b_near = b_near
    def motion_model(self):
        pass

class Slide():
    def __init__(self, q_rand, b_near):
        self.q_rand = q_rand
        self.b_near = b_near
    def motion_model(self):
        pass





