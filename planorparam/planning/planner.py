from operators import *
from pillar_state_py import State
import numpy as np
robot_pos_fqn = "frame:pose/position"
robot_orn_fqn = "frame:pose/quaternion"
block_pos_fqn = "frame:block:pose/position"
block_orn_fqn = "frame:block:pose/quaternion"
block_on_color_fqn = "frame:block:on_color"


class Node:
    def __init__(self, state_str, op=None, parent=None):
        self.state_str =  state_str
        self.state = State.create_from_serialized_string(state_str)
        self.values = np.array(self.state.get_values_as_vec([block_pos_fqn, block_on_color_fqn]))
        if parent is None:
            self.cost = 0
        else:
            self.op = op  # op taken to get to this state
            self.cost = parent.cost + op.cost()
        self.f = self.cost
    def compute_heuristic(self, goal):
        goal_values = goal.get_values_as_vec([block_pos_fqn])
        my_values = self.state.get_values_as_vec([block_pos_fqn])
        return np.linalg.norm(goal_values-my_values)

    def __eq__(self, other):
        my_values = self.state.get_values_as_vec([robot_pos_fqn, robot_orn_fqn, block_pos_fqn])
        other_values = self.other.get_values_as_vec([robot_pos_fqn, robot_orn_fqn, block_pos_fqn])
        return np.linalg.norm(my_values-other_values) < 0.02


class Planner:
    def __init__(self):
        pass
    def plan(self, start, goal):
        """
        Given a pillar start state, output a series of operators to reach the goal state with high probability
        A* search
        """
        start_node = Node(start)
        open = [start_node]
        closed = []
        #make discrete action space
        discrete_actions = []
        for dir in [0,1,2,3]:
            discrete_actions.append(GoToSide(dir))
            for amounts in [0.05, 0.1, 0.15]:
                T = 10*amounts
                discrete_actions.append(PushInDir(amount, T))
        while (len(open) > 0):
            best_i = max(open, key = lambda node: node.g)
            curr_node = open.pop(best_i)
            closed[curr_node] = curr_node.f
            for op in discrete_actions:
                #make sure precond satisfied
                if not op.precond(curr_node.state):
                    continue
                new_node = Node(op.transition_model(curr_node.state.to_serialized_string(), op))
                if new_node in closed:
                    continue
                if collision_fn(new_node.state):
                    closed.append(new_node)
                    continue
                #check for collision
                open.append(new_node)
        #backtrack
        plan = []
        while (curr_node.parent != start_node):
            plan.append(curr_node.copy())
            curr_node = curr_node.parent
        return plan



def collision_fn(state):
    #TODO will there be any collision from this state applying this action?
    obs_pos = state.get_values_as_vec(["frame:obstacle/position"])
    obs_width =state.get_values_as_vec(["constants/obstacle_width"])
    block_pos = state.get_values_as_vec(["frame:block/position"])
    block_width =state.get_values_as_vec(["constants/block_width"])
    #approximate with circles
    max_dist = block_width*((2)**0.5)+obs_width((2)**0.5)
    if np.linalg.norm(obs_pos-block_pos) < max_dist:
        return True
    return False
    #does the line pass through any points in the obstace?










