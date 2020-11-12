from planning.operators import *
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
        self.parent=parent
        if parent is None:
            self.cost = 0
            self.op = None
        else:
            self.op = op  # op taken to get to this state
            self.cost = parent.cost + op.cost()
        self.f = self.cost

    def compute_heuristic(self, goal):
        goal_values = goal.state.get_values_as_vec([block_pos_fqn])
        my_values = self.state.get_values_as_vec([block_pos_fqn])
        return np.linalg.norm(np.array(goal_values)-np.array(my_values))

    def __eq__(self, other):
        if other is None:
            return False
        my_values = self.state.get_values_as_vec([robot_pos_fqn, robot_orn_fqn, block_pos_fqn])
        other_values = other.state.get_values_as_vec([robot_pos_fqn, robot_orn_fqn, block_pos_fqn])
        return np.linalg.norm(np.array(my_values)-np.array(other_values)) < 0.02

    def __hash__(self):
        return 17

    def __str__(self):
        block_pos = self.state.get_values_as_vec([block_pos_fqn])
        robot_pos = self.state.get_values_as_vec([robot_pos_fqn])
        return "Cost : " + str(self.cost) + " block pose: "+str(np.round(block_pos, 2))+ "robot pose :" +str(np.round(robot_pos, 2)) + " op: "+str(self.op)

class Planner:
    def __init__(self, env_config):
        self.env_config =env_config
    def plan(self, start, goal, goal_tol = 0.02):
        """
        Given a pillar start state, output a series of operators to reach the goal state with high probability
        A* search
        """
        start_node = Node(start)
        goal_node = Node(goal)
        open = [start_node]
        closed = {}
        #make discrete action space
        discrete_actions = []
        for dir in [1]:
            discrete_actions.append(GoToSide(dir))
            for amount in [0.05, 0.1, 0.15, 0.2]:
                T = (5+(10*amount))/self.env_config['scene']['gym']['dt']
                discrete_actions.append(PushInDir(dir, amount, T))
        num_expanded = 0
        while (len(open) > 0):
            best_i = np.argmin([node.f for node in open])
            curr_node = open.pop(best_i)
            #print("Expanding", curr_node)
            num_expanded +=1
            if num_expanded % 100 == 0:
                print("Num expanded", num_expanded)
            if is_goal(curr_node, goal_node, tol=goal_tol):
                print("Found goal!")
                break
            closed[curr_node] = curr_node.f
            for op in discrete_actions:
                #make sure precond satisfied
                if not op.precond(curr_node.state.get_serialized_string()):
                    #print("Precond for ", op, "Not satisfied by", curr_node)
                    #import ipdb; ipdb.set_trace()
                    continue
                new_node = Node(op.transition_model(curr_node.state.get_serialized_string(), op), parent=curr_node, op=op)
                new_node.f = new_node.cost + new_node.compute_heuristic(goal_node)
                #print("New node", new_node, " from ", op, " parent is ", curr_node)
                if new_node in closed:
                    continue
                if collision_fn(new_node.state):
                    closed[new_node] = new_node.cost
                    continue
                #check for collision
                open.append(new_node)
        #backtrack
        plan = []
        print(curr_node)
        while (curr_node.parent is not None):
            plan.append(curr_node)
            curr_node = curr_node.parent
        print("Found plan of length", len(plan))
        return plan[::-1]

def is_goal(node, goal_node, tol=0.02):
    my_values = node.state.get_values_as_vec([block_pos_fqn])
    goal_values = goal_node.state.get_values_as_vec([block_pos_fqn])
    return np.linalg.norm(np.array(my_values)-np.array(goal_values)) < tol



def collision_fn(state):
    #TODO will there be any collision from this state applying this action?
    obs_pos = state.get_values_as_vec(["frame:obstacle/position"])
    obs_width =state.get_values_as_vec(["constants/obstacle_width"])[0]
    block_pos = state.get_values_as_vec(["frame:block:pose/position"])
    block_width =state.get_values_as_vec(["constants/block_width"])[0]
    #TODO get these from cfg
    min_x = -0.3
    max_x = 0.3
    min_y = 0.1
    max_y = 0.6
    if block_pos[0] < min_x or block_pos[2] > max_x:
        return True
    if block_pos[2] < min_y or block_pos[2] > max_y:
        return True
    #approximate with circles
    max_dist = block_width*((2)**0.5)+obs_width*((2)**0.5)
    if np.linalg.norm(np.array(obs_pos)-np.array(block_pos)) < max_dist:
        return True
    return False
    #does the line pass through any points in the obstace?
