import copy
import numpy as np
from odium.utils.graph_search_utils.astar import Node, Astar
from odium.utils.simulation_utils import set_sim_state_and_goal, apply_dynamics_residual
from odium.controllers.controller import Controller
from odium.agents.fetch_agents.discrepancy_utils import apply_discrepancy_penalty
from planning.transition_models import BlockPushSimpleTransitionModel


class RobotPushController(Controller):
    def __init__(self,
                 env,
                 delta_goal,
                 num_expansions=10,
                 discrete=False,
                 reward_type='sparse'):
        '''
        model - gym env (should be wrapper env)
        num_expansions - Number of expansions to be done by A*
        discrete - Is the env discrete action space or not?
        reward_type - env cost function sparse or dense?
        '''
        super(RobotPushController, self).__init__()
        self.model = BlockPushSimpleTransitionModel()
        self.env = env
        self.env.reset()
        self.goal = self.env.env.get_delta_goal(delta_goal)
        self.num_expansions = num_expansions
        self.discrete = discrete
        self.reward_type = reward_type
        self.n_bins = 3  # TODO: Make an argument. Also needs to be the same as that of env
        if discrete:
            self.discrete_actions = self.env.discrete_actions

        self.astar = Astar(self.heuristic,
                           self.get_successors,
                           self.check_goal,
                           num_expansions,
                           self.actions())

        self.residual_heuristic_fn = None
        self.discrepancy_fn = None
        self.residual_dynamics_fn = None

    def actions(self):
        return self.env.discrete_actions_list.copy()

    def heuristic(self, node, augment=True):
        if isinstance(node.simple_state, dict):
            simple_state = self.obs_to_simple_state(node.simple_state)
        else:
            simple_state = node.simple_state
        return np.linalg.norm(simple_state - self.goal)

    def obs_to_simple_state(self, obs):
        return obs["observation"]#[self.env.env.real_env_idx,19:19+2]

    def heuristic_obs_g(self, observation, g):
        obs = {}
        obs['observation'] = observation
        obs['desired_goal'] = g
        node = Node(obs)
        return self.heuristic(node, augment=False)

    def get_qvalue(self, state, ac):
        # Set the model state
        # Get next observation
        next_state = self.model.step(state, np.array(ac))
        # Get heuristic of the next observation
        next_node = Node(next_state)
        next_heuristic = self.heuristic(next_state, augment=False)
        rew = 0
        return (-rew) + next_heuristic

    def get_qvalue_obs_ac(self, obs, g, qpos, qvel, ac):
        set_sim_state_and_goal(self.model,
                               qpos.copy(),
                               qvel.copy(),
                               g.copy())
        next_observation, rew, _, _ = self.model.step(np.array(ac))
        next_node = Node(next_observation)
        next_heuristic = self.heuristic(next_node, augment=False)

        return (-rew) + next_heuristic

    def get_all_qvalues(self, observation):
        qvalues = []
        for ac in self.model.discrete_actions_list:
            qvalues.append(self.get_qvalue(observation, ac))
        return np.array(qvalues)

    def get_successors(self, node, action):
        simple_state = node.simple_state
        #set_sim_state_and_goal(
        #    self.model,
        #    obs['qpos'].copy(),
        #    obs['qvel'].copy(),
        #    obs['desired_goal'].copy()
        #)

        #redesign these nodes. They really dont need the whole planning state.
        next_state = self.model.step(simple_state, np.array(action))
        """
        AL redesign this
        if self.discrepancy_fn is not None:
            rew = apply_discrepancy_penalty(
                obs, action, rew, self.discrepancy_fn)

        if self.residual_dynamics_fn is not None:
            next_obs, rew = apply_dynamics_residual(self.model,
                                                    self.residual_dynamics_fn,
                                                    obs,
                                                    info,
                                                    action,
                                                    next_obs)

        #mj_sim_state = copy.deepcopy(self.model.env.get_state())
        #next_obs['sim_state'] = mj_sim_state
        """
        next_node = Node(next_state)
        rew = 0
        return next_node, -rew

    def check_goal(self, node):
        simple_state = node.simple_state
        if isinstance(simple_state, dict):
            simple_state = self.obs_to_simple_state(simple_state)
        return self.env.env.is_success(simple_state, self.goal)

    def act(self, obs):
        start_node = Node(obs)
        best_action, info = self.astar.act(
            start_node)
        info['start_node_h_estimate'] = self.heuristic(
            start_node, augment=False)
        return np.array(best_action), info

    def reconfigure_heuristic(self, residual_heuristic_fn):
        self.residual_heuristic_fn = residual_heuristic_fn
        return True

    def reconfigure_num_expansions(self, n_expansions):
        self.num_expansions = n_expansions
        self.astar = Astar(self.heuristic,
                           self.get_successors,
                           self.check_goal,
                           self.num_expansions,
                           self.actions())
        return True

    def reconfigure_discrepancy(self, discrepancy_fn):
        self.discrepancy_fn = discrepancy_fn
        return True

    def reconfigure_residual_dynamics(self, residual_dynamics_fn):
        self.residual_dynamics_fn = residual_dynamics_fn
        return True
