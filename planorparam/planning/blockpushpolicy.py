from carbongym import gymapi
from sklearn.exceptions import  NotFittedError
from carbongym_utils.math_utils import RigidTransform_to_transform, transform_to_RigidTransform
import numpy as np
import matplotlib.pyplot as plt
from autolab_core import RigidTransform
from planning.transition_models import BlockPushSimpleTransitionModel, LearnedTransitionModel
from agent.model_selection import ModelSelector
class BlockPushPolicy():
    def __init__(self, use_history=False):
        self.manual_transition_model  = BlockPushSimpleTransitionModel()
        self.learned_transition_model = LearnedTransitionModel()
        self.manual_transition_model_idx = 0
        self.learned_transition_model_idx = 1
        self.tolerance = 0.2 #in percentage of motion in cm
        self.model_selector = ModelSelector(use_history=use_history)
        self.model_selector.add(self.manual_transition_model, model_type = "manual")
        self.model_selector.add(self.learned_transition_model, model_type = "GP")

    """
    :param goal desired block pose in np array
    :return [(state,action)] tuple for each env
    states and action is  each a 3D array where [env_id, state, t] 
    ex) the state for env 1 at time 3 = states[1,:,3]
    """

    def choose_transition_model(self, state, action):
        try:
            return self.model_selector.select_model(state, action, self.tolerance)
        except NotFittedError:
            return self.manual_transition_model

    def sample_actions(self, state, goal, delta = 0.05):
        #Adapted for simpler action space
        stiffness = 1000
        return np.array([-0.02, stiffness])
        if len(state.shape) == 2:
            dir = (goal[:,:3]-state[:,:3])/(np.linalg.norm(goal[:,:3]-state[:,:3]))
        else:
            dir = (goal[:3] - state[:3]) / (np.linalg.norm(goal[:3] - state[:3]))
        pos_action = delta*dir
        rest_of_actions = np.multiply(np.ones((state.shape[0], 5)), np.array([0,0,0,1,stiffness]).T )
        return np.hstack([pos_action, rest_of_actions])
    """
    Planner that uses DFS; nothing fancy
    """
    def plan(self, start, goal,  delta = 0.001, horizon = 50):
        states = None
        action_shape = (8,)
        actions = None
        current_state = start.copy()
        goal_tol = 0.01
        model_per_t = [] #0 for planned 1 for learned
        i = 0
        states = [current_state] #next states after taking an action, including the first though.
        actions = []
        while np.max(np.linalg.norm(current_state - goal, axis=0)) > goal_tol:
            #print("planner distance", np.linalg.norm(current_state[:,:3]-goal[:,:3]))
            #if in learned region, use learned one, else use normal one
            #we never reject actions rn, until we use a real planner
            print("curr distance", np.linalg.norm(current_state - goal))
            action = self.sample_actions(current_state, goal, delta = delta)
            model = self.choose_transition_model(current_state, action)
            if model == self.manual_transition_model:
                model_idx = self.manual_transition_model_idx
            else:
                model_idx = self.learned_transition_model_idx
            model_per_t.append(model_idx)
            next_state = model.predict(current_state, action)
            current_state = next_state.copy()
            #action = action.reshape((1,) + action.shape)
            #next_state = next_state.reshape((1,) + next_state.shape)
            actions.append(action)
            states.append(next_state)
            assert(len(actions) == len(states)-1 == len(model_per_t))
            i +=1
        states = np.vstack(states)
        actions = np.vstack(actions)
        states = states.T
        actions= actions.T
        model_per_t = np.array(model_per_t)
        return states, actions, model_per_t
    """
    tol: percentage error that's OK relative to the magnitude of the motion
    """
    def monitored_execution(self, vec_env, states, actions, custom_draws,
                            model_per_t = None,
                            tol = 0.01,
                            n_steps = 20,
                            use_history=False,
                            plot_deviations=False):
        """
        :param states, actions states and high level actions to execute
        :return list of states where the agent deviated, amount of each deviation
        """
        deviations = np.zeros(states.shape[1]-1)
        actual_states = []
        deviated_states = []
        learned_model_training_s = []
        learned_model_training_sprime = []
        learned_model_training_a = []

        for t in range(states.shape[-1]-1):
            vec_env.step(actions[:,t])
            vec_env.render(custom_draws=custom_draws)
            next_state = vec_env._compute_obs(None)["observation"]
            actual_states.append(next_state)
            expected_state = states[:,t+1]
            new_deviations = np.linalg.norm(next_state-expected_state, axis=0)
            step_distances = np.linalg.norm(states[:,t+1]-states[:,t], axis=0)
            deviations[t] = new_deviations/step_distances
            if (deviations[t]) > tol:
                    deviated_states.append(states[:,t])
                    learned_model_training_sprime.append(actual_states[-1])
                    learned_model_training_a.append(actions[:,t])
                    learned_model_training_s.append(states[:,t])
        actual_states = np.vstack(actual_states)
        if len(deviated_states) > 0:
            deviated_states = np.vstack(deviated_states)
        timesteps = np.array(list(range(actions.shape[-1])))
        if plot_deviations:
            manual_timesteps = timesteps[model_per_t == self.manual_transition_model_idx]
            learned_timesteps = timesteps[model_per_t == self.learned_transition_model_idx]
            manual_state_preds = states[1,1:][model_per_t == self.manual_transition_model_idx]
            learned_state_preds = states[1,1:][model_per_t == self.learned_transition_model_idx]
            plt.scatter(manual_timesteps, manual_state_preds ,label="manually predicted z", marker = "o" )
            plt.scatter(learned_timesteps, learned_state_preds, label="learned model predicted z" , marker="x" )
            plt.plot(timesteps, actual_states[:,1], label = "actual z")
            plt.legend()
            plt.show()
        manual_states = states[:,1:][:,model_per_t == self.manual_transition_model_idx]
        manual_deviations = deviations[model_per_t == self.manual_transition_model_idx]
        self.model_selector.add_history(manual_states, manual_deviations, self.manual_transition_model)
        learned_model_training_s, learned_model_training_a,learned_model_training_sprime = [np.vstack(data) for data in [learned_model_training_s, learned_model_training_a, learned_model_training_sprime]]
        self.learned_transition_model.train(learned_model_training_s, learned_model_training_a, learned_model_training_sprime, save_data = True, load_data = use_history)

        return deviations, deviated_states

    def go_to_push_start(self, vec_env, z_offset=0.09):
        for env_index, env_ptr in enumerate(vec_env._scene.env_ptrs):
            block_ah = vec_env._scene.ah_map[env_index][vec_env._block_name]
            block_transform = vec_env._block.get_rb_transforms(env_ptr, block_ah)[0]
            ee_transform = vec_env._franka.get_ee_transform(env_ptr, vec_env._franka_name)
            grasp_transform = gymapi.Transform(p=block_transform.p, r=ee_transform.r)
            pre_grasp_transfrom = gymapi.Transform(p=grasp_transform.p, r=grasp_transform.r)
            pre_grasp_transfrom.p.z += z_offset
            pre_grasp_transfrom.p.y -= 0.01
            rt = transform_to_RigidTransform(pre_grasp_transfrom)
            rot_matrix = RigidTransform.y_axis_rotation(np.pi/2)
            rt.rotation = np.dot(rot_matrix, rt.rotation)
            pre_grasp_transfrom = RigidTransform_to_transform(rt)
            stiffness = 1e3
            vec_env._franka.set_attractor_props(env_index, env_ptr, vec_env._franka_name,
                                             {
                                                 'stiffness': stiffness,
                                                 'damping': 6 * np.sqrt(stiffness)
                                             })
            vec_env._franka.set_ee_transform(env_ptr, env_index, vec_env._franka_name, pre_grasp_transfrom)
        #for i in range(50):
        #    vec_env.step([0.1,0.1,0.1,0,0,0,1])
    def go_to_block(self, vec_env):
        eps = 0.01
        for env_index, env_ptr in enumerate(vec_env._scene.env_ptrs):
            block_ah = vec_env._scene.ah_map[env_index][vec_env._block_name]
            block_transform = vec_env._block.get_rb_transforms(env_ptr, block_ah)[0]
            ee_transform = vec_env._franka.get_ee_transform(env_ptr, vec_env._franka_name)
            grasp_transform = gymapi.Transform(p=block_transform.p, r=ee_transform.r)
            pre_grasp_transfrom = gymapi.Transform(p=grasp_transform.p, r=grasp_transform.r)
            pre_grasp_transfrom.p.z += vec_env._cfg['block']['dims']['width']/2. -eps

            pre_grasp_transfrom.p.y -=0.01
            stiffness = 20
            vec_env._franka.set_attractor_props(env_index, env_ptr, vec_env._franka_name,
                                                {
                                                    'stiffness': stiffness,
                                                    'damping': 2 * np.sqrt(stiffness)
                                                })
            vec_env._franka.set_ee_transform(env_ptr, env_index, vec_env._franka_name, pre_grasp_transfrom)

