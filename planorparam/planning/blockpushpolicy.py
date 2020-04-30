from carbongym import gymapi
from carbongym_utils.math_utils import RigidTransform_to_transform, transform_to_RigidTransform
import numpy as np
import matplotlib.pyplot as plt
from autolab_core import RigidTransform
from planning.transition_models import BlockPushSimpleTransitionModel, LearnedTransitionModel
from agent.model_selection import ModelSelector
class BlockPushPolicy():
    def __init__(self):
        self.manual_transition_model  = BlockPushSimpleTransitionModel()
        self.learned_transition_model = LearnedTransitionModel()
        self.model_selector = ModelSelector()
        self.model_selector.add(self.manual_transition_model, model_type = "manual")
        self.model_selector.add(self.learned_transition_model, model_type = "GP")

    """
    :param goal desired block pose in np array
    :return [(state,action)] tuple for each env
    states and action is  each a 3D array where [env_id, state, t] 
    ex) the state for env 1 at time 3 = states[1,:,3]
    """

    def choose_transition_model(self, state, action):
        return self.model_selector.select_model()

    def sample_actions(self, state, goal, delta = 0.05):
        stiffness = 100
        if len(state.shape) == 2:
            dir = (goal[:,:3]-state[:,:3])/(np.linalg.norm(goal[:,:3]-state[:,:3]))
        else:
            dir = (goal[:3] - state[:3]) / (np.linalg.norm(goal[:3] - state[:3]))
        pos_action = delta*dir
        rest_of_actions = np.multiply(np.ones((state.shape[0], 5)), np.array([1,0,0,0,stiffness]).T )
        return np.hstack([pos_action, rest_of_actions])
    """
    Planner that uses DFS; nothing fancy
    """
    def plan(self, start, goal,  horizon = 5):
        states = None
        action_shape = (8,)
        actions = None
        current_state = start.copy()
        tol = 0.02
        model_per_t = []
        while np.linalg.norm(current_state - goal) > tol:
            #if in learned region, use learned one, else use normal one
            #we never reject actions rn, until we use a real planner
            action = self.sample_actions(current_state, goal)
            model = self.choose_transition_model(current_state, action)
            model_per_t.append(model)
            next_state = model.predict(current_state, action)
            if actions is None:
                actions = action
                states = next_state
            else:
                states = np.concatenate([states, next_state],axis=-1)
                actions = np.concatenate([actions, action],axis=-1)

        return states, actions, model_per_t
    """
    tol: percentage error that's OK relative to the magnitude of the motion
    """
    def monitored_execution(self, vec_env, states, action, custom_draws,
                            model_per_t = None,
                            tol = 0.01,
                            n_steps = 20,
                            plot_deviations=False):
        """
        :param states, actions states and high level actions to execute
        :return list of states where the agent deviated, amount of each deviation
        """
        deviations = np.zeros((states.shape[0], states.shape[1], states.shape[-1]-1))
        actual_states = []
        deviated_states = []
        learned_model_training_s = []
        learned_model_training_sprime = []
        learned_model_training_a = []

        for t in range(states.shape[-1]-1):
            for i in range(n_steps):
                vec_env.step(action[:,:,t])
                vec_env.render(custom_draws=custom_draws)
            next_state = vec_env.get_states()
            actual_states.append(next_state.T)
            expected_state = states[:,:,t+1]
            new_deviations = np.linalg.norm(next_state-expected_state, axis=0)
            deviations[:,:,t] = new_deviations
            step_distances = np.linalg.norm(states[:,:,t+1]-states[:,:,t], axis=0)
            for env_id in range(states.shape[0]): #TODO vectorize
                if (np.linalg.norm(new_deviations[env_id])/step_distances[env_id]) > tol:
                    deviated_states.append(states[env_id,:,t])
                    learned_model_training_sprime.append(actual_states[env_id])
                    learned_model_training_a.append(actual_states[env_id])
                    learned_model_training_s.append(states[env_id,:,t])
        actual_states = np.hstack(actual_states)
        deviated_states = np.hstack(deviated_states)
        if plot_deviations:
            plt.plot(states[0,2,1:], label="expected z")
            plt.plot(actual_states[2,:], label = "actual z")
            plt.plot(model_per_t == self.learned_transition_model, label="used learned model")
            plt.legend()
            plt.show()
        manual_states = states[model_per_t]
        manual_deviations = deviations[model_per_t]
        self.model_selector.add_history(manual_states, manual_deviations, self.manual_transition_model)
        learned_model_training_s, learned_model_training_a,learned_model_training_sprime = [np.vstack(data) for data in [learned_model_training_s, learned_model_training_a, learned_model_training_sprime]]
        self.learned_transition_model.train(learned_model_training_s, learned_model_training_a, learned_model_training_sprime)
        np.save("data/states.npy", learned_model_training_s)
        np.save("data/actions.npy", learned_model_training_a)
        np.save("data/next_states.npy", learned_model_training_sprime)
        return deviations, deviated_states

    def go_to_push_start(self, vec_env, z_offset=0.09):
        for env_index, env_ptr in enumerate(vec_env._scene.env_ptrs):
            block_ah = vec_env._scene.ah_map[env_index][vec_env._block_name]
            block_transform = vec_env._block.get_rb_transforms(env_ptr, block_ah)[0]
            ee_transform = vec_env._franka.get_ee_transform(env_ptr, vec_env._franka_name)
            grasp_transform = gymapi.Transform(p=block_transform.p, r=ee_transform.r)
            pre_grasp_transfrom = gymapi.Transform(p=grasp_transform.p, r=grasp_transform.r)
            pre_grasp_transfrom.p.z += z_offset
            pre_grasp_transfrom.p.y -= 0.02
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

