import GPy as gpy
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import Matern
from sklearn import preprocessing

class BlockPushSimpleTransitionModel():
    def __init__(self):
        self.step = self.predict


    def predict(self, state, action):
        next_state = state.copy()
        if action[-1] < 5:
            return next_state #stiffness too low; won't move anywhere
        if len(state.shape)  ==2:
            next_state[:,:3] = next_state[:,:3]+action[:,:3]
        else:
            next_state[1] += action[0]
        return next_state #ignore\\ln p(y_{i}|\\lambda(f_{i})) = -\\frac{N \\ln 2\\pi}{2} - \\frac{\\ln |K|}{2} - \\frac{(y_{i} - \\lambda(f_{i}))^{T}\\sigma^{-2}(y_{i} - \\lambda(f_{i}))}{2} stiffness

class LearnedTransitionModel():
    def __init__(self, fit_to_default = True):
        self.high_state = np.array([0.1,0.1,0.1])
        self.low_state = np.array([-0.1, -0.1, -0.1])
        lengthscale = (self.high_state-self.low_state) * 0.001
        self.k = gpy.kern.Matern52(self.high_state.shape[0], ARD=True, lengthscale=lengthscale)
        self.k =  Matern()
        self.model = GPR(kernel = self.k, random_state=0, optimizer="fmin_l_bfgs_b", n_restarts_optimizer = 200, normalize_y=True) #TODO fill in with better prior
        self.states_fn = "data/states.npy"
        self.actions_fn = "data/actions.npy"
        self.next_states_fn = "data/next_states.npy"
        if fit_to_default:
            print("Training on old data")
            self.train_on_old_data_if_present()

    def train_on_old_data_if_present(self):
        old_states, old_actions, old_next_states = self.load_data_if_present()
        if old_states is not None:
            inputs = self.get_features(old_states, old_actions)
            self.model.fit(inputs, old_next_states)


    def load_data_if_present(self):
        try:
            old_states = np.load(self.states_fn)
            old_actions = np.load(self.actions_fn)
            old_next_states = np.load(self.next_states_fn)
        except FileNotFoundError:
            old_states, old_actions, old_next_states = None, None, None
        return old_states, old_actions, old_next_states

    def train(self, states, actions, next_states, save_data=False, load_data=False):
        old_states, old_actions, old_next_states = self.load_data_if_present()
        if load_data:
            if old_states is None:
                training_states = states
                training_actions = actions
                training_next_states = next_states
            else:
                training_states =  np.vstack([old_states, states])
                training_actions = np.vstack([old_actions, actions])
                training_next_states =  np.vstack([old_next_states, next_states])
        else:
            training_states = states
            training_actions = actions
            training_next_states = next_states

        inputs = self.get_features(training_states, training_actions)
        #self.model = gpy.models.GPRegression(inputs, next_states, self.k)
        #self.model['.*variance'].constrain_bounded(1e-1,2., warning=False)
        #self.model['Gaussian_noise.variance'].constrain_bounded(1e-4,0.01, warning=False)
        # These GP hyper parameters need to be calibrated for good uncertainty predictions.
        #self.model.optimize(messages=False)
        self.model.fit(inputs, training_next_states)
        if save_data:
            if not load_data:
                if old_states is None:
                    states_to_save = states
                    actions_to_save = actions
                    next_states_to_save = next_states
                else:
                    states =  np.vstack([old_states, states])
                    actions_to_save = np.vstack([old_actions, actions])
                    next_states_to_save =  np.vstack([old_next_states, next_states])
            else:
                states_to_save = training_states
                actions_to_save = training_actions
                next_states_to_save = training_next_states
            np.save(self.states_fn, states_to_save)
            np.save(self.actions_fn, actions_to_save)
            np.save(self.next_states_fn, next_states_to_save)


    def get_features(self, states, actions):
        actions_rescaled = actions.copy()
        scalar = 0.001
        if len(actions_rescaled.shape) == 1:
            actions_rescaled[-1] *= scalar
        else:
            actions_rescaled[:,-1] *= scalar

        return preprocess(np.hstack([states, actions])) #all manual for now


    #requires it be of shape N X M where N is number of samples
    def predict(self, states, actions, flatten=True):
        inputs = self.get_features(states, actions)
        if len(inputs.shape) == 1:
            inputs = inputs.reshape(1,-1)
        mean, sigma = self.model.predict(inputs, return_std=True)
        action_dist = np.linalg.norm(mean-states)
        if action_dist < 0.0001:
            print("Low action dist")
            print(states, "states")
            print(actions, "actions")
        if flatten:
            return mean.flatten()
        return mean #, 2*sigma
def preprocess(data):
    return preprocessing.scale(data)

