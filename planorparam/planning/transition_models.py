import GPy as gpy
import numpy as np

class BlockPushSimpleTransitionModel():
    def __init__(self):
        pass

    def predict(self, state, action):
        next_state = state.copy()
        next_state[:,:3] = next_state[:,:3]+action[:,:3]
        return next_state #ignore stiffness

class LearnedTransitionModel():
    def __init__(self):
        self.high_state = np.array([0.1,0.1,0.1])
        self.low_state = np.array([-0.1, -0.1, -0.1])
        lengthscale = (self.high_state-self.low_state) * 0.001
        self.k = gpy.kern.Matern52(self.high_state.shape[0], ARD=True, lengthscale=lengthscale)

    def train(self, states, actions, next_states):
        inputs = self.get_features(states, actions)
        self.model = gpy.models.GPRegression(inputs, next_states, self.k)
        self.model['.*variance'].constrain_bounded(1e-1,2., warning=False)
        self.model['Gaussian_noise.variance'].constrain_bounded(1e-4,0.01, warning=False)
        # These GP hyper parameters need to be calibrated for good uncertainty predictions.
        self.model.optimize(messages=False)

    def get_features(self, states, actions):
        if len(states.shape) == 1:
            return np.hstack([states[2], actions[2], 0.001*actions[-1]]) #all manual for now
        else:
            return np.hstack([states[:,:3], actions[:,:3], 0.001*actions[:,-1].reshape(-1,1)])


    def predict(self, state, action):
        inputs = self.get_features(state, action)
        mean, sigma =self.model.predict(inputs)
        return mean, 2*sigma

