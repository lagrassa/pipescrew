import GPy as gpy
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import Matern

class BlockPushSimpleTransitionModel():
    def __init__(self):
        self.step = self.predict
    def predict(self, state, action):
        next_state = state.copy()
        if len(state.shape)  ==2:
            next_state[:,:3] = next_state[:,:3]+action[:,:3]
        else:
            next_state[1] += action[0]
        return next_state #ignore\\ln p(y_{i}|\\lambda(f_{i})) = -\\frac{N \\ln 2\\pi}{2} - \\frac{\\ln |K|}{2} - \\frac{(y_{i} - \\lambda(f_{i}))^{T}\\sigma^{-2}(y_{i} - \\lambda(f_{i}))}{2} stiffness

class LearnedTransitionModel():
    def __init__(self):
        self.high_state = np.array([0.1,0.1,0.1])
        self.low_state = np.array([-0.1, -0.1, -0.1])
        lengthscale = (self.high_state-self.low_state) * 0.001
        self.k = gpy.kern.Matern52(self.high_state.shape[0], ARD=True, lengthscale=lengthscale)
        self.k =  Matern()
        self.model = GPR(kernel = self.k, random_state=0, optimizer="fmin_l_bfgs_b", n_restarts_optimizer = 10, normalize_y=True) #TODO fill in with better prior

    def train(self, states, actions, next_states):
        inputs = self.get_features(states, actions)
        #self.model = gpy.models.GPRegression(inputs, next_states, self.k)
        #self.model['.*variance'].constrain_bounded(1e-1,2., warning=False)
        #self.model['Gaussian_noise.variance'].constrain_bounded(1e-4,0.01, warning=False)
        # These GP hyper parameters need to be calibrated for good uncertainty predictions.
        #self.model.optimize(messages=False)
        import ipdb; ipdb.set_trace()
        self.model.fit(inputs, next_states)

    def get_features(self, states, actions):
        if len(states.shape) == 1:
            return np.hstack([states[2], actions[2], 0.001*actions[-1]]) #all manual for now
        else:
            return np.hstack([states[:,:3], actions[:,:3], 0.001*actions[:,-1].reshape(-1,1)])


    def predict(self, state, action):
        inputs = self.get_features(state, action)
        #mean, sigma =self.model.predict(inputs)
        import ipdb; ipdb.set_trace()
        mean, sigma = self.model.predict(inputs, return_std=True)
        return mean, 2*sigma

