import GPy as gpy
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import Matern
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.model_selection import RandomizedSearchCV
from autolab_core import YamlConfig
from pillar_state_py import State

def color_block_is_on(cfg, pose):
    #return the first color that the block is on.
    board_names = [name for name in cfg.keys() if "boardpiece" in name]
    for board_name in board_names:
        center = np.array([cfg[board_name]['pose']['y'], 0,cfg[board_name]['pose']['x'],]) #yzx
        width = cfg[board_name]['dims']['width'],
        depth = cfg[board_name]['dims']['depth'],
        if (pose[2] > center[2]-depth/0.5 and pose[2] < center[2]+depth/0.5 ):
            if (pose[0] > center[0]-width/0.5 and pose[0] < center[0]+depth/0.5):
                return cfg[board_name]["rb_props"]["color"]

class BlockPushSimpleTransitionModel():
    def __init__(self, config_fn = "cfg/franka_block_push_two_d.yaml"):
        self.step = self.predict
        self.cfg = YamlConfig(config_fn)
        self.dir_to_rpy= {0:[-np.pi/2,np.pi/2,0],
                          1:[-np.pi/2,np.pi/2,np.pi/2],
                          2:[-np.pi/2,np.pi/2,np.pi],
                          3:[-np.pi/2,np.pi/2,1.5*np.pi]}
    """
    Pillar State object
    action = dir, amount, T. 
    """
    def predict(self, state_str, action):
        state = State.create_from_serialized_string(state_str)
        new_state = State.create_from_serialized_string(state_str)
        #take current state
        dir = action[0]
        amount = action[1]
        T = action[2]
        robot_pos_fqn = "frame:pose/position"
        robot_orn_fqn = "frame:pose/quaternion"
        block_pos_fqn = "frame:block:pose/position"

        #precondition is that the robot is at the appropriate side, taken care of at another layer of abstraction
        #if robot is "below" block, otherwise it's the same
        current_block_state = state.get_values_as_vec([block_pos_fqn])
        next_block_state_np = np.array([current_block_state[0] - amount * np.sin(self.dir_to_rpy[dir][2]),
                             current_block_state[1],
                             current_block_state[2] - amount * np.cos(self.dir_to_rpy[dir][2])])
        new_state.set_values_from_vec([block_pos_fqn],next_block_state_np.tolist())
        return new_state.get_serialized_string()


    def predict_simple(self, state, action):
        next_state = state.copy()
        if action[-1] < 5:
            return next_state #stiffness too low; won't move anywhere
        if len(state.shape)  ==2:
            next_state[:,:3] = next_state[:,:3]+action[:,:3]
        else:
            next_state[1] += action[0]
        return next_state #ignore\\ln p(y_{i}|\\lambda(f_{i})) = -\\frac{N \\ln 2\\pi}{2} - \\frac{\\ln |K|}{2} - \\frac{(y_{i} - \\lambda(f_{i}))^{T}\\sigma^{-2}(y_{i} - \\lambda(f_{i}))}{2} stiffness

class LearnedTransitionModel():
    def __init__(self, fit_to_default = True, use_GP = False):
        self.high_state = np.array([0.1])
        self.low_state = np.array([-0.1])
        lengthscale = (self.high_state-self.low_state) * 0.001
        #self.k = gpy.kern.Matern52(self.high_state.shape[0], ARD=True, lengthscale=lengthscale)
        self.k =  Matern(length_scale= 0.4, length_scale_bounds=(0.00001, 1),nu=5/2.)
        self.scaler = preprocessing.StandardScaler()
        self.use_GP = use_GP
        if use_GP:
            self.model = GPR(kernel = self.k, random_state=17, optimizer="fmin_l_bfgs_b", n_restarts_optimizer = 200, normalize_y=True) #TODO fill in with better prior
        else:
            n_estimators = [int(x) for x in np.linspace(start=20, stop=1000, num=10)]
            # Number of features to consider at every split
            max_features = ['auto', 'sqrt']
            # Maximum number of levels in tree
            max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
            max_depth.append(None)
            # Minimum number of samples required to split a node
            min_samples_split = [2, 5, 10]
            # Minimum number of samples required at each leaf node
            min_samples_leaf = [1, 2, 4]
            # Method of selecting samples for training each tree
            bootstrap = [True, False]  # Create the random grid
            rf = RFR()
            random_grid = {'n_estimators': n_estimators,
                           'max_features': max_features,
                           'max_depth': max_depth,
                           'min_samples_split': min_samples_split,
                           'min_samples_leaf': min_samples_leaf,
                           'bootstrap': bootstrap}
            self.rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=3, cv=3, verbose=0,
                                           random_state=42, n_jobs=12)
            self.model = self.rf_random

        self.states_fn = "data/states.npy"
        self.actions_fn = "data/actions.npy"
        self.next_states_fn = "data/next_states.npy"
        if fit_to_default:
            print("Training on old data")
            self.train_on_old_data_if_present()

    def train_on_old_data_if_present(self):
        old_states, old_actions, old_next_states = self.load_data_if_present()
        if old_states is not None:
            inputs = self.get_features(old_states, old_actions, train=True)
            if self.model is None:
                self.model = gpy.models.GPRegression(inputs, old_next_states[:,1].reshape(-1,1),self.k)
                self.model['.*variance'].constrain_bounded(1e-4, 2., warning=False)
                self.model['Gaussian_noise.variance'].constrain_bounded(1e-4, 0.1, warning=False)
                for i in range(20):
                    self.model.optimize(messages=True)
                self.model.fit = lambda x,y: 3
            self.model.fit(inputs, old_next_states[:,1])


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

        inputs = self.get_features(training_states, training_actions, train=True)
        #self.model = gpy.models.GPRegression(inputs, next_states, self.k)
        #self.model['.*variance'].constrain_bounded(1e-1,2., warning=False)
        #self.model['Gaussian_noise.variance'].constrain_bounded(1e-4,0.01, warning=False)
        # These GP hyper parameters need to be calibrated for good uncertainty predictions.
        #self.model.optimize(messages=False)

        self.model.fit(inputs, training_next_states[:, 1])
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


    def get_features(self, states, actions, train=False):
        #unprocessed_input = np.hstack([states, actions])
        unprocessed_input = states
        if len(unprocessed_input.shape) == 1:
            unprocessed_input = unprocessed_input.reshape(1,-1)
        unprocessed_input = unprocessed_input[:,1].reshape(-1,1) #hack
        if train:
            self.scaler.fit(unprocessed_input)
        return self.scaler.transform(unprocessed_input)


    #requires it be of shape N X M where N is number of samples
    def predict(self, states, actions, flatten=True):
        #assuming same x
        inputs = self.get_features(states, actions)
        if self.use_GP:
            try:
                mean, sigma = self.model.predict(inputs, return_std=True)
            except TypeError: #wrong GP library
                mean, sigma = self.model.predict(inputs)
        else:
            try:
                mean = self.model.predict(inputs)
            except sklearn.exceptions.NotFittedError:
                mean = self.model.predict(inputs)
        if len(mean.shape) == 1:
            mean = mean.reshape(-1,1)
        if mean.shape[0] < mean.shape[1]:
            mean = mean.T
        if len(states.shape) == 1:
            next_state = np.hstack([states[0], mean.flatten()])
        else:
            next_state = np.hstack([states[:,0].reshape(-1,1), mean.reshape(-1,1)])  # , 2*sigma
        action_dist = np.linalg.norm(next_state-states)
        if action_dist < 1e-3:
            print("Low action dist")
            print(states, "states")
        return next_state
def preprocess(data):
    return preprocessing.scale(data)

