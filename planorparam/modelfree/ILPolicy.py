import keras
import numpy as np
from autolab_core import RigidTransform
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.model_selection import RandomizedSearchCV# Number of trees in random forest

import keras.layers as kl
class ILPolicy:
    """
    Fits NN policy of observation data to action data as a behaviour cloning loss.
    observations expected to be in some autoencoded format
    """
    def __init__(self, observation_data, action_data, load_fn=None, model_type ="neural"):
        observation_data_shape = observation_data.shape[1:]
        action_data_shape = action_data.shape[1]
        self.model_type = model_type
        #also add the current action data
        self.make_model(observation_data_shape, action_data_shape)
        if load_fn is not None and model_type == "neural":
            self.model.load_weights(load_fn)
        if load_fn is not None and model_type == "forest":
            params = np.load(load_fn, allow_pickle=True)
            self.model= params.item()['estimator']

    def __call__(self, obs, reset = False):
        if self.model_type == "neural":
            if reset:
                self.model.reset_state()
            return self.model.predict(obs.reshape(1,-1))
        else:
            return self.model.predict(obs.reshape(1,-1))

    def low_pass_filter_actions(self,data):
        #run a low pass filter on the actions
        cutoff = 8
        nyq = 30*0.5
        from scipy.signal import butter,filtfilt
        normal_cutoff = cutoff / nyq
        # Get the filter coefficients 
        order = 6
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        scalar = 1
        y = np.hstack([scalar*filtfilt(b, a, data[:,i]).reshape(-1,1) for i in range(data.shape[1])])
        return y
    def save_model(self, fn):
        if self.model_type == "neural":
            self.model.save_weights(fn)
        else:
            np.save("models/rfparams.npy", self.model.get_params())
    def train_model(self,observation_data, action_data, n_epochs = 1, validation_split = 0.05, params_file = None):
        if self.model_type == "neural":
            #action_data = self.low_pass_filter_actions(action_data)
            self.model.fit(observation_data, action_data, epochs = n_epochs, validation_split=validation_split)
        else:
            if params_file is None:
                n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
                # Number of features to consider at every split
                max_features = ['auto', 'sqrt']
                # Maximum number of levels in tree
                max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
                max_depth.append(None)
                # Minimum number of samples required to split a node
                min_samples_split = [2, 5, 10]
                # Minimum number of samples required at each leaf node
                min_samples_leaf = [1, 2, 4]
                # Method of selecting samples for training each tree
                bootstrap = [True, False]# Create the random grid
                rf = RFR()
                random_grid = {'n_estimators': n_estimators,
                            'max_features': max_features,
                            'max_depth': max_depth,
                            'min_samples_split': min_samples_split,
                            'min_samples_leaf': min_samples_leaf,
                            'bootstrap': bootstrap}
                rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 12, cv = 3, verbose=1, random_state=42, n_jobs = 12)# Fit the random search model 70 works
                rf_random.fit(observation_data,action_data) 
                self.model=rf_random
            else:
                self.model.fit(observation_data, action_data)

    def make_model(self, obs_shape, act_shape):
        if self.model_type == "neural":
            input_shape = obs_shape
            inputs = kl.Input(shape=input_shape, name='encoder_input')
            x = kl.Dense(16)(inputs)
            x = kl.Dropout(0.5)(x)
            #x = kl.LSTM(10, return_sequences=True, stateful=True)(x)
            x = kl.GaussianNoise(0.00001)(x)
            x = kl.Dense(8)(x)
            x = kl.Dropout(0.5)(x)
            il_output = kl.Dense(act_shape)(x)
            self.model = keras.models.Model(inputs, [il_output], name='IL node')
            self.model.compile(optimizer='adam', loss="mse")
        else:
            self.model = RFR(max_depth = 20, criterion="mse", oob_score=True)

def process_action_data(action_data):
    """
    returns in x,y,z, quat form
    """
    fourbyfour = action_data.reshape((action_data.shape[0], 4,4))
    trans = fourbyfour[:,-1,0:3]
    quats = []
    for i in range(action_data.shape[0]):
        rot = RigidTransform(rotation=fourbyfour[i,:3,:3])
        quat = rot.quaternion
        quats.append(quat)
    quat = np.vstack(quats)
    return np.hstack([trans, quat]) 

    #right 3 is the rot, then bototm 3 for pos.
