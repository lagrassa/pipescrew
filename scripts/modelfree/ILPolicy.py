import keras
import numpy as np
from autolab_core import RigidTransform
import keras.layers as kl
class ILPolicy:
    """
    Fits NN policy of observation data to action data as a behaviour cloning loss.
    observations expected to be in some autoencoded format
    """
    def __init__(self, observation_data, action_data, load_fn=None):
        observation_data_shape = observation_data.shape[1:]
        action_data = process_action_data(action_data)
        action_data_shape = action_data.shape[1]
        self.make_model(observation_data_shape, action_data_shape)
        if load_fn is not None:
            self.model.load_weights(load_fn)

    def __call__(self, obs ):
        return self.model.predict(obs.reshape(1,-1))

    def save_model(self, fn):
        self.model.save_weights(fn)
    def train_model(self,observation_data, action_data, n_epochs = 1, validation_split = 0.99):
        action_data = process_action_data(action_data)
        self.model.fit(observation_data[:-1], action_data[1:], epochs = n_epochs, validation_split=validation_split)

    def make_model(self, obs_shape, act_shape):
        input_shape = obs_shape
        inputs = kl.Input(shape=input_shape, name='encoder_input')
        x = kl.Dense(64)(inputs)
        x = kl.Dropout(0.5)(x)
        x = kl.Dense(32)(x)
        x = kl.Dropout(0.5)(x)
        x = kl.Dense(16)(x)
        il_output = kl.Dense(act_shape)(x)
        self.model = keras.models.Model(inputs, [il_output], name='IL node')
        self.model.compile(optimizer='adam', loss="mse")

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
