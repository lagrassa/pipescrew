import keras
import keras.layers as kl
class ILPolicy:
    """
    Fits NN policy of observation data to action data as a behaviour cloning loss.
    observations expected to be in some autoencoded format
    """
    def __init__(self, observation_data, action_data):
        observation_data_shape = observation_data.shape[1:]
        action_data_shape = action_data.shape[1]
        self.make_model(observation_data_shape, action_data_shape)

    def __call__(self, obs ):
        return self.model.predict(obs.reshape(1,-1))

    def train_model(self,observation_data, action_data, n_epochs = 1):
        self.model.fit(observation_data[:-1], action_data[1:], epochs = n_epochs)

    def make_model(self, obs_shape, act_shape):
        input_shape = obs_shape
        inputs = kl.Input(shape=input_shape, name='encoder_input')
        x = kl.Dense(16)(inputs)
        x = kl.Dense(32)(x)
        x = kl.Dense(64)(x)
        il_output = kl.Dense(act_shape)(x)
        self.model = keras.models.Model(inputs, [il_output], name='IL node')
        self.model.compile(optimizer='adam', loss="mse")

