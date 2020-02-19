import keras
import keras.layers as kl
class ILPolicy:
    """
    Fits NN policy of observation data to action data as a behaviour cloning loss.
    observations expected to be in some autoencoded format
    """
    def __init__(self, observation_data, action_data, load_fn=None):
        observation_data_shape = observation_data.shape[1:]
        action_data_shape = action_data.shape[1]
        self.make_model(observation_data_shape, action_data_shape)
        if load_fn is not None:
            self.model.load_weights(load_fn)

    def __call__(self, obs ):
        return self.model.predict(obs.reshape(1,-1))

    def save_model(self, fn):
        self.model.save_weights(fn)
    def train_model(self,observation_data, action_data, n_epochs = 1, validation_split = 0.99):
        self.model.fit(observation_data, action_data, epochs = n_epochs, validation_split=validation_split)

    def make_model(self, obs_shape, act_shape):
        input_shape = obs_shape
        inputs = kl.Input(shape=input_shape, name='encoder_input')
        x = kl.Dense(64)(inputs)
        x = kl.Dense(32)(x)
        x = kl.Dense(16)(x)
        il_output = kl.Dense(act_shape)(x)
        self.model = keras.models.Model(inputs, [il_output], name='IL node')
        self.model.compile(optimizer='adam', loss="mse")

