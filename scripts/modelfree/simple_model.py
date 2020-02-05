import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

"""
[x,y] -> [xt+1 , yt+1]
"""
def process_data(history):
    xts = None
    xtplus1s = None
    for path in history.paths:
        if xts is None:
            xts = np.vstack(path[:-1])
            xtplus1s = np.vstack(path[1:])
        else:
            xts = np.vstack([xts, path[:-1]])
            xtplus1s = np.vstack([xtplus1s, path[1:]])
    return xts, xtplus1s




"""
trains a model based on the history. The history predicts the next state
"""

def train_policy(history):
    x_train, y_train = process_data(history)
    model = Sequential()
    #model.add(Flatten(input_shape=(2,)))
    model.add(Dense(128, activation='relu', input_shape=(x_train.shape[1],)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(x_train.shape[1], activation='linear'))

    model.compile(loss=keras.losses.mean_squared_error,
                  optimizer=keras.optimizers.Adadelta(),
                                metrics=['accuracy'])

    model.fit(x_train, y_train,
              batch_size=10,
                        epochs=10,
                                  verbose=1)
    return model
