from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Add, Subtract, Lambda, LSTM
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K


class Dueling_model():
    def build_model(self, state_size, neurons, action_size):
        state_input = Input(shape=state_size)
        lstm = LSTM(neurons, activation='sigmoid')(state_input)
        d1 = Dense(2*neurons, activation='relu')(lstm)
        d2 = Dense(neurons,activation='relu')(d1)
        
        advantage = Dense(action_size,activation='linear')(d2)
        value = Dense(1,activation='linear')(d2)
        
        policy = Lambda(lambda x: x[0]-K.mean(x[0])+x[1],
        output_shape=(action_size,))([advantage, value])

        model = Model(inputs=state_input, outputs=policy)
        model.compile(loss='mse', optimizer=Adam(lr=0.001))
        
        return model