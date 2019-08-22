from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Add, Subtract, Lambda, LSTM, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K

#Tensorflow 2.0 Beta

class Dueling_model():
    def build_model(self, state_size, neurons, action_size):
        #前面的LSTM層們
        state_input = Input(shape=state_size)
        lstm1 = LSTM(neurons, activation='sigmoid',return_sequences=True)(state_input)
        lstm1 = Dropout(0.2)(lstm1)
        lstm2 = LSTM(neurons, activation='sigmoid')(lstm1)
        lstm2 = Dropout(0.2)(lstm2)

        #連結層
        d2 = Dense(neurons,activation='relu')(lstm2)
        
        #dueling
        d3_a = Dense(neurons/2, activation='relu')(d2)
        d3_v = Dense(neurons/2, activation='relu')(d2)
        a = Dense(action_size,activation='linear')(d3_a)
        value = Dense(1,activation='linear')(d3_v)
        a_mean = Lambda(lambda x: K.mean(x, axis=1, keepdims=True))(a)
        advantage = Subtract()([a, a_mean])
        q = Add()([value,advantage])
        
        #最後compile
        model = Model(inputs=state_input, outputs=q)
        model.compile(loss='mse', optimizer=Adam(lr=0.001))
        
        return model
    
    def noisy_dense(self, inputs, units, c_names, w_i, b_i=None,
        activation=tf.keras.activations.linear,
        noisy_distribution='factorised'
        ):
        
        flatten_shape = inputs.shape[1]
        weights = tf.get_variable(name='weights', shape=[flatten_shape, units], initializer=w_i)
        w_sigma = tf.get_variable(name='w_sigma', shape=[flatten_shape, units], initializer=w_i, collections=c_names)
        if noisy_distribution == 'independent':
            weights += tf.multiply(tf.random_normal(shape=w_sigma.shape), w_sigma)
        elif noisy_distribution == 'factorised':
            noise_1 = f(tf.random_normal(tf.TensorShape([flatten_shape, 1]), dtype=tf.float32))  # 注意是列向量形式，方便矩阵乘法
            noise_2 = f(tf.random_normal(tf.TensorShape([1, units]), dtype=tf.float32))
            weights += tf.multiply(noise_1 * noise_2, w_sigma)
        dense = tf.matmul(inputs, weights)
        return activation(dense) if activation is not None else dense