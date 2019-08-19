from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
import os, random
from collections import deque
import numpy as np

class Agent:
	def __init__(self, ticker, state_size, neurons, m_path, is_eval=False):
		self.state_size = state_size # normalized previous days
		self.action_size = 4 
		self.neurons = neurons
		self.memory = deque(maxlen=1000) #記憶長度
		self.gamma = 0.95
		self.epsilon = 1
		self.epsilon_min = 0.01
		self.epsilon_decay = 0.995
		self.is_eval = is_eval
		self.checkpoint_path = m_path
		self.checkpoint_dir = os.path.dirname(self.checkpoint_path)
		self.check_index = self.checkpoint_path + '.index'   #checkpoint裡面的檔案多加了一個.index
		self.model = self._model('  Model')
		
	def _model(self, model_name):
		model = keras.Sequential()
		model.add(keras.layers.LSTM(units=self.neurons, input_shape=self.state_size, activation="sigmoid"))
		model.add(keras.layers.Dropout(0.2))
		model.add(keras.layers.Dense(units=self.neurons, activation="relu"))
		model.add(keras.layers.Dense(units=0.5*self.neurons, activation="relu"))
		model.add(keras.layers.Dense(self.action_size, activation="linear"))
		model.compile(loss="mse", optimizer=keras.optimizers.Adam(lr=0.001))
		#output為各action的機率(要轉換)
		if os.path.exists(self.check_index):
			#如果已經有訓練過，就接著load權重
			print('-'*42+'{} Weights loaded!!'.format(model_name)+'-'*42)
			model.load_weights(self.checkpoint_path)
		else:
			print('-'*43+'Create new model!!'+'-'*43)
		return model

	def act(self, state):
		if not self.is_eval and np.random.rand() <= self.epsilon:
			return random.randrange(self.action_size)
		
		options = self.model.predict(state)
		return np.argmax(options[0]) #array裡面最大值的位置號
		#return options
		
	def expReplay(self, batch_size): #用memory來訓練神經網路
		mini_batch = []

		mini_batch = random.sample(self.memory, batch_size) #隨機取記憶出來訓練
			
		for state, action, reward, next_state, done in mini_batch:
			target = self.model.predict(state)
			if done:
				target[0,action] = reward
			else:
				t = self.model.predict(next_state)   
				target[0,action] = reward + self.gamma * np.amax(t[0])
	
			#checkpoint設定
			cp_callback = tf.keras.callbacks.ModelCheckpoint(
			filepath=self.checkpoint_path,
			save_weights_only=True,
			verbose=0)
				
			self.model.fit(state, target, epochs=1,
			 verbose=0, callbacks = [cp_callback])

		if self.epsilon > self.epsilon_min:
			#貪婪度遞減   
			self.epsilon *= self.epsilon_decay 