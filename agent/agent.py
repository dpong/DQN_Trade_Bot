from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
import os, random
from collections import deque
import numpy as np
from agent.prioritized_memory import Memory

class Agent:
	def __init__(self, ticker, state_size, neurons, m_path, is_eval=False):
		self.state_size = state_size # normalized previous days
		self.action_size = 4 
		self.neurons = neurons
		self.memory_size = 1000 #記憶長度
		self.memory = Memory(self.memory_size)
		self.gamma = 0.95
		self.epsilon = 1
		self.epsilon_min = 0.01
		self.epsilon_decay = 0.995
		self.batch_size = 32
		self.is_eval = is_eval
		self.checkpoint_path = m_path
		self.checkpoint_dir = os.path.dirname(self.checkpoint_path)
		self.check_index = self.checkpoint_path + '.index'   #checkpoint裡面的檔案多加了一個.index
		self.model = self._model('  Model')
		if is_eval==False:
			self.target_model = self._model(' Target')
		self.cp_callback = self._check_point()
		
	def _model(self, model_name):
		model = keras.Sequential()
		model.add(keras.layers.LSTM(units=self.neurons, input_shape=self.state_size, activation="sigmoid"))
		model.add(keras.layers.Dropout(0.2))
		model.add(keras.layers.Dense(units=2*self.neurons, activation="relu"))
		model.add(keras.layers.Dense(units=self.neurons, activation="relu"))
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

	#設定check point
	def _check_point(self):
		cp_callback = tf.keras.callbacks.ModelCheckpoint(
		filepath=self.checkpoint_path,
		save_weights_only=True,
		verbose=0)
		return cp_callback

	def update_target_model(self):
		self.target_model.set_weights(self.model.get_weights())

	def act(self, state):
		if not self.is_eval and np.random.rand() <= self.epsilon:
			return random.randrange(self.action_size)
		
		options = self.model.predict(state)
		return np.argmax(options[0]) #array裡面最大值的位置號
		#return options

	# save sample (error,<s,a,r,s'>) to the replay memory
	def append_sample(self, state, action, reward, next_state, done):
		target, error = self.get_target_error(state, action, reward, next_state, done)
		self.memory.add(error,(state, action, reward, next_state, done))

	# pick samples from prioritized replay memory (with batch_size)
	def train_model(self):
		if self.epsilon > self.epsilon_min:
			self.epsilon *= self.epsilon_decay #貪婪度遞減  

		mini_batch, idxs, is_weights = self.memory.sample(self.batch_size)

		for i in range(self.batch_size):
			target, error = self.get_target_error(mini_batch[i][0],mini_batch[i][1],mini_batch[i][2],mini_batch[i][3],mini_batch[i][4])
			idx = idxs[i] # update priority
			self.memory.update(idx, error)
			#train
			self.model.fit(mini_batch[i][0], target, epochs = 1,
			 verbose=0, callbacks = [self.cp_callback])

	#迭代Q值
	def get_target_error(self, state, action, reward, next_state, done):
		target = self.model.predict(state)
		old_val = target[0,action]
		if done:
			target[0,action] = reward
		else:
			t = self.target_model.predict(next_state)   
			target[0,action] = reward + self.gamma * np.amax(t[0])
		
		error = abs(old_val - target[0,action])
		return target, error

	