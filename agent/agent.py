from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import os, random
from collections import deque
import numpy as np
from agent.prioritized_memory import Memory
from agent.dueling_model import Dueling_model

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
		ddqn = Dueling_model()
		model = ddqn.build_model(self.state_size,self.neurons, self.action_size)
		if os.path.exists(self.check_index):
			#如果已經有訓練過，就接著load權重
			print('-'*52+'{} Weights loaded!!'.format(model_name)+'-'*52)
			model.load_weights(self.checkpoint_path)
		else:
			print('-'*53+'Create new model!!'+'-'*53)
		return model

	#設定check point
	def _check_point(self):
		cp_callback = tf.keras.callbacks.ModelCheckpoint(
		filepath=self.checkpoint_path,
		save_weights_only=True,
		verbose=0)
		return cp_callback

	#把model的權重傳給target model
	def update_target_model(self):
		self.target_model.set_weights(self.model.get_weights())

	def act(self, state):
		if not self.is_eval and np.random.rand() <= self.epsilon:
			return random.randrange(self.action_size)
		
		options = self.model.predict(state)
		return np.argmax(options[0]) #array裡面最大值的位置號
		#return options

	# Prioritized experience replay
	# save sample (error,<s,a,r,s'>) to the replay memory
	def append_sample(self, state, action, reward, next_state, done):
		target, error = self.get_target_n_error(state, action, reward, next_state, done)
		self.memory.add(error,(state, action, reward, next_state, done))

	def train_model(self):
		if self.epsilon > self.epsilon_min:
			self.epsilon *= self.epsilon_decay  #貪婪度遞減

		# pick samples from prioritized replay memory (with batch_size)
		mini_batch, idxs, is_weights = self.memory.sample(self.batch_size)

		for i in range(self.batch_size):
			target, error = self.get_target_n_error(
				mini_batch[i][0], #state
				mini_batch[i][1], #action
				mini_batch[i][2], #reward
				mini_batch[i][3], #next_state
				mini_batch[i][4]  #done
				)

			idx = idxs[i] # update priority
			self.memory.update(idx, error)
			#train model
			self.model.fit(mini_batch[i][0], target, epochs = 1,
			 verbose=0, callbacks = [self.cp_callback])

	# 更新Q和error
	def get_target_n_error(self, state, action, reward, next_state, done):
		#主model動作
		result = self.model.predict(state)
		old_result = result[0,action]
		next_result = self.model.predict(next_state)
		next_action = np.argmax(next_result[0])
		#target model動作
		t_next_result = self.target_model.predict(next_state)
		#更新Q值: Double DQN的概念
		result[0,action] = reward
		if not done:
			result[0,action] += self.gamma * t_next_result[0,next_action]
		#計算error給PER
		error = abs(old_val - result[0,action])
		return result, error



	