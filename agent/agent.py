from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
import os, random
from collections import deque
import numpy as np

class Agent:
	def __init__(self, ticker, state_size, m_path, is_eval=False):
		self.state_size = state_size # normalized previous days
		self.action_size = 3 # sit, buy, sell
		self.memory = deque(maxlen=1000) #記憶長度
		self.inventory = []
		self.gamma = 0.95
		self.epsilon = 1.0
		self.epsilon_min = 0.01
		self.epsilon_decay = 0.995
		self.is_eval = is_eval
		self.checkpoint_path = m_path
		self.checkpoint_dir = os.path.dirname(self.checkpoint_path)
		self.check_index = self.checkpoint_path + '.index'   #checkpoint裡面的檔案多加了一個.index
		self.model = self._model('  Model')
		if not self.is_eval:
			self.target_model = self._model(' Target')

	def _model(self, model_name):
		self.optimizer = keras.optimizers.Adam(lr=0.001)  #學習率0.001
		model = keras.Sequential()
		model.add(keras.layers.LSTM(units=64, input_shape=self.state_size, activation="relu",return_sequences=True))
		model.add(keras.layers.LSTM(units=32, activation="relu", return_sequences=True))
		model.add(keras.layers.LSTM(units=8, activation="relu"))
		model.add(keras.layers.Dense(self.action_size, activation="linear"))
		model.compile(loss="mse", optimizer=self.optimizer)
		#output為各action的機率(要轉換)
		if os.path.exists(self.check_index):
			#如果已經有訓練過，就接著load權重
			print('-'*30+'{} Weights loaded!!'.format(model_name)+'-'*30)
			model.load_weights(self.checkpoint_path)
		else:
			print('-'*31+'Create new model!!'+'-'*31)
		return model

	def update_target_model(self):
        # copy weights from model to target_model
		self.target_model.set_weights(self.model.get_weights())

	def act(self, state):
		if not self.is_eval and np.random.rand() <= self.epsilon:
			return random.randrange(self.action_size)
		
		options = self.model.predict(state)
		return np.argmax(options[0]) #array裡面最大值的位置號

	def expReplay(self, batch_size): #用memory來訓練神經網路
		mini_batch = []
		l = len(self.memory)
		mini_batch = random.sample(self.memory, batch_size) #隨機取記憶出來訓練
			
		for state, action, reward, next_state, done in mini_batch:
			target = self.model.predict(state)
			if done:
				target[0][action] = reward
			else:
				t = self.target_model.predict(next_state)[0]   #DDQN
				target[0][action] = reward + self.gamma * np.amax(t)

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