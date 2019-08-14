from functions import *
from agent.agent import Agent
import sys

ticker, window_size, episode_count = 'AAPL', 20, 3
init_cash = 10000
commission = 0.003  #千分之三的手續費
unit = 10            #都是操作固定單位
#要給checkpoint個路徑
c_path = "models/{}/training.ckpt".format(ticker)
m_path = "models/{}/model.h5".format(ticker)
#取得歷史資料，沒給時間就是從有資料到最近
df = pdr.DataReader('{}'.format(ticker),'yahoo',start='2018-1-1',end='2019-1-1')
#資料整合轉換
data = init_data(df)

#給agent初始化輸入的緯度
input_shape = get_shape(data[:window_size+1],window_size)
agent = Agent(ticker, input_shape, c_path)

l = len(data)-1
batch_size = 32


for e in range(1, episode_count + 1):
	total_profit = 0
	agent.inventory = []
	cash = init_cash
	data[:,5], data[:,6] = 0,0   #cash跟profit歸零

	for t in range(window_size+1, l):         #前面的資料要來預熱一下
		state = getState(data, t, window_size)
		action = agent.act(state)
		next_state = getState(data, t + 1, window_size)
		reward = 0
		#這邊交易的價格用當日的收盤價(t+1)代替，實際交易就是成交價格
		if action == 1 and cash > data[t+1][0] * unit: # long
			cost = (1+commission) * data[t+1][0] * unit     #費用
			cash -= cost
			data[t][5] = 
			agent.inventory.append(data[t+1][0])     #只放入價格
			print("Ep " + str(e) + "/" + str(episode_count)+" %.2f%%" % round(t*(100/l),2) + " Cash: " + formatPrice(cash)
			 + " | Holding: "+ str(len(agent.inventory) * unit) + " | Long : " + formatPrice(data[t+1][0]))
		
		elif action ==1 and cash <= data[t+1][0] * unit: # no cash
			action = 3   #把action更新成看戲

		elif action == 2 and len(agent.inventory) > 0: # short
			bought_price = agent.inventory.pop(0)
			profit = ((1-commission*2) * data[t+1][0] - bought_price) * unit 
			cash += data[t+1][0] * unit * (1 - commission)    #為了算profit所以重複扣了手續費，這邊要加回cash裡
			if profit > 0:
				reward = 1
			else:
				reward = -1 
			total_profit += profit
			print("Ep " + str(e) + "/" + str(episode_count)+" %.2f%%" % round(t*(100/l),2) + " Cash: " + formatPrice(cash)
			+ " | Holding: "+ str(len(agent.inventory) * unit) +" | Short: " + formatPrice(data[t+1][0]) 
			+ " | Profit: " + formatPrice(profit)+ " | Total Profit: " + formatPrice(total_profit))

		

		done = True if t == l - 1 else False
		agent.memory.append((state, action, reward, next_state, done))

		if done:
			agent.update_target_model()
			print("-"*80)
			print("Episode " + str(e) + "/" + str(episode_count)
			+ " | Profolio: " + formatPrice(cash + data[t+1][0] * len(agent.inventory) * unit) 
			+ " | Total Profit: " + formatPrice(total_profit))
			print("-"*80)
			if e == episode_count:
				agent.model.save(m_path)

		if len(agent.memory) > batch_size:
			agent.expReplay(batch_size)
