from functions import *
import sys
from agent.agent import Agent
import caffeine

caffeine.on(display=False) #電腦不休眠
ticker, window_size, episode_count = 'TSLA', 20, 50
init_cash = 1000000
commission = 0.003  #千分之三的手續費      
#要給checkpoint個路徑
c_path = "models/{}/training.ckpt".format(ticker)
m_path = "models/{}/model.h5".format(ticker)
#取得歷史資料
df = pdr.DataReader('{}'.format(ticker),'yahoo',start='2018-1-1',end='2019-1-1')
unit = get_unit(df['Close'].mean(),init_cash) #目前都是操作固定單位
#資料整合轉換
data = init_data(df)
#給agent初始化輸入的緯度
input_shape, neurons = get_shape(data[:window_size+1],window_size)
agent = Agent(ticker, input_shape, neurons, c_path)

l = len(data) -1
batch_size = 32
n_close = 0

for e in range(1, episode_count + 1):
	total_profit = 0
	inventory = []
	cash = init_cash
	total_reward = 0
	for t in range(window_size+1, l):         #前面的資料要來預熱一下
		state = getState(data, t, window_size)
		next_state = getState(data, t + 1, window_size)

		if t == l - 1: #最後一個state
			action = 3
		else:
			action = agent.act(state)
		reward = 0

		#這邊交易的價格用當日的收盤價(t+1)代替，實際交易就是成交價格
		if action == 1 and cash > data[t+1][n_close] * unit: # long
			if len(inventory) > 0 and inventory[0][1]=='short':
				sold_price = inventory.pop(0)
				price = data[t+1][n_close] * (1+commission)
				profit = (sold_price[0] - price) * unit
				total_profit += profit
				reward = profit / (sold_price[0] * unit)
				cash += profit + sold_price[0] * unit
				print("Ep " + str(e) + "/" + str(episode_count)+" %.2f%%" % round(t*(100/l),2) + " Cash: " + formatPrice(cash)
				+ " | Bear: "+ str(len(inventory) * unit) +" | Long: " + formatPrice(price) 
				+ " | Profit: " + formatPrice(profit)+ " | Total Profit: " + formatPrice(total_profit))
			else:
				price = data[t+1][n_close] * (1+commission)
				cost = price * unit
				cash -= cost
				inventory.append([price,'long']) #存入進場資訊
				print("Ep " + str(e) + "/" + str(episode_count)+" %.2f%%" % round(t*(100/l),2) + " Cash: " + formatPrice(cash)
			 	+ " | Bull: "+ str(len(inventory) * unit) + " | Long : " + formatPrice(price))
		
		elif action == 1 and cash <= data[t+1][n_close] * unit: # cash不足
			pass

		elif action == 2 and cash > data[t+1][n_close] * unit: # short
			if len(inventory) > 0 and inventory[0][1]=='long':
				price = data[t+1][n_close] * (1-commission)
				bought_price = inventory.pop(0)
				profit = (price - bought_price[0]) * unit
				total_profit += profit
				reward = profit / (bought_price[0] * unit)
				cash += profit + bought_price[0] * unit
				print("Ep " + str(e) + "/" + str(episode_count)+" %.2f%%" % round(t*(100/l),2) + " Cash: " + formatPrice(cash)
				+ " | Bull: "+ str(len(inventory) * unit) +" | Short: " + formatPrice(price) 
				+ " | Profit: " + formatPrice(profit)+ " | Total Profit: " + formatPrice(total_profit))
			else: #放空
				price = data[t+1][n_close] * (1-commission)
				cost = price * unit #做空一樣要付出成本，保證金的概念
				cash -= cost
				inventory.append([price,'short']) #存入進場資訊
				print("Ep " + str(e) + "/" + str(episode_count)+" %.2f%%" % round(t*(100/l),2) + " Cash: " + formatPrice(cash)
			 	+ " | Bear: "+ str(len(inventory) * unit) + " | Short : " + formatPrice(price))

		elif action == 2 and cash <= data[t+1][n_close] * unit: #手上沒現貨
			pass

		elif action == 3 and len(inventory) > 0: #全部平倉
			if inventory[0][1] == 'long':
				account_profit, avg_price = get_long_account(inventory,data[t+1][n_close],commission)
				reward = (account_profit / avg_price) * len(inventory)
				profit = account_profit * unit * len(inventory)
				total_profit += profit
				cash += avg_price * len(inventory) * unit + profit
				print("Ep " + str(e) + "/" + str(episode_count)+" %.2f%%" % round(t*(100/l),2) + " Cash: " + formatPrice(cash)
					+ " | Clean Inventory"+ ' | Profit: ' + formatPrice(profit)
					+ " | Total Profit: " + formatPrice(total_profit))
			else:
				account_profit, avg_price = get_short_account(inventory,data[t+1][n_close],commission)
				reward = (account_profit / avg_price) * len(inventory)
				profit = account_profit * unit * len(inventory)
				total_profit += profit
				cash += avg_price * len(inventory) * unit + profit
				print("Ep " + str(e) + "/" + str(episode_count)+" %.2f%%" % round(t*(100/l),2) + " Cash: " + formatPrice(cash)
					+ " | Clean Inventory"+ ' | Profit: ' + formatPrice(profit)
					+ " | Total Profit: " + formatPrice(total_profit))
			inventory = [] #全部平倉

		if action == 0: #不動作
			if len(inventory) > 0:
				if inventory[0][1] == 'long':
					account_profit, avg_price = get_long_account(inventory,data[t+1][n_close],commission)
					print("Ep " + str(e) + "/" + str(episode_count)+" %.2f%%" % round(t*(100/l),2) + " Cash: " + formatPrice(cash)
					+ " | Bull: "+ str(len(inventory) * unit) + ' | Potential: ' + formatPrice(account_profit * unit * len(inventory)))		
				else:
					account_profit, avg_price = get_short_account(inventory,data[t+1][n_close],commission)
					print("Ep " + str(e) + "/" + str(episode_count)+" %.2f%%" % round(t*(100/l),2) + " Cash: " + formatPrice(cash)
					+ " | Bear: "+ str(len(inventory) * unit) + ' | Potential: ' + formatPrice(account_profit * unit * len(inventory)))		
			else:
				print("Ep " + str(e) + "/" + str(episode_count)+" %.2f%%" % round(t*(100/l),2) + " Cash: " + formatPrice(cash)
				+ " | Nuetrual")

		done = True if t == l - 1 else False
		#放大reward來加速訓練
		reward *= 100 
		total_reward += reward
		agent.memory.append((state, action, reward, next_state, done))
		
		if done:
			#agent.update_target_model()
			print("-"*80)
			print("Episode " + str(e) + "/" + str(episode_count)
			+ " | Cash: " + formatPrice(cash) 
			+ " | Total Profit: " + formatPrice(total_profit)
			+ " | Return Ratio: " + str(round(total_profit/init_cash,2))
			+ " | Total Reward: " + str(round(total_reward,2)))
			print("-"*80)
			if e == episode_count:
				agent.model.save(m_path)
				caffeine.off() #讓電腦回去休眠

		if len(agent.memory) > batch_size:
			agent.expReplay(batch_size)
