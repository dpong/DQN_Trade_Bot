from functions import *
import sys
from agent.agent import Agent
import caffeine

if len(sys.argv) != 4:
	print('Usage: python3 train.py [stock] [window] [episodes]')
	exit()

caffeine.on(display=False) #電腦不休眠
ticker, window_size, episode_count = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
init_cash = 1000000
commission = 0.003  #千分之三的手續費
stop_pct = 0.1      #停損%數
safe_pct = 0.8      #現金安全水位
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
n_close = 0

for e in range(1, episode_count + 1):
	total_profit, cash, total_reward = 0, init_cash, 0
	inventory = []
	highest_value = np.array([0,0])  #0是多倉位，1是空倉位
	max_drawdown = 0
	for t in range(window_size+1, l):         #前面的資料要來預熱一下
		state = getState(data, t, window_size)
		next_state = getState(data, t + 1, window_size)

		if t == l - 1: #最後一個state
			action = 3
		else:
			action = agent.act(state)
		reward = 0

		#這邊交易的價格用當日的收盤價(t+1)代替，實際交易就是成交價格
		if action == 1 and safe_pct*cash > data[t+1][n_close] * unit: # long
			if len(inventory) > 0 and inventory[0][1]=='short':
				sold_price = inventory.pop(0)
				price = data[t+1][n_close] * (1+commission)
				profit = (sold_price[0] - price) * unit
				total_profit += profit
				reward = profit / (sold_price[0] * unit)
				cash += profit + sold_price[0] * unit
				print("Ep " + str(e) + "/" + str(episode_count)+" %.2f%%" % round(t*(100/l),2) + " Cash: " + formatPrice(cash)
				+ " | Bear: "+ str(len(inventory) * unit) +" | Long: " + formatPrice(price) 
				+ " | Profit: " + formatPrice(profit)+ " | Total Profit: " + formatPrice(total_profit)
				+ " | Reward: " + str(round(reward,2)))
			else:
				price = data[t+1][n_close] * (1+commission)
				cost = price * unit
				cash -= cost
				inventory.append([price,'long']) #存入進場資訊
				print("Ep " + str(e) + "/" + str(episode_count)+" %.2f%%" % round(t*(100/l),2) + " Cash: " + formatPrice(cash)
			 	+ " | Bull: "+ str(len(inventory) * unit) + " | Long : " + formatPrice(price))
		
		elif action == 1 and safe_pct*cash <= data[t+1][n_close] * unit: # cash不足
			action = 0

		elif action == 2 and safe_pct*cash > data[t+1][n_close] * unit: # short
			if len(inventory) > 0 and inventory[0][1]=='long':
				price = data[t+1][n_close] * (1-commission)
				bought_price = inventory.pop(0)
				profit = (price - bought_price[0]) * unit
				total_profit += profit
				reward = profit / (bought_price[0] * unit)
				cash += profit + bought_price[0] * unit
				print("Ep " + str(e) + "/" + str(episode_count)+" %.2f%%" % round(t*(100/l),2) + " Cash: " + formatPrice(cash)
				+ " | Bull: "+ str(len(inventory) * unit) +" | Short: " + formatPrice(price) 
				+ " | Profit: " + formatPrice(profit)+ " | Total Profit: " + formatPrice(total_profit)
				+ " | Reward: " + str(round(reward,2)))
			else: #放空
				price = data[t+1][n_close] * (1-commission)
				cost = price * unit #做空一樣要付出成本，保證金的概念
				cash -= cost
				inventory.append([price,'short']) #存入進場資訊
				print("Ep " + str(e) + "/" + str(episode_count)+" %.2f%%" % round(t*(100/l),2) + " Cash: " + formatPrice(cash)
			 	+ " | Bear: "+ str(len(inventory) * unit) + " | Short : " + formatPrice(price))

		elif action == 2 and safe_pct*cash <= data[t+1][n_close] * unit: 
			action = 0

		elif action == 3 and len(inventory) > 0: #全部平倉
			if inventory[0][1] == 'long':
				account_profit, avg_price = get_long_account(inventory,data[t+1][n_close],commission)
				reward = (account_profit / avg_price) * len(inventory)
				profit = account_profit * unit * len(inventory)
				total_profit += profit
				cash += avg_price * len(inventory) * unit + profit
				print("Ep " + str(e) + "/" + str(episode_count)+" %.2f%%" % round(t*(100/l),2) + " Cash: " + formatPrice(cash)
					+ " | Clean Inventory"+ ' | Profit: ' + formatPrice(profit)
					+ " | Total Profit: " + formatPrice(total_profit)
					+ " | Reward: " + str(round(reward,2)))
			else:
				account_profit, avg_price = get_short_account(inventory,data[t+1][n_close],commission)
				reward = (account_profit / avg_price) * len(inventory)
				profit = account_profit * unit * len(inventory)
				total_profit += profit
				cash += avg_price * len(inventory) * unit + profit
				print("Ep " + str(e) + "/" + str(episode_count)+" %.2f%%" % round(t*(100/l),2) + " Cash: " + formatPrice(cash)
					+ " | Clean Inventory"+ ' | Profit: ' + formatPrice(profit)
					+ " | Total Profit: " + formatPrice(total_profit)
					+ " | Reward: " + str(round(reward,2)))
			inventory = [] #全部平倉
			highest_value[:] = 0	
		
		elif action == 3 and len(inventory) == 0:
			action = 0

		if action == 0: #不動作
			if len(inventory) > 0:
				if inventory[0][1] == 'long':
					account_profit, avg_price = get_long_account(inventory,data[t+1][n_close],commission)
					account_value = account_profit * unit * len(inventory)   #價差乘上有多少單位才是價值
					avg_value = avg_price * unit * len(inventory)
					value_diff = (account_value - highest_value[0]) / avg_value
					if account_value > highest_value[0]: 
						highest_value[0] = account_value
					elif value_diff <= -stop_pct and highest_value[0] > 0:  #帳面獲利減少的懲罰
						reward = value_diff
					elif account_profit / avg_price < -stop_pct:  #帳損超過的懲罰
						reward = account_value / avg_value
					print("Ep " + str(e) + "/" + str(episode_count)+" %.2f%%" % round(t*(100/l),2) + " Cash: " + formatPrice(cash)
					+ " | Bull: "+ str(len(inventory) * unit) + ' | Potential: ' + formatPrice(account_value)
					+ " | Highest: " + str(round(highest_value[0],2))
					+ " | Reward: " + str(round(reward,2)))	
					highest_value[1] = 0	
				else:
					account_profit, avg_price = get_short_account(inventory,data[t+1][n_close],commission)
					account_value = account_profit * unit * len(inventory)
					avg_value = avg_price * unit * len(inventory)
					value_diff = (account_value - highest_value[1]) / avg_value
					if account_value > highest_value[1]: 
						highest_value[1] = account_value
					elif value_diff <= -stop_pct and highest_value[1] > 0:  #帳面獲利減少的懲罰
						reward = value_diff
					elif account_value / avg_value < -stop_pct:  #帳損超過的懲罰
						reward = account_value / avg_value
					print("Ep " + str(e) + "/" + str(episode_count)+" %.2f%%" % round(t*(100/l),2) + " Cash: " + formatPrice(cash)
					+ " | Bear: "+ str(len(inventory) * unit) + ' | Potential: ' + formatPrice(account_value)
					+ " | Highest: " + str(round(highest_value[1],2))
					+ " | Reward: " + str(round(reward,2)))	
					highest_value[0] = 0		
			else:
				print("Ep " + str(e) + "/" + str(episode_count)+" %.2f%%" % round(t*(100/l),2) + " Cash: " + formatPrice(cash)
				+ " | Nuetrual")
				highest_value[:] = 0	

		done = True if t == l - 1 else False
		total_reward += reward
		agent.append_sample(state, action, reward, next_state, done)
		#計算max drawdown
		if len(inventory) > 0:
			inventory_value = get_inventory_value(inventory,data[t+1][n_close],commission)
			inventory_value *= unit
			profolio = inventory_value + cash
		else:
			profolio = cash

		if profolio - init_cash < 0:  #虧損時才做
			drawdown = (profolio - init_cash) / init_cash
			if drawdown < max_drawdown:
				max_drawdown = drawdown

		if done:
			agent.update_target_model()
			print("-"*104)
			print("Episode " + str(e) + "/" + str(episode_count)
			+ " | Cash: " + formatPrice(cash) 
			+ " | Total Profit: " + formatPrice(total_profit)
			+ " | Return Ratio: %.2f%%" % round(100*total_profit/init_cash,2)
			+ " | Max DrawDown: %.2f%%" % round(-max_drawdown*100,2)
			+ " | Total Reward: " + str(round(total_reward,2)))
			print("-"*104)
			if e == episode_count:
				agent.model.save(m_path)
				caffeine.off() #讓電腦回去休眠
		
		if agent.memory.tree.n_entries > agent.batch_size:
			agent.train_model()
			