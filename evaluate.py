from functions import *
from agent.agent import Agent
import sys


ticker, window_size = 'TSLA', 20
init_cash = 1000000
commission = 0.003  #千分之三的手續費
stop_pct = 0.1      #停損%數
safe_pct = 0.8      #現金安全水位      
#要給checkpoint個路徑
c_path = "models/{}/training.ckpt".format(ticker)
#取得歷史資料
df = pdr.DataReader('{}'.format(ticker),'yahoo')
unit = get_unit(df['Close'].mean(),init_cash) #目前都是操作固定單位
#資料整合轉換
data = init_data(df)
#給agent初始化輸入的緯度
input_shape, neurons = get_shape(data[:window_size+1],window_size)
agent = Agent(ticker, input_shape, neurons, c_path, is_eval=True)

l = len(data) - 1
batch_size = 32
n_close = 0

total_profit = 0
inventory = []
cash = init_cash
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
			profit = (sold_price[0] - data[t+1][n_close]*(1+commission)) *unit
			total_profit += profit
			reward = profit / (sold_price[0]*unit)
			cash += profit + sold_price[0]*unit
			print(" Cash: " + formatPrice(cash)
			+ " | Bear: "+ str(len(inventory) * unit) +" | Long: " + formatPrice(data[t+1][0]) 
			+ " | Profit: " + formatPrice(profit)+ " | Total Profit: " + formatPrice(total_profit))
		else:
			price = data[t+1][n_close] * (1+commission)
			cost = price * unit
			cash -= cost
			inventory.append([price,'long']) #存入進場資訊
			print(" Cash: " + formatPrice(cash)
		 	+ " | Bull: "+ str(len(inventory) * unit) + " | Long : " + formatPrice(data[t+1][n_close]))
		
	elif action == 1 and safe_pct*cash <= data[t+1][n_close] * unit: # cash不足
		action = 0

	elif action == 2 and safe_pct*cash > data[t+1][n_close] * unit: # short
		if len(inventory) > 0 and inventory[0][1]=='long':
			bought_price = inventory.pop(0)
			profit = (data[t+1][n_close]*(1-commission) - bought_price[0])*unit
			total_profit += profit
			reward = profit / (bought_price[0]*unit)
			cash += profit + bought_price[0]*unit
			print(" Cash: " + formatPrice(cash)
			+ " | Bull: "+ str(len(inventory) * unit) +" | Short: " + formatPrice(data[t+1][0]) 
			+ " | Profit: " + formatPrice(profit)+ " | Total Profit: " + formatPrice(total_profit))
		else: #放空
			price = data[t+1][n_close] * (1-commission)
			cost = price * unit #做空一樣要付出成本，保證金的概念
			cash -= cost
			inventory.append([price,'short']) #存入進場資訊
			print(" Cash: " + formatPrice(cash)
		 	+ " | Bear: "+ str(len(inventory) * unit) + " | Short : " + formatPrice(data[t+1][n_close]))

	elif action == 2 and safe_pct*cash <= data[t+1][n_close] * unit: #手上沒現貨
		action = 0

	elif action == 3 and len(inventory) > 0: #全部平倉
		if inventory[0][1] == 'long':
			account_profit = get_long_account(inventory,data[t+1][n_close],commission)
			reward = account_profit * len(inventory)
			account_profit, avg_price = get_long_account(inventory,data[t+1][n_close],commission)
			profit = account_profit * unit *len(inventory)
			total_profit += profit
			cash += avg_price * len(inventory) * unit + profit
			print(" Cash: " + formatPrice(cash)
				+ " | Clean Inventory"+ ' | Profit: ' + formatPrice(profit)
				+ " | Total Profit: " + formatPrice(total_profit))
		else:
			account_profit = get_short_account(inventory,data[t+1][n_close],commission)
			reward = account_profit * len(inventory)
			account_profit, avg_price = get_short_account(inventory,data[t+1][n_close],commission)
			profit = account_profit * unit *len(inventory)
			total_profit += profit
			cash += avg_price * unit * len(inventory) + profit
			print(" Cash: " + formatPrice(cash)
				+ " | Clean Inventory"+ ' | Profit: ' + formatPrice(profit)
				+ " | Total Profit: " + formatPrice(total_profit))
		inventory = [] #全部平倉

	elif action == 3 and len(inventory) == 0:
			action = 0	

	if action == 0: #不動作
		if len(inventory) > 0:
			if inventory[0][1] == 'long':
				account_profit, avg_price = get_long_account(inventory,data[t+1][n_close],commission)
				print(" Cash: " + formatPrice(cash)
				+ " | Bull: "+ str(len(inventory) * unit) + ' | Potential: ' + formatPrice(account_profit * unit * len(inventory)))		
			else:
				account_profit, avg_price = get_short_account(inventory,data[t+1][n_close],commission)
				print(" Cash: " + formatPrice(cash)
				+ " | Bear: "+ str(len(inventory) * unit) + ' | Potential: ' + formatPrice(account_profit * unit * len(inventory)))		
		else:
			print(" Cash: " + formatPrice(cash)
			+ " | Nuetrual")

	done = True if t == l - 1 else False
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
		#agent.update_target_model()
		print("-"*104)
		print(
		" Cash: " + formatPrice(cash) 
		+ " | Total Profit: " + formatPrice(total_profit)
		+ " | Return Ratio: %.2f%%" % round(100*total_profit/init_cash,2)
			+ " | Max DrawDown: %.2f%%" % round(-max_drawdown*100,2))
		print("-"*104)
	

