from functions import *
from agent.agent import Agent
from action import Action
import sys


ticker, window_size = 'TSLA', 20
init_cash = 1000000
commission = 0.003  #千分之三的手續費
stop_pct = 0.1      #停損%數
safe_pct = 0.8      #現金安全水位      
#要給checkpoint個路徑
c_path = "models/{}/training.ckpt".format(ticker)
#取得歷史資料
df = pdr.DataReader('{}'.format(ticker),'yahoo',start='2018-1-1',end='2019-8-1')
unit = get_unit(df['Close'].mean(),init_cash) #目前都是操作固定單位
trading = Action(unit)
#資料整合轉換
data = init_data(df)
#給agent初始化輸入的緯度
input_shape, neurons = get_shape(data[:window_size+1],window_size)
agent = Agent(ticker, input_shape, neurons, c_path, is_eval=True)

l = len(data) - 1
n_close = 0

total_profit = 0
inventory = []
cash = init_cash
max_drawdown = 0

e ,episode_count = 1,1

for t in range(window_size+1, l):         #前面的資料要來預熱一下
	state = getState(data, t, window_size)
	next_state = getState(data, t + 1, window_size)

	if t == l - 1: #最後一個state
		action = 3
	else:
		action = agent.act(state)
	reward = 0

	#這邊交易的價格用當日的收盤價(t+1)代替，實際交易就是成交價格
	if action == 1 and len(inventory) > 0 and inventory[0][1]=='short':
		cash, inventory, total_profit = trading._long_clean(data[t+1][n_close] , cash, inventory, total_profit, e, episode_count,t,l)
		
	elif action == 1 and len(inventory) > 0 and inventory[0][1]=='long':
		if trading.safe_margin * cash > data[t+1][n_close] * unit:
			cash, inventory, total_profit = trading._long_new(data[t+1][n_close] , cash, inventory, total_profit, e, episode_count,t,l)
		else:
			action = 0

	elif action == 1 and len(inventory) == 0:
		if trading.safe_margin * cash > data[t+1][n_close] * unit:
			cash, inventory, total_profit = trading._long_new(data[t+1][n_close] , cash, inventory, total_profit, e, episode_count,t,l)
		else:
			action = 0

	elif action == 2 and len(inventory) > 0 and inventory[0][1]=='long':
		cash, inventory, total_profit = trading._short_clean(data[t+1][n_close] , cash, inventory, total_profit, e, episode_count,t,l)

	elif action == 2 and len(inventory) > 0 and inventory[0][1]=='short':
		if trading.safe_margin * cash > data[t+1][n_close] * unit:
			cash, inventory, total_profit = trading._short_new(data[t+1][n_close] , cash, inventory, total_profit, e, episode_count,t,l)
		else:
			action = 0
		
	elif action == 2 and len(inventory) == 0:
		if trading.safe_margin * cash > data[t+1][n_close] * unit:
			cash, inventory, total_profit = trading._short_new(data[t+1][n_close] , cash, inventory, total_profit, e, episode_count,t,l)
		else:
			action = 0

	elif action == 3 and len(inventory) > 0:
		cash, inventory, total_profit = trading._clean_inventory(data[t+1][n_close] , cash, inventory, total_profit, e, episode_count,t,l)
		
	elif action == 3 and len(inventory) == 0:
		action = 0

	if action == 0: #不動作
		trading._hold(data[t+1][n_close] , cash, inventory, e, episode_count,t,l)
	
	done = True if t == l - 1 else False
	#計算max drawdown
	if len(inventory) > 0:
		inventory_value = get_inventory_value(inventory,data[t+1][n_close],trading.commission)
		inventory_value *= trading.unit
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
	

