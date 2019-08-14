import numpy as np
import random
import pandas_datareader as pdr
import pandas as pd
from scipy import special
from sklearn import preprocessing

# prints formatted price
def formatPrice(n):
	return ("-$" if n < 0 else "$") + "{0:.2f}".format(abs(n))

#標準化
def standardscale(x):
	scaler = preprocessing.StandardScaler()
	return scaler.fit_transform(x)

# returns an n-day state representation ending at time t
def getState(data, t, n):
	block = data[t-n-1:t]
	block_1 = block[1:]
	block_0 = block[:n]
	keep = block_1[:,5:]    #後面的資料不計算差值，也不標準化
	res = block_1[:,:5] - block_0[:,:5]
	res = standardscale(res)
	res = np.concatenate((res,keep),axis=1)
	return np.array([res])  #修正input形狀

#model的輸入值起始
def get_shape(data,window_size):
	input_shape = getState(data,window_size+1,window_size)
	return input_shape.shape[1:]

#轉換cash成state資料，設定為1～-1之間
def get_cash(cash,init_cash):
	return cash / init_cash

#轉換profit成state資料，設定為1～-1之間
def get_profit(profit,init_cash):
	return profit/init_cash

#初始化輸入資料
def init_data(df):
	df['shift_close'] = df['Close'].shift(1)   #交易的時候，只知道昨天的收盤價
	df.dropna(how='any',inplace=True)
	df['Cash'] = 1     #起始比例
	df['Profit'] = 0
	data = df[['shift_close','Open','High','Low','Volume','Cash','Profit']].values
	return data
