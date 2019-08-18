# DQN_Trade_Bot

用Deep Q Network為基礎的交易機器人

經過了一連串的試驗和常識，終於有了初步的成果了！

Bot有四個動作：
0.不動
1.買一單位
2.賣一單位
3.全部平倉

Tensorflow 2.0Beta如果用LSTMreturn sequance就會報警告。
然後activation function也要用sigmoid來避免報錯，這就等待修正。

原本以為是我自己reward系統給的不好，結果是神經搭建的問題，
縮小規模可以提高訓練效率，但也要花點時間就是了。

下面回測結果是拿TSLA股票2018年整年的資料來訓練，因為是個盤整期間，沒有明顯的趨勢。
總共訓練了50次，用我的macbook pro大概要花個5小時。
回測的資料是從有資料開始到最近(2010-1-4～2019-8-16)，
所以訓練跟回測的尺度是不同。

![image](https://github.com/dpong/DQN_Trade_Bot/blob/master/fig_result_50run.png)

大概快10年下來賺了336%，這樣的結果還滿驚人的。
等我把訓練次數拉到100次看看有沒有差異。

