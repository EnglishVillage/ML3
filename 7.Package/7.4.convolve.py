# !/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


if __name__ == "__main__":
	# unpack表示读取数据散开
	stock_max, stock_min, stock_close, stock_amount = np.loadtxt('7.SH600000.txt', delimiter='\t', skiprows=2, usecols=(2, 3, 4, 5), unpack=True)
	N = 100
	stock_close = stock_close[:N]
	print stock_close

	n = 5
	weight = np.ones(n)
	weight /= weight.sum()
	print weight
	# 使用[ 0.2  0.2  0.2  0.2  0.2]做卷积
	stock_sma = np.convolve(stock_close, weight, mode='valid')  # simple moving average

	weight = np.linspace(1, 0, n)
	weight = np.exp(weight)
	weight /= weight.sum()
	print weight
	# 使用[0.31002201  0.24144538  0.18803785  0.14644403  0.11405072]做卷积,内部会进行翻转
	stock_ema = np.convolve(stock_close, weight, mode='valid')  # exponential moving average

	t = np.arange(n-1, N)
	# 使用多项式进行拟合(总共11项,10,9,...0)
	poly = np.polyfit(t, stock_ema, 10)
	print poly
	# 多项式预测值
	stock_ema_hat = np.polyval(poly, t)

	mpl.rcParams['font.sans-serif'] = [u'SimHei']
	mpl.rcParams['axes.unicode_minus'] = False
	plt.plot(np.arange(N), stock_close, 'ro-', linewidth=2, label=u'原始收盘价')
	t = np.arange(n-1, N)
	plt.plot(t, stock_sma, 'b-', linewidth=2, label=u'简单移动平均线')
	plt.plot(t, stock_ema, 'g-', linewidth=2, label=u'指数移动平均线')
	plt.legend(loc='upper right')
	plt.grid(True)
	plt.show()

	plt.figure(figsize=(9, 6))
	plt.plot(np.arange(N), stock_close, 'r-', linewidth=1, label=u'原始收盘价')
	plt.plot(t, stock_ema, 'g-', linewidth=2, label=u'指数移动平均线')
	plt.plot(t, stock_ema_hat, 'm-', linewidth=3, label=u'指数移动平均线估计')
	plt.legend(loc='upper right')
	plt.grid(True)
	plt.show()
