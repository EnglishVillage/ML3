#!/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import GridSearchCV

from utils import wkzutils


def read_file():
	path = '10.Advertising.csv'
	# # 手写读取数据 - 请自行分析，在10.2.Iris代码中给出类似的例子
	# f = file(path)
	# x = []
	# y = []
	# for i, d in enumerate(f):
	# 	if i == 0:
	# 		continue
	# 	d = d.strip()
	# 	if not d:
	# 		continue
	# 	d = map(float, d.split(','))
	# 	x.append(d[1:-1])
	# 	y.append(d[-1])
	# pprint(x)
	# pprint(y)
	# # 转成ndarray的格式
	# x = np.array(x)
	# y = np.array(y)
	# # Python自带库
	# f = file(path, 'rb')
	# print f
	# d = csv.reader(f)
	# for line in d:
	#	 print line
	# f.close()
	# # numpy读入
	# p = np.loadtxt(path, delimiter=',', skiprows=1)
	# print p
	# print type(p)
	# print '\n\n===============\n\n'
	# exit()
	# pandas读入
	data = pd.read_csv(path)  # TV、Radio、Newspaper、Sales
	# x = data[['TV', 'Radio', 'Newspaper']]
	x = data[['TV', 'Radio']]
	y = data['Sales']
	print(data.head())  # pandas.core.frame.DataFrame
	# print x	#pandas.core.frame.DataFrame
	# print y
	# exit()
	return data, x, y

if __name__ == "__main__":
	# 读取文件
	data, x, y = wkzutils.read_file_csv("10.Advertising.csv", ['TV', 'Radio', 'Newspaper'], "Sales")

	# 返回的是一个list列表,random_state设置相同之後,下次运行会得到同一次的训练数据
	x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=1)

	alpha_can = np.logspace(-3, 2, 10)
	np.set_printoptions(suppress=True)
	# print(alpha_can)
	# exit()

	model = Lasso()	# L1正则
	# model = Ridge()		# L2一般使用Ridge效果更好些
	# 使用5折交叉验证(默认3折)挑选最优的Lasso/Ridge中alpha参数
	nice_model = GridSearchCV(model, param_grid={'alpha': alpha_can}, cv=5)

	nice_model.fit(x_train, y_train)

	# 最好的alpha参数,它是从上面我们提供的等比数列中得到的,必须写在训练模型之後
	print '超参数：\n', nice_model.best_params_

	y_hat = nice_model.predict(np.array(x_test))

	wkzutils.print_statistics(x_test,y_test,y_hat,nice_model.score)

	wkzutils.show_predict(x_test,y_test,y_hat)

