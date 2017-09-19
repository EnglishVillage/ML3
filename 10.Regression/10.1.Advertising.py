#!/usr/bin/python
# -*- coding:utf-8 -*-


"""
线性回归
对广告投资进行回归分析
"""

import csv
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from pprint import pprint

from utils import wkzutils


if __name__ == "__main__":
	# 读取文件
	data, x, y = wkzutils.read_file_csv("10.Advertising.csv", ['TV', 'Radio', 'Newspaper'], "Sales")

	# 图表显示原数据分布情况
	wkzutils.show_data(data, y, ["TV", "Radio", "Newspaper"])

	# 返回的是一个list列表,random_state设置相同之後,下次运行会得到同一次的训练数据
	x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=1)
	# print x_train.head()
	# print y_train.head()

	# 使用默认线性回归模型
	linreg = LinearRegression()
	model = linreg.fit(x_train, y_train)
	print model
	# fit_intercept:是否来拟合它的截矩项,normalize:是否对数据进行标准化,copy_X:对X是否拷贝,n_jobs:CPU数
	# LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
	# coef_是[w1,w2,...wn]多元线性方程的值
	print "coef:", linreg.coef_
	# 截距:w0的值
	print "intercept:", linreg.intercept_

	# 预测值
	# y_hat = linreg.predict(np.array(x_test))
	y_hat = linreg.predict(x_test)
	# 打印mse,rmse,R^2
	wkzutils.print_statistics(x_test,y_test,y_hat,linreg.score)

	# 图表显示预测情况
	wkzutils.show_predict(x_test, y_test, y_hat)
