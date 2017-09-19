#!/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn import preprocessing
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from utils import wkzutils

if __name__ == "__main__":
	x, y = wkzutils.read_file_csv2("10.iris.data")
	show_graph=False

	# 仅使用前两列特征
	if show_graph:
		x = x[:, :2]

	# 不做标准化,效果会不一样
	# lr = LogisticRegression()
	# 管道:先做标准化,然做logistic回归
	lr = Pipeline([('sc', StandardScaler()), ('clf', LogisticRegression())])
	# y是一个[150,1]的矩阵,y.ravel()则是一个[150,]行向量
	# lr.fit(x, y.ravel())
	lr.fit(x, y)

	y_hat = lr.predict(x)

	# 判断的概率情况
	y_hat_prob = lr.predict_proba(x)
	np.set_printoptions(suppress=True)
	print 'y_hat = \n', y_hat
	print 'y_hat_prob = \n', y_hat_prob
	print u'准确度：%.2f%%' % (100 * np.mean(y_hat == y.ravel()))


	if show_graph:
		# 画图
		N, M = 500, 500  # 横纵各采样多少个值
		x1_min, x1_max = x[:, 0].min(), x[:, 0].max()  # 第0列的范围
		x2_min, x2_max = x[:, 1].min(), x[:, 1].max()  # 第1列的范围
		t1 = np.linspace(x1_min, x1_max, N)
		t2 = np.linspace(x2_min, x2_max, M)
		x1, x2 = np.meshgrid(t1, t2)  # 生成网格采样点
		x_test = np.stack((x1.flat, x2.flat), axis=1)  # 测试点

		# # 无意义，只是为了凑另外两个维度
		# x3 = np.ones(x1.size) * np.average(x[:, 2])
		# x4 = np.ones(x1.size) * np.average(x[:, 3])
		# x_test = np.stack((x1.flat, x2.flat, x3, x4), axis=1)  # 测试点

		mpl.rcParams['font.sans-serif'] = [u'simHei']
		mpl.rcParams['axes.unicode_minus'] = False
		cm_light = mpl.colors.ListedColormap(['#77E0A0', '#FF8080', '#A0A0FF'])
		cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])
		y_hat = lr.predict(x_test)  # 预测值
		y_hat = y_hat.reshape(x1.shape)  # 使之与输入的形状相同
		plt.figure(facecolor='w')
		plt.pcolormesh(x1, x2, y_hat, cmap=cm_light)  # 预测值的显示
		plt.scatter(x[:, 0], x[:, 1], c=y, edgecolors='k', s=50, cmap=cm_dark)  # 样本的显示
		plt.xlabel(u'花萼长度', fontsize=14)
		plt.ylabel(u'花萼宽度', fontsize=14)
		plt.xlim(x1_min, x1_max)
		plt.ylim(x2_min, x2_max)
		plt.grid()
		plt.title(u'鸢尾花Logistic回归分类效果 - 标准化', fontsize=17)
		plt.show()
