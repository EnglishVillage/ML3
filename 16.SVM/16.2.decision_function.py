#!/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
from sklearn import svm
from scipy import stats
from sklearn.metrics import accuracy_score
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


def extend(a, b, r):
	x = a - b
	m = (a + b) / 2
	return m - r * x / 2, m + r * x / 2


if __name__ == "__main__":
	np.random.seed(0)
	N = 20
	x = np.empty((4 * N, 2))
	#  设置均值
	means = [(-1, 1), (1, 1), (1, -1), (-1, -1)]
	# 设置方差
	sigmas = [np.eye(2), 2 * np.eye(2), np.diag((1, 2)), np.array(((2, 1), (1, 2)))]
	for i in range(4):
		# 多元正态随机变量的类
		mn = stats.multivariate_normal(means[i], sigmas[i] * 0.3)
		# 产生随机变量
		# temp=mn.rvs(N)	# 生成20*2的随机变量数组
		x[i*N:(i+1)*N, :] = mn.rvs(N)	# x是80*2的数组
	a = np.array((0, 1, 2, 3)).reshape((-1, 1))
	# 拷贝数组里的元素多次,变成一个新的数组;将多维数组压平成为一个一维的数组
	y = np.tile(a, N).flatten()
	# exit()
	# decision_function_shape:‘ovo’ 一对一, ‘ovr’ 多对多  or None 无, default=None
	# clf = svm.SVC(C=1, kernel='rbf', gamma=1, decision_function_shape='ovr')
	clf = svm.SVC(C=1, kernel='rbf', gamma=1, decision_function_shape='ovo')
	# clf = svm.SVC(C=1, kernel='linear', decision_function_shape='ovr')
	clf.fit(x, y)
	y_hat = clf.predict(x)
	acc = accuracy_score(y, y_hat)
	np.set_printoptions(suppress=True)
	# ovr多个值里选择最大的那个作为分类
	print u'预测正确的样本个数：%d，正确率：%.2f%%' % (round(acc * 4 * N), 100 * acc)
	# decision_function
	print clf.decision_function(x)
	# print y_hat

	x1_min, x2_min = np.min(x, axis=0)
	x1_max, x2_max = np.max(x, axis=0)
	x1_min, x1_max = extend(x1_min, x1_max, 1.05)
	x2_min, x2_max = extend(x2_min, x2_max, 1.05)
	x1, x2 = np.mgrid[x1_min:x1_max:500j, x2_min:x2_max:500j]
	x_test = np.stack((x1.flat, x2.flat), axis=1)
	y_test = clf.predict(x_test)
	y_test = y_test.reshape(x1.shape)
	cm_light = mpl.colors.ListedColormap(['#FF8080', '#A0FFA0', '#6060FF', '#F080F0'])
	cm_dark = mpl.colors.ListedColormap(['r', 'g', 'b', 'm'])
	mpl.rcParams['font.sans-serif'] = [u'SimHei']
	mpl.rcParams['axes.unicode_minus'] = False
	plt.figure(facecolor='w')
	plt.pcolormesh(x1, x2, y_test, cmap=cm_light)
	plt.scatter(x[:, 0], x[:, 1], s=40, c=y, cmap=cm_dark, alpha=0.7)
	plt.xlim((x1_min, x1_max))
	plt.ylim((x2_min, x2_max))
	plt.grid(b=True)
	plt.tight_layout(pad=2.5)
	plt.title(u'SVM多分类方法：One/One or One/Other', fontsize=18)
	plt.show()
