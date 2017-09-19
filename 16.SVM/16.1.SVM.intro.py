#!/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
import matplotlib as mpl
import matplotlib.pyplot as plt

from utils import wkzutils


def iris_type(s):
	it = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
	return it[s]


# 'sepal length', 'sepal width', 'petal length', 'petal width'
iris_feature = u'花萼长度', u'花萼宽度', u'花瓣长度', u'花瓣宽度'


def show_accuracy(a, b, tip):
	acc = a.ravel() == b.ravel()
	print tip + '正确率：', np.mean(acc)


if __name__ == "__main__":
	path = wkzutils.get_path_sources("10.iris.data")  # 数据文件路径
	data = np.loadtxt(path, dtype=float, delimiter=',', converters={4: iris_type})
	x, y = np.split(data, (4,), axis=1)
	x = x[:, :2]
	x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, train_size=0.6)

	#（1）C: 目标函数的惩罚系数C，用来平衡分类间隔margin和错分样本的，default C = 1.0；
	#（2）kernel：参数选择有rbf高斯核, linear线性核, poly多项式核, sigmoid是S核,等 默认的是"rbf";
	#（3）degree：if you choose 'Poly' in param 2, this is effective, degree决定了多项式的最高次幂；
	#（4）gamma：【越大越容易过拟合】核函数的系数('Poly', 'RBF' and 'Sigmoid',线性核无效), 默认是gamma = 1 / n_features;
	#（5）coef0：核函数中的独立项，'RBF' and 'Poly'有效；
	#（6）probablity: 可能性估计是否使用(true or false)；
	#（7）shrinking：是否进行启发式；
	#（8）tol（default = 1e - 3）: svm结束标准的精度;
	#（9）cache_size: 制定训练所需要的内存（以MB为单位）；
	#（10）class_weight: 每个类所占据的权重，不同的类设置不同的惩罚参数C, 缺省的话自适应；
	#（11）verbose: 跟多线程有关，不大明白啥意思具体；
	#（12）max_iter: 最大迭代次数，default = 1， if max_iter = -1, no limited;
	#（13）decision_function_shape ： ‘ovo’ 一对一, ‘ovr’ 多对多 or None无, default = None
	#（14）random_state ：用于概率估计的数据重排时的伪随机数生成器的种子。
	#ps：7, 8, 9一般不考虑。

	# clf = svm.SVC(C=0.1, kernel='linear', decision_function_shape='ovr')
	clf = svm.SVC(C=0.8, kernel='rbf', gamma=20, decision_function_shape='ovr')
	clf.fit(x_train, y_train.ravel())

	# 准确率
	print clf.score(x_train, y_train)  # 精度
	y_hat = clf.predict(x_train)
	show_accuracy(y_hat, y_train, '训练集')
	print clf.score(x_test, y_test)
	y_hat = clf.predict(x_test)
	show_accuracy(y_hat, y_test, '测试集')

	# decision_function
	print 'decision_function:\n', clf.decision_function(x_train)
	print '\npredict:\n', clf.predict(x_train)

	# 画图
	x1_min, x1_max = x[:, 0].min(), x[:, 0].max()  # 第0列的范围
	x2_min, x2_max = x[:, 1].min(), x[:, 1].max()  # 第1列的范围
	x1, x2 = np.mgrid[x1_min:x1_max:200j, x2_min:x2_max:200j]  # 生成网格采样点
	grid_test = np.stack((x1.flat, x2.flat), axis=1)  # 测试点
	# print 'grid_test = \n', grid_test
	# Z = clf.decision_function(grid_test)	# 样本到决策面的距离
	# print Z
	grid_hat = clf.predict(grid_test)	   # 预测分类值
	print grid_hat
	grid_hat = grid_hat.reshape(x1.shape)  # 使之与输入的形状相同
	mpl.rcParams['font.sans-serif'] = [u'SimHei']
	mpl.rcParams['axes.unicode_minus'] = False

	cm_light = mpl.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])
	cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])
	x1_min, x1_max = x[:, 0].min(), x[:, 0].max()  # 第0列的范围
	x2_min, x2_max = x[:, 1].min(), x[:, 1].max()  # 第1列的范围
	x1, x2 = np.mgrid[x1_min:x1_max:200j, x2_min:x2_max:200j]  # 生成网格采样点
	grid_test = np.stack((x1.flat, x2.flat), axis=1)  # 测试点
	plt.pcolormesh(x1, x2, grid_hat, cmap=cm_light)

	plt.scatter(x[:, 0], x[:, 1], c=y, edgecolors='k', s=50, cmap=cm_dark)	  # 样本
	plt.scatter(x_test[:, 0], x_test[:, 1], s=120, facecolors='none', zorder=10)	 # 圈中测试集样本
	plt.xlabel(iris_feature[0], fontsize=13)
	plt.ylabel(iris_feature[1], fontsize=13)
	plt.xlim(x1_min, x1_max)
	plt.ylim(x2_min, x2_max)
	plt.title(u'鸢尾花SVM二特征分类', fontsize=15)
	plt.grid()
	plt.show()
