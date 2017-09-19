#!/usr/bin/python
# -*- coding:utf-8 -*-

import os, sys, re, time, math
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import preprocessing

mpl.rcParams['font.sans-serif'] = [u'simHei']
mpl.rcParams['axes.unicode_minus'] = False
plt_format = ["ro", "g^", "b*"]
plt_format2=["r--","r-"]

def get_path_sources(filename):
	"""
		根据文件名获取本项目sources文件路径
		:param file:文件名
		:return:
		"""
	return os.path.join("..", "sources", filename)

def get_path_target(filename):
	"""
	根据文件名获取本项目target文件路径
	:param filename:文件名
	:return:
	"""
	return os.path.join("..", "wkztarget", filename)


def read_file_csv(filename, xfields, yfield):
	"""
	读取CSV文件
	:param filename:文件名
	:param xfields:要作为样本的维度的字段,list类型
	:param yfield:要作为样本的结果的字段
	:return:data, x, y
	"""
	filename = get_path_sources(filename)

	# # 手写读取数据 - 请自行分析，在10.2.Iris代码中给出类似的例子
	# f = file(filename)
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
	# f = file(filename, 'rb')
	# print f
	# d = csv.reader(f)
	# for line in d:
	#	 print line
	# f.close()
	# # numpy读入
	# p = np.loadtxt(filename, delimiter=',', skiprows=1)
	# print p
	# print type(p)
	# print '\n\n===============\n\n'
	# exit()
	# pandas读入

	data = pd.read_csv(filename)  # TV、Radio、Newspaper、Sales
	# x = data[['TV', 'Radio', 'Newspaper']]
	x = data[xfields]
	y = data[yfield]
	# print(data.head())  # pandas.core.frame.DataFrame
	# print x	#pandas.core.frame.DataFrame
	# print y
	# exit()
	return data, x, y


def read_file_csv2(filename):
	filename = get_path_sources(filename)

	# # 手写读取数据
	# f = file(filename)
	# x = []
	# y = []
	# for d in f:
	# 	d = d.strip()
	# 	if d:
	# 		d = d.split(',')
	# 		y.append(d[-1])
	# 		x.append(map(float, d[:-1]))
	# print '原始数据X：\n', x
	# print '原始数据Y：\n', y
	# x = np.array(x)
	# print 'Numpy格式X：\n', x
	# y = np.array(y)
	# print 'Numpy格式Y - 1:\n', y
	# y[y == 'Iris-setosa'] = 0
	# y[y == 'Iris-versicolor'] = 1
	# y[y == 'Iris-virginica'] = 2
	# print 'Numpy格式Y - 2:\n', y
	# y = y.astype(dtype=np.int)
	# print 'Numpy格式Y - 3:\n', y
	# print '\n\n============================================\n\n'


	# 使用np进行读取
	# def iris_type(s):
	# 	it = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
	# 	return it[s]
	#
	# # 路径，浮点型数据，逗号分隔，第4列使用函数iris_type单独处理
	# df = np.loadtxt(filename, dtype=float, delimiter=',', converters={4: iris_type})
	# print df


	# # 使用sklearn的数据预处理
	# df = pd.read_csv(filename, header=None)
	# x = df.values[:, :-1]
	# y = df.values[:, -1]
	# print x.shape
	# print y.shape
	# print 'x = \n', x
	# print 'y = \n', y
	# le = preprocessing.LabelEncoder()
	# # 预处理中将文本使用数字进行替换
	# le.fit(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])
	# print le.classes_
	# y = le.transform(y)
	# print 'Last Version, y = \n', y



	df = pd.read_csv(filename, header=None)
	iris_types = df[4].unique()
	for i, type in enumerate(iris_types):
		df.set_value(df[4] == type, 4, i)
	x, y = np.split(df.values, [4], axis=1)
	# print x.dtype	# object
	# print y.dtype	# object
	x = x.astype(np.float)
	y = y.astype(np.int)
	# print 'x = \n', x
	# print 'y = \n', y
	return x,y




def show_data(data, y, fields):
	"""
	图表显示原数据分布情况(离散)
	:param data: 样本数据源
	:param y: 样本结果
	:param fields: 样本的维度
	:return:None
	"""

	# # 绘制1(3个维度合并到一张图表中)
	# plt.plot(data['TV'], y, 'ro', label='TV')
	# plt.plot(data['Radio'], y, 'g^', label='Radio')
	# plt.plot(data['Newspaper'], y, 'mv', label='Newspaer')
	# plt.legend(loc='lower right')	#设置图例位置
	# # plt.legend(loc='upper left')	#设置图例位置
	# plt.grid()
	# # 添加图表网格线，设置网格线颜色，线形，宽度和透明度
	# # plt.grid(color='#95a5a6', linestyle='--', linewidth=1, axis='x', alpha=0.9)
	# plt.show()
	# exit()

	# 绘制2(3个维度分别绘制)
	plt.figure(figsize=(9, 12))
	length = math.ceil(float(len(fields)) / 2)
	begin = length * 100 + 2 * 10
	for index, field in enumerate(fields):
		plt.subplot(begin + index + 1)
		plt.plot(data[field], y, plt_format[index])
		plt.title(field)
		plt.grid()
	plt.tight_layout()
	plt.show()


def print_statistics(x_test, y_test, y_hat, score_func):
	"""
	打印输出均方误差mse,均方根误差/标准误差rmse,分数R^2
	:param x_test: 测试数据样本
	:param y_test: 测试数据值
	:param y_hat: 测试数据的预测值
	:param score_func: 评分函数,不同的模型不同的评分函数
	:return: None
	"""
	if not isinstance(y_test, np.ndarray):
		y_test = np.array(y_test)
	# 均方误差,越小越好
	mse = np.average((y_hat - np.array(y_test)) ** 2)
	# 均方根误差/标准误差,越小越好
	rmse = np.sqrt(mse)
	print "\n"
	print("mse =", mse)  # Mean Squared Error
	print("rmse =", rmse)  # Root Mean Squared Error
	# RSS = (y1_hat - y1) ^ 2 + (y2_hat - y2) ^ 2 + ... + (yi_hat - yi) ^ 2,y1_hat是x1的对应的预测值
	# TSS = (y1 - y_mean) ^ 2 + (y2 - y_mean) ^ 2 + ... + (yi - y_mean) ^ 2,y_mean是y1,y2,yn平均值
	# R2 = 1 - RSS/TSS,越大越好
	# R^2传入的是测试数据的样本及真实值
	print("R2 =", score_func(x_test, y_test))
	# 正确率
	print("Accuracy =",np.mean(y_hat == y_test))
	print "\n"


def show_predict(x_test, y_test, y_hat):
	"""
	图表显示原数据及预测值的分布情况(连续)
	:param x_test:测试数据样本
	:param y_test:测试数据值
	:param y_hat:测试数据的预测值
	:return:None
	"""
	if not isinstance(x_test, np.ndarray):
		x_test = np.arange(len(x_test))
	plt.plot(x_test, y_test, 'r-', linewidth=2, label=u'真实数据')
	plt.plot(x_test, y_hat, 'g-', linewidth=1, label=u'预测数据')
	plt.legend(loc='upper right')
	plt.title(u'真实及预测结果', fontsize=18)
	plt.grid()
	plt.show()
