# /usr/bin/python
# -*- encoding:utf-8 -*-

import xgboost as xgb
import numpy as np


# 判断蘑菇是否有毒
# 1、xgBoost的基本使用
# 2、自定义损失函数的梯度和二阶导
# 3、binary:logistic/logitraw


# 定义f: theta * x
from utils import wkzutils


def log_reg(y_hat, y):
	# 这里p是sigmod函数
	p = 1.0 / (1.0 + np.exp(-y_hat))
	# 目标函数J(f)的一阶导【这里导函数是自己设置】
	g = p - y.get_label()
	# 目标函数J(f)的二阶导【这里导函数是自己设置(h实际是p的一阶导函数)】
	h = p * (1.0-p)
	return g, h


def error_rate(y_hat, y):
	return 'error', float(sum(y.get_label() != (y_hat > 0.5))) / len(y_hat)


if __name__ == "__main__":
	# 读取数据
	data_train = xgb.DMatrix(wkzutils.get_path_sources("14.agaricus_train.txt"))
	data_test = xgb.DMatrix(wkzutils.get_path_sources("14.agaricus_test.txt"))
	print data_train
	print type(data_train)

	# 设置参数max_depth树深度,eta防止过拟合(衰减因子里面学习率v,推荐v<0.1,可以防止过拟合,但会造成计算次数增多),
	# silent是否输出其它多余信息,objective目标函数(二分类问题)
	param = {'max_depth': 3, 'eta': 0.1, 'silent': 0, 'objective': 'binary:logistic'} # logitraw
	# param = {'max_depth': 3, 'eta': 0.3, 'silent': 1, 'objective': 'reg:logistic'}
	# 训练数据和测试数据一起封闭成元组
	watchlist = [(data_test, 'eval'), (data_train, 'train')]
	n_round = 3
	# dtrain训练数据,num_boost_round迭代计算次数(树的数量?),evals训练数据时使用的评估数据
	# obj自定义目标函数(objective目标函数有设置,这个可不设置),feval自定义评估函数
	# bst = xgb.train(param, data_train, num_boost_round=n_round, evals=watchlist)
	bst = xgb.train(param, dtrain=data_train, num_boost_round=n_round, evals=watchlist, obj=log_reg, feval=error_rate)

	# 计算错误率
	y_hat = bst.predict(data_test)
	y = data_test.get_label()
	print y_hat
	print y
	error = sum(y != (y_hat > 0.5))
	error_rate = float(error) / len(y_hat)
	print '样本总数：\t', len(y_hat)
	print '错误数目：\t%4d' % error
	print '错误率：\t%.5f%%' % (100*error_rate)
