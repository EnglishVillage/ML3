# /usr/bin/python
# -*- encoding:utf-8 -*-

import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split   # cross_validation

from utils import wkzutils


def iris_type(s):
	# xgboost中预测值必须从0开始
	it = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
	return it[s]


if __name__ == "__main__":
	path = wkzutils.get_path_sources("10.iris.data")  # 数据文件路径
	data = np.loadtxt(path, dtype=float, delimiter=',', converters={4: iris_type})
	x, y = np.split(data, (4,), axis=1)
	x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, test_size=50)
	# print x_train
	# [[6.3  2.5  4.9  1.5]
	# [6.7  3.1  5.6  2.4]
	# [4.9  3.1  1.5  0.1]]

	data_train = xgb.DMatrix(x_train, label=y_train)
	data_test = xgb.DMatrix(x_test, label=y_test)

	# objective这里是多分类,num_class是分类个数
	param = {'max_depth': 2, 'eta': 1, 'silent': 1, 'objective': 'multi:softmax', 'num_class': 3}
	watch_list = [(data_test, 'eval'), (data_train, 'train')]
	bst = xgb.train(param, data_train, num_boost_round=6, evals=watch_list)
	y_hat = bst.predict(data_test)
	# y_test是多行一列的矩阵,y_hat是一行多列向量
	result = y_test.reshape(1, -1) == y_hat
	print '正确率:\t', float(np.sum(result)) / len(y_hat)
	print 'END.....\n'
