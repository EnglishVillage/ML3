#!/usr/bin/python
# -*- coding:utf-8 -*-


from sklearn.datasets import load_iris   #skit-learn是一个机器学习的开源包，需另外安装
from sklearn import tree
from sklearn.externals.six import StringIO
import pydot
iris = load_iris()#载入数据集
clf = tree.DecisionTreeClassifier()#算法模型
clf = clf.fit(iris.data, iris.target)#模型训练
dot_data = StringIO()
tree.export_graphviz(clf, out_file=dot_data)
graph = pydot.graph_from_dot_data(dot_data.getvalue())
graph[0].write_pdf("iris.pdf")#写入pdf