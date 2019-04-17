# -*- coding: UTF-8 -*-
import numpy as np
import operator
import collections

"""
函数说明:创建一个Toy数据集

Parameters:
	无
Returns:
	group - 训练数据集
	labels - 分类标签
Time:
	2019-04-17
"""
def createDataSet():
	# 创建4个训练样本，每一个样本有两个特征（实际上KNN算法并没有显示的训练过程，是一种基于实例的训练算法）
	group = np.array([[1,101],[5,89],[108,5],[115,8]])
	# 创建每个样本所对应的标签组成的标签向量，这是一个监督学习算法所必须有的
	labels = ['爱情片','爱情片','动作片','动作片']
	return group, labels

"""
函数说明:kNN算法,分类模型

Parameters:
	inX - 用于待分类的数据(测试样本)
	dataSet - 用于参与模型训练的数据(训练集)，虽然是基于实例的学习算法，暂且按机器学习的标准来理解吧
	labes - 分类标签
	k - kNN算法参数,选择距离测试样本inX最近的k个样本，采用投票法来预测待测试数据的标签
Returns:
	sortedClassCount[0][0] - 模型的预测结果

Time:
	2019-04-17
"""
def classify0(inx, dataset, labels, k):
	# 计算欧氏距离，这是KNN算法经常使用的距离度量，这里使用了numpy的广播机制，相比于原书中的tile方法更为先进
	dist = np.sum((inx - dataset)**2, axis=1)**0.5
	# 得到距离测试样本数据最近的k个样本的标签
	k_labels = [labels[index] for index in dist.argsort()[0 : k]]
	# 采用投票法，出现次数最多的标签即为测试样本的标签值
	label = collections.Counter(k_labels).most_common(1)[0][0]
	return label

if __name__ == '__main__':
	# 创建用于训练的训练数据集，包括特征矩阵和标签向量
	group, labels = createDataSet()
	# 创建待测试的测试样本
	test_sample = [101,20]
	# 使用kNN模型进行预测
	test_sample_label = classify0(test_sample, group, labels, 3)
	#打印分类结果
	print("the test_sample's label is " + test_sample_label)
