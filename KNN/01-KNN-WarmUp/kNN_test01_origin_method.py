# -*- coding: UTF-8 -*-
import numpy as np
import operator

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
def classify0(inX, dataSet, labels, k):
	# 计算训练数据集的样本的数量
	dataSetSize = dataSet.shape[0]
	# 以inX为单位，将其reshape成(dataSetSize, 1)，使其与dataSet形状相同，为了后面的计算
	diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
	sqDiffMat = diffMat**2
	# 矩阵的每一行对应一个样本的对应维度的平方，然后按行求和，及计算出(x-x1)**2+(y-y1)**2
	sqDistances = sqDiffMat.sum(axis=1)
	# 计算出欧氏距离
	distances = sqDistances**0.5
	# 返回distances中元素从小到大排序后的索引值
	sortedDistIndices = distances.argsort()
	# 定义一个记录类别次数的字典
	classCount = {}
	for i in range(k):
		# 取出前k个元素的标签
		voteIlabel = labels[sortedDistIndices[i]]
		# dict.get(key,default=None),字典的get()方法,返回指定键的值,如果值不在字典中返回默认值。
		# 计算标签对应的次数
		classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
	# python3中用items()替换python2中的iteritems()
	# key=operator.itemgetter(1)根据字典的值进行排序
	# key=operator.itemgetter(0)根据字典的键进行排序
	# reverse降序排序字典
	sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
	# 返回次数最多的标签,即所要分类的类别
	return sortedClassCount[0][0]

if __name__ == '__main__':
	# 创建用于训练的训练数据集，包括特征矩阵和标签向量
	group, labels = createDataSet()
	# 创建待测试的测试样本
	test_sample = [101, 20]
	# 使用kNN模型进行预测
	test_sample_label = classify0(test_sample, group, labels, 3)
	# 打印分类结果
	print("the test_sample's label is " + test_sample_label)
