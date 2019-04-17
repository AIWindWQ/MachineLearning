# -*- coding: UTF-8 -*-
import numpy as np
import operator
from os import listdir

"""
函数说明:kNN算法,分类器

Parameters:
	inX - 用于待测试的测试数据(测试集)
	dataSet - 用于参与模型训练的训练数据(训练集)，暂且这样理解吧
	labes - 分类标签
	k - kNN算法参数,选择距离测试样本最近的k个样本
Returns:
	sortedClassCount[0][0] - 分类结果

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

"""
函数说明:将32x32的二进制图像矩阵转换为1x1024向量，数据的预处理过程。

Parameters:
	filename - 文件名
Returns:
	returnVect - 返回的二进制图像的1x1024向量

Time:
	2019-04-17
"""
def img2vector(filename):
	#创建1x1024零向量
	returnVect = np.zeros((1, 1024))
	#打开文件
	fr = open(filename)
	#按行读取
	for i in range(32):
		#读一行数据
		lineStr = fr.readline()
		#每一行的前32个元素依次添加到returnVect中
		for j in range(32):
			returnVect[0, 32*i+j] = int(lineStr[j])
	#返回转换后的1x1024向量
	return returnVect

"""
函数说明:手写数字分类测试

Parameters:
	无
Returns:
	无

Time:
	2019-04-17
"""
def handwritingClassTest():
	#测试集的Labels
	hwLabels = []
	#返回trainingDigits目录下的文件名
	trainingFileList = listdir('trainingDigits')
	#返回文件夹下文件的个数
	m = len(trainingFileList)
	#初始化训练的Mat矩阵,测试集
	trainingMat = np.zeros((m, 1024))
	#从文件名中解析出训练集的类别
	for i in range(m):
		#获得文件的名字
		fileNameStr = trainingFileList[i]
		#获得分类的数字
		classNumber = int(fileNameStr.split('_')[0])
		#将获得的类别添加到hwLabels中
		hwLabels.append(classNumber)
		#将每一个文件的1x1024数据存储到trainingMat矩阵中
		trainingMat[i,:] = img2vector('trainingDigits/%s' % (fileNameStr))
	#返回testDigits目录下的文件名
	testFileList = listdir('testDigits')
	#错误检测计数
	errorCount = 0.0
	#测试数据的数量
	mTest = len(testFileList)
	#从文件中解析出测试集的类别并进行分类测试
	for i in range(mTest):
		#获得文件的名字
		fileNameStr = testFileList[i]
		#获得分类的数字
		classNumber = int(fileNameStr.split('_')[0])
		#获得测试集的1x1024向量,用于训练
		vectorUnderTest = img2vector('testDigits/%s' % (fileNameStr))
		#获得预测结果
		classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
		print("分类返回结果为%d\t真实结果为%d" % (classifierResult, classNumber))
		if(classifierResult != classNumber):
			errorCount += 1.0
	print("总共错了%d个数据\n错误率为%f%%" % (errorCount, errorCount/mTest))


"""
函数说明:main函数

Parameters:
	无
Returns:
	无

Time:
	2019-04-17
"""
if __name__ == '__main__':
	handwritingClassTest()
