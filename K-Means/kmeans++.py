# -*- coding:utf-8 -*-
'''
KMeans++算法是KMeans算法的改进，其改进目的是在聚类中心的选择过程中选择较优的聚类中心
原因在于，原始的KMeans算法一方面受到簇的个数K的影响；另一方面还会受到初始簇中心位置选择的影响。
簇中心选择的效果也会影响到最后的聚簇效果。KMeans++本质上是在簇中心的初始化策略上作了改进，其它的步骤
并未发生改变。
'''

import numpy as np
from random import random
from kmeans import load_data, kmeans, distance, save_result

FLOAT_MAX = 1e100  # 设置一个较大的值作为初始化的最小的距离


def nearest(point, cluster_centers):
    '''
    计算point和cluster_centers之间的最小距离
    input:  point(mat):当前的样本点
            cluster_centers(mat):当前已经初始化的聚类中心
    output: min_dist(float):点point和当前的聚类中心之间的最短距离
    '''
    min_dist = FLOAT_MAX
    m = np.shape(cluster_centers)[0]  # 当前已经初始化的聚类中心的个数
    for i in range(m):
        # 计算point与每个聚类中心之间的距离
        d = distance(point, cluster_centers[i, ])
        # 选择最短距离
        if min_dist > d:
            min_dist = d
    return min_dist


def get_centroids(points, k):
    '''
    KMeans++的初始化聚类中心的方法
    input:  points(mat):样本
            k(int):聚类中心的个数
    output: cluster_centers(mat):初始化后的聚类中心
    '''
    m, n = np.shape(points)
    cluster_centers = np.mat(np.zeros((k, n)))
    # 1、随机选择一个样本点为第一个聚类中心
    index = np.random.randint(0, m)
    cluster_centers[0, ] = np.copy(points[index, ])
    # 2、初始化一个距离的序列
    d = [0.0 for _ in range(m)]

    for i in range(1, k):
        sum_all = 0
        for j in range(m):
            # 3、对每一个样本找到最近的聚类中心点
            d[j] = nearest(points[j, ], cluster_centers[0:i, ])
            # 4、将所有的最短距离相加
            sum_all += d[j]
        # 5、取得sum_all之间的随机值
        sum_all *= random()
        # 6、获得距离最远的样本点作为聚类中心点
        for j, di in enumerate(d):
            sum_all -= di
            if sum_all > 0:
                continue
            cluster_centers[i] = np.copy(points[j, ])
            break
    return cluster_centers


if __name__ == "__main__":
    k = 4  # 聚类中心的个数
    file_path = "./Data/data.txt"
    # 1、导入数据
    print("---------- 1.load data ------------")
    data = load_data(file_path)
    # 2、KMeans++的聚类中心初始化方法
    print("---------- 2.K-Means++ generate centers ------------")
    centroids = get_centroids(data, k)
    # 3、聚类计算
    print("---------- 3.kmeans ------------")
    subCenter = kmeans(data, k, centroids)
    # 4、保存所属的类别文件
    print("---------- 4.save subCenter ------------")
    save_result("./TrainingResult/kmeans++_sub", subCenter)
    # 5、保存聚类中心
    print("---------- 5.save centroids ------------")
    save_result("./TrainingResult/kmeans++_center", centroids)
