# -*- coding:utf-8 -*-

import numpy as np

def load_data(file_path):
    '''
    导入测试数据
    input:  file_path(string):训练数据
    output: feature(mat):特征矩阵
    '''
    f = open(file_path)
    feature = []
    for line in f.readlines():
        feature_tmp = []
        lines = line.strip().split("\t")
        feature_tmp.append(1)  # x0
        for i in range(len(lines)):
            feature_tmp.append(float(lines[i]))
        feature.append(feature_tmp)         # 构建增广特征矩阵
    f.close()
    return np.mat(feature)


def load_model(model_file):
    '''
    导入模型
    input:  model_file(string):线性回归模型
    output: w(mat):权重值
    '''
    w = []
    f = open(model_file)
    for line in f.readlines():
        w.append(float(line.strip()))
    f.close()
    return np.mat(w).T


def get_prediction(data, w):
    '''
    得到预测值
    input:  data(mat):测试数据
            w(mat):权重值
    output: 最终的预测
    '''
    return data * w


def save_predict(file_name, predict):
    '''
    保存最终的预测值
    input:  file_name(string):需要保存的文件名
            predict(mat):对测试数据的预测值
    '''
    m = np.shape(predict)[0]
    result = []
    for i in range(m):
        result.append(str(predict[i, 0]))
    f = open(file_name, "w")
    f.write("\n".join(result))
    f.close()


if __name__ == "__main__":
    # 1、导入测试数据
    testData = load_data("./BasicLinearRegressionData/test_data.txt")
    # 2、导入线性回归模型
    w = load_model("./BasicLinearRegressionTrain/weights")
    # 3、得到预测结果
    predict = get_prediction(testData, w)
    # 4、保存最终的结果
    save_predict("./BasicLinearRegressionTrain/model_predict_result", predict)

