#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/11/13下午14:06
# @Author  : GLJ
# @File    : Sklearn_DecisionTrees.py
# @Software: PyCharm + Python3.6

import xlrd
import xlwt
import numpy as np
import graphviz

from sklearn.model_selection import train_test_split #划分数据
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import tree

# 取出数据方法
# 参数：文件路径
def getClassData(path):
    data = xlrd.open_workbook(path)  # 打开xls文件
    table = data.sheets()[0]  # 打开第一张表
    nrows = table.nrows  # 获取表的行数

    workbook = xlwt.Workbook(encoding='ascii')
    Data_Sheet = workbook.add_sheet('Data_Sheet')
    Class_Sheet = workbook.add_sheet('Class_Sheet')
    num = 0
    for i in range(nrows):  # 循环逐行打印
        if i == 0:  # 跳过第一行
            continue
        if table.row_values(i)[5] != '':
            num += 1
            for j in range(0, 5):
                Data_Sheet.write(num-1, j, label=table.row_values(i)[j])
            Class_Sheet.write(num - 1, 0, label=table.row_values(i)[5])
    workbook.save('homework_08_Data.xls')

# 将Excel转为矩阵
# 参数：path 传入文件路径 sheet：表
# 返回值：数据矩阵
def excel2m(path,sheet):
    data = xlrd.open_workbook(path)
    table = data.sheets()[sheet]
    nrows = table.nrows  # 行数
    ncols = table.ncols  # 列数
    datamatrix = np.zeros((nrows, ncols))
    for x in range(ncols):
        cols = table.col_values(x)
        cols1 = np.matrix(cols)   # 把list转换为矩阵进行矩阵操作
        datamatrix[:, x] = cols1  # 把数据进行存储
    return datamatrix

def save(data, path):
    f = xlwt.Workbook()  # 创建工作簿
    sheet1 = f.add_sheet(u'sheet1', cell_overwrite_ok=True)  # 创建sheet
    [h, l] = data.shape  # h为行数，l为列数
    for i in range(h):
        for j in range(l):
            sheet1.write(i, j, data[i, j])
    f.save(path)


if __name__ == "__main__":
    getClassData('样本集.xls')

    All_Data = excel2m('homework_08_Data.xls',0)
    Class_data = excel2m('homework_08_Data.xls',1)

    # 这里划分数据以1/3的来划分 训练集 训练结果 测试集 测试结果
    #               数据个数   400   400      200   200
    train_X, test_X, train_y, test_y = train_test_split(All_Data, Class_data[:, 0], test_size=1 / 3, random_state=1)
    # print(train_X, test_X, train_y, test_y )
    save(train_X,'homework_08_train_Data_X.xls')
    #print((train_y))
    #save(np.matrix.tolist(train_y), 'homework_08_train_Data_X.xls')
    # 决策树模型
    dtc = tree.DecisionTreeClassifier(criterion='entropy', max_features=5)
    dtc = dtc.fit(train_X,train_y)
    print("决策树模型预测类别：")
    print(dtc.predict(test_X))
    print("识别率为：%s"%dtc.score(test_X, test_y))

    # Boosting模型
    gbc = GradientBoostingClassifier(learning_rate=0.1, n_estimators=100, max_features='sqrt', subsample=0.8, random_state=10)
    gbc = gbc.fit(train_X, train_y)
    print("Boosting模型预测类别：")
    print(gbc.predict(test_X))
    print("识别率为：%s"%gbc.score(test_X, test_y))

    # AdaBoost模型
    abc = AdaBoostClassifier(base_estimator=dtc,n_estimators=120)
    abc = abc.fit(train_X, train_y)
    print("AdaBoost模型预测类别：")
    print(abc.predict(test_X))
    print("识别率为：%s" % abc.score(test_X, test_y))

    # 生成决策树
    feature_names = ['dos','cod','tn','nh3n','tp']
    target_names = ['1','2','3','4','5','6']
    dot_data = tree.export_graphviz(dtc, out_file=None,
                                    feature_names=feature_names,
                                    class_names=target_names,
                                    filled=True, rounded=True,
                                    special_characters=True)
    graph = graphviz.Source(dot_data)
    graph.render("水质检测决策树")