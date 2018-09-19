import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc  #ROC曲线
from sklearn.metrics import confusion_matrix    #混淆矩阵
from sklearn.metrics import f1_score    #F1度量;2/(1/r + 1/p) r,p:召回率和精度
from itertools import chain
import time
import os
import sys



def time_to_date(time_stamp):
    """把时间戳转换成日期的形式"""
    time_array = time.localtime(time_stamp) #传入秒数，得到相应的时间(年月日等)
    date_style_time = time.strftime("%Y-%m-%d %H:%M:%S", time_array)    #格式化输出
    return date_style_time

def check_path(_path):
    """查看是否存在路径，如果不存在就创造一个以便保存"""
    #获取_path的目录;
    if os.path.dirname(_path):
        if not os.path.exists(os.path.dirname(_path)):
            os.makedirs(os.path.dirname(_path))

def feature_analyze(model, to_print = False, to_plot = False, csv_path = None):
    """XGBOOST 模型特征重要性分析
    Args:
        model:训练好的xgb模型
        to_print：bool,是否输出每个特征重要性
        to_plot：bool,是否绘制特征重要性图表
        csv_path：str, 保存分析结果的csv文件路径
    """
    #获得每个特征的重要性，一般有三种标准，
        #weight - 该特征在所有树中被用作分割样本的特征的次数。
        #gain - 在所有树中的平均增益。
        #cover - 在树中使用该特征时的平均覆盖范围。
    feature_score = model.get_fscore()  #默认为weight
    feature_score = sorted(feature_score.items(), key = lambda x:x[1], reverse = True)  #降序排列
    if to_plot:
        features = list()
        scores = list()
        for (key, value) in feature_score:
            features.append(key)
            scores.append(value)
        plt.barh(range(len(scores)), scores)
        plt.yticks(range(len(scores)), features)
        for i in range(len(scores)):
            plt.text(scores[i] + 0.75, i - 0.25, scores[i])
        plt.xlabel('feature score')
        plt.title('feature score evaluate')
        plt.grid()
        plt.show()
    fs = []
    for (key, value) in feature_score:
        fs.append("{0},{1}\n".format(key, value))
    if to_print:
        print(''.join(fs))
    if csv_path is not None:
        with open(csv_path, 'w') as f:
            f.writelines('feature,score\n')
            f.writelines(fs)
    return feature_score

def get_grid_params(search_params):
    """遍历grid search 的所有参数组合
    Args:
        search_params;需要搜索的参数字典
    Returns:
        grid_params: lis, 每个元素为一个dict，对应每次搜索的参数
    """
    keys = list(search_params.keys())
    values = list(search_params.values())
    grid_params = list()
    if len(keys) == 1:
        for value in values[0]:
            dict_params = dict()
            dict_params[keys[0]] = value
            grid_params.append(dict_params.copy())
        return grid_params
    list_params_left = [[p] for p in values[0]]
    for i in range(1, len(values)):
        list_params_right = values[i]
        list_params_left = params_append(list_params_left, list_params_right)
    for params in list_params_left:
        dict_param = dict()
        for i in range(len(keys)):
            dict_param[keys[i]] = params[i]
        grid_params.append(dict_param.copy())
    return grid_params

def params_append(list_params_left, list_params_right):
    if type(list_params_left) is not list:
        list_params_left = list(map(lambda p:list([p]), list_params_left))
    n_left = len(list_params_left)
    n_right = len(list_params_right)
    list_params_left *= n_right         #就是将list_params_left重复n_right道[2,3]*2 --->[2,3,2,3]
    list_params_left = [x[:] for x in list_params_left] #这里必须这样作，因为，复制过来的与原数据指向的地址一样，改变一个引起连锁反应
    #如果n_left = 3, m = [2, 7],进行下面操作变为[[2, 2, 2], [7, 7, 7]]
        #加上chain是形成迭代器，但此处在外围加上了list，就有形成了list， 不加chain变成[[[7,7,7], [7, 7, 7], [7, 7,7]]]
    list_params_right = list(chain([[p] * n_left for p in list_params_right]))
    list_params_right = np.array(list_params_right).flatten()   #进行合并[[2, 2], [3, 3]]--->[2, 2, 3, 3]
    for i in range(len(list_params_left)):
        list_params_left[i].append(list_params_right[i])
    return list_params_left








