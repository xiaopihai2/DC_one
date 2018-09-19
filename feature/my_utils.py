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




def check_path():
    pass