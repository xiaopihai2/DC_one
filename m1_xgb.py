from data_helper import load_feat, df_future_test
from my_utils import feature_analyze, check_path
import pandas as pd
import numpy as np
from sklearn import preprocessing

import xgboost as xgb
from sklearn.externals import joblib    #joblib更适合大数据量的模型，且只能往硬盘存储，不能往字符串存储
from sklearn.model_selection import train_test_split
import time
import logging.handlers     #日志库
import os

#日志保存文件
LOG_FILE = 'log/xgb_train.log'
check_path(LOG_FILE)
#参数maxBytes和backupCount允许日志文件在达到maxBytes时rollover.当文件大小达到或者超过maxBytes时，就会新创建一个日志文件。
handler = logging.handlers.RotatingFileHandler(LOG_FILE, maxBytes=1024 * 1024, backupCount=1)   #实例化handler
fmt = '%(asctime)s - %(filename)s:%(lineno)s - %(message)s'
#定义handler的输出格式
formatter = logging.Formatter(fmt)
handler.setFormatter(formatter)
#指定name，返回一个名称为name的Logger实例。如果再次使用相同的名字，是实例化一个对象。未指定name，返回Logger实例，名称是root，即根Logger。
logger = logging.getLogger('train')
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)


def xgb_DMatrix(dataframe):
    features = dataframe.columns.values
    for f in dataframe.columns.values:
        if dataframe[f].dtype == 'object':
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(dataframe[f].values))
            dataframe[f] = lbl.transform(list(dataframe[f].values))
    dataframe[features] = np.array(dataframe)
    dataframe[features] = dataframe[features].astype(float)
    return dataframe

class Config(object):
    def __init__(self):
        self.params = {
            'learning_rate': 0.05,      #学习效率
            'eval_metric': 'auc',   #对于有效数据的度量方法。,这里是曲线下面积，具体可以分回归和分类来说，其他的度量也可以选，此处分类
            'n_estimators': 5000,   #这里好像是树的总数，
            'max_depth': 6,         #每棵树的最大深度
            'min_child_weight': 7,  #决定最小叶子节点样本权重和。这个参数用于避免过拟合。当它的值较大时，可以避免模型学习到局部的特殊样本。，但过高又会欠拟合
            'gamma': 0,     #在节点分裂时，只有分裂后损失函数的值下降了，才会分裂这个节点。Gamma指定了节点分裂所需的最小损失函数下降值。
            'subsample': 0.8,       #和GBM中的subsample参数一模一样。这个参数控制对于每棵树，随机采样的比例。应该是选取的数据比例
            'colsample_betree': 0.6,    #和GBM里面的max_features参数类似。用来控制每棵随机采样的列数的占比, 即所用的特征占全部特征的最大60%
            'eta': 0.05,     # 同 learning rate, Shrinkage（缩减），每次迭代完后叶子节点乘以这系数，削弱每棵树的权重，通过减少每一步的权重，可以提高模型的鲁棒性。
            'silent': 1,       #当这个参数值为1时，静默模式开启，不会输出任何信息。
            'objective': 'binary:logistic', #定义最小化损失函数,binary:logistic 二分类的逻辑回归，返回预测的概率(不是类别)
            #'nthread': 6,  #这个参数用来进行多线程控制，应当输入系统的核数。不输入，算法自动检测
            'scale_pos_weight': 1       #在各类别样本十分不平衡时，把这个参数设定为一个正值，可以使算法更快收敛。
        }
        self.max_round = 3000
        self.cv_folds = 10
        self.early_stop_round = 50
        self.seed = 3
        self.save_model_path = 'model/xgb.dat'


def run_cv(X_train, X_test, y_train):
    config = Config()
    #train model
    tic = time.time()
    data_message = 'X_train.shape={}, X_test.shape={}'.format(X_train.shape, X_test.shape)
    print(len(data_message))
    logger.info(data_message)   #日志库不明白， 以后再看看
    xgb_model, best_auc, best_round, cv_result = xgb_fit(config, X_train, y_train)
    print("用时{}s".format(time.time() - tic))
    result_message = '最好迭代次数{},最优auc面积{}'.format(best_round, best_auc)
    logger.info(result_message)
    print(result_message)
    #predict
    now = time.strftime("%m%d-%H%M%S")
    result_path = 'result/result_xgb_{}-{:.4f}.csv'.format(now, best_auc)
    check_path(result_path)
    xgb_predict(xgb_model, X_test, result_path)
    #feature analyze    特征分析
    feature_score_path = 'features/xgb_feature_score.csv'
    check_path(feature_score_path)
    feature_analyze(xgb_model, csv_path = feature_score_path)

def xgb_fit(config, X_train, y_train):
    """模型(交叉验证)训练， 并返回最优迭代次数和最优的结果
    Args:
        config：xgb模型参数{params, max_round, cv_folds, early_stop_round, seed, save_path}
        X_train：array like, shape = n_sample * n_feature
        y_train：shape = n_sample * 1
    Returns：
        best_model：训练好的最优模型
        best_auc：float, 在测试集上面的AUC值
        best_round：int,最优迭代次数
    """
    params = config.params
    max_round = config.max_round    #最大迭代次数
    cv_folds = config.cv_folds      #10折交叉
    early_stop_round = config.early_stop_round
    seed = config.seed              #随机数的种子
    save_model_path = config.save_model_path
    X_train = xgb_DMatrix(X_train)
    y_train = np.array(y_train)
    y_train = y_train.astype(float)
    if cv_folds is not None:
        dtrain = xgb.DMatrix(X_train, label=y_train)
        #show_stdv:是否显示标准偏差
        #verbose_eval = True, 是否显示进度， 为True， 进度显示在boosting阶段
        #early_stopping_rounds：防止因为迭代次数过多而过拟合的状态，比如为10时， 表示10轮迭代中， 都没有提升的话就stop，按最后一个指标为准
        cv_result = xgb.cv(params, dtrain, max_round, nfold=cv_folds, seed=seed, verbose_eval=True,
                           metrics='auc', early_stopping_rounds=early_stop_round,show_stdv=False)
        #xgb.cv()返回一个dataframe
        #最优模型， 最优迭代次数
        best_round = cv_result.shape[0]
        print(cv_result.info())
        best_auc = cv_result['test-auc-mean'].values[-1]    #最好的 auc 值
        best_model = xgb.train(params, dtrain, best_round)
    else:
        X_train,X_vaild, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=100)
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dvaild = xgb.DMatrix(X_vaild, label=y_valid)
        watchlist = [(dtrain, 'train'), (dvaild, 'vaild')]
        best_model = xgb.train(params, dtrain, max_round, evals=watchlist, early_stopping_rounds=early_stop_round)
        best_round = best_model.best_iteration
        best_auc = best_model.best_score
        cv_result = None
    if save_model_path:
        check_path(save_model_path)
        joblib.dump(best_model, save_model_path)
    return best_model, best_auc, best_round, cv_result

def xgb_predict(model, X_test, save_result_path = None):
    X_test = xgb_DMatrix(X_test)
    dtest = xgb.DMatrix(X_test)
    print(dtest.num_row())
    y_pred_prob = model.predict(dtest)
    if save_result_path:
        """因为我们在处理数据时使用的左连接，导致我们要处理的数据记录数变成了12000多条，
        然而，题目要求的记录数是10076条，(因为左连接导致的)所以我们要用X_test['userid']来初始DataFrame
        最后我们按照id分组， 计算平均值，得到概率。我看了一下，其实该操作之前查看同id数据的概率差不多相等
       """
        df_result = pd.DataFrame()
        df_result['userid'] = X_test['userid']
        print(len(df_result))
        df_result['orderType'] = y_pred_prob
        df_result = df_result.groupby('userid', as_index=False)['orderType'].mean()
        df_result.to_csv(save_result_path, index=False)
        print('保存结果到{}'.format(save_result_path))
    return y_pred_prob

def run_feat_search(X_train, X_test, y_train, feature_name):
    config = Config()
    #train model
    tic = time.time()
    preds_data = np.load('xgb_feat_search_pred_105.npz')


if __name__ == "__main__":
    """执行顺序
    1.首先执行run_csv()，会得到一份feature_score.
    2.执行get_no_used_feature.py生成特征排序表
    3.执行feature_score进行特征搜索，并将预测结果进行保存
    """
    #get feature
    #熊本市, 莱顿, 美因茨, 二世古, 拜县, 和歌山, 汉诺威, 湖区, 千岁, 利物浦, 剑桥, 格拉斯哥, 大城, 富士五湖地区
    feature_path = 'features/'
    train_data, test_data = load_feat(re_get=False, feature_path = feature_path)
    train_feats = train_data.columns.values
    test_feats = test_data.columns.values
    drop_columns = list(filter(lambda x:x not in test_feats, train_feats))
    drop_columns2 = list(filter(lambda x:x not in train_feats, test_data))
    print(drop_columns, drop_columns2)
    X_train = train_data.drop(drop_columns, axis = 1)
    y_train = train_data['label']
    X_test = test_data.drop(drop_columns2, axis=1)
    data_message = 'X_train.shape={}, X_test.shape={}'.format(X_train.shape, X_test.shape)
    print(data_message)
    logger.info(data_message)
    run_cv(X_train, X_test, y_train)
