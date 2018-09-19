from data_helper import load_feat, df_future_test
from my_utils import check_path
import numpy as np
import pandas as pd
import pickle

import lightgbm as  lgb
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import time
import logging.handlers

"""Train the lightGBM model."""
LOG_FILE = 'log/lgb_train.log'
check_path(LOG_FILE)
handler = logging.handlers.RotatingFileHandler(LOG_FILE, maxBytes=1024 * 1024, backupCount=1)   #实例化handler
fmt = '%(asctime)s - %(filename)s:%(lineno)s - %(name)s - %(message)s'
formatter = logging.Formatter(fmt)
handler.setFormatter(formatter)
logger = logging.getLogger('train')
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)
"""
    Lightgbm是基于决策树的分布式梯度提升框架，以选取最大信息增益作为特征选择的目标(ID3?,C4.5?)
"""

#主要是集成算法在数理数据上要，xgboost要DMatrix,必须转换成float且最好按dataFrame传入
#但不知道lightgbm的Dataset需要不？
def lgb_DMatrix(dataframe):
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
            'objective': 'binary',       #二分类的
            'metric': {'auc'},
            'learning_rate': 0.05,       #学习效率
            'num_leaves': 30,            #叶子设置为50 线下过拟合严重。决策树的叶子节点数(Lightgbm用决策树叶子节点数来确定树的复杂度，而XGboost用max_depth确定树的复杂度)
            'min_sum_hessian_in_leaf': 0.1,     #好像是一个叶子节点上的海森矩阵的和，用来处理过拟合的
            'feature_fraction': 0.3,    #每次迭代中随机选择30%的参数来键树
            'bagging_fraction': 0.5,    #每次迭代时用的数据比例
            'lambda_l1': 0, #L1正则化
            'lambda_l2': 5, #L2正则化
            'num_thread': 6     #线程数设置为真实的cpu数， 一般12线程的机器有6个物理核
        }
        self.max_round = 3000
        self.cv_folds = 5
        self.early_stop_round = 30
        self.seed = 3
        self.save_model_path = 'model/lgb.txt'



def run_cv(X_train, X_test, y_train):
    config = Config()
    #train model
    tic = time.time()
    data_message = 'X_train.shape={}, X_test.shape={}'.format(X_train.shape, X_test.shape)
    print(data_message)
    logger.info(data_message)
    lgb_model, best_auc, best_round, cv_result = lgb_fit(config, X_train, y_train)
    print("用时{}s".format(time.time() - tic))
    result_message = 'best_round={}, best_result={}'.format(best_round, best_auc)
    logger.info(result_message)
    print(result_message)
    #predict
    #lgb_model = lgb.Booster(model_file=config.save_model_path)     #应该是加载模型吧
    now = time.strftime('%m%d-%H%M%S')
    result_path = 'result/result_lgb_{}-{:.4f}.csv'.format(now, best_auc)
    check_path(result_path)
    lgb_predict(lgb_model, X_test, result_path)
def lgb_fit(config, X_train, y_train):

    """模型(交叉验证)训练，并返回最优迭代此时和最优的结果
    Args:
       config: xgb 模型参数 {params, max_round, cv_folds, early_stop_round, seed, save_model_path}
        X_train：array like, shape = n_sample * n_feature
        y_train:  shape = n_sample * 1

    Returns:
        bast_model: 训练好的最优模型
        best_auc:float，在测试集上的AUC最大线下面积
        best_round: int,最优迭代次数
    """
    params = config.params      #参数字典
    max_round = config.max_round    #最大迭代次数
    cv_folds = config.cv_folds      #交叉验证的几折
    early_stop_round = config.early_stop_round  #每early_stop_round轮迭代，都没有提升，就停止
    seed = config.seed      #随机种子
    #seed = np.random.randint(0, 1000)
    save_model_path = config.save_model_path
    X_train =lgb_DMatrix(X_train)
    y_train = np.array(y_train).astype(float)
    if cv_folds is not None:
        dtrain = lgb.Dataset(X_train, label=y_train)
        #与xgboost的参数一样
        cv_result = lgb.cv(params, dtrain, max_round, nfold=cv_folds, seed =seed, verbose_eval=True,
                           metrics='auc', early_stopping_rounds=early_stop_round, show_stdv=False)
        #最优模型， 最优迭代次数
        print(type(cv_result))
        best_round = len(cv_result['auc-mean'])
        best_auc = cv_result['auc-mean'][-1]    #最好的auc值
        best_model = lgb.train(params, dtrain, best_round)
    else:
        X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=100)
        dtrain = lgb.Dataset(X_train, y_train)
        dvalid = lgb.Dataset(X_valid, y_valid)
        watchlist = [dtrain, dvalid]
        best_model = lgb.train(params, dtrain, max_round, valid_sets=watchlist, early_stopping_rounds=early_stop_round)
        best_round = best_model.best_iteration
        best_auc = best_model.best_score
        cv_result = None
    if save_model_path:
        check_path(save_model_path)
        best_model.save_model(save_model_path)
    return best_model, best_auc, best_round, cv_result

def lgb_predict(model, X_test, save_result_path = None):
    X_test =lgb_DMatrix(X_test)
    y_pred_prob = model.predict(X_test)
    if save_result_path:
        df_result = pd.DataFrame()
        df_result['userid'] = X_test['userid']
        df_result['orderType'] = y_pred_prob
        df_result = df_result.groupby('userid', as_index=False)['orderType'].mean()
        df_result.to_csv(save_result_path, index = False)
        print('保存结果至{}'.format(save_result_path))
    return y_pred_prob

if __name__ == '__main__':
    #get feature
    feature_path = 'features/'
    train_data, test_data = load_feat(re_get = False, feature_path=feature_path)
    train_feats = train_data.columns.values
    test_feats = test_data.columns.values
    drop_columns = list(filter(lambda x:x not in test_feats, train_feats))
    drop_columns2 = list(filter(lambda x:x not in train_feats, test_feats))
    X_train = train_data.drop(drop_columns, axis = 1)
    y_train = train_data['label']
    X_test = test_data
    data_message = 'X_train.shape={}, X_test.shape={}'.format(X_train.shape, X_test.shape)
    print(data_message)
    logger.info(data_message)
    run_cv(X_train, X_test, y_train)
