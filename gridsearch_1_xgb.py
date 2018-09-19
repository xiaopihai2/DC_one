from data_helper import load_feat
from my_utils import feature_analyze, get_grid_params
from m1_xgb import xgb_fit, xgb_predict, check_path
import numpy as np

from sklearn.externals import joblib
import time
import logging.handlers

"""
    通过网格搜索来选择最优参数
    保存最好的模型，预测结果最佳的参数
        60组参数，运算太耗时了
"""
LOG_FILE = 'log/xgb_grid_search.log'
check_path(LOG_FILE)
handler = logging.handlers.RotatingFileHandler(LOG_FILE, maxBytes=1024 * 1024, backupCount=5)
fmt = "%(asctime)s - %(filename)s:%(lineno)s - %(name)s - %(message)s"
formatter = logging.Formatter(fmt)
handler.setFormatter(formatter)
logger = logging.getLogger('search')
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)


class Config(object):
    def __init__(self):
        self.params = {
            'learning_rate': 0.025,
            'eval_metric': 'auc',
            'n_estimators': 5000,
            'max_depth': 5,
            'min_child_weight': 7,
            'gamma': 0,
            'subsample': 0.8,   #随机采样的比例，即每棵树用80%样本数据(行)
            'colsample_bytree': 0.6, #随机采样的比例(列， 其实就是特征选择比例)
            'eta': 0.05,
            'silent': 1,    #静默模式开启，不会输出任何信息
            'objective': 'binary:logistic',
            'scale_pos_weight': 1
        }
        self.max_round = 5000
        self.cv_folds = 5
        self.seed = 3
        self.early_stop_round = 30
        self.save_model_path = None
        self.best_model_path = 'model/best_xgb_model.dat'

def my_grid_search(config, search_params, model_fit, X_train, y_train):
    """grid search."""

    best_auc = 0
    best_model = None
    best_params = None
    #感觉这里好像多此一举，在sklearn中有这样网格搜索是params排列组合的函数，为什么要写一个这样的函数
        #使params排列组合，保证出现每一种情况，一共60组
    grid_params = get_grid_params(search_params)
    print(grid_params)
    message = '开始网格搜索，参数组的种类:{}'.format(len(grid_params))
    logger.info(message)
    for i in range(len(grid_params)):
        dict_params = grid_params[i]
        logger.info('进度：{}/{}'.format(i+1, len(grid_params)))
        for k, v in dict_params.items():
            config.params[k] = v
        _model, _auc, _round, _= model_fit(config, X_train, y_train)
        if _auc >= best_auc:    #寻找最好的auc
            best_params = dict_params.copy()
            best_auc = _auc
            best_model = _model
        message = 'Best_auc={}, auc={}, round={}, params={}'.format(best_auc, _auc, _round, dict_params)
        logger.info(message)
        print(message)
    search_messages = "搜索完成;Best auc={}, Best paras are\n={}".format(best_auc, best_params)
    logger.info(search_messages)
    print(search_messages)
    return best_model, best_params, best_auc



if __name__ == '__main__':
    #get feature
    feature_path = 'features/'
    train_data, test_data = load_feat(re_get=False, feature_path=feature_path)
    train_feats = train_data.columns.values
    test_feats = test_data.columns.values
    drop_columns = list(filter(lambda x: x not in test_feats, train_feats))
    drop_columns2 = list(filter(lambda x: x not in train_feats, test_feats))
    X_train = train_data.drop(drop_columns, axis=1)
    y_train = np.array(train_data['label']).astype(float)
    X_test = test_data.drop(drop_columns2, axis=1)
    data_message = 'X_train.shape={}, X_test.shape={}'.format(X_train.shape, X_test.shape)
    print(data_message)
    logger.info(data_message)

    tic = time.time()
    config = Config()
    search_params = {'learning_rate':[0.01, 0.025, 0.05],
                     'max_path': [5, 6, 7, 8],
                     'subsample': [0.6, 0.8],
                     'colsample_bytree': [0.5, 0.6, 0.8]}
    logger.info(search_params)
    #进行网格搜索找到最优的参数，得到最好的模型
    best_model, best_params, best_auc = my_grid_search(config, search_params, xgb_fit, X_train, y_train)
    print('花费时间{}s'.format(time.time() - tic))
    check_path(config.best_model_path)
    joblib.dump(best_model, config.best_model_path)

    #predict
    #best_model = joblib.load(config.best_model_path)   #读取最好的xgboost模型
    now = time.strftime("%m%d-%H%M%S")
    result_path = 'result/result_xgb_search_{}.csv'.format(now)
    check_path(result_path)
    xgb_predict(best_model, X_test, result_path)

    #feature analyze
    feature_score_path = 'model/xgb_scarch_feature_score.csv'
    check_path(feature_score_path)
    feature_analyze(best_model, csv_path =feature_score_path)
