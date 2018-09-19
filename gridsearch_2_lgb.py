"""
lightgbm模型的网格搜索，寻找最佳参数和最优模型
"""
from data_helper import load_feat
from my_utils import get_grid_params, check_path
from m2_lgb import lgb_fit, lgb_predict

import time
import logging.handlers
import numpy as np
from sklearn.externals import joblib

LOG_FILE = 'log/log_grid_search.log'
check_path(LOG_FILE)
handler = logging.handlers.RotatingFileHandler(LOG_FILE, maxBytes=1024 * 1024, backupCount=5)
fmt = '%(asctime)s - %(filename)s:%(lineno)s - %(name)s - %(messages)s'
formatter = logging.Formatter(fmt)
handler.setFormatter(formatter)
logger = logging.getLogger('search')
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

class Config(object):
    def __init__(self):
        self.params = {
            'objective': 'binary',
            'metric': {'auc'},
            'learning_rate': 0.05,
            'num_leaves': 30,       #叶子设置为50， 线下过拟合严重
            'min_sum_hessian_in_leaf': 0.1,
            'feature_fraction':0.3,
            'bagging_fraction':0.5,
            'lambda_l1': 0,
            'lambda_l2': 5
        }
        self.max_round = 5000
        self.cv_folds = 5
        self.early_stop_round = 30
        self.seed = 3
        self.save_model_path = None
        self.best_model_path = 'model/best_lgb_model.txt'

def my_grid_search(config, search_params, model_fit, X_train, y_train):
    """grid search"""
    best_auc = 0.0
    best_model = None
    best_params = None
    grid_params = get_grid_params(search_params)
    messages = "开始搜索，参数组的大小为{}".format(len(grid_params))
    print(messages)
    logger.info(messages)
    for i in range(len(grid_params)):
        dict_params = grid_params[i]
        logger.info('进度：{}/{}'.format(i+1, len(grid_params)))
        for k, v in dict_params.items():
            config.params[k] = v
        _model, _auc, _round, _ = model_fit(config, X_train, y_train)
        if _auc >= best_auc:
            best_params = dict_params.copy()
            best_auc = _auc
            best_model =_model
        messages = 'Best_auc={}, auc={}, round={}, params={}'.format(best_auc, _auc, _round, dict_params)
        logger.info(messages)
        print(messages)
    search_messages = '搜索完毕;Best auc={}, Best params are \n{}'.format(best_auc, best_params)
    logger.info(search_params)
    print(search_messages)
    return best_model, best_params, best_auc
if __name__ == '__main__':
    #get feature
    feature_path = 'features/'
    train_data, test_data = load_feat(re_get = False, feature_path = feature_path)
    train_feats = train_data.columns.values
    test_feats = test_data.columns.values
    drop_columns = list(filter(lambda x:x not in test_feats, train_feats))
    drop_columns2 = list(filter(lambda x:x not in train_feats, test_feats))
    X_train = train_data.drop(drop_columns, axis=1)
    y_train = np.array(train_data['label']).astype(float)
    X_test = test_data.drop(drop_columns2, axis=1)
    data_messages = 'X_train.shape={}, X_test.shape={}'.format(X_train.shape, X_test.shape)
    logger.info(data_messages)
    print(data_messages)

    #grid search
    tic = time.time()
    config = Config()
    search_params = {'num_leaves':[20, 30, 40, 50],
                     'learning_rate': [0.025, 0.05, 0.1, 0.15, 0.20]}
    logger.info(search_params)
    best_model, best_params, best_auc = my_grid_search(config, search_params, lgb_fit, X_train, y_train)
    print("花费时间{}s".format(time.time() - tic))
    check_path(config.best_model_path)
    best_model.save_model(config.best_model_path)

    #predict
    #best_model = joblib.load(config.best_model_path)   #加载模型
    now= time.strftime('%m%d-%H%M%S')
    result_path = 'result/result_lgb_search_{}.csv'.format(now)
    check_path(result_path)
    lgb_predict(best_model, X_test, result_path)

