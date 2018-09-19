from m1_xgb import xgb_fit, xgb_predict
from m1_xgb import Config as XGB_Config
from m2_lgb import lgb_fit, lgb_predict
from m2_lgb import Config as LGB_Config
from m3_cgb import cgb_fit, cgb_predict
from m3_cgb import Config as CGB_Config
from data_helper import load_feat
from my_utils import check_path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, GridSearchCV #K折交叉验证和网格搜索调参
from sklearn.linear_model import LogisticRegression

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from sklearn.svm import SVR

import logging.handlers
import time

"""模型融合
每个单模型需要具备两个函数：
    1.model_fit，返回best model
    2.返回best model具有predict预测类别概率
    sklearn_stacking:所有best_model都是sklearn中的模型， 这样具有统一的fit,predict接口
"""

LOG_FILE = 'log/stacking.log'
check_path(LOG_FILE)
handler = logging.handlers.RotatingFileHandler(LOG_FILE, maxBytes=1024 * 1024, backupCount=1)
fmt = '%(asctime)s - %(filename)s:%(lineno)s - %(name)s - %(messags)s'
formatter = logging.Formatter(fmt)
handler.setFormatter(formatter)
logger = logging.getLogger('stack')
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

SEED = 3


def run_my_stack():
    """模型融合函数对于测试集"""
    X_train, y_train, X_test = load_features()
    fit_funcs = list()
    predict_func = list()
    configs = list()
    MAX_ROUND = 3

    #lgb
    num_leaves = [31, 41, 51, 61, 71, 81, 91]   #lgb_model的叶子节点
    feature_fractions = [0.4, 0.4, 0.4, 0.3, 0.3, 0.3, 0.3] #所用特征占比
    for i in range(len(num_leaves)):
        lgb_config = LGB_Config()
        lgb_config.params['num_leaves'] = num_leaves[i]
        lgb_config.params['feature_fraction'] = feature_fractions[i]
        lgb_config.seed = np.random.randint(0, 10000)
        lgb_config.save_model_path = None
        #lgb_config.max_round = MAX_ROUND
        #这里存放了LGB_Config的参数不同组合
        configs.append(lgb_config)
        fit_funcs.append(lgb_fit)
        predict_func.append(lgb_predict)
    #xgb_model参数，
    max_depths = [6, 7] #树的深度
    colsample_bytrees = [0.7, 0.6]  #所用特征的占比
    for i in range(len(max_depths)):
        xgb_config = XGB_Config()
        xgb_config.params['max_depth'] = max_depths[i]
        xgb_config.params['colsample_bytrees'] = colsample_bytrees[i]
        xgb_config.seed = np.random.randint(0, 10000)
        xgb_config.save_model_path = None
        #xgb_config.max_ronnd = MAX_ROUND
        #这里有将xgb_model的参数，训练函数，测试函数放进个列表中
        configs.append(xgb_config)
        fit_funcs.append(xgb_fit)
        predict_func.append(xgb_predict)

    #cgb
    max_depths = [8]
    for i in range(len(max_depths)):
        cgb_config = CGB_Config()
        cgb_config.params['depth'] = max_depths[i]
        cgb_config.seed = np.random.randint(0, 10000)
        cgb_config.save_model_path = None
        #cgb_config.max_round = MAx_ROUND
        configs.append(cgb_config)
        fit_funcs.append(cgb_fit)
        predict_func.append(cgb_predict)

    X_train_stack, y_train_stack, X_test_stack = my_stacking(fit_funcs, predict_func, configs, X_train, y_train,X_test)
    result_path = 'result/my_stack_result-{}.csv'.format(time.strftime('%m%d-%H%M%S'))
    y_pred_prob = final_fit_predict(X_train_stack, y_train_stack, X_test_stack, save_result_path=result_path)





def load_features(feature_path = 'features/'):
    """加载数据"""
    train_data, test_data = load_feat(re_get = False, feature_path= feature_path)
    train_feats = train_data.columns.values
    test_feats = test_data.columns.values
    drop_columns = list(filter(lambda x:x not in test_feats, train_feats))
    drop_columns2 = list(filter(lambda x:x not in train_feats, test_feats))
    X_train = train_data.drop(drop_columns, axis=1)
    y_train = np.array(train_data['label']).astype(float)
    X_test = test_data.drop(drop_columns2, axis=1)
    data_messages = 'X_train.shape={}, X_test.shape={}'.format(X_train.shaoe, X_test.shape)
    print(data_messages)
    logger.info(data_messages)
    return X_train, y_train, X_test


def my_stacking(fit_funcs, predict_funcs, configs, X_train, y_train, X_test, n_fold = 5):
    """融合模型，xgb,lgb,cgb
    对于每个模型应该指定它们的训练模型和预测模型函数
    Args:
        fit_funcs:训练函数，返回最佳模型
        predict_funcs:预测函数，返回正类的概率
        X_train:shape=[n_sample_train, n_feats], 训练数据
        y_train:训练数据标签
        X_test:shape=[n_sample_test,n_feats],需要预测的数据
        n_fold:交叉验证的折数
        save_path:保存模型融合后的特征
    Returns:
        X_train_stack:shape=[n_sample_train, n_model]
        y_train_stack:shape=[n_sample_test, 1]
        X_test_stack:shape=[n_sample_test, n_model]
        """
    #df_lgb_feat_score = pd.read_csv('features/lgb_features.csv')
    #features_names = df_lgb_feat_score.feature.values
    #y_train = y_train.values
    if type(X_train) == pd.DataFrame:
        X_train = X_train.values
    if type(X_test) == pd.DataFrame:
        X_test = X_test.values
    if (type(y_train) == pd.DataFrame) | (type(y_train) == pd.Series):
        y_train = y_train.values
    n_train = len(X_train)
    n_test = len(X_test)
    n_model = len(fit_funcs)
    #打乱训练数据
        #与shuffle一样都是随机对数据洗牌， 但shuffle是在原数据上操作，而permutation不是，是生成另一个
            #这里是打乱0-n_train中的数值，然后当做索引
    new_idx = np.random.permutation(n_train)
    y_train = y_train[new_idx]
    X_train = X_train[new_idx]
    print('X_train.shape={}, X_test.shape={}'.format(X_train.shape, X_test.shape))
    kf = KFold(n_splits=n_fold, shuffle=False)

    X_train_stack = None
    X_test_stack = None
    tic = time.time()
    #训练模型
    for k in range(n_model):
        messages = '训练模型;{}/{}，用时：{}s'.format(k+1, n_model, time.time() - tic)
        print(messages)
        logger.info(messages)
        #取出每个模型函数及对应的参数字典
        fit_func = fit_funcs[k]
        predict_func = predict_funcs[k]
        config = configs[k]
        #这里创建元素0的数组，根据数据大小
            #off_train只有一行
        off_train = np.zeros((n_train,))
        off_test_skf = np.zeros((n_test, n_fold))
        #kf是k折交叉验证的对象，这里传入X_train是将其划分为n_fold等分，取n_fold-1分为train_data, 剩下一份为test_data
            #而split属性是返回train_data和test_dat对应的索引
        for i, (train_idx, test_idx) in enumerate(kf.split(X_train)):
            X_tr = X_train[train_idx]
            y_tr = y_train[train_idx]
            X_te = X_test[test_idx]
            best_model, best_auc, _, _,=fit_func(config, X_tr, y_tr)
            messages = 'k-折进度：{}/{}, auc={}'.format(i+1, n_fold, best_auc)
            logger.info(messages)
            y_pred_prob = predict_func(best_model, X_te)

            off_train[test_idx] = y_pred_prob
            off_test_skf[:, i] = predict_func(best_model, X_test)
        off_train = off_train.reshape(-1, 1)
        off_test = np.mean(off_test_skf, axis = 1).reshape(-1, 1)
        if X_train_stack is None:
            #不明白为什么X_train_stack = off_train, off_train只是随机的k折中的1折
                #我知道了，上面k-fold中的参数shuffle = False,没有打乱，则循环后的off_train是全部的X_train的结果
            X_train_stack = off_train
            X_test_stack = off_test
        else:
            X_train_stack = np.hstack((X_train_stack, off_train))   #水平(按列的顺序)进行合并
            X_test_stack = np.hstack((X_test_stack, off_test))
        stack_feats_path = 'feature/stack_feats/round_{}.npz'.format(k+1)   #训练过程进行中
        check_path(stack_feats_path)
        #将多个数组(可以是不同类型,如;list, dict）保存在一个文件中。
            #不明白为什么此处的y_train = y_train, y_train是打乱后的标签，难道X_train也保留打乱的顺序？？？
        np.savez(stack_feats_path, X_train=X_test_stack, y_train=y_train, X_test=X_test_stack)
    messages = 'X_train_stack.shape={}, X_test_stack.shape={}'.format(X_train_stack.shape, X_test_stack.shape)
    print(messages)
    logger.info(messages)
    save_path = 'features/stack_feat_{}.npz'.format(time.strftime('%m%d-%H%M%S'))
    check_path(save_path)
    np.savez(save_path, X_train=X_train_stack, y_train=y_train, X_test=X_train_stack)
    messages = '模型融合，保存特征到{}'.format(save_path)
    logger.info(messages)
    print(messages)
    return X_train_stack, y_train, X_train_stack

def final_fit_predict(X_train, y_train, X_test, save_result_path):
    """最终使用融合后的特征进行训练
    使用逻辑回归对结果的0-1规划
    """
    params_grids = {
        'C':list(np.linspace(0.0001, 10, 100))
    }
    print('Begin final fit with params:{}'.format(params_grids))
    #逻辑回归模型来作交叉验证
    grid = GridSearchCV(LogisticRegression(penalty='l2', max_iter=120), param_grid=params_grids, cv = 5, scoring='roc_auc')
    grid.fit(X_train, y_train)
    try:
        message = 'Final fit: 参数组合;{}；\n最优参数:{}；\n最好评分:{}；\n最佳模型:{}'.format(params_grids, grid.best_params_, grid.best_score_, grid.best_estimator_)
        logger.info(message)
        print(message)
    except Exception as e:
        print(e.message)
    y_pred_prob = grid.predict_proba(X_test)[:,1]
    if save_result_path is not None:
        df_result = pd.DataFrame()
        df_result['userid'] = X_test['userid']
        df_result['orderType'] = y_pred_prob
        df_result = df_result.groupby('userid', as_index=False)['orderType'].mean()
        df_result.to_csv(save_result_path, index=False)
        print('保存结果到{}'.format(save_result_path))
    return y_pred_prob



if __name__ == "__main__":
    run_my_stack()