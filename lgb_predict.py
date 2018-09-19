from data_helper import load_feat
from my_utils import check_path
import lightgbm as lgb
import numpy as np
import time
import pandas as pd
from sklearn import preprocessing

"""
这里是用lgb模型预测train的结果与真实结果的比较,然后输出保存在文件夹中
"""

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


def lgb_predict_on_train(model, X_train, y_train, save_result_path=None):
    X_train = lgb_DMatrix(X_train)
    y_pred_prob = model.predict(X_train)
    df_result = pd.DataFrame()
    df_result['userid'] = X_train['userid']
    df_result['label'] = y_train
    df_result['predict_type'] = y_pred_prob
    df_result = df_result.groupby('userid', as_index=False)[['label', 'predict_type']].mean()
    df_result.to_csv(save_result_path, index=False)
    print('保存结果到：{}'.format(save_result_path))


if __name__ == '__main__':
    save_model_path = 'model/lgb.txt'
    now = time.strftime('%m%d-%H%M%S')
    result_path = 'result/result_train_lab_{}'.format(now)
    check_path(result_path)
    #get feature
    feature_path = 'features/'
    train_data, test_data = load_feat(re_get=False, feature_path = feature_path)
    train_feats = train_data.columns.values
    test_feats = test_data.columns.values
    drop_columns = list(filter(lambda x:x not in test_feats, train_feats))
    drop_columns2 = list(filter(lambda x:x not in train_feats, test_feats))
    X_train = train_data.drop(drop_columns, axis = 1)
    y_train = train_data['label']
    X_test = test_data.drop(drop_columns2, axis=1)
    data_messages = 'X_train.shape={}, X_test.shape={}'.format(X_train.shape, X_test.shape)
    print(data_messages)

    lgb_model = lgb.Booster(model_file=save_model_path)
    lgb_predict_on_train(lgb_model, X_train, y_train, result_path)