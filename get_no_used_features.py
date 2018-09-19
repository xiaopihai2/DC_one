import pandas as pd
import lightgbm as lgb

def get_xgb_features(data_path='features/test_data.csv',
                     feature_score_path='features/xgb_feature_score.csv',
                     xgb_features_path = 'features/xgb_features.csv'):
    """统计没有用到的特征"""
    df_feature_score = pd.read_csv(feature_score_path)
    df_test_data = pd.read_csv(data_path)
    all_features = df_test_data.columns.values
    used_features = df_feature_score.feature.values
    print('n_all_features={}, n_feature_used={}'.format(len(all_features)), len(used_features))
    #作差集得到没有使用的特征，并得到它到的大小
    no_used_features = list(set(all_features).difference(set(used_features)))   #difference：取差集
    n_no_used = len(no_used_features)
    print('n_no_used_feature={}'.format(n_no_used))
    #这里将每用的特征重要程度设为0， 并追加到原特征重要程度表的末尾

    df_no_used = pd.DataFrame({'feature': no_used_features, 'score':[0] * n_no_used})
    df_xgb_features = pd.concat([df_feature_score, df_no_used])
    df_xgb_features.to_csv(xgb_features_path, index=False)

def get_lgb_features(lgb_model_path='model/lgb.txt', lgb_feature_path='features/lgb_features.csv'):
    lgb_model = lgb.Booster(model_file=lgb_model_path)
    feature_name = lgb_model.feature_name()
    feature_importance = lgb_model.feature_importance()
    df_lgb_features = pd.DataFrame({'feature':feature_name, 'score':feature_importance})
    df_lgb_features = df_lgb_features.sort_values('score', ascending=False)
    df_lgb_features.to_csv(lgb_feature_path, index=False)

if __name__ == '__main__':
    get_xgb_features()
    get_lgb_features(lgb_model_path='model/lgb.txt')

