
import feature.action as action
import feature.comment as  comment
import feature.history as history
import feature.profile as profile
from my_utils import check_path

import numpy as np
import pandas as pd
import os
import time
import pickle


df_profile_train = pd.read_csv('data/trainingset/userProfile_train.csv')
df_action_train = pd.read_csv('data/trainingset/action_train.csv')
df_history_train = pd.read_csv('data/trainingset/orderHistory_train.csv')
df_comment_train = pd.read_csv('data/trainingset/userComment_train.csv')
df_future_train = pd.read_csv('data/trainingset/orderFuture_train.csv')

df_profile_test = pd.read_csv('data/test/userProfile_test.csv')
df_action_test = pd.read_csv('data/test/action_test.csv')
df_history_test = pd.read_csv('data/test/orderHistory_test.csv')
df_comment_test = pd.read_csv('data/test/userComment_test.csv')
df_future_test = pd.read_csv('data/test/orderFuture_test.csv')
df_future_train.rename(columns = {'orderType': 'label'}, inplace=True)      #把表中'orderType'改为'label'



def load_feat(re_get=False, feature_path = None):
    #加载特征：
    """
    如果re_get为真，则运行get_feat获得新的特征，否则按照加载处理后的数据特征
    feature_path:已经处理好的特征数据路径
    返回特征处理好的数据(训练和测试)
    """
    train_data_path = '{}train_data.csv'.format(feature_path)
    test_data_path = '{}test_data.csv'.format(feature_path)
    if not re_get and os.path.exists(feature_path):
        print("加载处理好的数据(路径)：{}".format(feature_path))
        train_data = pd.read_csv(train_data_path)
        test_data = pd.read_csv(test_data_path)
    else:
        tic = time.time()
        print('数据处理中。。。。')
        train_data = get_feat(df_future_train, df_history_train, df_action_train, df_profile_train, df_comment_train)
        test_data = get_feat(df_future_test, df_history_test, df_action_test, df_profile_test, df_comment_test)
        print("数据处理耗费时间{}s".format(time.time() - tic))
        #保存处理好的数据
        if feature_path:
            check_path(feature_path)
            train_data.to_csv(train_data_path, index = False)
            test_data.to_csv(test_data_path, index = False)
            message = '保存数据特征到{}'.format(feature_path)
            print(message)
    return train_data, test_data



def get_feat(df_future, df_history, df_action, df_profile, df_comment):
    dict_last_action_rate_0 = get_action_rate_before_jp(0)
    dict_last_action_rate_1 = get_action_rate_before_jp(1)
    dict_last_pair_rate_0 = get_pair_rate_bafore_jp(0)
    dict_last_pair_rate_1 = get_pair_rate_bafore_jp(1)
    dict_last_triple_rate_0 = get_triple_rate_before_jp(0)
    dict_last_triple_rate_1 = get_triple_rate_before_jp(1)

    df_history_feat = history.get_history_feat(df_history)
    print(df_future.shape)
    #将lable添加进去
    df_features = pd.merge(df_future, df_history_feat, on = 'userid', how = 'left')
    print(df_features.shape)
    #用户信息特征：
    df_profile_feat = profile.get_profile_feat(df_profile)
    df_features = pd.merge(df_features, df_profile_feat, on = 'userid', how = 'left')
    #评论特征;
    df_comment_feat = comment.get_comment_feat(df_comment)
    df_features = pd.merge(df_features, df_comment_feat, on = 'userid', how='left')
    #行为特征;
    df_action_feat = action.get_action_feat(df_action,
                                            dict_last_action_rate_0,
                                            dict_last_action_rate_1,
                                            dict_last_pair_rate_0,
                                            dict_last_pair_rate_1,
                                            dict_last_triple_rate_0,
                                            dict_last_triple_rate_1)
    df_features = pd.merge(df_features, df_action_feat, on = 'userid', how = 'left')
    print(df_features.shape)
    return df_features





def get_action_rate_before_jp(order_type = 1):
    """统计订单下单前的单个动作的比例
        这里传入的是订单类型，即将两种类型分开处理
    """
        #还是先判断是否处理过该特征以保存它的路径，如果没有(第一次运行)就进行处理数据
    save_path = 'dict_last_action_rate_{}.pkl'.format(order_type)
    if os.path.exists(save_path):
        dict_last_action_rate = pickle.load(open(save_path, 'rb'))
        return dict_last_action_rate
    #这里是将训练集和测试集纵向合并，应该是为了一并处理
    df_history = pd.concat([df_history_train, df_history_test])
    #订单类型与传入数据判断，不明白为什么action的数据与history有什么联系
    df_history = df_history[df_history.orderType == order_type].copy()
    df_history.rename(columns = {'orderTime':'orderTime_{}'.format(order_type)}, inplace=True)
    df_action = pd.concat([df_action_train, df_action_test])
    # print(df_action.head(10), df_history.head(10))
    df_action = pd.merge(df_action, df_history[['userid', 'orderTime_{}'.format(order_type)]], on = 'userid', how = 'left')
    #得到行为时间与订单时间之差，看是否订单下单前的行为
    df_action['time_after_order'] = df_action['actionTime'] - df_action['orderTime_{}'.format(order_type)]
    df_action = df_action[df_action['time_after_order'] <=0]
    #某用户在形成订单的最后的行为类型
    df_user_last_action_type = df_action[['userid', 'actionType']].groupby('userid', as_index=False).tail(1)    #tail(1)是返回数据的最后一行,但这里好像是纯粹的想返回数据
    #最后行为的统计量
    sr_action_count = df_user_last_action_type.actionType.value_counts()
    action_types = sr_action_count.index
    action_counts = sr_action_count.values
    n_total_actions = len(df_user_last_action_type)
    dict_last_action_rate = dict()
    for i in range(len(action_types)):
        #得到最后行为的占比
        dict_last_action_rate[action_types[i]] = action_counts[i] / n_total_actions
    pickle.dump(dict_last_action_rate, open(save_path, 'wb'))   #保存
    print("保存dict_last_action_rate到：{}".format(save_path))
    print(dict_last_action_rate)
    return dict_last_action_rate

def get_pair_rate_bafore_jp(order_Type = 1):
    '''统计订单下单前的两个连续的行为类型的比例'''
    save_path = 'dict_last_pair_rate_{}.pkl'.format(order_Type)
    if os.path.exists(save_path):
        dict_last_pair_rate = pickle.load(open(save_path, 'rb'))
        return dict_last_pair_rate
    df_history = pd.concat([df_history_train, df_history_test])
    df_history = df_history[df_history.orderType == order_Type].copy()
    df_history.rename(columns = {'orderTime':'orderTime_{}'.format(order_Type)}, inplace=True)
    df_actions = pd.concat([df_action_train, df_action_test])
    df_actions = pd.merge(df_actions, df_history[['userid', 'orderTime_{}'.format(order_Type)]], on = 'userid', how = 'left')

    #构造两列是id，行为类型列下移一个位置形成的
    df_actions['last_userid'] = df_actions.userid.shift(1)
    df_actions['last_actionType'] = df_actions.actionType.shift(1)
    #这里进行错位，是的每个用户的最后一行数据保留了最后的行为与倒数第二个行为类型，秒啊!
    df_actions = df_actions.loc[df_actions.last_userid == df_actions.userid].copy()
    action_pairs = list(zip(df_actions.last_actionType.values, df_actions.actionType.values))
    action_pairs = list(map(lambda s:str(int(s[0])) + '_' + str(int(s[1])), action_pairs))
    df_actions['action_pair'] = action_pairs

    df_actions['time_after_order'] = df_actions['actionTime'] - df_actions['orderTime_{}'.format(order_Type)]
    df_actions = df_actions[df_actions['time_after_order'] <= 0]
    print(df_actions.head())
    df_user_last_pair_type = df_actions[['userid', 'action_pair']].groupby('userid', as_index=False).tail(1)
    sr_pair_count = df_user_last_pair_type.action_pair.value_counts()
    print(sr_pair_count)
    pair_types = sr_pair_count.index
    pair_counts = sr_pair_count.values
    n_total_pairs = len(df_user_last_pair_type)
    dict_last_pair_rate = dict()
    for i in range(len(pair_types)):
        dict_last_pair_rate[pair_types[i]] = pair_counts[i] / n_total_pairs
    pickle.dump(dict_last_pair_rate, open(save_path, 'wb'))
    print('保存订单前两次的行动类型占比：{}'.format(save_path))
    return dict_last_pair_rate


def get_triple_rate_before_jp(order_type = 1):
    '''下单前的连续三个行为动作的统计比例'''
    save_path = 'dict_last_triple_rate_{}.pkl'.format(order_type)
    if os.path.exists(save_path):
        dict_last_triple_rate = pickle.load(open(save_path, 'rb'))
        return dict_last_triple_rate
    df_history = pd.concat([df_history_train, df_history_test])
    df_history = df_history[df_history.orderType == order_type].copy()
    df_history.rename(columns = {'orderTime':'orderTime_{}'.format(order_type)}, inplace = True)
    df_actions = pd.concat([df_action_train, df_action_test])
    df_actions = pd.merge(df_actions, df_history[['userid', 'orderTime_{}'.format(order_type)]], on='userid', how = 'left')
    #故技重施，只是为了得到订单前最后连续三次的占比
    df_actions['last_userid_2'] = df_actions.userid.shift(2)
    df_actions['last_actionType_2'] = df_actions.actionType.shift(2)
    df_actions['last_userid'] = df_actions.userid.shift(1)
    df_actions['last_actionType'] = df_actions.actionType.shift(1)
    df_actions = df_actions.loc[df_actions.last_userid_2 == df_actions.userid].copy()

    action_triples = list(zip(df_actions.last_actionType_2.values, df_actions.last_actionType.values, df_actions.actionType.values))
    action_triples = list(map(lambda s: str(int(s[0])) + '_' + str(int(s[1])) + '_' + str(int(s[2])), action_triples))
    df_actions['action_triples'] = action_triples
    df_actions['time_after_order'] = df_actions['actionTime'] - df_actions['orderTime_{}'.format(order_type)]
    df_actions = df_actions[df_actions['time_after_order'] <= 0]
    df_user_last_triple_type = df_actions[['userid', 'action_triples']].groupby('userid', as_index = False).tail(1)
    sr_triple_count = df_user_last_triple_type.action_triples.value_counts()
    triple_types = sr_triple_count.index
    triple_count = sr_triple_count.values
    n_total_triples = len(df_user_last_triple_type)
    dict_last_triple_rate = dict()
    for i in range(len(triple_types)):
        dict_last_triple_rate[triple_types[i]] = triple_count[i] / n_total_triples
    pickle.dump(dict_last_triple_rate, open(save_path, 'wb'))
    print("下单前连续三个行为的占比保存。")
    return dict_last_triple_rate
