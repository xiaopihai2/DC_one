
from my_utils import time_to_date
import pandas as pd
import numpy as np
"""
    处理df_actions用户行为数据，构造一些数据特征：一共是327维,
        主要熟练运用pandas和numpy对数据的处理，选择，构造， 筛选等操作，真的很棒， 虽然代码中含有许多重复操作
        但是构造什么特征，怎样构造，是对数据处理问题的核心问题？
"""






def get_action_feat(df_actions,
                    dict_last_action_rate_0,
                    dict_last_action_rate_1,
                    dict_last_pair_rate_0,
                    dict_last_pair_rate_1,
                    dict_last_triple_rate_0,
                    dict_last_triple_rate_1):
    """
    :param df_action: 用户行为信息数据
    #订单下单前的单个行为的比例
    :param dict_last_action_rate_0:
    :param dict_last_action_rate_1:
    #订单下单前的连续两个行为的比例
    :param dict_last_pair_rate_0:
    :param dict_last_pair_rate_1:
    #订单下单前的连续三个行为的比例：
    :param dict_last_triple_rate_0:
    :param dict_last_triple_rate_1:
    :return:df_action_feat
    """
    """获取所有的动作相关的特征："""
    #***1.所有行为的比例;
    df_action_feat = get_action_ratio(df_actions.copy(), n_seconds = None)
    print("1:",df_action_feat.shape)
    #***2.最早一次行为的时间和最晚一次行为的时间，还有两次行为时间距离
    df_first_last_action_time = get_first_last_action_time(df_actions.copy())
    df_action_feat = pd.merge(df_action_feat, df_first_last_action_time, on = 'userid', how = 'left')
    print("2:",df_action_feat.shape)
    #***3.行为actionType的最早ICI时间和最后一次时间，还有两次时间差
    for action_type in range(1, 10):
        df_each_action_feat = get_each_action_last_time(df_actions.copy(), action_type = action_type)
        df_action_feat = pd.merge(df_action_feat, df_each_action_feat.copy(), on = 'userid', how ='left')
    print("3:",df_action_feat.shape)
    #***4.行为action_type到用户最后一次行为的时间距离
    for action_type in range(1, 10):
        df_action_to_end_diff = get_each_action_to_end_diff(df_actions.copy(), action_type = action_type)
        df_action_feat = pd.merge(df_action_feat, df_action_to_end_diff.copy(), on = 'userid', how = 'left')
    print("4:",df_action_feat.shape)
    #***5.最后10个行为之间的时间间隔
    df_last_diffs = get_last_diff(df_actions, n_last_diff = 10)
    df_action_feat = pd.merge(df_action_feat, df_last_diffs, on = 'userid', how = 'left')
    print("5:",df_action_feat.shape)
    #***6.所有行为之间时间间隔统计特征：
    df_last_diff_statistic_3 = get_last_diff_statistic(df_actions.copy(), n_last_diff=3)
    df_action_feat = pd.merge(df_action_feat, df_last_diff_statistic_3, on = 'userid', how = 'left')
    print("6:",df_action_feat.shape)
    #***7.最后10个行为之间时间间隔的统计特征：
    df_last_diff_statistic_10 = get_last_diff_statistic(df_actions.copy(), n_last_diff=10)
    df_action_feat = pd.merge(df_action_feat, df_last_diff_statistic_10, on = 'userid', how = 'left')
    print("7:",df_action_feat.shape)
    #最后3个行为和10个行为间隔比值;
    df_last_diff_statistic_3_10 = get_last_diff_divide_statistic(df_last_diff_statistic_3, df_last_diff_statistic_10, 3, 10)
    df_action_feat = pd.merge(df_action_feat, df_last_diff_statistic_3_10, on = 'userid', how = 'left')
    print("7.5:",df_action_feat.shape)
    #***8.最后n(=1)个行为的类型：
    df_last_action_types = get_last_action_type(df_actions.copy(), n_action=1, to_one_hot = False)
    df_action_feat = pd.merge(df_action_feat, df_last_action_types, on = 'userid', how = 'left')
    print("8:",df_action_feat.shape)
    #***最后一个行为转化为对应的概率：映射：dict_last_action_rate_0中存放的是每种行为类型在所有用户最后一次行为类型集合中所占的比例
        #这里值得注意的是，这里提到的最后一个行为类型是指df_actions中的数据中每个用户的最后一次行为类型，而字典中是下单之前的最后一个行为类型的占比。要搞清楚两者的区别
    df_action_feat['last_action_type_rate_0'] = df_action_feat['last_action_type'].apply(lambda s:dict_last_action_rate_0.get(s, 0))
    df_action_feat['last_action_type_rate_1'] = df_action_feat['last_action_type'].apply(lambda s:dict_last_action_rate_1.get(s, 0))
    print("8.3:",df_action_feat.shape)
    #***最后一组的行为及其转化概率(与data.py对应，一次是)
    df_last_pair_rate = get_last_pair(df_actions.copy(), dict_last_pair_rate_0, dict_last_pair_rate_1)
    df_action_feat = pd.merge(df_action_feat, df_last_pair_rate, on = 'userid', how = 'left')
    print("8.6:",df_action_feat.shape)
    #***最后3个行为组合及其转化概率
    df_last_triple_rate = get_last_triple(df_actions.copy(), dict_last_triple_rate_0, dict_last_triple_rate_1)
    df_action_feat = pd.merge(df_action_feat, df_last_triple_rate, on = 'userid', how = 'left')
    print("8.9:",df_action_feat.shape)
    #***9.最后一个行为(如;actionType为6)后面的统计;
    for action_type in range(1, 10):
        df_after_action_feats = get_after_action_feat(df_actions.copy(), action_type, n_diff = 2)
        df_action_feat = pd.merge(df_action_feat, df_after_action_feats.copy(), on = 'userid', how = 'left')
    print("9:",df_action_feat.shape)
    #***10.获取两个特特定行为之间的时间距离和行为距离：
    #action_pairs = [[5, 5], [5, 6], [1, 1], [6, 6], [6, 7], [5, 7], [7, 7]], 为啥是这几个？
    action_pairs = [[5, 5], [5, 6], [1, 1], [6, 6], [6, 7], [5, 7], [7, 7]]
    for action1, action2 in action_pairs:
        df_action1_action_2_feats = get_action_pair_feats(df_actions.copy(), action1, action2)
        df_action_feat = pd.merge(df_action_feat, df_action1_action_2_feats.copy(), on = 'userid', how = 'left')
    print("10:",df_action_feat.shape)
    #***11.统计用户出现(action1, action2)连续行为的次数
    action_pairs = [[5, 6]]
    for action1, action2 in action_pairs:
        df_action_pair_count_feats = get_action_pair_count(df_actions.copy(), action1, action2)
        df_action_feat = pd.merge(df_action_feat, df_action_pair_count_feats, on = 'userid', how = 'left')
    print("11:",df_action_feat.shape)
    #***12.将2-4类行为全部替换为‘2’统计‘2’和其他行为之间的时间距离
    df_actions_24 = df_actions.copy()
    df_actions_24.actionType = df_actions_24.actionType.apply(lambda x: x if x not in [2, 3, 4] else 2)
    action_pairs = [[2, 5], [2, 6]]
    for action1, action2 in action_pairs:
        df_action1_action_2_feats = get_action_pair_feats(df_actions_24, action1, action2)
        df_action_feat = pd.merge(df_action_feat, df_action1_action_2_feats, on = 'userid', how = 'left')
    print("12:",df_action_feat.shape)
    #***13.最后一个行为的年月日时分：
    action_types = ['all', 1, 5, 6, 7, 8, 2, 3, 4, 9]
    for action_type in action_types:
        df_last_action_date = get_last_action_date_time(df_actions.copy(), action_type).copy()
        df_action_feat = pd.merge(df_action_feat, df_last_action_date, on = 'userid', how = 'left')
    print("13:",df_action_feat.shape)
    #***14.获取各种行为组合的比例
    df_pair_rate = get_pair_rate(df_actions.copy())
    df_action_feat = pd.merge(df_action_feat, df_pair_rate, on = 'userid', how = 'left')
    print("14:",df_action_feat.shape)
    return df_action_feat



def get_action_ratio(df_actions, n_seconds = None):
    """统计每个用户的各种行为比例，取用户最后的n_seconds的行为进行统计
    Args:
        df_actions:用户行为数据
        n_seconds:取该用户最后的n_secnds时间段进行统计
        如果n_seconds == None，统计用户所有的行为。此外，主要统计：1小时(3600), 12小时(12 * 3600), 48小时(48 * 3600)
    Returns:
        action{}_ratio_{}.format(1~9,n_seconds):行为比例
        action_count_{}.format(n_seconds):行为总数
    """
    if n_seconds is not None:
        #得到用户的最后一次操作时间与原数据合并，应该是为了得到在其他行为与最后行为时间之差
        df_last_action_time = df_actions[['userid', 'actionTime']].groupby('userid').tail(1).copy()
        df_last_action_time.rename(columns = {'actionTime':'last_actionTime'}, inplace = True)
        df_actions = pd.merge(df_actions, df_last_action_time, on = 'userid', how = 'left')
        df_actions['diff_with_last_action'] = df_actions.last_actionTime - df_actions.actionTime
        #取距离小于n_seconds的行为
        df_actions = df_actions[df_actions['diff_with_last_action'] <= n_seconds]
    else:
        n_seconds = 'all'

    #统计行为的比例：
    df_actions['action_count'] = 1
    df_count = df_actions[['userid', 'action_count']].groupby('userid', as_index = False).count()   #用户行为总数
    df_action_type = pd.get_dummies(df_actions['actionType'], prefix = 'actionType_{}'.format(n_seconds))   #这里独热编码，再加上前缀就构成了所要求的列
    df_action_ratio = pd.concat([df_actions['userid'], df_action_type], axis = 1)   #横向合并
    df_action_ratio = df_action_ratio.groupby('userid', as_index=False).sum()   #按id分组，对每种行为进行求和
    for action_type in range(1, df_action_ratio.shape[1]):
        col_name = 'actionType_{}_{}'.format(n_seconds, action_type)
        #这里求得每个行为在总行为中所占比例
        df_action_ratio[col_name] = df_action_ratio[col_name] / df_count['action_count']
    df_action_ratio['n_action_{}'.format(n_seconds)] = df_count['action_count'].values
    return df_action_ratio

def get_first_last_action_time(df_actions):
    """获取最早一次的时间和最晚一次的行为时间， 还有两次行为时间的间隔"""
    df_actions = df_actions.sort_values(['userid', 'actionTime'])
    grouped_action_time = df_actions[['userid', 'actionTime']].groupby('userid', as_index = False)
    df_mintime = grouped_action_time.min()
    df_mintime.rename(columns = {'actionTime':'first_action_time'}, inplace = True)
    #最后一次行为距离"当前时刻"
    df_maxtime = grouped_action_time.max()
    df_maxtime.rename(columns = {'actionTime':'last_action_time'}, inplace = True)
    df_actions_feat = pd.merge(df_mintime, df_maxtime, on = 'userid', how = 'left')
    df_actions_feat['diff_last_first_action'] = df_actions_feat['last_action_time'] - df_actions_feat['first_action_time']
    return df_actions_feat

def get_each_action_last_time(df_actions, action_type):
    """获取行为action_type的最早一次时间和最后一次时间"""
    #先筛选出actionType的数据
    df_actions = df_actions.loc[df_actions.actionType == action_type][['userid', 'actionTime']]
    grouped = df_actions.groupby('userid')
    #分别取出最早和最晚的数据，再相减得到之差
    df_first_action = grouped.head(1).copy()
    df_first_action.rename(columns = {'actionTime': 'first_action_{}_time'.format(action_type)}, inplace = True)
    df_last_actions = grouped.tail(1).copy()
    df_last_actions.rename(columns = {'actionTime': 'last_action_{}_time'.format(action_type)}, inplace =True)
    df_each_action_feat = pd.merge(df_first_action, df_last_actions, on = 'userid', how = 'left')
    diff_col = 'first_last_action_{}_diff'.format(action_type)
    df_each_action_feat[diff_col] = df_each_action_feat['last_action_{}_time'.format(action_type)] - df_each_action_feat['first_action_{}_time'.format(action_type)]
    return df_each_action_feat

def get_each_action_to_end_diff(df_actions, action_type):
    """获取最后一次行为action_type的距离最后一次时间距离
    TODO：把2,3,4归为一类。
    TODO：获取用户每次历史订单时间距离每个动作的时间距离
    """
    df_actions = df_actions.sort_values(['userid', 'actionTime'])
    #得到每个用户的最后行为的时间
    df_last_action_time = df_actions[['userid', 'actionTime']].groupby('userid', as_index = False).tail(1).copy()
    df_last_action_time.rename(columns = {'actionTime': 'last_action_time'}, inplace = True)
    #进行合并，好让最后行为时间减去其他行为时间
    df_actions = pd.merge(df_actions, df_last_action_time, on = 'userid', how = 'left')
    #确定actionType
    df_actions = df_actions[df_actions['actionType'] == action_type][['userid', 'actionTime', 'last_action_time']]
    df_actions['{}_to_end_diff'.format(action_type)] = df_actions['last_action_time'] - df_actions['actionTime']
    #对于每个用户来说，行为可能一样，只是时间不同，我们取最近的一次
    df_action_to_end_diff = df_actions[['userid', '{}_to_end_diff'.format(action_type)]].groupby('userid', as_index =False).tail(1).copy()
    return df_action_to_end_diff

def get_last_diff(df_actions, n_last_diff = 10):
    """获取最后n_last_diff个行为之间的时间间隔
    Args：
        df_actions：用户行为数据
        n_last_diff：最近的n_last_diff个动作
    Return：
        final_diff：该用户最近的n_last_diff个行为之间的时间间隔
    """
    df_actions = df_actions.sort_values(['userid', 'actionTime'])
    df_actions['next_userid'] = df_actions.userid.shift(-1) #id列向上移动一个单位
    df_actions['next_actionTime'] = df_actions.actionTime.shift(-1)
    #这样进行筛选会将剔除每个用户的是。比如原：1->n,变变成了1->n-1,但next一列变成2->n
    df_actions = df_actions.loc[df_actions.next_userid == df_actions.userid].copy()
    #其实上条语句就进行了错位，使行为时间错开了一位，这样可以相邻的两次行为的时间间隔
    df_actions['action_diff'] = df_actions['next_actionTime'] - df_actions['actionTime']

    #取每个用户的最后一行(原始的df_actions是倒数第二行)赋值，减去其他行为的代号数值，再比较n_last_diff
    #感觉技巧满满！！！
    df_actions['rank_'] = range(df_actions.shape[0])
    df_max_rank = df_actions.groupby('userid', as_index = False)['rank_'].agg({'user_max_rank':np.max})
    df_actions = pd.merge(df_actions, df_max_rank, on = 'userid', how = 'left')
    df_actions['last_rank'] = df_actions['user_max_rank'] - df_actions['rank_']
    df_actions = df_actions[df_actions['last_rank'] < n_last_diff]     #取最后n_last_diff的时间间隔

    df_last_diffs = df_actions[['userid', 'last_rank', 'action_diff']].copy()
    #set_index:将‘userid’和‘last_rank’作为索引，变成双索引，但使用unstack()后，变成了相当于透视表的形式，列名为last_rank的值，值为action_diff
    df_last_diffs = df_last_diffs.set_index(['userid', 'last_rank']).unstack()
    df_last_diffs.reset_index(inplace=True)
    df_last_diffs.columns = ['userid'] + ['final_diff_{}'.format(i) for i in range(n_last_diff)]
    return df_last_diffs

def get_last_diff_statistic(df_actions, n_last_diff = 'all'):
    """获取最后n_last_diff个行为之间时间间隔的统计特征"""
    #先上移，再选择对应的行数相减，得到的值作数理统计
    df_actions['next_userid'] = df_actions.userid.shift(-1)
    df_actions['next_actionTime'] = df_actions.actionTime.shift(-1)
    df_actions = df_actions.loc[df_actions.next_userid == df_actions.userid].copy()
    df_actions['action_diff'] = df_actions['next_actionTime'] - df_actions['actionTime']
    if n_last_diff is not 'all':
        #按理说，上个特征g构建也可以用tail()函数
        df_n_last_diff = df_actions.groupby('userid', as_index = False).tail(n_last_diff).copy()
        df_last_diff_statistic = df_n_last_diff.groupby('userid', as_index = False).action_diff.agg({
            #这里可不可以加上其他统计特征，如中位数，最小值，最大值等
            'last_{}_action_diff_mean'.format(n_last_diff):np.mean,
            'last_{}_action_diff_std'.format(n_last_diff):np.std
        })
    else:
        grouped_user = df_actions.groupby('userid', as_index = False)
        df_last_diff_statistic = grouped_user.action_diff.agg({
            'last_{}_action_diff_mean'.format(n_last_diff):np.mean,
            'last_{}_action_diff_std'.format(n_last_diff):np.std
        })
    return df_last_diff_statistic

def get_last_diff_divide_statistic(df_1, df_2, n_last_diff1, n_last_diff2):
    df_last_diff_divide = pd.merge(df_1, df_2, on='userid', how = 'left')
    df_last_diff_divide['mean_divide_{}_{}'.format(n_last_diff1, n_last_diff2)] = df_last_diff_divide['last_{}_action_diff_mean'.format(n_last_diff1)] / \
        df_last_diff_divide['last_{}_action_diff_mean'.format(n_last_diff2)]
    df_last_diff_divide['std_divide_{}_{}'.format(n_last_diff1, n_last_diff2)] = df_last_diff_divide['last_{}_action_diff_std'.format(n_last_diff1)] / \
        df_last_diff_divide['last_{}_action_diff_std'.format(n_last_diff2)]
    df_last_diff_divide = df_last_diff_divide[['userid', 'mean_divide_{}_{}'.format(n_last_diff1, n_last_diff2), 'std_divide_{}_{}'.format(n_last_diff1, n_last_diff2)]].copy()
    return df_last_diff_divide

def get_last_action_type(df_actions, n_action = 1, to_one_hot = True):
    """ 获取最后n_action个行为类型
    Args:
        df_actions:用户行为数据
        n_action:最后n_action个行为
        to_one_hot:是否进行one_hot处理
    Returns:
        最后的n_action个特征
    """
    df_actions = df_actions.sort_values(['userid', 'actionTime'])
    grouped_user = df_actions.groupby('userid', as_index = False)
    if n_action == 1:
        #经过排序后拿到每个用户最后一行数据
        df_last_action_types = grouped_user['userid', 'actionType'].tail(1).copy()
        df_last_action_types.rename(columns = {'actionType':'last_action_type'}, inplace = True)
        if to_one_hot:
            df_last_action_types = pd.get_dummies(df_last_action_types, columns=['last_action_type'])
        return df_last_action_types
    df_actions['rank_'] = range(df_actions.shape[0])
    #每个用户最后的行为rank
    user_rank_max = df_actions.groupby('userid', as_index = False)['rank_'].agg('user_rank_max',np.max)
    df_actions = pd.merge(df_actions, user_rank_max, on = 'userid', how = 'left')
    #每个用户的倒数第几个行为：
    df_actions['user_rank_sub'] = df_actions['user_rank_max'] - df_actions['rank_']
    df_last_action_types = user_rank_max[['userid']]
    for k in range(n_action):
        #这里选择每个用户的k个行为类型
        df_last_action = df_actions[df_actions.user_rank_sub == k][['userid', 'actionType']]
        if to_one_hot:
            df_last_action = pd.get_dummies(df_last_action, columns=['actionType'], prefix='user_last_{}_action_type'.format(k))
        else:
            df_last_action.renam(columns = {'actionType':'last_action_type_{}'.format(k)})
        df_last_action_types = pd.merge(df_last_action_types, df_last_action, on = 'userid', how = 'left')
    return df_last_action_types

def get_last_pair(df_actions, dict_pair_rate_0, dict_pair_rate_1):
    """获取最后一组行为类型， 并将其转化为对应的概率"""
    df_actions['last_userid'] = df_actions.userid.shift(1)
    df_actions['last_actionType'] = df_actions.actionType.shift(1)
    df_actions = df_actions.loc[df_actions['last_userid'] == df_actions['userid']].copy()
    #将错开的两列的actio_type进行对应成对，最后进行合并
    action_pairs = list(zip(df_actions.last_actionType.values, df_actions.actionType.values))
    #合并
    action_pairs = list(map(lambda s:str(int(s[0])) + "_" +str(int(s[1])), action_pairs ))
    df_actions['action_pair'] = action_pairs
    #取得每个用户的最后两个行为类型的组合数据
    df_user_last_pair_type = df_actions[['userid', 'action_pair']].groupby('userid', as_index = False).tail(1)
    df_user_last_pair_type['last_pair_rate_0'] = df_user_last_pair_type['action_pair'].apply(lambda s:dict_pair_rate_0.get(s, 0))
    df_user_last_pair_type['last_pair_rate_1'] = df_user_last_pair_type['action_pair'].apply(lambda s:dict_pair_rate_1.get(s, 0))
    #这里算的是最后两个行为在精品中的占比在总占比中占多少
    df_user_last_pair_type['last_pair_convert_rate'] = df_user_last_pair_type['last_pair_rate_1'] / (df_user_last_pair_type['last_pair_rate_0'] + df_user_last_pair_type['last_pair_rate_1'])
    #按道理不应该出现无限大的数啊？？？算了也没用这列作为特征
    df_user_last_pair_type['last_pair_convert_rate'] = df_user_last_pair_type['last_pair_convert_rate'].apply(lambda x:0 if x == np.inf else x)
    df_last_pair_rate = df_user_last_pair_type[['userid', 'last_pair_rate_0', 'last_pair_rate_1']]
    return df_last_pair_rate

def get_last_triple(df_actions, dict_last_triple_rate_0, dict_last_triple_rate_1):
    """获取最后连续3个行为，并将其转化为对应的转化概率。"""
    df_actions['last_userid_2'] = df_actions.userid.shift(2)
    df_actions['last_actionType_2'] = df_actions.actionType.shift(2)
    df_actions['last_userid'] = df_actions.userid.shift(1)
    df_actions['last_actionType'] = df_actions.actionType.shift(1)
    df_actions = df_actions.loc[df_actions.last_userid_2== df_actions.userid].copy()
    action_triple = list(zip(df_actions.last_actionType_2.values, df_actions.last_actionType.values, df_actions.actionType.values))
    action_triple = list(map(lambda s:str(int(s[0])) + "_" + str(int(s[1])) + "_" + str(int(s[2])), action_triple))
    df_actions['action_triple'] = action_triple
    df_user_last_triple_type = df_actions[['userid', 'action_triple']].groupby('userid', as_index = False).tail(1)
    df_user_last_triple_type['last_triple_rate_0'] = df_user_last_triple_type['action_triple'].apply(lambda s:dict_last_triple_rate_0.get(s, 0))
    df_user_last_triple_type['last_triple_rate_1'] = df_user_last_triple_type['action_triple'].apply(lambda s:dict_last_triple_rate_1.get(s, 0))
    df_user_last_triple_type['last_triple_convert_rate'] = df_user_last_triple_type['last_triple_rate_1'] / (df_user_last_triple_type['last_triple_rate_0'] + df_user_last_triple_type['last_triple_rate_1'])
    df_user_last_triple_type['last_triple_convert_rate'] = df_user_last_triple_type['last_triple_convert_rate'].apply(lambda x:0 if x == np.inf else x)
    df_last_triple_rate = df_user_last_triple_type[['userid', 'last_triple_rate_0', 'last_triple_rate_1']]
    return df_last_triple_rate


def get_after_action_feat(df_actions, action_type, n_diff = 2):
    """获取最近一个行为后面的统计特征：这里的最后指的是action_type(每个用户)的最后
    Args:
        df_actions:用户行为数据;
        action_type:行为类型；
        n_diff：action_type后面的n_diff个行为的时间间隔
    Returns:
        'n_action_after_{}'.action_type：行为个数；
        'diff1_after_{}'.action_type:该行为后第一个行为的时间间隔
        'diff2_after_{}':该行为后第2个行为的时间间隔
        'mean_diff_after_{}'.action_type时间间隔均值
        'std_diff_after_{}'.action_type时间间隔标准差
    """
    df_actions = df_actions.sort_values(['userid', 'actionTime'])
    df_after_action_feats = df_actions.groupby('userid', as_index = False).tail(1)[['userid']]
    df_actions['rank_'] = range(df_actions.shape[0])
    df_actions_op = df_actions[df_actions.actionType == action_type].copy()
    #每个用户的该行为类型的最后一个行为op的rank, 相当于取得了每个用户该行为(传的参数的action_type)最后一次的的代号
    op_max_rank = df_actions_op.groupby('userid', as_index = False)['rank_'].agg({'op_max_rank':np.max})
    df_actions = pd.merge(df_actions, op_max_rank, on='userid', how = 'left')
    df_actions['after_op'] = df_actions['rank_'] - df_actions['op_max_rank']
    #只要>=0的行即是该类型行为结束后的行为
    df_actions = df_actions[df_actions['after_op'] >= 0 ]

    #这里应该是该行为后的行为总数，后面还减一， 暂时还不明白是为什么？之所以减一应该是上面>=0造成的，只有减一才是该行为之后的行为总数
    df_actions['count'] = 1
    df_count = df_actions[['userid', 'count']].groupby('userid', as_index = False).count()
    df_count.rename(columns = {'count':'n_action_after_{}'.format(action_type)}, inplace = True)
    df_count['n_action_after_{}'.format(action_type)] = df_count['n_action_after_{}'.format(action_type)].apply(lambda x:x - 1)

    #统计后面行为的时间间隔
    df_actions['next_userid'] = df_actions.userid.shift(-1) #上移
    df_actions['next_action_time'] = df_actions.actionTime.shift(-1)
    df_actions['next_action_diff'] = df_actions['next_action_time'] - df_actions['actionTime']
    df_actions = df_actions[df_actions.next_userid == df_actions.userid].copy()     #得到action_type后的所有行为的时间间隔(两两之间)
    #统计每个用户的时间间隔的统计量
    user_op_time = df_actions.groupby('userid', as_index = False)['next_action_diff'].agg({
        'mean_diff_after_{}'.format(action_type):np.mean,
        'std_diff_after_{}'.format(action_type):np.std,
        'min_diff_after_{}'.format(action_type):np.min,
        'max_diff_after_{}'.format(action_type):np.max
    })
    #这里可以理解为我们所要数据的初始化，前者是id，后者是每个用户该行为后的行为总数(我是这么理解的)
    df_after_action_feats = pd.merge(df_after_action_feats, df_count, on = 'userid', how = 'left')

    #获取该行为后面的n_diff个行为
    for k in range(n_diff):
        #这里得到是每个用户在该类型后的n_diff个行为的时间间隔，并没有计算它们的统计量，因为n_diff太小
        df_last_diff = df_actions[df_actions.after_op == k][['userid', 'next_action_diff']]
        df_last_diff.rename(columns = {'next_action_diff': 'diff_{}_after_{}'.format(k ,action_type)}, inplace = True)
        df_after_action_feats = pd.merge(df_after_action_feats, df_last_diff, on = 'userid', how ='left')
    #这里可以看出，我们所要的数据中包含了每个用户在该行为后所有行为的时间间隔的数学统计量和n_diff个行为的时间间隔
    df_after_action_feats = pd.merge(df_after_action_feats, user_op_time, on = 'userid', how = 'left')
    return df_after_action_feats

def get_action_pair_feats(df_actions, action1 = 5, action2 = 6):
    """获取两个特定行为之间的时间距离
        必须action1在之前，否则返回np.nan
    Args:
        action_pair:行为对
    Returns:
        mean_diff_action1_action2:action_1到action_2的平均时间距离
        last_diff_action1_action2:最后一次action1到action2的时间距离
        diff_rate_action1_action2: = last_diff_action1_action2 / mean_diff_action1_action2
    """
    df_actions_pair_feat = df_actions.groupby('userid', as_index = False).tail(1)[['userid']].copy()
    df_actions['rank_'] = range(df_actions.shape[0])
    df_actions = df_actions.loc[(df_actions.actionType == action1) | (df_actions.actionType == action2)].copy() #选出相对应的行为类型数据
    #还是老规矩，错位选择
    df_actions['next_userid'] = df_actions.userid.shift(-1)
    df_actions['next_actionType'] = df_actions.actionType.shift(-1)
    df_actions['next_actionTime'] = df_actions.actionTime.shift(-1)
    df_actions['next_action_rank'] = df_actions.rank_.shift(-1)

    df_actions = df_actions.loc[df_actions.userid == df_actions.next_userid].copy()
    df_actions['next_action_diff'] = df_actions['next_actionTime'] - df_actions['actionTime']
    df_actions['next_action_count'] = df_actions['next_action_rank'] - df_actions['rank_']
    df_actions =df_actions[['userid', 'next_action_count', 'next_action_diff']]
    #这里的df_action1_action2_count可以理解为action1的行为与邻近的action2之间的行为距离：
        #如;[5, 5]中用户第3次行为出现第一次5， 第7次行为出现第二次5，则就是7-3 = 4
            #但我还是有一点疑问，当action2 != action1,这样处理感觉没有理由
    df_action1_action2_count = df_actions.groupby('userid', as_index = False)['next_action_count'].agg({
        '{}_{}_action_count_mean'.format(action1, action2): np.mean,
        '{}_{}_action_count_max'.format(action1, action2): np.max,
        '{}_{}_action_count_min'.format(action1, action2): np.min,
        '{}_{}_action_count_std'.format(action1, action2): np.std
    })
    #同理：这里的操作只是由行为距离变为了时间距离
    df_action1_action2_time = df_actions.groupby('userid', as_index = False)['next_action_diff'].agg({
        '{}_{}_action_diff_mean'.format(action1, action2): np.mean,
        '{}_{}_action_diff_max'.format(action1, action2): np.max,
        '{}_{}_action_diff_min'.format(action1, action2): np.min,
        '{}_{}_action_diff_std'.format(action1, action2):np.std
    })
    #这里是选择行为距离和时间距离的最后一次
    df_last_a1_a2_action = df_actions.groupby('userid', as_index = False).tail(1).copy()
    df_last_a1_a2_action.rename(columns = {
        'next_action_count': 'last_{}_{}_action_count'.format(action1, action2),
        'next_action_diff': 'last_{}_{}_action_diff'.format(action1,action2)
    }, inplace = True)

    df_actions_pair_feat = pd.merge(df_actions_pair_feat, df_action1_action2_count, on = 'userid', how = 'left')
    df_actions_pair_feat = pd.merge(df_actions_pair_feat, df_action1_action2_time, on = 'userid', how = 'left')
    df_actions_pair_feat = pd.merge(df_actions_pair_feat, df_last_a1_a2_action, on = 'userid', how = 'left')

    #最后一次的占比;
    df_actions_pair_feat['last_count_divide_men_{}_{}'.format(action1, action2)] = df_actions_pair_feat['last_{}_{}_action_count'.format(action1, action2)] / \
        df_actions_pair_feat['{}_{}_action_count_mean'.format(action1, action2)]
    df_actions_pair_feat['last_diff_divide_men_{}_{}'.format(action1, action2)] = df_actions_pair_feat['last_{}_{}_action_diff'.format(action1, action2)] / \
        df_actions_pair_feat['{}_{}_action_diff_mean'.format(action1, action2)]
    return df_actions_pair_feat

def get_action_pair_count(df_actions, action1, action2):
    """统计用户出现(action1, action2)连续行为的次数(TODO：和时间间隔)
    TODO:划分时间窗口统计， 如最近一天内的统计
    """
    df_actions['next_userid'] = df_actions.userid.shift(-1)
    df_actions['next_actionType'] = df_actions.actionType.shift(-1)
    df_actions['next_actionTime'] = df_actions.actionTime.shift(-1)
    df_actions = df_actions.loc[df_actions.userid == df_actions.next_userid]
    #还是错位选择两个行为同时出现的行， 并计算它们的时间距离
    df_actions = df_actions.loc[(df_actions.actionType == action1) & (df_actions.next_actionType == action2)]
    df_actions['diff'] = df_actions['next_actionTime'] - df_actions['actionTime']
    df_actions['action_{}_{}_count'.format(action1, action2)] = 1
    #联合连续行为组合出现的次数;
    grouped_user = df_actions.groupby('userid', as_index = False)
    #得到每个用户该组合出现的次数
    df_action_pair_count = grouped_user['action_{}_{}_count'.format(action1, action2)].count()
    #统计时间间隔(时间间隔的统计量)：
    df_action_pair_diff = grouped_user['diff'].agg({
        'pair_count_action_diff_mean_{}_{}'.format(action1, action2): np.mean,
        'pair_count_action_diff_std_{}_{}'.format(action1, action2): np.std
    })
    df_action_pair_count_feats = pd.merge(df_action_pair_count, df_action_pair_diff, on = 'userid', how ='left')
    return df_action_pair_count_feats

def get_last_action_date_time(df_actions, action_type):
    """获取最后一次行为action_type的日期"""
    #没有取年，月，天，取得是周， 小时， 分钟
    if action_type is not 'all':
        df_actions = df_actions[df_actions.actionType == action_type].copy()
    df_actions = df_actions.sort_values(['userid', 'actionTime'])
    #得到每个用户的最后行为的数据
    df_actions = df_actions[['userid', 'actionTime']].groupby('userid', as_index = False).tail(1).copy()
    df_actions['last_action_date'] = df_actions['actionTime'].apply(time_to_date)
    df_actions['last_action_date'] = pd.to_datetime(df_actions.last_action_date, format='%Y-%m-%d %H:%M:%S')
    # df_actions['last_action_year'] = df_actions.last_action_date.apply(lambda dt:dt.year)
    # df_actions['last_action_month'] = df_actions.last_action_date.apply(lambda dt:dt.month)
    # df_actions['last_action_day'] = df_actions.last_action_date.apply(lambda dt:dt.day)
    df_actions['last_action_weekday_{}'.format(action_type)] = df_actions.last_action_date.apply(lambda dt:dt.weekday())
    df_actions['last_action_hour_{}'.format(action_type)] = df_actions.last_action_date.apply(lambda dt:dt.hour)
    df_actions['last_action_minute_{}'.format(action_type)] = df_actions.last_action_date.apply(lambda dt:dt.minute)
    df_last_action_date = df_actions.drop(['actionTime', 'last_action_date'], axis = 1)
    return df_last_action_date



def merge_action(action_type):
    """将行为进行归类"""
    if action_type in [2, 3, 4]:
        return 2
    if action_type in [7, 8, 9]:
        return 7
    else:return action_type


def get_pair_rate(df_actions):
    """获取不同行为组合的比例：
    统计每种行为最后一次出现距离最后的时间
    """
    df_actions = df_actions.sort_values(['userid', 'actionTime'])
    df_end_time = df_actions[['userid', 'actionTime']].groupby('userid', as_index = False).tail(1)
    df_end_time.rename(columns = {'actionTime':'end_time'}, inplace = True)
    df_actions = pd.merge(df_actions, df_end_time, on = 'userid', how = 'left')
    df_actions.actionType = df_actions.actionType.apply(merge_action)   #行为归类
    df_actions['last_userid'] = df_actions.userid.shift(1)
    df_actions['last_actionType'] = df_actions.actionType.shift(1)
    df_actions = df_actions.loc[df_actions.last_userid == df_actions.userid].copy()
    action_type = df_actions.actionType.values.astype(int)
    last_action_type = df_actions.last_actionType.astype(int)
    action_pairs = list(zip(last_action_type, action_type))
    action_pairs = list(map(lambda s: str(s[0]) + '_' + str(s[1]), action_pairs))
    df_actions['action_pair'] = action_pairs
    df_actions['diff_to_end'] = df_actions['end_time'] - df_actions['actionTime']
    #统计每种行为最后出现的时间
    df_last_pair_feat = df_actions.drop_duplicates(['userid'])[['userid']].copy()
    #因为action_pairs:表示的是用户连续行为的组合，如：第n次行为_第n+1次行为， 这里set变成集合
    action_pair_types = set(action_pairs)
    for action_pair in  action_pair_types:
        #筛选符合标准的数据
        df_actions_ = df_actions[df_actions.action_pair == action_pair][['userid', 'diff_to_end']].copy()
        #按id分组,选出该行为组合最后出现的数据:
        df_last_pair_to_end = df_actions_.groupby('userid', as_index = False).tail(1)
        #这里可以得到用户该组合行为最后出现的时间(按s[1]的时间算)到用户最后行为时间的距离
        df_last_pair_to_end.rename(columns = {'diff_to_end':'pair_to_end_{}'.format(action_pair)}, inplace = True)
        df_last_pair_feat = pd.merge(df_last_pair_feat, df_last_pair_to_end, on = 'userid', how = 'left')
    #统计每种行为组合出现的频次
    df_action_pair = pd.get_dummies(df_actions[['userid', 'action_pair']], columns = ['action_pair'], prefix='action_pair')
    df_action_pair['total_count'] = 1
    grouped_user = df_action_pair.groupby('userid', as_index=False)
    #默认在0维上求和，则，行为id，列为action_pair_action1_action2，total_count， 而前者求和的值是每个用户出现该组合的次数，而后者相当于所有的总和
    df_pair_rate = grouped_user.sum()
    #过滤器：筛选组合列
    pair_columns = list(filter(lambda x:x.startswith('action_pair'), df_pair_rate.columns.values))      #过滤器
    for pair_column in pair_columns:
        #计算该组合的占比
        df_pair_rate[pair_column] = df_pair_rate[pair_column] / df_pair_rate['total_count']
    df_pair_rate.drop(['total_count'], axis = 1, inplace = True)
    df_last_pair_feat = pd.merge(df_last_pair_feat, df_pair_rate, on = 'userid', how = 'left')
    return df_last_pair_feat


