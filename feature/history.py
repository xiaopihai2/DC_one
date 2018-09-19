
from my_utils import time_to_date
import pandas as pd
import numpy as np

"""
    对用户订单历史数据表进行特征处理：
        1.得到精品订单在用户订单中的占比，且得到最后一单的具体时间
        2.得到用户去过的国家在旅行中所占比例(可以看出爱去那个国家)，注意这里的国家进行过处理，合并
        3.与2相同的城市占比
        4.与2相同洲的占比
        注：2,3,4 得到都是一个很大的稀疏矩阵
"""





def get_history_feat(df_history):

    """获取历史订单特征。时间相同的订单去重处理
    统计每个用户的下单数量，和每个用户购买精品订单数量，统计每个用户购买精品订单的比例
    """
    #或取精品订单的比例：
    df_jp_ratio_feat = get_jp_ratio(df_history)
    df_first_order_feats = get_first_last_order_feat(df_history)
    df_history_feat = pd.merge(df_jp_ratio_feat, df_first_order_feats, on = 'userid', how = 'left')
    #去过的国家次数：(这里包含了用户去过的国家在去过的所有国家中的占比)
    df_country_count_feat = get_country_count(df_history)
    df_history_feat = pd.merge(df_history_feat, df_country_count_feat, on = 'userid', how = 'left')
    #去过的城市次数,这个稀疏矩阵更大，而且他的columns是城市的集合，如果要用的话应该在精炼一些。
    df_city_count_feat = get_city_count(df_history)
    df_history_feat = pd.merge(df_history_feat, df_city_count_feat, on = 'userid', how = 'left')
    #去过的洲次数：
    df_continent_count_feat = get_continent_count(df_history)
    df_history_feat = pd.merge(df_history_feat, df_continent_count_feat, on = 'userid', how = 'left')
    return df_history_feat


def get_jp_ratio(df_history):
    """统计每个用户下单的数量和购买精品的比例"""
    #按订单时间和用户id进行分类，然后将它们的'orderType'进行求和，列名为‘real_type’， reset_index，原行索引作为一列保留，列名为index
        #注意这里，虽然分组后调用了sum函数，但仔细想用户在某一时刻是不可能一下订几单，只能是一单，而这里是按时间和id分组的，所以，sum只是具体化groupby函数的结果
    df_real_type = df_history.groupby(['userid', 'orderTime'])['orderType'].agg([('real_type', np.sum)]).reset_index()
    #这里是判断如果真出现上面出现的结果，一定是订单重复，或记录错误(存在大于2的数据)，所以将大于0的取1
    df_real_type.real_type = df_real_type.real_type.apply(lambda x:1 if x>0 else 0)
    grouped_user = df_real_type[['userid', 'real_type']].groupby('userid', as_index = False)
    #统计该用户订单总数：
    df_total_count = grouped_user.count()
    df_total_count.rename(columns = {'real_type':'n_total_order'}, inplace = True)
    df_jp_count = grouped_user.sum()        #这里得到精品订单的数量
    df_jp_count.rename(columns = {'real_type':'n_jp_order'}, inplace = True)
    df_jp_ratio_feat = pd.merge(df_total_count, df_jp_count, on='userid', how='left')
    df_jp_ratio_feat['jp_rate'] = df_jp_ratio_feat['n_jp_order'] / df_jp_ratio_feat['n_total_order']
    return df_jp_ratio_feat

def get_first_last_order_feat(df_history):
    """获取最后一单的类型和最后一单距离当前时间
    Args:
        df_history:历史订单数据表
    Returns:
        first_order_type: 0或1最后一单的类型
        first_order_time:最后一单到现在的时间
    """
    #按‘orderTime’升序排列
    df_history = df_history.sort_values(by = 'orderTime', ascending=True)[['userid', 'orderType', 'orderTime']]
    #最早，最后一单的类型：
    grouped_user = df_history.groupby(['userid'], as_index = False)
    df_first_order_type = grouped_user['userid', 'orderType'].head(1)   #最早的一单
    df_first_order_type.rename(columns = {'orderType':'first_order_type'}, inplace = True)
    df_last_order_type = grouped_user['userid', 'orderType'].tail(1)
    df_last_order_type.rename(columns = {'orderType': 'last_ordeer_type'}, inplace = True)
    df_first_order_feats = pd.merge(df_first_order_type, df_last_order_type, on='userid', how = 'left')

    #最早，最后一单的时间：
    df_first_order_time = grouped_user['userid', 'orderTime'].head(1)
    df_first_order_time.rename(columns = {'orderTime':'first_order_time'}, inplace = True)
    df_last_order_time = grouped_user['userid', 'orderTime'].tail(1)
    df_last_order_time.rename(columns = {'orderTime':'last_order_time'}, inplace = True)

    #获取最后一单的年月日等
    df_last_order_time['last_order_date'] = df_last_order_time['last_order_time'].apply(time_to_date)
    df_last_order_time['last_order_date'] = pd.to_datetime(df_last_order_time.last_order_date, format='%Y-%m-%d %H:%M:%S')
    df_last_order_time['last_order_year'] = df_last_order_time.last_order_date.apply(lambda dt:dt.year)
    df_last_order_time['last_order_month'] = df_last_order_time.last_order_date.apply(lambda dt:dt.month)
    df_last_order_time['last_order_day'] = df_last_order_time.last_order_date.apply(lambda dt:dt.day)
    df_last_order_time['last_order_weekday'] = df_last_order_time.last_order_date.apply(lambda dt:dt.weekday())
    df_last_order_time['last_order_hour'] = df_last_order_time.last_order_date.apply(lambda dt:dt.hour)
    df_last_order_time['last_order_minute'] = df_last_order_time.last_order_date.apply(lambda dt:dt.minute)
    df_last_order_time.drop('last_order_date', axis =1, inplace = True)

    df_first_order_feats = pd.merge(df_first_order_feats,df_first_order_time, on = 'userid', how = 'left')
    df_first_order_feats = pd.merge(df_first_order_feats,df_last_order_time, on = 'userid', how = 'left')
    return df_first_order_feats

def get_country_count(df_history):
    """统计每个用户到每个国家的比例。注意去重，用户同一时间订单去重"""
    df_history = df_history.drop_duplicates(['userid', 'orderTime']).copy()
    df_history.country = df_history.country.apply(map_country)

    #n_total_trip：用户出去过几次(注意这里的国家是合并后的国家)
    #country_count：用户去这个国家几次(此处的国家也是合并的国家)
    #每个用户所有的国家和：先将同类城市合并再计算次数
    df_total_count = df_history[['userid', 'country']].groupby(['userid'], as_index = False).count()
    df_total_count.rename(columns = {'country':'n_total_trip'}, inplace = True)
    #每个用户每个国家的次数:即某一个用户去去过几次
    df_country_count = df_history.groupby(['userid', 'country'])['country'].agg([('country_count', np.size)]).reset_index()
    #制作透视表， id为行， 然后国家编号1,2等为每列，值为该id用户去某国家的次数
    df_country_count = pd.pivot_table(df_country_count, values='country_count', index='userid', columns = 'country').reset_index()
    df_country_count = df_country_count.fillna(0)   #缺失值补0
    df_country_count.columns = ['userid'] + ['country_{}'.format(i) for i in range(8)]
    df_country_count = pd.merge(df_total_count, df_country_count, on='userid', how = 'left')
    for i in range(8):
        #这里可以得到用户去过的国家在出去的次数中占多少(注意此处国家是合并后的国家)
        df_country_count['country_{}'.format(i)] = df_country_count['country_{}'.format(i)] / df_country_count['n_total_trip']
    df_country_count_feat = df_country_count.drop('n_total_trip', axis = 1)
    #得到一个巨大的稀疏矩阵
    return df_country_count_feat

def map_country(country):
    """根据每个国家的精品率进行分组合并不同国家"""
    if country in ['斐济', '中国香港', '新加坡']:
        return 1
    if country in ['爱尔兰', '澳大利亚', '泰国', '越南', '马来西亚']:
        return 2
    if country in ['法国', '英国', '菲律宾', '加拿大', '柬埔寨', '新西兰', '美国']:
        return 3
    if country in ['日本', '韩国', '中国台湾']:
        return 4
    if country in ['印度尼西亚', '阿联酋', '中国澳门', '荷兰', '挪威', '希腊']:
        return 5
    if country in ['缅甸', '比利时', '土耳其', '瑞典', '埃及', '德国', '芬兰', '西班牙', '意大利']:
        return 6
    if country in ['波兰', '冰岛', '巴西', '毛里求斯', '瑞士', '丹麦', '奥地利', '匈牙利', u'葡萄牙', '捷克', u'俄罗斯']:
        return 7
    else: # [u'老挝', u'摩洛哥', u'尼泊尔', u'墨西哥', u'卡塔尔', u'南非', u'中国']
        return 0

def get_city_count(df_history):
    """统计每个用户到每个城市的次数"""
    df_city_count_feat = df_history.drop_duplicates(['userid', 'orderTime'])
    df_history = df_history.drop_duplicates(['userid', 'orderTime'])
    df_city_count = df_history.groupby(['userid', 'city'])['city'].agg([('city_count', np.size)]).reset_index()
    df_city_count = pd.pivot_table(df_city_count, values='city_count', index = 'userid', columns = 'city').reset_index()
    df_city_count = df_city_count.fillna(0)
    df_city_count_feat = pd.merge(df_city_count_feat, df_city_count, on='userid', how = 'left')
    return df_city_count_feat


def get_continent_count(df_history):
    """
    统计每个用户到每个洲的比例。注意去重，用户同一时间的订单去重：
    """
    df_history = df_history.drop_duplicates(['userid', 'orderTime']).copy() #去重
    df_history.continent = df_history.continent.apply(map_continent)    #将不同洲划分为不同等级
    #每个用户所有的洲和：n_total_trip:用户去出去过几次，这里应该与在统计国家是应该一致
    df_total_count = df_history[['userid', 'continent']].groupby('userid', as_index = False).count()
    df_total_count.rename(columns = {'continent':'n_total_trip'}, inplace = True)
    #每个用户每个洲的次数：
    df_continent_count = df_history.groupby(['userid', 'continent'])['continent'].agg([('continent_count', np.size)]).reset_index()
    df_continent_count = pd.pivot_table(df_continent_count, values='continent_count', index='userid', columns='continent').reset_index()
    df_continent_count = df_continent_count.fillna(0)
    df_continent_count.columns = ['userid'] + ['continent_{}'.format(i) for i in range(3)]
    df_continent_count = pd.merge(df_total_count, df_continent_count, on = 'userid', how = 'left').reset_index()
    for i in range(3):
        df_continent_count['continent_{}'.format(i)] = df_continent_count['continent_{}'.format(i)] / df_continent_count['n_total_trip']
    df_continent_count_feat = df_continent_count.drop('n_total_trip', axis = 1)
    return df_continent_count_feat

def map_continent(continent):
    if continent in ['大洋洲', '北美洲']:
        return 0
    if continent in ['亚洲']:
        return 1
    else:   #['南美洲', '非洲', '欧洲']
        return 2