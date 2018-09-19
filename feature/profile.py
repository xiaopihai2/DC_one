import pandas as pd


"""
    同样这里也是合并省会，
    但将性别，省会，年龄按照one_hot进行编码，但感觉太粗糙了
"""


def get_profile_feat(df_profile):
    """
    获取个人信息特征
    """
    #删除用户信息表中id重复的数据，并将userid列的数据复制到df_profile_feat里
    df_profile_feat = df_profile.drop_duplicates(['userid'])[['userid']].copy()
    #省会列的数据处理
    df_province = get_province(df_profile.copy())
    df_profile_feat = pd.merge(df_profile_feat, df_province, on = 'userid', how = 'left')   #按照'userid'进行左连接
    df_profile_ = df_profile[['userid', 'gender', 'age']].copy()
    df_dummy_feat = pd.get_dummies(df_profile_)
    df_profile_feat = pd.merge(df_profile_feat, df_dummy_feat, on='userid', how = 'left')
    return df_profile_feat

def get_province(df_profile):
    #按照省会来划分几线城市
    df_profile.province = df_profile.province.apply(map_province)
    #one_hot编码
    df_province = pd.get_dummies(df_profile[['userid', 'province']], columns = ['province'])
    return df_province


def map_province(province):
    """合并省会"""
    if (province == '北京') | (province == '上海'):
        return province
    if province in ['广东','四川', '辽宁','浙江', '湖北', '江苏', '重庆', '湖南']:
        return '二线'
    else:
        return '三线'

if __name__ == '__main__':
    df_profile = pd.read_csv('../data/trainingset/userProfile_train.csv')
    df_profile_feat = get_profile_feat(df_profile)
    print(df_profile_feat.head(10))