import pandas as pd

def get_comment_feat(df_comment):
    """获取评论特征
    加上评分有提高， 加上tags数量下降
    """
    """
        个人觉得可以对评论内容进行文本情感分析，感觉可能会有所提高
    """
    #df_comment = df_comment[['userid', 'orderid', 'rating', 'tags', 'commentsKeyWords']]
    df_comment_feat = df_comment[['userid', 'rating']].copy()
    df_comment.rename(columns = {'rating': 'comment_rating'}, inplace = True)
    return df_comment_feat