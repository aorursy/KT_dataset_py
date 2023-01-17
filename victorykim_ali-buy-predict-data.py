import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

from functools import reduce

warnings.filterwarnings("ignore")

%matplotlib inline
user_data = pd.read_csv("../input/tianchi_fresh_comp_train_user.csv")

item_data = pd.read_csv("../input/tianchi_fresh_comp_train_item.csv")
# 转换时间类型

user_data["time"] = pd.to_datetime(user_data["time"], format="%Y%m%d %H")
# 用户相关特征提取函数

# 用户执行各项type操作的总次数

def user_action_total_counts(data):

    data["user_action_total_counts"] = 1

    feature = data[["user_id", "user_action_total_counts"]].groupby(["user_id"], as_index=False).count()

    return feature

# 用户执行type操作的次数  data:数据 type:操作类型 name:新增列名称

def user_type_counts(data, type, name):

    data[name] = 1

    feature = data[data["behavior_type"] == type][["user_id", name]].groupby(["user_id"], as_index=False).count()

    return feature



# 商品相关特征提取函数

# 商品被执行各项type操作的总次数

def item_action_total_counts(data):

    data["item_action_total_counts"] = 1

    feature = data[["item_id", "item_action_total_counts"]].groupby(["item_id"], as_index=False).count()

    return feature

# 对商品执行各项type操作的用户总数

def item_total_user_counts(data):

    data = data[["user_id", "item_id"]].drop_duplicates()

    data["item_total_user_counts"] = 1

    feature = data[["item_id", "item_total_user_counts"]].groupby(["item_id"], as_index=False).count()

    return feature

# 商品被执行type操作的次数

def item_type_counts(data, type, name):

    data[name] = 1

    feature = data[data["behavior_type"] == type][["item_id", name]].groupby(["item_id"], as_index=False).count()

    return feature



# 商品类型相关特征提取函数

# 此类型商品被执行各项type操作的总次数

def category_action_total_counts(data):

    data["category_action_total_counts"] = 1

    feature = data[["item_category", "category_action_total_counts"]].groupby(["item_category"], as_index=False).count()

    return feature

# 对此类型商品执行各项type操作的用户总数

def category_total_user_counts(data):

    data = data[["user_id", "item_category"]].drop_duplicates()

    data["category_total_user_counts"] = 1

    feature = data[["item_category", "category_total_user_counts"]].groupby(["item_category"], as_index=False).count()

    return feature

# 此类型商品被执行type操作的次数

def category_type_counts(data, type, name):

    data[name] = 1

    feature = data[data["behavior_type"] == type][["item_category", name]].groupby(["item_category"], as_index=False).count()

    return feature



# 用户-商品相关特征提取函数

# 用户对商品执行各项type操作的总次数

def user_item_action_total_counts(data):

    data["user_item_action_total_counts"] = 1

    feature = data[["user_id", "item_id", "user_item_action_total_counts"]].groupby(["user_id", "item_id"], as_index=False).count()

    return feature

# 用户对商品执行type操作的次数

def user_item_type_counts(data, type, name):

    data[name] = 1

    feature = data[data["behavior_type"] == type][["user_id", "item_id", name]].groupby(["user_id", "item_id"], as_index=False).count()

    return feature

# 用户对商品执行type操作的最后时间 

def user_item_last_type_time(data, type, name):

    feature = data[data["behavior_type"] == type][["user_id", "item_id", "time"]].groupby(["user_id", "item_id"], as_index=False).max()

    feature.rename(columns={"time": name}, inplace=True)

    return feature



# 用户第一次购买特定商品从浏览到购买的时间间隔、过程中的浏览次数和加入购物车到购买的时间间隔

def user_item_look_to_buy(data):

    # 用户购买过的用户-商品组合

    buy_user_item = data[data["behavior_type"] == 4][["user_id", "item_id"]].drop_duplicates()

    # 用户-商品组合的数据

    data = pd.merge(buy_user_item, data, how="left", on=["user_id", "item_id"])[["user_id", "item_id", "behavior_type", "time"]]

    # 第一次浏览商品的时间

    earliest_look = data[data["behavior_type"] == 1].groupby(["user_id", "item_id"], as_index=False).min()

    earliest_look.rename(columns={"time": "earliest_look_time"}, inplace=True)

    # 第一次购买商品的时间

    earliest_buy = data[data["behavior_type"] == 4].groupby(["user_id", "item_id"], as_index=False).min()

    earliest_buy.rename(columns={"time": "earliest_buy_time"}, inplace=True)

    # 第一次将商品加入购物车的时间

    earliest_add = data[data["behavior_type"] == 3].groupby(["user_id", "item_id"], as_index=False).min()

    earliest_add.rename(columns={"time": "earliest_add_time"}, inplace=True)

    # 第一次购买商品中浏览到购买的时间间隔(单位：小时)

    feature = pd.merge(earliest_buy, earliest_look, how="left", on=["user_id", "item_id"])

    feature["earliest_user_item_timedelta_look_to_buy"] = (feature["earliest_buy_time"] - feature["earliest_look_time"]).dt.total_seconds()/360000

    feature = feature[feature["earliest_user_item_timedelta_look_to_buy"] >= 0]

    feature = feature[["user_id", "item_id", "earliest_look_time", "earliest_buy_time", "earliest_user_item_timedelta_look_to_buy"]]

    # 第一次购买商品过程中的浏览次数

    data = pd.merge(feature, data, how="left", on=["user_id", "item_id"])

    data = data[(data["behavior_type"] == 1)&(data["time"] <= data["earliest_buy_time"])]

    data["item_look_counts_before_buy"] = 1

    item_look_counts_before_buy = data[["user_id", "item_id", "item_look_counts_before_buy"]].groupby(["user_id", "item_id"], as_index=False).count()

    feature = pd.merge(feature, item_look_counts_before_buy, how="left", on=["user_id", "item_id"])

    # 返回结果

    return feature[["user_id", "item_id", "item_look_counts_before_buy", "earliest_user_item_timedelta_look_to_buy"]]



# 用户-商品类型相关特征提取函数

# 用户对同种类型商品执行各项type操作的总次数

def user_category_action_total_counts(data):

    data["user_category_action_total_counts"] = 1

    feature = data[["user_id", "item_category", "user_category_action_total_counts"]].groupby(["user_id", "item_category"], as_index=False).count()

    return feature

# 用户对同种类型商品执行type操作的次数

def user_category_type_counts(data, type, name):

    data[name] = 1

    feature = data[data["behavior_type"] == type][["user_id", "item_category", name]].groupby(["user_id", "item_category"], as_index=False).count()

    return feature

# 用户对同种类型商品执行type操作的最后时间

def user_category_last_type_time(data, type, name):

    feature = data[data["behavior_type"] == type][["user_id", "item_category", "time"]].groupby(["user_id", "item_category"], as_index=False).max()

    feature.rename(columns={"time": name}, inplace=True)

    return feature

# 用户第一次购买同种类型商品从浏览到购买的时间间隔和过程中的浏览次数

def user_category_look_to_buy(data):

    # 用户购买过的用户-商品类型组合

    buy_user_item = data[data["behavior_type"] == 4][["user_id", "item_category"]].drop_duplicates()

    # 用户-商品类型组合的数据

    data = pd.merge(buy_user_item, data, how="left", on=["user_id", "item_category"])[["user_id", "item_category", "behavior_type", "time"]]

    # 第一次浏览同种类型商品的时间

    earliest_look = data[data["behavior_type"] == 1].groupby(["user_id", "item_category"], as_index=False).min()

    earliest_look.rename(columns={"time": "earliest_look_time"}, inplace=True)

    # 第一次购买同种类型商品的时间

    earliest_buy = data[data["behavior_type"] == 4].groupby(["user_id", "item_category"], as_index=False).min()

    earliest_buy.rename(columns={"time": "earliest_buy_time"}, inplace=True)

    # 第一次将同种类型商品加入购物车的时间

    earliest_add = data[data["behavior_type"] == 3].groupby(["user_id", "item_category"], as_index=False).min()

    earliest_add.rename(columns={"time": "earliest_add_time"}, inplace=True)

    # 第一次购买同种类型商品中浏览到购买的时间间隔(单位：小时)

    feature = pd.merge(earliest_buy, earliest_look, how="left", on=["user_id", "item_category"])

    feature["earliest_user_category_timedelta_look_to_buy"] = (feature["earliest_buy_time"] - feature["earliest_look_time"]).dt.total_seconds()/3600

    feature = feature[feature["earliest_user_category_timedelta_look_to_buy"] >= 0]

    feature = feature[["user_id", "item_category", "earliest_look_time", "earliest_buy_time", "earliest_user_category_timedelta_look_to_buy"]]

    # 第一次购买同种类型商品过程中的浏览次数

    data = pd.merge(feature, data, how="left", on=["user_id", "item_category"])

    data = data[(data["behavior_type"] == 1)&(data["time"] <= data["earliest_buy_time"])]

    data["category_look_counts_before_buy"] = 1

    category_look_counts_before_buy = data[["user_id", "item_category", "category_look_counts_before_buy"]].groupby(["user_id", "item_category"], as_index=False).count()

    feature = pd.merge(feature, category_look_counts_before_buy, how="left", on=["user_id", "item_category"])

    # 返回结果

    return feature[["user_id", "item_category", "category_look_counts_before_buy", "earliest_user_category_timedelta_look_to_buy"]]
def merge_user(data1, data2):

    data = pd.merge(data1, data2, how="left", on="user_id")

    return data

def merge_item(data1, data2):

    data = pd.merge(data1, data2, how="left", on="item_id")

    return data

def merge_category(data1, data2):

    data = pd.merge(data1, data2, how="left", on="item_category")

    return data

def merge_user_item(data1, data2):

    data = pd.merge(data1, data2, how="left", on=["user_id", "item_id"])

    return data

def merge_user_category(data1, data2):

    data = pd.merge(data1, data2, how="left", on=["user_id", "item_category"])

    return data
# 构造特征函数 predict_date:预测日期

def get_feature(predict_date):

    train_data = user_data[user_data["time"] < predict_date]

#     # 用户相关特征

#     # u1:用户执行各项type操作的总次数

#     u1 = user_action_total_counts(train_data)

#     # u2:用户浏览商品的次数

#     u2 = user_type_counts(train_data, 1, "user_look_counts")

#     # u3:用户收藏商品的次数

#     u3 = user_type_counts(train_data, 2, "user_like_counts")

#     # u4:用户加入购物车的次数

#     u4 = user_type_counts(train_data, 3, "user_add_counts")

#     # u5:用户购买商品的次数

#     u5 = user_type_counts(train_data, 4, "user_buy_counts")

    

#     # 商品相关特征

#     # i1:商品被执行各项type操作的总次数

#     i1 = item_action_total_counts(train_data)

#     # i2:对商品执行各项type操作的用户总数

#     i2 = item_total_user_counts(train_data)

#     # i3:商品被浏览的次数

#     i3 = item_type_counts(train_data, 1, "item_look_counts")

#     # i4:商品被收藏的次数

#     i4 = item_type_counts(train_data, 2, "item_like_counts")

#     # i5:商品被加入购物车的次数

#     i5 = item_type_counts(train_data, 3, "item_add_counts")

#     # i6:商品被购买的次数

#     i6 = item_type_counts(train_data, 4, "item_buy_counts")

    

#     # 商品类型特征

#     # c1:商品被执行各项type操作的总次数

#     c1 = category_action_total_counts(train_data)

#     # c2:对商品执行各项type操作的用户总数

#     c2 = category_total_user_counts(train_data)

#     # c3:此类型商品被浏览的次数

#     c3 = category_type_counts(train_data, 1, "category_look_counts")

#     # c4:此类型商品被收藏的次数

#     c4 = category_type_counts(train_data, 2, "category_like_counts")

#     # c5:此类型商品被加入购物车的次数

#     c5 = category_type_counts(train_data, 3, "category_add_counts")

#     # c6:此类型商品被购买的次数

#     c6 = category_type_counts(train_data, 4, "category_buy_counts")

    

    # 用户-商品相关特征

    # ui1:用户对商品执行各项type操作的总次数

#     ui1 = user_item_action_total_counts(train_data)

    # ui2:用户浏览特定商品的次数

    ui2 = user_item_type_counts(train_data, 1, "user_item_look_counts")

#     # ui3:用户收藏特定商品的次数

#     ui3 = user_item_type_counts(train_data, 2, "user_item_like_counts")

#     # ui4:用户将特定商品加入购物车的次数

#     ui4 = user_item_type_counts(train_data, 3, "user_item_add_counts")

    # ui5:用户购买特定商品的次数

    ui5 = user_item_type_counts(train_data, 4, "user_item_buy_counts")

    # ui6:用户浏览特定商品的最后时间

    ui6 = user_item_last_type_time(train_data, 1, "user_item_last_look_time")

    # ui7:用户收藏特定商品的最后时间

    ui7 = user_item_last_type_time(train_data, 2, "user_item_last_like_time")

    # ui8:用户将特定商品加入购物车的最后时间

    ui8 = user_item_last_type_time(train_data, 3, "user_item_last_add_time")

    # ui9:用户购买特定商品的最后时间

    ui9 = user_item_last_type_time(train_data, 4, "user_item_last_buy_time")

    # ui10:用户第一次购买特定商品从浏览到购买的时间间隔和过程中的浏览次数

    ui10 = user_item_look_to_buy(train_data)

    

    # 用户-商品类型相关特征

    # uc1:用户对同种类型商品执行各项type操作的总次数

#     uc1 = user_category_action_total_counts(train_data)

#     # uc2:用户浏览同种类型商品的次数

    uc2 = user_category_type_counts(train_data, 1, "user_category_look_counts")

#     # uc3:用户收藏同种类型商品的次数

#     uc3 = user_category_type_counts(train_data, 2, "user_category_like_counts")

#     # uc4:用户将同种类型商品加入购物车的次数

#     uc4 = user_category_type_counts(train_data, 3, "user_category_add_counts")

    # uc5:用户购买同种类型商品的次数

    uc5 = user_category_type_counts(train_data, 4, "user_category_buy_counts")

    # uc6:用户浏览同种类型商品的最后时间

    uc6 = user_category_last_type_time(train_data, 1, "user_category_last_look_time")

    # uc7:用户收藏同种类型商品的最后时间

    uc7 = user_category_last_type_time(train_data, 2, "user_category_last_like_time")

    # uc8:用户将同种类型商品加入购物车的最后时间

    uc8 = user_category_last_type_time(train_data, 3, "user_category_last_add_time")

    # uc9:用户购买同种类型商品的最后时间

    uc9 = user_category_last_type_time(train_data, 4, "user_category_last_buy_time")

    # uc10:用户第一次购买同种类型商品从浏览到购买的时间间隔和过程中的浏览次数

    uc10 = user_category_look_to_buy(train_data)

    

    # 联结表

    train_data = train_data[["user_id", "item_id", "item_category"]].drop_duplicates()

#     train_data = reduce(merge_user, [train_data, u1, u2, u3, u4, u5])

#     train_data = reduce(merge_item, [train_data, i1, i2, i3, i4, i5, i6])

#     train_data = reduce(merge_category, [train_data, c1, c2, c3, c4, c5, c6])

#     train_data = reduce(merge_user_item, [train_data, ui1, ui2, ui3, ui4, ui5, ui6, ui7, ui8, ui9, ui10])

#     train_data = reduce(merge_user_category, [train_data, uc1, uc2, uc3, uc4, uc5, uc6, uc7, uc8, uc9, uc10])

    train_data = reduce(merge_user_item, [train_data, ui2, ui5, ui6, ui7, ui8, ui9, ui10])

    train_data = reduce(merge_user_category, [train_data, uc2, uc5, uc6, uc7, uc8, uc9, uc10])

    # 距预测日期的时间间隔 单位：小时

    train_data["user_item_last_look_to_now"] = (pd.to_datetime(predict_date) - train_data["user_item_last_look_time"]).dt.total_seconds()/3600

    train_data["user_item_last_like_to_now"] = (pd.to_datetime(predict_date) - train_data["user_item_last_like_time"]).dt.total_seconds()/3600

    train_data["user_item_last_add_to_now"] = (pd.to_datetime(predict_date) - train_data["user_item_last_add_time"]).dt.total_seconds()/3600

    train_data["user_item_last_buy_to_now"] = (pd.to_datetime(predict_date) - train_data["user_item_last_buy_time"]).dt.total_seconds()/3600

    train_data["user_category_last_look_to_now"] = (pd.to_datetime(predict_date) - train_data["user_category_last_look_time"]).dt.total_seconds()/3600

    train_data["user_category_last_like_to_now"] = (pd.to_datetime(predict_date) - train_data["user_category_last_like_time"]).dt.total_seconds()/3600

    train_data["user_category_last_add_to_now"] = (pd.to_datetime(predict_date) - train_data["user_category_last_add_time"]).dt.total_seconds()/3600

    train_data["user_category_last_buy_to_now"] = (pd.to_datetime(predict_date) - train_data["user_category_last_buy_time"]).dt.total_seconds()/3600

    # 删除无用特征

    drop_columns = ["user_item_last_look_time", "user_item_last_like_time", "user_item_last_add_time", "user_item_last_buy_time"]

    drop_columns += ["user_category_last_look_time", "user_category_last_like_time", "user_category_last_add_time", "user_category_last_buy_time"]

    train_data = train_data.drop(drop_columns, axis=1)

    # 处理缺失值

#     fill_columns = ["user_look_counts", "user_like_counts", "user_add_counts", "user_buy_counts"]

#     fill_columns += ["item_look_counts", "item_like_counts", "item_add_counts", "item_buy_counts"]

#     fill_columns += ["category_look_counts", "category_like_counts", "category_add_counts", "category_buy_counts"]

    fill_columns = ["user_item_look_counts", "user_item_buy_counts"]

    fill_columns += ["user_category_look_counts", "user_category_buy_counts"]

    train_data[fill_columns] = train_data[fill_columns].fillna(0)

    return train_data
data_train = get_feature("2014-12-17")

data_eval = get_feature("2014-12-18")

data_test = get_feature("2014-12-19")

data_train.to_csv("data_train.csv", index=False)

data_eval.to_csv("data_eval.csv", index=False)

data_test.to_csv("data_test.csv", index=False)

user_data.to_csv("user_data.csv", index=False)

item_data.to_csv("item_data.csv", index=False)