import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings("ignore")

%matplotlib inline
user_data = pd.read_csv("../input/tianchi_fresh_comp_train_user.csv")

item_data = pd.read_csv("../input/tianchi_fresh_comp_train_item.csv")
user_data.head()
user_data.isnull().sum()/len(user_data)
item_data.head()
item_data.isnull().sum()/len(item_data)
item_list = item_data["item_id"].unique().tolist()
len(user_data[user_data["behavior_type"] == 4])/len(user_data)
# 转换时间类型

user_data["time"] = pd.to_datetime(user_data["time"], format="%Y%m%d %H")

user_data["date"] = user_data["time"].dt.date

user_data["weekday"] = user_data["time"].dt.weekday

user_data["hour"] = user_data["time"].dt.hour
# 商品每天的购买情况 in_itemlist表示是否特指待预测商品子集的商品

def erveryday_buy_counts(in_itemlist=False):

    if in_itemlist == True:

        condition = user_data["item_id"].isin(item_list)

    else:

        condition = True

    item_day_buy_count = user_data[(user_data["behavior_type"] == 4)&(condition)][["user_id", "item_id", "date"]].drop_duplicates()

    item_day_buy_count["buy_count"] = 1

    item_day_buy_count["date"] = item_day_buy_count["date"].apply(lambda x : str(x)[5:]) # 日期只显示月日

    item_day_buy_count = item_day_buy_count[["date", "buy_count"]].groupby("date", as_index=False).count().sort_values(by="date")

    # 绘制柱状图

    f, ax = plt.subplots(figsize=(20, 5))

    fig = sns.barplot(x=item_day_buy_count["date"], y=item_day_buy_count["buy_count"])
# 商品全集每天的购买情况

erveryday_buy_counts(in_itemlist=False)
# 商品子集每天的购买情况

erveryday_buy_counts(in_itemlist=True)
# 用户type操作的星期分布

def weekday_buy_counts(type, in_itemlist=False):

    if in_itemlist == True:

        condition = user_data["item_id"].isin(item_list)

    else:

        condition = True

    weekday_buy_counts = user_data[(user_data["behavior_type"] == type)&(condition)][["weekday"]]

    weekday_buy_counts["count"] = 1

    weekday_buy_counts["weekday"] = weekday_buy_counts["weekday"].apply(lambda x: x+1)

    weekday_buy_counts = weekday_buy_counts.groupby("weekday", as_index=False).count().sort_values(by="count", ascending=False).reset_index(drop=True)

    return weekday_buy_counts

# 用户type操作的时间段分布

def hour_buy_counts(type, in_itemlist=False):

    if in_itemlist == True:

        condition = user_data["item_id"].isin(item_list)

    else:

        condition = True

    hour_buy_counts = user_data[(user_data["behavior_type"] == type)&(condition)][["hour"]]

    hour_buy_counts["count"] = 1

    hour_buy_counts = hour_buy_counts.groupby("hour", as_index=False).count().sort_values(by="count", ascending=False).reset_index(drop=True)

    return hour_buy_counts
# 商品全集购买星期分布图

data = weekday_buy_counts(4, False)

fig = sns.barplot(x="weekday", y="count", data=data)
# 商品子集购买星期分布图

data = weekday_buy_counts(4, True)

fig = sns.barplot(x="weekday", y="count", data=data)
# 商品全集购买时间段分布图

data = hour_buy_counts(4,True)

fig = sns.barplot(x="hour", y="count", data=data)
# 商品子集购买时间段分布图

data = hour_buy_counts(4,True)

fig = sns.barplot(x="hour", y="count", data=data)