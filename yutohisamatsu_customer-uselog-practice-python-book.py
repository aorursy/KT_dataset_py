# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

uselog = pd.read_csv('../input/use_log.csv')

print(len(uselog))

uselog.head()
customer = pd.read_csv('../input/customer_master.csv')

print(len(customer))

customer.head()
class_master = pd.read_csv('../input/class_master.csv')

print(len(class_master))

class_master.head()
campaign_master = pd.read_csv('../input/campaign_master.csv')

print(len(campaign_master))

campaign_master.head()
# 会員区分、キャンペーン区分を結合

customer_join = pd.merge(customer, class_master, on="class", how="left")

customer_join = pd.merge(customer_join, campaign_master, on="campaign_id", how="left")

customer_join.head()
# データ数にジョイン前後で変化がないか確認

print(len(customer))

print(len(customer_join))
# 欠損値の確認　ジョインする際に、上手くジョインできないと欠損値を自動で入れる

customer_join.isnull().sum()
# 会員区分

customer_join.groupby("class_name").count()["customer_id"]
#　キャンペーン区分

customer_join.groupby("campaign_name").count()["customer_id"]
#　性別

customer_join.groupby("gender").count()["customer_id"]
#　すでに退会しているかどうか

customer_join.groupby("is_deleted").count()["customer_id"]
#　入会人数

customer_join["start_date"] = pd.to_datetime(customer_join["start_date"])

customer_start = customer_join.loc[customer_join["start_date"] > pd.to_datetime("20180401")]

print(len(customer_start))
# 2019年３月(2019-03-31)に退会したユーザー、もしくは、在籍しているユーザーで絞り込む

customer_join["end_date"] = pd.to_datetime(customer_join["end_date"])

customer_newer = customer_join.loc[(customer_join["end_date"] >= pd.to_datetime("20190331")) | (customer_join["end_date"].isna())]



print(len(customer_newer))

customer_newer["end_date"].unique()



# NaTは、datetimeの欠損値、このデータにおいては退会していない顧客
#　最新顧客の把握

#　会員区分

customer_newer.groupby("class_name").count()["customer_id"]
#　キャンペーン区分

customer_newer.groupby("campaign_name").count()["customer_id"]
#　性別

customer_newer.groupby("gender").count()["customer_id"]
# 利用履歴データ

# 顧客データと違い、時間的な要素の分析が可能

uselog["usedate"] = pd.to_datetime(uselog["usedate"])



uselog["年月"] = uselog["usedate"].dt.strftime("%Y/%m")

#年月かつ顧客ID毎に集計

uselog_months = uselog.groupby(["年月", "customer_id"], as_index = False).count()



#引数inplaceをTrueにすると、元のDataFrameが変更される。新しいDataFrameは返されず、返り値はNone

uselog_months.rename(columns={"log_id": "count"}, inplace = True)



#余分なusedateは削除

del uselog_months["usedate"]



uselog_months.head()
# 顧客毎に、平均値、中央値、最大値、最小値を集計　（月内の利用回数の集計）

# agg()は、groupbyがもつメソッド

uselog_customer = uselog_months.groupby("customer_id").agg(["mean", "median", "max", "min"])["count"]



# groupbyをした影響でindexに入っているcustomer_id列をカラムに変更して、indexの振り直し

uselog_customer = uselog_customer.reset_index(drop = False)

uselog_customer.head()
# 定期的にジムを利用している場合のフラグ作成

# 顧客毎に、月/曜日別に集計を行い、最大値が4以上の曜日が1ヶ月でもあったユーザーは、フラグ1にする

uselog["weekday"] = uselog["usedate"].dt.weekday



# 顧客、年月、曜日毎に、log_idを数える

uselog_weekday = uselog.groupby(["customer_id", "年月", "weekday"], as_index = False).count()[["customer_id", "年月", "weekday", "log_id"]]

uselog_weekday.rename(columns = {"log_id" : "count"}, inplace = True)

uselog_weekday.head()
# 顧客毎の各月の最大値を取得し、その最大値が4以上の場合、フラグ

uselog_weekday = uselog_weekday.groupby("customer_id", as_index = False).max()[["customer_id", "count"]]

uselog_weekday["routine_flg"] = 0

# 4以上の場合は、1を代入

uselog_weekday["routine_flg"] = uselog_weekday["routine_flg"].where(uselog_weekday["count"] < 4, 1)

uselog_weekday.head()
# 顧客データと利用履歴データを結合

customer_join = pd.merge(customer_join, uselog_customer, on="customer_id", how="left")

customer_join = pd.merge(customer_join, uselog_weekday[["customer_id", "routine_flg"]], on="customer_id", how="left")

customer_join.head()
customer_join.isnull().sum()
from dateutil.relativedelta import relativedelta

customer_join["calc_date"] = customer_join["end_date"]

# 欠損値に2019年4月30日を代入

customer_join["calc_date"] = customer_join["calc_date"].fillna(pd.to_datetime("20190430"))

customer_join["membership_period"] = 0

for i in range(len(customer_join)):

    delta = relativedelta(customer_join["calc_date"].iloc[i], customer_join["start_date"].iloc[i])

    customer_join["membership_period"].iloc[i] = delta.years*12 + delta.months

customer_join.head()
customer_join[["mean", "median", "max", "min"]].describe()
customer_join.groupby("routine_flg").count()["customer_id"]
import matplotlib.pyplot as plt

%matplotlib inline

plt.hist(customer_join["membership_period"])
# 退会ユーザー1350人、継続ユーザー2840人

customer_end = customer_join.loc[customer_join["is_deleted"] == 1]

customer_end.describe()
customer_stay = customer_join.loc[customer_join["is_deleted"] == 0]

customer_stay.describe()