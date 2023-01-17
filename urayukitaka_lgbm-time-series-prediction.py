import numpy as np

import pandas as pd



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Libraries

import datetime



# Visualization

from matplotlib import pyplot as plt

plt.style.use('fivethirtyeight')

import seaborn as sns



import statsmodels.api as sm



# Statistics library

from scipy.stats import norm

from scipy import stats

import scipy



# Data preprocessing

from sklearn.model_selection import train_test_split



# Machine learning

import lightgbm as lgb

from sklearn.linear_model import Ridge

from sklearn.linear_model import Lasso

from sklearn.linear_model import ElasticNet

from sklearn.model_selection import GridSearchCV



# Validataion

from sklearn.metrics import mean_squared_error

from sklearn.metrics import r2_score
df_items = pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/items.csv", header=0)

df_shops = pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/shops.csv", header=0)

df_sales_train = pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/sales_train.csv", header=0)

df_test = pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/test.csv", header=0)

df_category = pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/item_categories.csv", header=0)

sample = pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/sample_submission.csv")
df_items.head()
df_items.shape
df_shops.head()
df_shops.shape
df_sales_train.head()
df_sales_train.shape
df_test.head()
df_test.shape
df_category.head()
df_category.shape
sample.head()
sample.shape
print("Null data df_sales_train:{}".format(df_sales_train.isnull().sum().sum()))

print("Null data df_items:{}".format(df_items.isnull().sum().sum()))

print("Null data df_shops:{}".format(df_shops.isnull().sum().sum()))

print("Null data df_category:{}".format(df_category.isnull().sum().sum()))
### Data preprocessing, df_train, datetime

df_sales_train["date_dt"] = pd.to_datetime(df_sales_train["date"], format='%d.%m.%Y')



df_sales_train["year"] = df_sales_train["date_dt"].dt.year

df_sales_train["month"] = df_sales_train["date_dt"].dt.month

df_sales_train["day"] = df_sales_train["date_dt"].dt.day
df_sales_train["item_sales"] = df_sales_train["item_price"]*df_sales_train["item_cnt_day"]

df_sales_train = pd.merge(df_sales_train, df_items[["item_id", "item_category_id"]], left_on="item_id", right_on="item_id", how="left")

train_df = df_sales_train.drop("date", axis=1).sort_index()
tot_daily_sales = train_df.groupby("date_dt").sum()["item_sales"]
# Time series

fig, ax1 = plt.subplots(figsize=(20,6))

ax1.plot(tot_daily_sales.index, tot_daily_sales/1000, linewidth=1)

ax1.set_ylabel("Sales(k)")

ax2 = ax1.twinx()

ax2.plot(tot_daily_sales.index, tot_daily_sales.cumsum()/1000000, linewidth=1, color="red")

ax2.grid()

ax2.set_ylabel("Total Sales(M)")

plt.xlabel("time")
# item sales distribution

plt.figure(figsize=(10,6))

sns.distplot(tot_daily_sales, kde=False, bins=50)

plt.ylabel("Frequency")

plt.yscale("log")

plt.xlabel("Total daily sales")
fig, ax = plt.subplots(2,2,figsize=(20,12))

plt.subplots_adjust(hspace=0.5)

sns.boxplot(train_df["item_cnt_day"], ax=ax[0,0])

ax[0,0].set_title("item_cnt_day")



sns.boxplot(train_df[train_df["item_cnt_day"]<800]["item_cnt_day"], ax=ax[0,1])

ax[0,1].set_title("item_cnt_day Remove outlier")



sns.boxplot(train_df["item_price"], ax=ax[1,0])

ax[1,0].set_title("item_price")



sns.boxplot(train_df[train_df["item_price"]<70000]["item_price"], ax=ax[1,1])

ax[1,1].set_title("item_price Remove outlier")
# Update

train_df = train_df[train_df["item_cnt_day"]<800]

train_df = train_df[train_df["item_price"]<70000]
# Re aggrecate tot_daily_sales

tot_daily_sales = train_df.groupby("date_dt").sum()["item_sales"]
# freq = 365 day

res = sm.tsa.seasonal_decompose(tot_daily_sales, freq=365)



# Decomposition

trend = res.trend

seaso = res.seasonal

resid = res.resid
# Visualization

fig, ax = plt.subplots(4,1, figsize=(15,15))

plt.subplots_adjust(hspace=0.5)



ax[0].plot(tot_daily_sales.index, tot_daily_sales, color="black")

ax[0].set_title("Time series")

ax[0].set_ylabel("Daily_sales\n(Frequeycy:365day)")

ax[0].set_xlabel("Time")



ax[1].plot(trend.index, trend, color="red")

ax[1].set_title("Trend")

ax[1].set_ylabel("Daily_sales\n(Frequeycy:365day)")

ax[1].set_xlabel("Time")



ax[2].plot(seaso.index, seaso, color="blue")

ax[2].set_title("Seasonal")

ax[2].set_ylabel("Daily_sales\n(Frequeycy:365day)")

ax[2].set_xlabel("Time")



ax[3].plot(resid.index, resid, color="green")

ax[3].set_title("Resid")

ax[3].set_ylabel("Daily_sales\n(Frequeycy:365day)")

ax[3].set_xlabel("Time")
del trend, seaso, resid
# freq = 30 day

res = sm.tsa.seasonal_decompose(tot_daily_sales, freq=30)



# Decomposition

trend = res.trend

seaso = res.seasonal

resid = res.resid
# Visualization

fig, ax = plt.subplots(4,1, figsize=(15,15))

plt.subplots_adjust(hspace=0.5)



ax[0].plot(tot_daily_sales.index[-365:], tot_daily_sales[-365:], color="black")

ax[0].set_title("Time series")

ax[0].set_ylabel("Daily_sales\n(Frequeycy:30day)")

ax[0].set_xlabel("Time")



ax[1].plot(trend.index[-365:], trend[-365:], color="red")

ax[1].set_title("Trend")

ax[1].set_ylabel("Daily_sales\n(Frequeycy:30day)")

ax[1].set_xlabel("Time")



ax[2].plot(seaso.index[-365:], seaso[-365:], color="blue")

ax[2].set_title("Seasonal")

ax[2].set_ylabel("Daily_sales\n(Frequeycy:30day)")

ax[2].set_xlabel("Time")



ax[3].plot(resid.index[-365:], resid[-365:], color="green")

ax[3].set_title("Resid")

ax[3].set_ylabel("Daily_sales\n(Frequeycy:30day)")

ax[3].set_xlabel("Time")
del trend, seaso, resid, tot_daily_sales
# pivot by shops

shops_pivot = pd.pivot_table(train_df, index="date_dt", columns="shop_id", values="item_sales", aggfunc="sum", fill_value=0)



# Shops sample, id=0 & 2 & 3

sample_0 = shops_pivot[0]

sample_1 = shops_pivot[2]

sample_2 = shops_pivot[3]



# freq = 365 day

res_0 = sm.tsa.seasonal_decompose(sample_0, freq=365)

res_1 = sm.tsa.seasonal_decompose(sample_1, freq=365)

res_2 = sm.tsa.seasonal_decompose(sample_2, freq=365)



# Decomposition

trend_0 = res_0.trend

seaso_0 = res_0.seasonal

resid_0 = res_0.resid



trend_1 = res_1.trend

seaso_1 = res_1.seasonal

resid_1 = res_1.resid



trend_2 = res_2.trend

seaso_2 = res_2.seasonal

resid_2 = res_2.resid
# Visualization

fig, ax = plt.subplots(4,3, figsize=(25,15))

plt.subplots_adjust(hspace=0.5,)



ax[0,0].plot(sample_0.index, sample_0, color="black")

ax[0,0].set_title("Shop0_Time series")

ax[0,0].set_ylabel("Daily_sales\n(Frequeycy:365day)", fontsize=15)

ax[0,0].tick_params(axis='x', labelsize=10)



ax[1,0].plot(trend_0.index, trend_0, color="red")

ax[1,0].set_title("Shop0_Trend")

ax[1,0].set_ylabel("Daily_sales\n(Frequeycy:365day)", fontsize=15)

ax[1,0].set_xlabel("Time")

ax[1,0].tick_params(axis='x', labelsize=10)



ax[2,0].plot(seaso_0.index, seaso_0, color="blue")

ax[2,0].set_title("Shop0_Seasonal")

ax[2,0].set_ylabel("Daily_sales\n(Frequeycy:365day)", fontsize=15)

ax[2,0].set_xlabel("Time")

ax[2,0].tick_params(axis='x', labelsize=10)



ax[3,0].plot(resid_0.index, resid_0, color="green")

ax[3,0].set_title("Shop0_Resid")

ax[3,0].set_ylabel("Daily_sales\n(Frequeycy:365day)", fontsize=15)

ax[3,0].set_xlabel("Time")

ax[3,0].tick_params(axis='x', labelsize=10)



ax[0,1].plot(sample_1.index, sample_1, color="black")

ax[0,1].set_title("Shop2_Time series")

ax[0,1].set_ylabel("Daily_sales\n(Frequeycy:365day)", fontsize=15)

ax[0,1].set_xlabel("Time")

ax[0,1].tick_params(axis='x', labelsize=10)



ax[1,1].plot(trend_1.index, trend_1, color="red")

ax[1,1].set_title("Shop2_Trend")

ax[1,1].set_ylabel("Daily_sales\n(Frequeycy:365day)", fontsize=15)

ax[1,1].set_xlabel("Time")

ax[1,1].tick_params(axis='x', labelsize=10)



ax[2,1].plot(seaso_1.index, seaso_1, color="blue")

ax[2,1].set_title("Shop2_Seasonal")

ax[2,1].set_ylabel("Daily_sales\n(Frequeycy:365day)", fontsize=15)

ax[2,1].set_xlabel("Time")

ax[2,1].tick_params(axis='x', labelsize=10)



ax[3,1].plot(resid_1.index, resid_1, color="green")

ax[3,1].set_title("Shop2_Resid")

ax[3,1].set_ylabel("Daily_sales\n(Frequeycy:365day)", fontsize=15)

ax[3,1].set_xlabel("Time")

ax[3,1].tick_params(axis='x', labelsize=10)



ax[0,2].plot(sample_2.index, sample_2, color="black")

ax[0,2].set_title("Shop3_Time series")

ax[0,2].set_ylabel("Daily_sales(Frequeycy:365day)", fontsize=15)

ax[0,2].set_xlabel("Time")

ax[0,2].tick_params(axis='x', labelsize=10)



ax[1,2].plot(trend_2.index, trend_2, color="red")

ax[1,2].set_title("Shop3_Trend")

ax[1,2].set_ylabel("Daily_sales\n(Frequeycy:365day)", fontsize=15)

ax[1,2].set_xlabel("Time")

ax[1,2].tick_params(axis='x', labelsize=10)



ax[2,2].plot(seaso_2.index, seaso_2, color="blue")

ax[2,2].set_title("Shop3_Seasonal")

ax[2,2].set_ylabel("Daily_sales\n(Frequeycy:365day)", fontsize=15)

ax[2,2].set_xlabel("Time")

ax[2,2].tick_params(axis='x', labelsize=10)



ax[3,2].plot(resid_2.index, resid_2, color="green")

ax[3,2].set_title("Shop3_Resid")

ax[3,2].set_ylabel("Daily_sales\n(Frequeycy:365day)", fontsize=15)

ax[3,2].set_xlabel("Time")

ax[3,2].tick_params(axis='x', labelsize=10)
del res_0, res_1, res_2, trend_0, seaso_0, resid_0, trend_1, seaso_1, resid_1, trend_2, seaso_2, resid_2
# pivot by category

category_pivot = pd.pivot_table(train_df, index="date_dt", columns="item_category_id", values="item_sales", aggfunc="sum", fill_value=0)



# Shops sample, id=0 & 2 & 3

sample_0 = category_pivot[0]

sample_1 = category_pivot[2]

sample_2 = category_pivot[3]



# freq = 365 day

res_0 = sm.tsa.seasonal_decompose(sample_0, freq=365)

res_1 = sm.tsa.seasonal_decompose(sample_1, freq=365)

res_2 = sm.tsa.seasonal_decompose(sample_2, freq=365)



# Decomposition

trend_0 = res_0.trend

seaso_0 = res_0.seasonal

resid_0 = res_0.resid



trend_1 = res_1.trend

seaso_1 = res_1.seasonal

resid_1 = res_1.resid



trend_2 = res_2.trend

seaso_2 = res_2.seasonal

resid_2 = res_2.resid
# Visualization

fig, ax = plt.subplots(4,3, figsize=(25,15))

plt.subplots_adjust(hspace=0.5,)



ax[0,0].plot(sample_0.index, sample_0, color="black")

ax[0,0].set_title("Category0_Time series")

ax[0,0].set_ylabel("Daily_sales\n(Frequeycy:365day)", fontsize=15)

ax[0,0].set_xlabel("Time")

ax[0,0].tick_params(axis='x', labelsize=10)



ax[1,0].plot(trend_0.index, trend_0, color="red")

ax[1,0].set_title("Category0_Trend")

ax[1,0].set_ylabel("Daily_sales\n(Frequeycy:365day)", fontsize=15)

ax[1,0].set_xlabel("Time")

ax[1,0].tick_params(axis='x', labelsize=10)



ax[2,0].plot(seaso_0.index, seaso_0, color="blue")

ax[2,0].set_title("Category0_Seasonal")

ax[2,0].set_ylabel("Daily_sales\n(Frequeycy:365day)", fontsize=15)

ax[2,0].set_xlabel("Time")

ax[2,0].tick_params(axis='x', labelsize=10)



ax[3,0].plot(resid_0.index, resid_0, color="green")

ax[3,0].set_title("Category0_Resid")

ax[3,0].set_ylabel("Daily_sales\n(Frequeycy:365day)", fontsize=15)

ax[3,0].set_xlabel("Time")

ax[3,0].tick_params(axis='x', labelsize=10)



ax[0,1].plot(sample_1.index, sample_1, color="black")

ax[0,1].set_title("Category2_Time series")

ax[0,1].set_ylabel("Daily_sales\n(Frequeycy:365day)", fontsize=15)

ax[0,1].set_xlabel("Time")

ax[0,1].tick_params(axis='x', labelsize=10)



ax[1,1].plot(trend_1.index, trend_1, color="red")

ax[1,1].set_title("Category2_Trend")

ax[1,1].set_ylabel("Daily_sales\n(Frequeycy:365day)", fontsize=15)

ax[1,1].set_xlabel("Time")

ax[1,1].tick_params(axis='x', labelsize=10)



ax[2,1].plot(seaso_1.index, seaso_1, color="blue")

ax[2,1].set_title("Category2_Seasonal")

ax[2,1].set_ylabel("Daily_sales\n(Frequeycy:365day)", fontsize=15)

ax[2,1].set_xlabel("Time")

ax[2,1].tick_params(axis='x', labelsize=10)



ax[3,1].plot(resid_1.index, resid_1, color="green")

ax[3,1].set_title("Category2_Resid")

ax[3,1].set_ylabel("Daily_sales\n(Frequeycy:365day)", fontsize=15)

ax[3,1].set_xlabel("Time")

ax[3,1].tick_params(axis='x', labelsize=10)



ax[0,2].plot(sample_2.index, sample_2, color="black")

ax[0,2].set_title("Category3_Time series")

ax[0,2].set_ylabel("Daily_sales\n(Frequeycy:365day)", fontsize=15)

ax[0,2].set_xlabel("Time")

ax[0,2].tick_params(axis='x', labelsize=10)



ax[1,2].plot(trend_2.index, trend_2, color="red")

ax[1,2].set_title("Category3_Trend")

ax[1,2].set_ylabel("Daily_sales\n(Frequeycy:365day)", fontsize=15)

ax[1,2].set_xlabel("Time")

ax[1,2].tick_params(axis='x', labelsize=10)



ax[2,2].plot(seaso_2.index, seaso_2, color="blue")

ax[2,2].set_title("Category3_Seasonal")

ax[2,2].set_ylabel("Daily_sales\n(Frequeycy:365day)", fontsize=15)

ax[2,2].set_xlabel("Time")

ax[2,2].tick_params(axis='x', labelsize=10)



ax[3,2].plot(resid_2.index, resid_2, color="green")

ax[3,2].set_title("Category3_Resid")

ax[3,2].set_ylabel("Daily_sales\n(Frequeycy:365day)", fontsize=15)

ax[3,2].set_xlabel("Time")

ax[3,2].tick_params(axis='x', labelsize=10)
del res_0, res_1, res_2, trend_0, seaso_0, resid_0, trend_1, seaso_1, resid_1, trend_2, seaso_2, resid_2
# pivot by category

item_pivot_sales = pd.pivot_table(train_df, index="date_dt", columns="item_id", values="item_sales", aggfunc="mean", fill_value=0)

item_pivot_count = pd.pivot_table(train_df, index="date_dt", columns="item_id", values="item_price", aggfunc="count", fill_value=0)

item_pivot_price = pd.pivot_table(train_df, index="date_dt", columns="item_id", values="item_price", aggfunc="mean", fill_value=0)
# Sample 1000, 2000, 10000

fig, ax = plt.subplots(3,3, figsize=(25,20))



item_list = [1000,2000,10000]

pivot_list = [item_pivot_sales, item_pivot_count, item_pivot_price]



for i in range(len(item_list)):

    for k in range(len(pivot_list)):

        ax[i,k].plot(pivot_list[k][item_list[i]].index, pivot_list[k][item_list[i]])

        ax[i,k].set_xlabel("Time")

        ax[i,k].tick_params(axis='x', labelsize=10)

        ax[i,0].set_ylabel("Sales")

        ax[i,1].set_ylabel("Count")

        ax[i,2].set_ylabel("Price")

        ax[i,k].set_title("item_id:{}".format(item_list[i]))



del item_pivot_sales, item_pivot_count, item_pivot_price
# Prepairing dataset

master = train_df.copy()
# Define class

class feature_eng:

    def __init__(self, data_ser, data_df, seasonal_len, name):

        self.date = data_ser

        self.data = data_df

        self.seas = seasonal_len

        self.name = name

        self.col = data_df.columns

        

    def sep_lag_trend_seaso_train(self):

        max_list = []

        mean_list = []

        lag1_list = []

        lag2_list = []

        lag3_list = []

        lag4_6_list = []

        seas1_list = []

        seas2_list = []

        seas3_list = []

        seas4_6_list = []        

        

        for i in self.col:

            # Calculate trend and seasonal facter

            res = sm.tsa.seasonal_decompose(self.data[i], freq=self.seas)

            last = self.data[i].values[-2]

            max_ = self.data[i].values[-2].max()

            mean = self.data[i].values[-8:-2].mean()

            # Append to list

            max_list.append(self.data[i].values[:-2].max())

            mean_list.append(self.data[i].values[:-2].mean())

            lag1_list.append((last - self.data[i].values[-3])*1)

            lag2_list.append((last - self.data[i].values[-4])*2)

            lag3_list.append((last - self.data[i].values[-5])*3)

            lag4_6_list.append((last - (self.data[i].values[-6]+self.data[i].values[-7]+self.data[i].values[-8]))*15)

            seas1_list.append((res.seasonal.values[-self.seas-1])*1)

            seas2_list.append((res.seasonal.values[-self.seas-2])*2)

            seas3_list.append((res.seasonal.values[-self.seas-3])*3)

            seas4_6_list.append((res.seasonal.values[-self.seas-4]+res.seasonal.values[-self.seas-5]+res.seasonal.values[-self.seas-6])*15)

        # Output data frame

        out = pd.DataFrame({"id":self.col,

                            "{}_max".format(self.name):max_list,

                            "{}_mean".format(self.name):mean_list,

                            "{}_lag1".format(self.name):lag1_list,

                            "{}_lag2".format(self.name):lag2_list,

                            "{}_lag3".format(self.name):lag3_list,

                            "{}_lag4_6".format(self.name):lag4_6_list,

                            "{}_seas1".format(self.name):seas1_list,

                            "{}_seas2".format(self.name):seas2_list,

                            "{}_seas3".format(self.name):seas3_list,

                            "{}_seas4_6".format(self.name):seas4_6_list

                           })

        return out

    

    def sep_lag_trend_seaso_test(self):

        max_list = []

        mean_list = []

        lag1_list = []

        lag2_list = []

        lag3_list = []

        lag4_6_list = []

        seas1_list = []

        seas2_list = []

        seas3_list = []

        seas4_6_list = []   

        

        for i in self.col:

            # Calculate trend and seasonal facter

            res = sm.tsa.seasonal_decompose(self.data[i], freq=self.seas)

            last = self.data[i].values[-1]

            max_ = self.data[i][:-1].values.max()

            mean = self.data[i].values[-19:-1].mean()

            # Append to list

            max_list.append(self.data[i].values[:-1].max())

            mean_list.append(self.data[i].values[:-1].mean())

            lag1_list.append((last - self.data[i].values[-2])*1)

            lag2_list.append((last - self.data[i].values[-3])*2)

            lag3_list.append((last - self.data[i].values[-4])*3)

            lag4_6_list.append((last - (self.data[i].values[-5]+self.data[i].values[-6]+self.data[i].values[-7]))*15)

            seas1_list.append((res.seasonal.values[-self.seas-1])*1)

            seas2_list.append((res.seasonal.values[-self.seas-2])*2)

            seas3_list.append((res.seasonal.values[-self.seas-3])*3)

            seas4_6_list.append((res.seasonal.values[-self.seas-4]+res.seasonal.values[-self.seas-5]+res.seasonal.values[-self.seas-6])*15)

        # Output data frame

        out = pd.DataFrame({"id":self.col,

                            "{}_max".format(self.name):max_list,

                            "{}_mean".format(self.name):mean_list,

                            "{}_lag1".format(self.name):lag1_list,

                            "{}_lag2".format(self.name):lag2_list,

                            "{}_lag3".format(self.name):lag3_list,

                            "{}_lag4_6".format(self.name):lag4_6_list,

                            "{}_seas1".format(self.name):seas1_list,

                            "{}_seas2".format(self.name):seas2_list,

                            "{}_seas3".format(self.name):seas3_list,

                            "{}_seas4_6".format(self.name):seas4_6_list

                           })

        return out
# Define class

class feature_eng:

    def __init__(self, data_ser, data_df, seasonal_len, name):

        self.date = data_ser

        self.data = data_df

        self.seas = seasonal_len

        self.name = name

        self.col = data_df.columns

        

    def sep_lag_trend_seaso_train(self):

        max_list = []

        mean_list = []

        lag1_list = []

        lag2_list = []

        lag3_list = []

        lag4_6_list = []

        tre1_list = []

        tre3_list = []

        tre4_6_list = []

        seas1_list = []

        seas2_list = []

        seas3_list = []

        seas4_6_list = []        

        

        for i in self.col:

            # Calculate trend and seasonal facter

            res = sm.tsa.seasonal_decompose(self.data[i], freq=self.seas)

            last = self.data[i].values[-2]

            max_ = self.data[i].values[-2].max()

            mean = self.data[i].values[-8:-2].mean()

            # Append to list

            max_list.append(self.data[i].values[:-2].max())

            mean_list.append(self.data[i].values[:-2].mean())

            lag1_list.append((last - self.data[i].values[-3])*1)

            lag2_list.append((last - self.data[i].values[-4])*2)

            lag3_list.append((last - self.data[i].values[-5])*3)

            lag4_6_list.append((last - (self.data[i].values[-6]+self.data[i].values[-7]+self.data[i].values[-8]))*15)

            tre1_list.append((last - res.trend.values[-int(self.seas*0.5)-1])*1)

            tre3_list.append((last - res.trend.values[-int(self.seas*0.5)-3])*3)

            tre4_6_list.append((last - (res.trend.values[-int(self.seas*0.5)-4]+res.trend.values[-int(self.seas*0.5)-5]+res.trend.values[-int(self.seas*0.5)-6]))*15)

            seas1_list.append((res.seasonal.values[-self.seas-1])*1)

            seas2_list.append((res.seasonal.values[-self.seas-2])*2)

            seas3_list.append((res.seasonal.values[-self.seas-3])*3)

            seas4_6_list.append((res.seasonal.values[-self.seas-4]+res.seasonal.values[-self.seas-5]+res.seasonal.values[-self.seas-6])*15)

        # Output data frame

        out = pd.DataFrame({"id":self.col,

                            "{}_max".format(self.name):max_list,

                            "{}_mean".format(self.name):mean_list,

                            "{}_lag1".format(self.name):lag1_list,

                            "{}_lag2".format(self.name):lag2_list,

                            "{}_lag3".format(self.name):lag3_list,

                            "{}_lag4_6".format(self.name):lag4_6_list,

                            "{}_tre1".format(self.name):tre1_list,

                            "{}_tre3".format(self.name):tre3_list,

                            "{}_tre4_6".format(self.name):tre4_6_list,

                            "{}_seas1".format(self.name):seas1_list,

                            "{}_seas2".format(self.name):seas2_list,

                            "{}_seas3".format(self.name):seas3_list,

                            "{}_seas4_6".format(self.name):seas4_6_list

                           })

        return out

    

    def sep_lag_trend_seaso_test(self):

        max_list = []

        mean_list = []

        lag1_list = []

        lag2_list = []

        lag3_list = []

        lag4_6_list = []

        tre1_list = []

        tre3_list = []

        tre4_6_list = []

        seas1_list = []

        seas2_list = []

        seas3_list = []

        seas4_6_list = []   

        

        for i in self.col:

            # Calculate trend and seasonal facter

            res = sm.tsa.seasonal_decompose(self.data[i], freq=self.seas)

            last = self.data[i].values[-1]

            max_ = self.data[i][:-1].values.max()

            mean = self.data[i].values[-19:-1].mean()

            # Append to list

            max_list.append(self.data[i].values[:-1].max())

            mean_list.append(self.data[i].values[:-1].mean())

            lag1_list.append((last - self.data[i].values[-2])*1)

            lag2_list.append((last - self.data[i].values[-3])*2)

            lag3_list.append((last - self.data[i].values[-4])*3)

            lag4_6_list.append((last - (self.data[i].values[-5]+self.data[i].values[-6]+self.data[i].values[-7]))*15)

            tre1_list.append((last - res.trend.values[-int(self.seas*0.5)-1])*1)

            tre3_list.append((last - res.trend.values[-int(self.seas*0.5)-3])*3)

            tre4_6_list.append((last - (res.trend.values[-int(self.seas*0.5)-4]+res.trend.values[-int(self.seas*0.5)-5]+res.trend.values[-int(self.seas*0.5)-6]))*15)

            seas1_list.append((res.seasonal.values[-self.seas-1])*1)

            seas2_list.append((res.seasonal.values[-self.seas-2])*2)

            seas3_list.append((res.seasonal.values[-self.seas-3])*3)

            seas4_6_list.append((res.seasonal.values[-self.seas-4]+res.seasonal.values[-self.seas-5]+res.seasonal.values[-self.seas-6])*15)

        # Output data frame

        out = pd.DataFrame({"id":self.col,

                            "{}_max".format(self.name):max_list,

                            "{}_mean".format(self.name):mean_list,

                            "{}_lag1".format(self.name):lag1_list,

                            "{}_lag2".format(self.name):lag2_list,

                            "{}_lag3".format(self.name):lag3_list,

                            "{}_lag4_6".format(self.name):lag4_6_list,

                            "{}_tre1".format(self.name):tre1_list,

                            "{}_tre3".format(self.name):tre3_list,

                            "{}_tre4_6".format(self.name):tre4_6_list,

                            "{}_seas1".format(self.name):seas1_list,

                            "{}_seas2".format(self.name):seas2_list,

                            "{}_seas3".format(self.name):seas3_list,

                            "{}_seas4_6".format(self.name):seas4_6_list

                           })

        return out
# Each shops count feature

shop_ts = pd.pivot_table(data=master, index=["year","month"], columns="shop_id", values="item_cnt_day", aggfunc="sum", fill_value=0)



date_ser = shop_ts.reset_index().drop(["year", "month"], axis=1).index

data_df = shop_ts.reset_index().drop(["year", "month"], axis=1)

seasonal_len = 12

name = "shop_count"
# Apply class for train data

shop_time_count = feature_eng(date_ser, data_df, seasonal_len, name)

shop_time_count_train = shop_time_count.sep_lag_trend_seaso_train()

# Apply class for test data

shop_time_count_test = shop_time_count.sep_lag_trend_seaso_test()
shop_time_count_train.head()
# Each shops price feature

shop_ts = pd.pivot_table(data=master, index=["year","month"], columns="shop_id", values="item_price", aggfunc="mean").fillna(method="ffill").fillna(0)



date_ser = shop_ts.reset_index().drop(["year", "month"], axis=1).index

data_df = shop_ts.reset_index().drop(["year", "month"], axis=1)

seasonal_len = 12

name = "shop_price"
# Apply class for train data

shop_time_price = feature_eng(date_ser, data_df, seasonal_len, name)

shop_time_price_train = shop_time_price.sep_lag_trend_seaso_train()

# Apply class for test data

shop_time_price_test = shop_time_price.sep_lag_trend_seaso_test()
shop_time_price_train.head()
# Each Category count feature

cate_ts = pd.pivot_table(data=master, index=["year","month"], columns="item_category_id", values="item_cnt_day", aggfunc="sum", fill_value=0)



date_ser = cate_ts.reset_index().drop(["year", "month"], axis=1).index

data_df = cate_ts.reset_index().drop(["year", "month"], axis=1)

seasonal_len = 12

name = "category_count"
# Apply class for train data

cate_time_count = feature_eng(date_ser, data_df, seasonal_len, name)

cate_time_count_train = cate_time_count.sep_lag_trend_seaso_train()

# Apply class for test data

cate_time_count_test = cate_time_count.sep_lag_trend_seaso_test()
cate_time_count_train.head()
# Each Category price feature

cate_ts = pd.pivot_table(data=master, index=["year","month"], columns="item_category_id", values="item_price", aggfunc="mean").fillna(method="ffill").fillna(0)



date_ser = cate_ts.reset_index().drop(["year", "month"], axis=1).index

data_df = cate_ts.reset_index().drop(["year", "month"], axis=1)

seasonal_len = 12

name = "category_count"
# # Apply class for train data

cate_time_price = feature_eng(date_ser, data_df, seasonal_len, name)

cate_time_price_train = cate_time_price.sep_lag_trend_seaso_train()

# # Apply class for test data

cate_time_price_test = cate_time_price.sep_lag_trend_seaso_test()
cate_time_price_train.head()
# Each Item count feature

item_ts = pd.pivot_table(data=master, index=["year","month"], columns="item_category_id", values="item_cnt_day", aggfunc="sum", fill_value=0)



date_ser = item_ts.reset_index().drop(["year", "month"], axis=1).index

data_df = item_ts.reset_index().drop(["year", "month"], axis=1)

seasonal_len = 12

name = "item_count"
# Apply class for train data

item_time_count = feature_eng(date_ser, data_df, seasonal_len, name)

item_time_count_train = item_time_count.sep_lag_trend_seaso_train()

# Apply class for test data

item_time_count_test = item_time_count.sep_lag_trend_seaso_test()
item_time_count_train.head()
# Each Item Price feature

item_ts = pd.pivot_table(data=master, index=["year","month"], columns="item_id", values="item_price", aggfunc="mean").fillna(method="ffill").fillna(0)



date_ser = item_ts.reset_index().drop(["year", "month"], axis=1).index

data_df = item_ts.reset_index().drop(["year", "month"], axis=1)

seasonal_len = 12

name = "category_count"
# Apply class for train data

item_time_price = feature_eng(date_ser, data_df, seasonal_len, name)

item_time_price_train = item_time_price.sep_lag_trend_seaso_train()

# Apply class for test data

item_time_price_test = item_time_price.sep_lag_trend_seaso_test()
item_time_price_train.head()
del shop_time_count, shop_time_price, cate_time_count, cate_time_price, item_time_count, item_time_price
# Type conversion to preserve memory

def dtype_change(df):

    columns = df.dtypes.index

    dtype = df.dtypes

    dtype = [str(d) for d in dtype]

    for i in range(len(columns)):

        if dtype[i] == 'int64':

            df[columns[i]] = df[columns[i]].astype("int32")

        elif dtype[i] == 'float64':

            df[columns[i]] = df[columns[i]].astype("float32")

        else:

            pass

    return df
# Training data

shop_time_count_train = dtype_change(shop_time_count_train)

shop_time_price_train = dtype_change(shop_time_price_train)

cate_time_count_train = dtype_change(cate_time_count_train)

cate_time_price_train = dtype_change(cate_time_price_train)

item_time_count_train = dtype_change(item_time_count_train)

item_time_price_train = dtype_change(item_time_price_train)

# Test data

shop_time_count_test = dtype_change(shop_time_count_test)

shop_time_price_test = dtype_change(shop_time_price_test)

cate_time_count_test = dtype_change(cate_time_count_test)

cate_time_price_test = dtype_change(cate_time_price_test)

item_time_count_test = dtype_change(item_time_count_test)

item_time_price_test = dtype_change(item_time_price_test)
# Assign test data ID to training data

master = pd.merge(master, df_test, left_on=["shop_id", "item_id"], right_on=["shop_id", "item_id"], how="left")
# Group by shop_id and item_id, and group them in data blocks in the column direction.

pivot = pd.pivot_table(data=master, index=["shop_id", "item_id"], columns="date_block_num", values="item_cnt_day", aggfunc="sum")
# The last value (base point of test data), the last value of last year (target value of training) and the previous one (base point of training data) are extracted.

last_test_block = pivot.iloc[:,-1].reset_index()

last_train_block = pivot.iloc[:,-2].reset_index()

last_train_2ndblock = pivot.iloc[:,-14].reset_index()
# Combine test data frame

Base = pd.merge(df_test, last_train_2ndblock, left_on=["shop_id", "item_id"], right_on=["shop_id", "item_id"], how="left")

Base = pd.merge(Base, last_train_block, left_on=["shop_id", "item_id"], right_on=["shop_id", "item_id"], how="left")

Base = pd.merge(Base, last_test_block, left_on=["shop_id", "item_id"], right_on=["shop_id", "item_id"], how="left")



Base = dtype_change(Base)

del last_test_block, last_train_block, last_train_2ndblock
# Create data with a corresponding relationship between item_id and category_id

category = train_df[["item_id", "item_category_id"]].drop_duplicates()
# merge Base and category

Base = pd.merge(Base, category, left_on="item_id", right_on="item_id", how="left")



del category
# Data check

# Null data

Base.isnull().sum()
# Data shape

Base.shape
# Trainin data

# shop data

Train = pd.merge(Base, shop_time_count_train, left_on="shop_id", right_on="id", how="left")

Train = pd.merge(Train, shop_time_price_train, left_on="shop_id", right_on="id", how="left")



# category data

Train = pd.merge(Train, cate_time_count_train, left_on="item_category_id", right_on="id", how="left")

Train = pd.merge(Train, cate_time_price_train, left_on="item_category_id", right_on="id", how="left")



# item data

Train = pd.merge(Train, item_time_count_train, left_on="item_id", right_on="id", how="left")

Train = pd.merge(Train, item_time_price_train, left_on="item_id", right_on="id", how="left")



Train.fillna(0, inplace=True)



Train.head()
# Divide learning data into explanatory variables and target values

# Train data

X_Train = Train.drop(["ID", "item_id", 33, 20, "id_x", "id_y", "shop_id", "item_category_id"], axis=1)



y_Train = Train[33].clip(0,20)
# # Test data

# shop data

Test = pd.merge(Base, shop_time_count_test, left_on="shop_id", right_on="id", how="left")

Test = pd.merge(Test, shop_time_price_test, left_on="shop_id", right_on="id", how="left")

# category data

Test = pd.merge(Test, cate_time_count_test, left_on="item_category_id", right_on="id", how="left")

Test = pd.merge(Test, cate_time_price_test, left_on="item_category_id", right_on="id", how="left")

# item data

Test = pd.merge(Test, item_time_count_test, left_on="item_id", right_on="id", how="left")

Test = pd.merge(Test, item_time_price_test, left_on="item_id", right_on="id", how="left")



Test.fillna(0, inplace=True)
Test.head()
# Divide learning data into explanatory variables

# Test data

X_Test = Test.drop(["ID", "item_id", 32, 20, "id_x", "id_y", "shop_id", "item_category_id"], axis=1)
print("X_Train shape:{}".format(X_Train.shape))

print("y_Train shape:{}".format(y_Train.shape))

print("X_Test shape:{}".format(X_Test.shape))
# Train test data split

X_train, X_val, y_train, y_val = train_test_split(X_Train, y_Train, test_size=0.2, random_state=10)
# Create instance

lgbm = lgb.LGBMRegressor()



params = {'learning_rate': [0.14, 0.18, 0.20], 'max_depth': [8, 10, 12]}



# Fitting

cv_lg = GridSearchCV(lgbm, params, cv = 10, n_jobs =1)

cv_lg.fit(X_train, y_train)



print("Best params:{}".format(cv_lg.best_params_))



best_lg = cv_lg.best_estimator_



# prediction

y_train_pred_lg = best_lg.predict(X_train)

y_val_pred_lg = best_lg.predict(X_val)



# prediction

y_train_pred_lg = cv_lg.predict(X_train)

y_val_pred_lg = cv_lg.predict(X_val)



print("MSE train:{}".format(mean_squared_error(y_train, y_train_pred_lg)))

print("MSE val;{}".format(mean_squared_error(y_val, y_val_pred_lg)))



print("R2 score train:{}".format(r2_score(y_train, y_train_pred_lg)))

print("R2 score val:{}".format(r2_score(y_val, y_val_pred_lg)))
# Training and score

ridge = Ridge()

params = {'alpha': [10000, 3000, 2000, 1000, 100, 10, 1]}



# Fitting

cv_r = GridSearchCV(ridge, params, cv = 10, n_jobs =1)

cv_r.fit(X_train, y_train)



print("Best params:{}".format(cv_r.best_params_))



best_r = cv_r.best_estimator_



# prediction

y_train_pred_r = best_r.predict(X_train)

y_val_pred_r = best_r.predict(X_val)



print("MSE train:{}".format(mean_squared_error(y_train, y_train_pred_r)))

print("MSE val;{}".format(mean_squared_error(y_val, y_val_pred_r)))



print("R2 score train:{}".format(r2_score(y_train, y_train_pred_r)))

print("R2 score val:{}".format(r2_score(y_val, y_val_pred_r)))
plt.figure(figsize=(6,6))

plt.scatter(y_val_pred_lg, y_val_pred_lg - y_val, c="red", marker='o', alpha=0.5, label="LGBM")

plt.scatter(y_val_pred_r, y_val_pred_r - y_val, c="green", marker='o', alpha=0.5, label="Rigde")

plt.xlabel('Predicted values')

plt.ylabel('Residuals')

plt.legend(loc = 'upper left')
plt.figure(figsize=(6,6))

plt.scatter(y_val.clip(0,20), y_val_pred_lg.clip(0,20), c="red", marker='o', alpha=0.5, label="LGBM")

plt.scatter(y_val.clip(0,20), y_val_pred_r.clip(0,20), c="green", marker='o', alpha=0.2, label="Rigde")

plt.xlabel('y_val data')

plt.ylabel('y_predcition')

plt.xlim([-2,22])

plt.ylim([-2,22])

plt.legend(loc = 'upper left')



print("MSE val LGBM:{}".format(mean_squared_error(y_val.clip(0,20), y_val_pred_lg.clip(0,20))))

print("MSE val Ridge:{}".format(mean_squared_error(y_val.clip(0,20), y_val_pred_r.clip(0,20))))
## Test prediction

y_test_pred = best_lg.predict(X_Test).clip(0,20)
# Predictin visualization

plt.figure(figsize=(10,6))

sns.distplot(y_test_pred, kde=False, bins=20)

plt.xlabel("prediction")

plt.xlim([-0.5,20.5])

plt.xticks(range(21))

plt.ylabel("Frequency")

plt.yscale("log")
# submit dataframe

submit = sample.copy()

submit["item_cnt_month"] = y_test_pred



submit.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")