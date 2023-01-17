# Basic libraries

import numpy as np

import pandas as pd



# Directry check

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Libraries

import datetime



# Visualization

from matplotlib import pyplot as plt

import seaborn as sns

sns.set()



# Data preporcessing

from sklearn.preprocessing import MinMaxScaler



# LSTM

from keras.models import Sequential

from keras.layers import LSTM

from keras.layers.core import Activation

from keras.optimizers import Adam

from keras.layers import Dense

from keras.layers import Dropout
df_items = pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/items.csv", header=0)

df_shops = pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/shops.csv", header=0)

df_sales_train = pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/sales_train.csv", header=0)

df_test = pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/test.csv", header=0)

df_category = pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/item_categories.csv", header=0)
# data size

print("items_data:", df_items.shape)

print("shops_data:", df_shops.shape)

print("sales_train_data:", df_sales_train.shape)

print("test_data:", df_test.shape)

print("category_data:", df_category.shape)
# Null data count

print("Null_of_items_data:\n{}".format(df_items.isnull().sum()))

print("-"*30)

print("Null_of_shops_data:\n{}".format(df_shops.isnull().sum()))

print("-"*30)

print("Null_of_sales_train_data:\n{}".format(df_sales_train.isnull().sum()))

print("-"*30)

print("Null_of_test_data:\n{}".format(df_test.isnull().sum()))

print("-"*30)

print("Null_of_category_data:\n{}".format(df_category.isnull().sum()))
# Data information

print("Info_of_items_data")

print(df_items.info())

print("-"*30)

print("Info_of_shops_data")

print(df_shops.info())

print("-"*30)

print("Info_of_sales_train_data")

print(df_sales_train.info())

print("-"*30)

print("Info_of_test_data")

print(df_test.info())

print("-"*30)

print("Info_of_category_data")

print(df_category.info())
# items data

df_items.head()
# shop data

df_shops.head()
# sales_train_data

print("unique_shop_id_count:{}".format(df_sales_train["shop_id"].value_counts().shape[0]))

print("unique_item_id_count:{}".format(df_sales_train["item_id"].value_counts().shape[0]))

df_sales_train.head()
# test_data

print("unique_shop_id_count:{}".format(df_test["shop_id"].value_counts().shape[0]))

print("unique_item_id_count:{}".format(df_test["item_id"].value_counts().shape[0]))

df_test.head()
# category_data

df_category.head()
# Time data

df_sales_train["date_dt"] = pd.to_datetime(df_sales_train["date"], format='%d.%m.%Y')



df_sales_train["year"] = df_sales_train["date_dt"].dt.year

df_sales_train["month"] = df_sales_train["date_dt"].dt.month

df_sales_train["day"] = df_sales_train["date_dt"].dt.day

df_sales_train["dayofweek"] = df_sales_train["date_dt"].dt.dayofweek

df_sales_train["dayofyear"] = df_sales_train["date_dt"].dt.dayofyear

df_sales_train["weekofyear"] = df_sales_train["date_dt"].dt.weekofyear
df_sales_train["item_sales"] = df_sales_train["item_price"]*df_sales_train["item_cnt_day"]



df_sales_train.head()
# shop_id distribution in df_train

plt.figure(figsize=(20,5))



index = [str(i) for i in df_sales_train["shop_id"].value_counts().index]

value = df_sales_train["shop_id"].value_counts()

plt.bar(index, value, color="blue", alpha=0.5)

plt.xlabel("shop_id")

plt.ylabel("count")

plt.title("shop_id_count from sales_train_data")
df_sales_train["item_id"].value_counts()
# Top50 item_id distribution in df_train

plt.figure(figsize=(20,5))



index = [str(i) for i in df_sales_train["item_id"].value_counts()[:50].index]

value = df_sales_train["item_id"].value_counts()[:50]

plt.bar(index, value, color="blue", alpha=0.5)

plt.xlabel("item_id")

plt.xticks(rotation=90)

plt.ylabel("count")

plt.title("Top50 item_id_count from sales_train_data")
# shop_id=31(max_sales) time series data

shop_31 = df_sales_train.query("shop_id==31").reset_index()



# shop_id=31(2nd_sales) time series data

shop_25 = df_sales_train.query("shop_id==25").reset_index()



# shop_id=31(3rd_sales) time series data

shop_54 = df_sales_train.query("shop_id==54").reset_index()



# shop_id=31(4th_sales) time series data

shop_28 = df_sales_train.query("shop_id==28").reset_index()



# shop_id=31(5th_sales) time series data

shop_57 = df_sales_train.query("shop_id==57").reset_index()







shop_31.head()
# Time series of daily sales

shop_31_monthly = shop_31.groupby(["year","month"]).sum()[["item_cnt_day", "item_sales"]].reset_index()

shop_25_monthly = shop_25.groupby(["year","month"]).sum()[["item_cnt_day", "item_sales"]].reset_index()

shop_54_monthly = shop_54.groupby(["year","month"]).sum()[["item_cnt_day", "item_sales"]].reset_index()

shop_28_monthly = shop_28.groupby(["year","month"]).sum()[["item_cnt_day", "item_sales"]].reset_index()

shop_57_monthly = shop_57.groupby(["year","month"]).sum()[["item_cnt_day", "item_sales"]].reset_index()



# Visualization of monthly time series

fig, ax = plt.subplots(1,5,figsize=(25,4))

plt.subplots_adjust(wspace=0.3, hspace=0.3)



# shop_31

ax[0].plot(shop_31_monthly.query("year==2013")["month"], shop_31_monthly.query("year==2013")["item_sales"], color="blue")

ax[0].plot(shop_31_monthly.query("year==2014")["month"], shop_31_monthly.query("year==2014")["item_sales"], color="red")

ax[0].plot(shop_31_monthly.query("year==2015")["month"], shop_31_monthly.query("year==2015")["item_sales"], color="green")

ax[0].set_xlabel("month")

ax[0].set_xlim([0,13])

ax[0].set_xticks(np.arange(1,13,1))

ax[0].set_ylabel("sales")

ax[0].set_title("shop_31_sales")

ax[0].legend(labels=("2013","2014","2015"))



# shop_25

ax[1].plot(shop_25_monthly.query("year==2013")["month"], shop_25_monthly.query("year==2013")["item_sales"], color="blue")

ax[1].plot(shop_25_monthly.query("year==2014")["month"], shop_25_monthly.query("year==2014")["item_sales"], color="red")

ax[1].plot(shop_25_monthly.query("year==2015")["month"], shop_25_monthly.query("year==2015")["item_sales"], color="green")

ax[1].set_xlabel("month")

ax[1].set_xlim([0,13])

ax[1].set_xticks(np.arange(1,13,1))

ax[1].set_ylabel("sales")

ax[1].set_title("shop_25_sales")

ax[1].legend(labels=("2013","2014","2015"))



# shop_54

ax[2].plot(shop_54_monthly.query("year==2013")["month"], shop_54_monthly.query("year==2013")["item_sales"], color="blue")

ax[2].plot(shop_54_monthly.query("year==2014")["month"], shop_54_monthly.query("year==2014")["item_sales"], color="red")

ax[2].plot(shop_54_monthly.query("year==2015")["month"], shop_54_monthly.query("year==2015")["item_sales"], color="green")

ax[2].set_xlabel("month")

ax[2].set_xlim([0,13])

ax[2].set_xticks(np.arange(1,13,1))

ax[2].set_ylabel("sales")

ax[2].set_title("shop_54_sales")

ax[2].legend(labels=("2013","2014","2015"))



# shop_28

ax[3].plot(shop_28_monthly.query("year==2013")["month"], shop_28_monthly.query("year==2013")["item_sales"], color="blue")

ax[3].plot(shop_28_monthly.query("year==2014")["month"], shop_28_monthly.query("year==2014")["item_sales"], color="red")

ax[3].plot(shop_28_monthly.query("year==2015")["month"], shop_28_monthly.query("year==2015")["item_sales"], color="green")

ax[3].set_xlabel("month")

ax[3].set_xlim([0,13])

ax[3].set_xticks(np.arange(1,13,1))

ax[3].set_ylabel("sales")

ax[3].set_title("shop_28_sales")

ax[3].legend(labels=("2013","2014","2015"))



# shop_57

ax[4].plot(shop_57_monthly.query("year==2013")["month"], shop_57_monthly.query("year==2013")["item_sales"], color="blue")

ax[4].plot(shop_57_monthly.query("year==2014")["month"], shop_57_monthly.query("year==2014")["item_sales"], color="red")

ax[4].plot(shop_57_monthly.query("year==2015")["month"], shop_57_monthly.query("year==2015")["item_sales"], color="green")

ax[4].set_xlabel("month")

ax[4].set_xlim([0,13])

ax[4].set_xticks(np.arange(1,13,1))

ax[4].set_ylabel("sales")

ax[4].set_title("shop_57_sales")

ax[4].legend(labels=("2013","2014","2015"))
# Visualization of monthly auto correlation 

# define the function

def auto_corr(data, k):

    y_mean = np.mean(data)

    sum_of_covariance = 0

    for i in range(k+1, len(data)):

        covariance = ( data[i] - y_mean )*( data[i-(k+1)] - y_mean )

        sum_of_covariance += covariance

    sum_of_denominator = 0

    for u in range(len(data)):

        denominator = ( data[u] - y_mean )**2

        sum_of_denominator += denominator

    return sum_of_covariance / sum_of_denominator



# Visualization of monthly time series

fig, ax = plt.subplots(1,5,figsize=(25,4))

plt.subplots_adjust(wspace=0.3, hspace=0.3)



# shop_31

acorr_data = [1]

for i in range(0,12):

    acorr_data.append(auto_corr(shop_31_monthly["item_sales"], i))

delta_month = np.arange(0,13)

# figure

ax[0].stem(delta_month, acorr_data)

ax[0].set_title("Auto_correlation")

ax[0].set_xlabel("delta_month")

ax[0].set_xlim([-1,13])

ax[0].set_xticks(np.arange(0,13,1))

ax[0].set_ylabel("ACF")



# shop_25

acorr_data = [1]

for i in range(0,12):

    acorr_data.append(auto_corr(shop_25_monthly["item_sales"], i))

delta_month = np.arange(0,13)

# figure

ax[1].stem(delta_month, acorr_data)

ax[1].set_title("Auto_correlation")

ax[1].set_xlabel("delta_month")

ax[1].set_xlim([-1,13])

ax[1].set_xticks(np.arange(0,13,1))

ax[1].set_ylabel("ACF")



# shop_54

acorr_data = [1]

for i in range(0,12):

    acorr_data.append(auto_corr(shop_54_monthly["item_sales"], i))

delta_month = np.arange(0,13)

# figure

ax[2].stem(delta_month, acorr_data)

ax[2].set_title("Auto_correlation")

ax[2].set_xlabel("delta_month")

ax[2].set_xlim([-1,13])

ax[2].set_xticks(np.arange(0,13,1))

ax[2].set_ylabel("ACF")



# shop_28

acorr_data = [1]

for i in range(0,12):

    acorr_data.append(auto_corr(shop_28_monthly["item_sales"], i))

delta_month = np.arange(0,13)

# figure

ax[3].stem(delta_month, acorr_data)

ax[3].set_title("Auto_correlation")

ax[3].set_xlabel("delta_month")

ax[3].set_xlim([-1,13])

ax[3].set_xticks(np.arange(0,13,1))

ax[3].set_ylabel("ACF")



# shop_57

acorr_data = [1]

for i in range(0,12):

    acorr_data.append(auto_corr(shop_57_monthly["item_sales"], i))

delta_month = np.arange(0,13)

# figure

ax[4].stem(delta_month, acorr_data)

ax[4].set_title("Auto_correlation")

ax[4].set_xlabel("delta_month")

ax[4].set_xlim([-1,13])

ax[4].set_xticks(np.arange(0,13,1))

ax[4].set_ylabel("ACF")
# Visualization of monthly time series

fig, ax = plt.subplots(1,5,figsize=(25,4))

plt.subplots_adjust(wspace=0.3, hspace=0.3)



# shop_31

ax[0].plot(shop_31_monthly.query("year==2013")["month"], shop_31_monthly.query("year==2013")["item_cnt_day"], color="blue")

ax[0].plot(shop_31_monthly.query("year==2014")["month"], shop_31_monthly.query("year==2014")["item_cnt_day"], color="red")

ax[0].plot(shop_31_monthly.query("year==2015")["month"], shop_31_monthly.query("year==2015")["item_cnt_day"], color="green")

ax[0].set_xlabel("month")

ax[0].set_xlim([0,13])

ax[0].set_xticks(np.arange(1,13,1))

ax[0].set_ylabel("sales")

ax[0].set_title("shop_31_sales_volume")

ax[0].legend(labels=("2013","2014","2015"))



# shop_25

ax[1].plot(shop_25_monthly.query("year==2013")["month"], shop_25_monthly.query("year==2013")["item_cnt_day"], color="blue")

ax[1].plot(shop_25_monthly.query("year==2014")["month"], shop_25_monthly.query("year==2014")["item_cnt_day"], color="red")

ax[1].plot(shop_25_monthly.query("year==2015")["month"], shop_25_monthly.query("year==2015")["item_cnt_day"], color="green")

ax[1].set_xlabel("month")

ax[1].set_xlim([0,13])

ax[1].set_xticks(np.arange(1,13,1))

ax[1].set_ylabel("sales")

ax[1].set_title("shop_25_sales_volume")

ax[1].legend(labels=("2013","2014","2015"))



# shop_54

ax[2].plot(shop_54_monthly.query("year==2013")["month"], shop_54_monthly.query("year==2013")["item_cnt_day"], color="blue")

ax[2].plot(shop_54_monthly.query("year==2014")["month"], shop_54_monthly.query("year==2014")["item_cnt_day"], color="red")

ax[2].plot(shop_54_monthly.query("year==2015")["month"], shop_54_monthly.query("year==2015")["item_cnt_day"], color="green")

ax[2].set_xlabel("month")

ax[2].set_xlim([0,13])

ax[2].set_xticks(np.arange(1,13,1))

ax[2].set_ylabel("sales")

ax[2].set_title("shop_54_sales_volume")

ax[2].legend(labels=("2013","2014","2015"))



# shop_28

ax[3].plot(shop_28_monthly.query("year==2013")["month"], shop_28_monthly.query("year==2013")["item_cnt_day"], color="blue")

ax[3].plot(shop_28_monthly.query("year==2014")["month"], shop_28_monthly.query("year==2014")["item_cnt_day"], color="red")

ax[3].plot(shop_28_monthly.query("year==2015")["month"], shop_28_monthly.query("year==2015")["item_cnt_day"], color="green")

ax[3].set_xlabel("month")

ax[3].set_xlim([0,13])

ax[3].set_xticks(np.arange(1,13,1))

ax[3].set_ylabel("sales")

ax[3].set_title("shop_28_sales_volume")

ax[3].legend(labels=("2013","2014","2015"))



# shop_57

ax[4].plot(shop_57_monthly.query("year==2013")["month"], shop_57_monthly.query("year==2013")["item_cnt_day"], color="blue")

ax[4].plot(shop_57_monthly.query("year==2014")["month"], shop_57_monthly.query("year==2014")["item_cnt_day"], color="red")

ax[4].plot(shop_57_monthly.query("year==2015")["month"], shop_57_monthly.query("year==2015")["item_cnt_day"], color="green")

ax[4].set_xlabel("month")

ax[4].set_xlim([0,13])

ax[4].set_xticks(np.arange(1,13,1))

ax[4].set_ylabel("sales")

ax[4].set_title("shop_57_sales_volume")

ax[4].legend(labels=("2013","2014","2015"))
# item_id=20949(max_item) time series data

item_20949 = df_sales_train.query("item_id==20949").reset_index()



# shop_id=5822(2nd_sales) time series data

item_5822 = df_sales_train.query("item_id==5822").reset_index()



# item_id=17717(3rd_sales) time series data

item_17717 = df_sales_train.query("item_id==17717").reset_index()



# item_id=2808(4th_sales) time series data

item_2808 = df_sales_train.query("item_id==2808").reset_index()



# item_id=4181(5th_sales) time series data

item_4181 = df_sales_train.query("item_id==4181").reset_index()







item_20949.head()
# Time series of daily sales

item_20949_monthly = item_20949.groupby(["year","month"]).sum()[["item_sales"]].reset_index()

item_5822_monthly = item_5822.groupby(["year","month"]).sum()[["item_sales"]].reset_index()

item_17717_monthly = item_17717.groupby(["year","month"]).sum()[["item_sales"]].reset_index()

item_2808_monthly = item_2808.groupby(["year","month"]).sum()[["item_sales"]].reset_index()

item_4181_monthly = item_4181.groupby(["year","month"]).sum()[["item_sales"]].reset_index()



# Visualization of monthly time series

fig, ax = plt.subplots(1,5,figsize=(25,4))

plt.subplots_adjust(wspace=0.3, hspace=0.3)



# item_20949

ax[0].plot(item_20949_monthly.query("year==2013")["month"], item_20949_monthly.query("year==2013")["item_sales"], color="blue")

ax[0].plot(item_20949_monthly.query("year==2014")["month"], item_20949_monthly.query("year==2014")["item_sales"], color="red")

ax[0].plot(item_20949_monthly.query("year==2015")["month"], item_20949_monthly.query("year==2015")["item_sales"], color="green")

ax[0].set_xlabel("month")

ax[0].set_xlim([0,13])

ax[0].set_xticks(np.arange(1,13,1))

ax[0].set_ylabel("sales")

ax[0].set_title("item_20949_sales")

ax[0].legend(labels=("2013","2014","2015"))



# item_5822

ax[1].plot(item_5822_monthly.query("year==2013")["month"], item_5822_monthly.query("year==2013")["item_sales"], color="blue")

ax[1].plot(item_5822_monthly.query("year==2014")["month"], item_5822_monthly.query("year==2014")["item_sales"], color="red")

ax[1].plot(item_5822_monthly.query("year==2015")["month"], item_5822_monthly.query("year==2015")["item_sales"], color="green")

ax[1].set_xlabel("month")

ax[1].set_xlim([0,13])

ax[1].set_xticks(np.arange(1,13,1))

ax[1].set_ylabel("sales")

ax[1].set_title("item_5822_sales")

ax[1].legend(labels=("2013","2014","2015"))



# item_17717

ax[2].plot(item_17717_monthly.query("year==2013")["month"], item_17717_monthly.query("year==2013")["item_sales"], color="blue")

ax[2].plot(item_17717_monthly.query("year==2014")["month"], item_17717_monthly.query("year==2014")["item_sales"], color="red")

ax[2].plot(item_17717_monthly.query("year==2015")["month"], item_17717_monthly.query("year==2015")["item_sales"], color="green")

ax[2].set_xlabel("month")

ax[2].set_xlim([0,13])

ax[2].set_xticks(np.arange(1,13,1))

ax[2].set_ylabel("sales")

ax[2].set_title("item_17717_sales")

ax[2].legend(labels=("2013","2014","2015"))



# item_2808

ax[3].plot(item_2808_monthly.query("year==2013")["month"], item_2808_monthly.query("year==2013")["item_sales"], color="blue")

ax[3].plot(item_2808_monthly.query("year==2014")["month"], item_2808_monthly.query("year==2014")["item_sales"], color="red")

ax[3].plot(item_2808_monthly.query("year==2015")["month"], item_2808_monthly.query("year==2015")["item_sales"], color="green")

ax[3].set_xlabel("month")

ax[3].set_xlim([0,13])

ax[3].set_xticks(np.arange(1,13,1))

ax[3].set_ylabel("sales")

ax[3].set_title("item_2808_sales")

ax[3].legend(labels=("2013","2014","2015"))



# item_4181

ax[4].plot(item_4181_monthly.query("year==2013")["month"], item_4181_monthly.query("year==2013")["item_sales"], color="blue")

ax[4].plot(item_4181_monthly.query("year==2014")["month"], item_4181_monthly.query("year==2014")["item_sales"], color="red")

ax[4].plot(item_4181_monthly.query("year==2015")["month"], item_4181_monthly.query("year==2015")["item_sales"], color="green")

ax[4].set_xlabel("month")

ax[4].set_xlim([0,13])

ax[4].set_xticks(np.arange(1,13,1))

ax[4].set_ylabel("sales")

ax[4].set_title("item_4181_sales")

ax[4].legend(labels=("2013","2014","2015"))
# Visualization of monthly time series

fig, ax = plt.subplots(1,5,figsize=(25,4))

plt.subplots_adjust(wspace=0.3, hspace=0.3)



# item_20949

acorr_data = [1]

for i in range(0,12):

    acorr_data.append(auto_corr(item_20949_monthly["item_sales"], i))

delta_month = np.arange(0,13)

# figure

ax[0].stem(delta_month, acorr_data)

ax[0].set_title("Auto_correlation")

ax[0].set_xlabel("delta_month")

ax[0].set_xlim([-1,13])

ax[0].set_xticks(np.arange(0,13,1))

ax[0].set_ylabel("ACF")



# item_5822

acorr_data = [1]

for i in range(0,12):

    acorr_data.append(auto_corr(item_5822_monthly["item_sales"], i))

delta_month = np.arange(0,13)

# figure

ax[1].stem(delta_month, acorr_data)

ax[1].set_title("Auto_correlation")

ax[1].set_xlabel("delta_month")

ax[1].set_xlim([-1,13])

ax[1].set_xticks(np.arange(0,13,1))

ax[1].set_ylabel("ACF")



# item_17717

acorr_data = [1]

for i in range(0,12):

    acorr_data.append(auto_corr(item_17717_monthly["item_sales"], i))

delta_month = np.arange(0,13)

# figure

ax[2].stem(delta_month, acorr_data)

ax[2].set_title("Auto_correlation")

ax[2].set_xlabel("delta_month")

ax[2].set_xlim([-1,13])

ax[2].set_xticks(np.arange(0,13,1))

ax[2].set_ylabel("ACF")



# item_2808

acorr_data = [1]

for i in range(0,12):

    acorr_data.append(auto_corr(item_2808_monthly["item_sales"], i))

delta_month = np.arange(0,13)

# figure

ax[3].stem(delta_month, acorr_data)

ax[3].set_title("Auto_correlation")

ax[3].set_xlabel("delta_month")

ax[3].set_xlim([-1,13])

ax[3].set_xticks(np.arange(0,13,1))

ax[3].set_ylabel("ACF")



# item_4181

acorr_data = [1]

for i in range(0,12):

    acorr_data.append(auto_corr(item_4181_monthly["item_sales"], i))

delta_month = np.arange(0,13)

# figure

ax[4].stem(delta_month, acorr_data)

ax[4].set_title("Auto_correlation")

ax[4].set_xlabel("delta_month")

ax[4].set_xlim([-1,13])

ax[4].set_xticks(np.arange(0,13,1))

ax[4].set_ylabel("ACF")
# train data setting

# pivot_table

pivot_train = pd.pivot_table(df_sales_train, index=["shop_id", "item_id"], columns="date_block_num", values="item_cnt_day", aggfunc="sum", fill_value=0).reset_index()



# Only id of test data & keeping only the needed data..

df = pd.merge(df_test, pivot_train, on=["item_id", "shop_id"], how="left").fillna(0)

df.drop(['ID', 'shop_id','item_id','ID'],inplace = True, axis = 1)

data = df.T.values



# Min-Max Scaling

ms = MinMaxScaler()

data_norm = pd.DataFrame(ms.fit_transform(data).T)



# data shape

data_norm.shape
# Training_data

X_train = data_norm.iloc[:,-30:-12].values

y_train = data_norm.iloc[:,-12].values

X_train = np.expand_dims(X_train,axis = 2) # (214200,18)⇒(214200,18,1)



# Test_data

X_test = data_norm.iloc[:,-18:].values

X_test = np.expand_dims(X_test,axis = 2) # (214200,18)⇒(214200,18,1)



# Shape check 

print(X_train.shape,y_train.shape,X_test.shape)
# Model construction

length = 18

in_out_neurons = 1

n_hidden = 300



# Model

model = Sequential()

model.add(LSTM(n_hidden,

               batch_input_shape=(None, length, in_out_neurons), # (, 18, 1)

               return_sequences=False)

         )

model.add(Dropout(0.4))

model.add(Dense(in_out_neurons))

model.add(Activation("relu"))

optimizer = Adam(lr=0.001)



# Compile

model.compile(loss="mean_squared_error", optimizer=optimizer, metrics = ['mean_squared_error'])



model.summary()
# Fitting model

model.fit(X_train,y_train,batch_size = 1000,epochs = 10)
# creating submission file 

pred = model.predict(X_test)



# Scaling

submit_data = ms.inverse_transform(pred.T)

submit_data = submit_data.clip(0,20)
# Submitting data

submit = pd.DataFrame({'ID':df_test['ID'],'item_cnt_month':submit_data.reshape(-1,)})
# Data describe

submit.describe()
# Result distribution

plt.figure(figsize=(10,6))

sns.distplot(submit["item_cnt_month"], bins=20, kde=False)

plt.yscale("log")

plt.ylabel('log(count)')
submit.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")