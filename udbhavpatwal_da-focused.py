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
calendar = pd.read_csv("/kaggle/input/m5-forecasting-accuracy/calendar.csv")
sell_prices = pd.read_csv("/kaggle/input/m5-forecasting-accuracy/sell_prices.csv")
sales_train = pd.read_csv("/kaggle/input/m5-forecasting-accuracy/sales_train_validation.csv")
sample_sub = pd.read_csv("/kaggle/input/m5-forecasting-accuracy/sample_submission.csv")

def memory_reduction(dataset):
    column_types = dataset.dtypes
    temp = None
    for x in range(len(column_types)):
        column_types[x] = str(column_types[x])
    for x in range(len(column_types)):
        temp = dataset.columns[x]
        if dataset.columns[x] == "date":
            dataset[temp] = dataset[temp].astype("datetime64")
        if column_types[x] == "int64" and dataset.columns[x] != "date":
            dataset[temp] = dataset[temp].astype("int16")
        if column_types[x] == "object" and dataset.columns[x] != "date":
            dataset[temp] = dataset[temp].astype("category")
        if column_types[x] == "float64" and dataset.columns[x] != "date":
            dataset[temp] = dataset[temp].astype("float16")
    return dataset
calendar_df = memory_reduction(calendar)
sell_prices["id"] = sell_prices["item_id"] + "_" + sell_prices["store_id"] + "_validation" 
sell_prices = pd.merge(sell_prices, sales_train[["cat_id", "id", "state_id"]], on = "id")
sell_prices_df = memory_reduction(sell_prices)
sales_train_df = memory_reduction(sales_train)
calendar_df = calendar_df[:1913]
calendar_df["day"] = pd.DatetimeIndex(calendar_df["date"]).day
calendar_df["day"] = calendar_df["day"].astype("int8")
calendar_df["week_num"] = (calendar_df["day"] - 1) // 7 + 1
calendar_df["week_num"] = calendar_df["week_num"].astype("int8")
import gc
def make_dataframe():
    # Wide format dataset 
    df_wide_train = sales_train_df.drop(columns=["item_id", "dept_id", "cat_id", "state_id","store_id", "id"]).T
    df_wide_train.index = calendar_df["date"]
    df_wide_train.columns = sales_train_df["id"]
    
   
    # Convert wide format to long format
    df_long = df_wide_train.stack().reset_index(1)
    df_long.columns = ["id", "value"]

    del df_wide_train
    gc.collect()
    
    df = pd.merge(pd.merge(df_long.reset_index(), calendar_df, on="date"), sell_prices_df, on=["id", "wm_yr_wk"])
    df = df.drop(columns=["d"])
    #df[["cat_id", "store_id", "item_id", "id", "dept_id"]] = df[["cat_id"", store_id", "item_id", "id", "dept_id"]].astype("category")
    df["sell_price"] = df["sell_price"].astype("float16")   
    df["value"] = df["value"].astype("int32")
    df["state_id"] = df["store_id"].str[:2].astype("category")


    del df_long
    gc.collect()

    return df

df = make_dataframe()
del calendar, sales_train, sell_prices
gc.collect()

df.head()
#importing all necessary libraries
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import seaborn as sn
%matplotlib inline
df.columns
df.shape
plt.figure(figsize = (12,12))
sn.heatmap(df.corr(), annot=True)
df.dtypes
df.isnull().sum()
sales_train_df.describe()
calendar_df.describe()
sell_prices_df.describe()
#value
temp = df.groupby(["cat_id", "date"])["value"].sum()
x = temp[temp.index.get_level_values("cat_id") == 'FOODS'].values
plt.hist(x)
x = temp[temp.index.get_level_values("cat_id") == 'HOUSEHOLD'].values
plt.hist(x)
gc.collect()
x = temp[temp.index.get_level_values("cat_id") == 'HOBBIES'].values
plt.hist(x)
del x, temp
gc.collect()
# ALTERNATE PROCEDURE Of DOING THIS WILL BE SHOWN HERE after some time
calendar_df.head()
calendar_df['event_name_1'].value_counts().plot.bar()
calendar_df['event_type_1'].value_counts().plot.bar()
calendar_df['event_name_2'].value_counts().plot.bar()
calendar_df['event_type_2'].value_counts().plot.bar()
calendar_df["snap_CA"].value_counts()
calendar_df["snap_TX"].value_counts()
calendar_df["snap_WI"].value_counts()
df["snap_CA"].value_counts()
df["snap_TX"].value_counts()
df["snap_WI"].value_counts()
df["cat_id"].value_counts()
sell_prices_df["cat_id"].value_counts()
sell_prices_df["cat_id"].value_counts().plot.bar()
sell_prices_df["state_id"].value_counts()
sell_prices_df["state_id"].value_counts().plot.bar()
sn.countplot(sell_prices_df.store_id)
sell_prices_df.store_id.value_counts()
df.head()
temp  = df.groupby(["cat_id", "date"])["value"].sum()
plt.figure(figsize = (8,6))
plt.plot(temp[temp.index.get_level_values('cat_id') == "FOODS"].index.get_level_values("date"), temp[temp.index.get_level_values('cat_id') == "FOODS"].values, label ="FOODS")
plt.plot(temp[temp.index.get_level_values('cat_id') == "HOUSEHOLD"].index.get_level_values("date"), temp[temp.index.get_level_values('cat_id') == "HOUSEHOLD"].values, label ="HOUSEHOLD")
plt.plot(temp[temp.index.get_level_values('cat_id') == "HOBBIES"].index.get_level_values("date"), temp[temp.index.get_level_values('cat_id') == "HOBBIES"].values, label ="HOBBIES")
plt.legend()
plt.show()

df.head()
#So lets analyze events first like how do they affect the data 
#first we will take event_type_1 in considerations
event_call = df.groupby(["event_name_1", "date"])["value"].sum()
plt.figure(figsize = (8,6))
plt.plot(temp[temp.index.get_level_values('cat_id') == "FOODS"].index.get_level_values("date"), temp[temp.index.get_level_values('cat_id') == "FOODS"].values, label ="FOODS")
plt.plot(temp[temp.index.get_level_values('cat_id') == "HOUSEHOLD"].index.get_level_values("date"), temp[temp.index.get_level_values('cat_id') == "HOUSEHOLD"].values, label ="HOUSEHOLD")
plt.plot(temp[temp.index.get_level_values('cat_id') == "HOBBIES"].index.get_level_values("date"), temp[temp.index.get_level_values('cat_id') == "HOBBIES"].values, label ="HOBBIES")
plt.legend()
plt.show()
pd.get_dummies(calendar_df, columns=["event_name_1"]).head()
