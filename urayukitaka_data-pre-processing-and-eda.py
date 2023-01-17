import numpy as np

import pandas as pd

import os



import seaborn as sns

import matplotlib.pyplot as plt

plt.style.use("fivethirtyeight")



# Time series analysis

from statsmodels.graphics.tsaplots import plot_acf



# Normality test

from scipy import stats



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
path = "/kaggle/input/agricultural-raw-material-prices-19902020/"



data = pd.read_csv(os.path.join(path, "agricultural_raw_material.csv"))
# columns name change, no space

col_name = ['Month', 'Coarse_wool_Price', 'Coarse_wool_price_%_Change', 'Copra_Price',

       'Copra_price_%_Change', 'Cotton_Price', 'Cotton_price_%_Change',

       'Fine_wool_Price', 'Fine_wool_price_%_Change', 'Hard_log_Price',

       'Hard_log_price_%_Change', 'Hard_sawnwood_Price',

       'Hard_sawnwood_price_%_Change', 'Hide_Price', 'Hide_price_%_change',

       'Plywood_Price', 'Plywood_price_%_Change', 'Rubber_Price',

       'Rubber_price_%_Change', 'Softlog_Price', 'Softlog_price_%_Change',

       'Soft_sawnwood_Price', 'Soft_sawnwood_price_%_Change',

       'Wood_pulp_Price', 'Wood_pulp_price_%_Change']



data.columns = col_name
# split year and month from string data

data["year"] = [int(s.split("-")[1]) for s in data["Month"]]

data["month"] = [str(s.split("-")[0]) for s in data["Month"]]



# year are changed to thousand year

data["year"] = [2000 + i if i < 89 else 1900 + i for i in data["year"]]



# month are changed to int from string

mapping = {"Apr":4, "May":5, "Jun":6, "Jul":7, "Aug":8, "Sep":9, "Oct":10, "Nov":11, "Dec":12, "Jan":1, "Feb":2, "Mar":3}

data["month"] = data["month"].map(mapping).astype("int16")

data.drop("Month", axis=1, inplace=True)



# create datetime column and set index 

data["date"] = [str(y) + '/' + str(m) for y,m in zip(data["year"], data["month"])]

data["date_dt"] = pd.to_datetime(data["date"])

data.set_index("date_dt", inplace=True)

data.drop("date", axis=1, inplace=True)
# dtype df

dtype_df = pd.DataFrame({"columns":data.dtypes.index,

             "dtype":[str(s) for s in data.dtypes.values]})

str_col = dtype_df[dtype_df["dtype"]=="object"]["columns"].values

str_col
# roop process

for c in str_col:

    list_1 = []

    list_2 = []

    list_3 = []

    # change '-' to '0'

    for s in data[c]:

        if s == '-':

            res = '0'

            list_1.append(res)

        else:

            res = s

            list_1.append(res)



    # replace ',' to ''

    for s in list_1:

        if ',' in str(s):

            res = s.replace(",", "")

            list_2.append(res)

        else:

            res = s

            list_2.append(res)

    

    # replace '%' to '' and change to float

    for s in list_2:

        if '%' in str(s):

            res = float(s.replace("%", ""))

            list_3.append(res)

        else:

            res = float(s)

            list_3.append(res)



    data[c] = list_3

    del list_1, list_2, list_3
# data chenge to light

# create dtype dataframe

dtype_df_2 = pd.DataFrame({"columns":data.dtypes.index,

             "dtype":[str(s) for s in data.dtypes.values]})



# Chenge to dtype

float_col = dtype_df_2[dtype_df_2["dtype"] == "float64"]["columns"].values

int_col = dtype_df_2[dtype_df_2["dtype"] == "int64"]["columns"].values



data[float_col] = data[float_col].astype("float32")

data[int_col] = data[int_col].astype("int16")
data.info()
day_col = ['year', 'month']

data_col = ['Coarse_wool_Price', 'Copra_Price', 'Cotton_Price', 'Fine_wool_Price',

            'Hard_log_Price', 'Hard_sawnwood_Price', 'Hide_Price', 'Plywood_Price',

            'Rubber_Price', 'Softlog_Price', 'Soft_sawnwood_Price', 'Wood_pulp_Price']

ratio_col = ['Coarse_wool_price_%_Change', 'Copra_price_%_Change', 'Cotton_price_%_Change', 'Fine_wool_price_%_Change', 

             'Hard_log_price_%_Change', 'Hard_sawnwood_price_%_Change', 'Hide_price_%_change', 'Plywood_price_%_Change', 

             'Rubber_price_%_Change', 'Softlog_price_%_Change', 'Soft_sawnwood_price_%_Change', 'Wood_pulp_price_%_Change']
# barplot

def bar_plot(df, data_col, day_col, groupby_col='year', value_label="price"):

    df_ = df[data_col+day_col]

    ave = df_.groupby(groupby_col)[data_col].mean()

    

    fig, ax = plt.subplots(4, 3, figsize=(30, 24))

    plt.subplots_adjust(hspace=0.4)

    for i in range(0,4):

        for j in range(0,3):

            ax[i,j].bar(ave.index, ave.iloc[:,i*3+j])

            ax[i,j].set_xlabel(groupby_col)

            ax[i,j].set_ylabel(value_label)

            ax[i,j].set_title(data_col[i*3+j])



    plt.show()

    

# time series plot

def timeseries_plot(df, data_col, value_label="price"):

    df_ = df[data_col]

    



    fig, ax = plt.subplots(4, 3, figsize=(30, 24))

    plt.subplots_adjust(hspace=0.4)

    for i in range(0,4):

        for j in range(0,3):

            ax[i,j].plot(df_[data_col[i*3+j]].index, df_[data_col[i*3+j]], label="month price")

            ax[i,j].plot(df_[data_col[i*3+j]].rolling(12).mean().index, df_[data_col[i*3+j]].rolling(12).mean(), label="Rolling mean 12 month")

            ax[i,j].set_xlabel("date")

            ax[i,j].set_ylabel("price")

            ax[i,j].set_title(data_col[i*3+j])

            ax[i,j].legend()



    plt.show()   

    

# Auto correlation plot

def autocorrelation_plot(df, data_col, lags=24):

    df_ = df[data_col]



    fig, ax = plt.subplots(4, 3, figsize=(30, 24))

    plt.subplots_adjust(hspace=0.4)

    for i in range(0,4):

        for j in range(0,3):

            plot_acf(df_[data_col[i*3+j]].dropna(), lags=lags, ax=ax[i,j])

            ax[i,j].set_xlabel("lags")

            ax[i,j].set_ylabel("ACF")

            ax[i,j].set_title(data_col[i*3+j])



    plt.show()

    

# Distribution plot and test of normalize

def dist_plot(df, ratio_col):

    df_ = df[ratio_col]



    fig, ax = plt.subplots(4, 3, figsize=(30, 24))

    plt.subplots_adjust(hspace=0.4)

    for i in range(0,4):

        for j in range(0,3):

            sns.distplot(df_[ratio_col[i*3+j]].dropna(), ax=ax[i,j])

            ax[i,j].set_xlabel("percentage")

            ax[i,j].set_ylabel("frequency")

            ax[i,j].set_title(data_col[i*3+j])



    plt.show()

    

# Distribution plot and test of normalize

def prob_plot(df, ratio_col):

    df_ = df[ratio_col]



    fig, ax = plt.subplots(4, 3, figsize=(30, 24))

    plt.subplots_adjust(hspace=0.4)

    for i in range(0,4):

        for j in range(0,3):

            stats.probplot(df_[ratio_col[i*3+j]].dropna(), plot=ax[i,j])

            ax[i,j].set_xlabel("percentage")

            ax[i,j].set_ylabel("frequency")

            ax[i,j].set_title(data_col[i*3+j])



    plt.show()
plt.figure(figsize=(8,8))

hm = sns.heatmap(data[data_col].corr(), vmax=1, vmin=-1, cmap="bwr", square=True)
sns.pairplot(data[data_col])
bar_plot(data, data_col, day_col, groupby_col='year', value_label="price")
bar_plot(data, data_col, day_col, groupby_col='month', value_label="price")
timeseries_plot(data, data_col, value_label="price")
autocorrelation_plot(data, data_col, lags=24)
plt.figure(figsize=(8,8))

hm = sns.heatmap(data[ratio_col].corr(), vmax=1, vmin=-1, cmap="bwr", square=True)
bar_plot(data, ratio_col, day_col, groupby_col='year', value_label="price_change_ratio(%)")
bar_plot(data, ratio_col, day_col, groupby_col='month', value_label="price_change_ratio(%)")
timeseries_plot(data, ratio_col, value_label="price_change_ratio(%)")
autocorrelation_plot(data, ratio_col, lags=24)
dist_plot(data, ratio_col)
prob_plot(data, ratio_col)