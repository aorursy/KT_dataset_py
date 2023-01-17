# Basic libraries

import pandas as pd

import numpy as np

import time

import datetime

import gc



# Data preprocessing

import category_encoders as ce



# Visualization

import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use("fivethirtyeight")



# Time series analysis

from statsmodels.graphics.tsaplots import plot_acf

import statsmodels.api as sm



# Normality test

from scipy import stats



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Dataloading

path = "/kaggle/input/m5-forecasting-accuracy/"



calendar = pd.read_csv(os.path.join(path,"calendar.csv"))

train = pd.read_csv(os.path.join(path, "sales_train_validation.csv"))

price = pd.read_csv(os.path.join(path, "sell_prices.csv"))
# Change dtype to light

# Calendar data

def dtype_ch_calendar(df):

    # Columns name

    int16_col = ["wday", "month", "snap_CA", "snap_TX", "snap_WI","wm_yr_wk", "year"]



    # dtype change

    df["date"] = pd.to_datetime(df["date"])

    df[int16_col] = df[int16_col].astype("int16")



    return df



# price data

def dtype_ch_price(df):

    # Columns name

    int16_col = ["wm_yr_wk"]

    float16_col = ["sell_price"]



    # dtype change

    df[int16_col] = df[int16_col].astype("int16")

    df[float16_col] = df[float16_col].astype("float16")



    return df



# train data

def dtype_ch_train(df):

    # Columns name

    int16_col = df.loc[:,"d_1":].columns

    # dtype change

    df[int16_col] = df[int16_col].astype("int16")



    return df
def create_features_calendar(df):

    # Change dtype to light

    df = dtype_ch_calendar(df)



    # day of month variable

    df["mday"] = df["date"].dt.day.astype("int16")



    # event object to numerical, ordinal encoder

    list_col = ["event_name_1", "event_type_1", "event_name_2", "event_type_2"]

    for i in list_col:

        ce_oe = ce.OrdinalEncoder(cols=i, handle_unknown='impute') # Create instance of OrdinalEncoder

        df = ce_oe.fit_transform(df)

        df[i] = df[i].astype("int16") # change to light dtype

        

    return df
# dtype change to light

calendar = create_features_calendar(dtype_ch_calendar(calendar))

price = dtype_ch_price(price)

train = dtype_ch_train(train)
# Data merge

def data_merge_3df(train, calendar, price):

    df = pd.DataFrame({})

    id_col = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]

    df = train.melt(id_vars=id_col, var_name="d", value_name="volume")

    df.drop(["id", "cat_id", "state_id"], axis=1, inplace=True)



    # calendar data merge

    df = pd.merge(df, calendar, left_on="d", right_on="d", how="left")

    # price data merge

    df = pd.merge(df, price, left_on=["store_id", "item_id", "wm_yr_wk"], right_on=["store_id", "item_id", "wm_yr_wk"], how='left')

    

    df.drop("wm_yr_wk", axis=1, inplace=True)



    gc.collect()



    return df
gc.collect()
# Create merged dataframe

master = data_merge_3df(train, calendar, price)
del calendar

gc.collect()
# Distribution plot function

def distribution_plot(df, col_name="item_id", target_value="d_1913"):

    value = df.groupby(col_name)[target_value].mean().values



    # Visualization

    plt.figure(figsize=(10,6))

    sns.distplot(value)

    plt.xlabel("Volume")

    plt.ylabel("Frequency")

    plt.title("Volume distribution by each item_id at {}".format(target_value))

    

# Box plot function

def box_plot(x, y, df, size=(20,6), y_label="Volume", stripplot=True):

    fig, ax = plt.subplots(1, 2, figsize=size)



    # Including outliers

    if stripplot == True:

        sns.boxplot(x=x, y=y, data=df, showfliers=False, ax=ax[0])

        sns.stripplot(x=x, y=y, data=df, jitter=True, ax=ax[0])

    else:

        sns.boxplot(x=x, y=y, data=df, ax=ax[0])

    ax[0].set_ylabel(y_label)

    ax[0].set_title("box plot at {} with outliers".format(y))

    ax[0].tick_params(axis='x', labelrotation=45)

    

    # Not including outliers

    sns.boxplot(x=x, y=y, data=df, ax=ax[1], sym="")

    ax[1].set_ylabel(y_label)

    ax[1].set_title("box plot at {} without outliers".format(y))

    ax[1].tick_params(axis='x', labelrotation=45)



# 2 params bubble plot function

def bubble_plot(x="store_id", y="dept_id", s="d_1913", df=train, size=(20,10)):

    # mean value

    data_ave = df.groupby([x,y])[s].mean().reset_index()

    x_ave = data_ave[x]

    y_ave = data_ave[y]

    s_ave = data_ave[s]



    # max values

    data_max = df.groupby([x,y])[s].max().reset_index()

    x_max = data_max[x]

    y_max = data_max[y]

    s_max = data_max[s]



    # visualization

    fig, ax = plt.subplots(1, 2, figsize=size)



    ax[0].scatter(x_ave, y_ave, s=s_ave*100, alpha=0.5, color="blue")

    ax[0].set_xlabel(x)

    ax[0].set_ylabel(y)

    ax[0].set_title("Bubble chart of average volume on {}".format(s))



    ax[1].scatter(x_max, y_max, s=s_max*100, alpha=0.5, color="green")

    ax[1].set_xlabel(x)

    ax[1].set_ylabel(y)

    ax[1].set_title("Bubble chart of average volume on {}".format(s))



# correlation plot

def correlation_plot(x="sell_price", y="volume", df=master, sample_size=3000, size=(8,8)):

    samp = df.sample(sample_size)

    x_data = samp[x]

    y_data = samp[y]

    

    plt.figure(figsize=size)

    plt.scatter(x_data,y_data)

    plt.xlabel(x)

    plt.ylabel(y)

    plt.title(x+"vs"+y + ", Sampling {}".format(sample_size))
# Time series

def time_series_plot(data, freq=28, size=(20,12), title=""):

    index = data.index

    col = data.columns

    

    fig, ax = plt.subplots(2,1, figsize=size)

    

    # Raw data

    for i in col:

        ax[0].plot(index, data[i], label=i, linewidth=1)

        ax[0].legend()

        ax[0].set_title("{} : Time series plot of raw data".format(title))

    

    # Rolling mean data

    for i in col:

        ax[1].plot(index, data[i].rolling(freq).mean(), label=i, linewidth=1)

        ax[1].legend()

        ax[1].set_title("{} : Time series plot of rolling {} data".format(title, freq))



# R coefficient plot 

def r_coef_plot(df, max_lag=72, size=(20,6)):

    col = df.loc[:, "d_1":].columns

    

    r_corr = []

    lag = range(len(col))

    

    for i in range(len(col)):

        x = df.iloc[:,-1]

        y = df[col[-i-1]]

        r = np.corrcoef(x,y)[0,1]

        r_corr.append(r)

        

    fig, ax = plt.subplots(1,2, figsize=size)

    

    ax[0].plot(lag, r_corr)

    ax[0].set_xlabel("lag")

    ax[0].set_ylabel("R coefficient")

    ax[0].set_ylim([0,1])

    ax[0].set_title("Lag volume R coefficient")

    

    ax[1].plot(lag[:max_lag], r_corr[:max_lag])

    ax[1].set_xlabel("lag")

    ax[1].set_ylabel("R coefficient")

    ax[1].set_ylim([0,1])

    ax[1].set_title("Lag volume (Max lag range {}) R coefficient".format(max_lag))



# Auto correlation plot

def autocorrelation_plot(data, lags=28):

    col = data.columns

    fig, ax = plt.subplots(len(col), 2, figsize=(20, 6*len(col)))

    

    for c in range(len(col)):                 

        # autocorrelation

        plot_acf(data[col[c]], lags=lags, ax=ax[c,0])

        ax[c,0].set_title("Auto correlation of {}".format(col[c]))

        ax[c,0].set_xlabel("lag")

        ax[c,0].set_ylabel("auto correlation")

        # time series

        ax[c,1].plot(data[col[c]][-365:].index, data[col[c]][-365:], linewidth=1)

        ax[c,1].set_title("Time series data of {}".format(col[c]))

        ax[c,1].set_xlabel("day")

        ax[c,1].set_ylabel("volume")

        

    plt.show()



# Resid normality test

def normality_test(df, freq=7):

    col_name = df.columns

    resid_df = pd.DataFrame({})

    

    for c in col_name:

        res = sm.tsa.seasonal_decompose(df[c], period=freq)

        resid_df["Resid_{}".format(c)] = res.resid

        

    resid_df.dropna(inplace=True)

        

    fig, ax = plt.subplots(resid_df.shape[1], 2, figsize=(20, 6*resid_df.shape[1]))

    plt.subplots_adjust(hspace=0.4)

    col_name = resid_df.columns

    for i in range(len(resid_df.columns)):

        # Shapiro wilk test

        WS, p = stats.shapiro(resid_df[col_name[i]])

        # distribution plot

        sns.distplot(resid_df[col_name[i]], ax=ax[i, 0])

        ax[i, 0].set_xlabel("resid")

        ax[i, 0].set_title("Distribution of resid : {} \n p-value of Shapiro Wilk test : {:.3f}".format(col_name[i], p))

        # probability

        stats.probplot(resid_df[col_name[i]], plot=ax[i,1])

        ax[i, 1].set_title("Probability plot")
distribution_plot(train, col_name="item_id", target_value="d_1913")
# dept_id : N=7 separated from cat_id, volume distribution on latest day with boxplot

box_plot(x="dept_id", y="d_1913", df=train, size=(20,6), y_label="Volume", stripplot=True)
box_plot(x="store_id", y="d_1913", df=train, size=(20,6), y_label="Volume", stripplot=True)
bubble_plot(x="store_id", y="dept_id", s="d_1913", df=train, size=(20,6))
# for keeping memory

gc.collect()
# price distribution with boxplot on latest day

box_plot(x="store_id", y="sell_price", df=price, size=(20,6), y_label="Price", stripplot=False)
# create dept_id from item_id

price_copy = price.copy()

price_copy["dept_id"] = [s.rsplit("_",1)[0] for s in price_copy["item_id"]]



box_plot(x="dept_id", y="sell_price", df=price_copy, size=(20,6), y_label="Price", stripplot=False)



del price_copy

gc.collect()
del price

gc.collect()
# Correlation with volume

correlation_plot(x="sell_price", y="volume", df=master, sample_size=1500, size=(8,8))
# year

box_plot(x="year", y="volume", df=master, size=(20,6), y_label="Volume", stripplot=False)
# month

box_plot(x="month", y="volume", df=master[master["year"]==2015], size=(20,6), y_label="Volume", stripplot=False)
# weekday

box_plot(x="weekday", y="volume", df=master[master["year"]==2015], size=(20,6), y_label="Volume", stripplot=False)
master["day_of_month"] = pd.to_datetime(master["date"]).dt.day



# day of month

box_plot(x="day_of_month", y="volume", df=master[master["year"]==2015], size=(20,6), y_label="Volume", stripplot=False)
gc.collect()
# Omitted due to memory over on kaggle

# snap

# master["snap_flag"] = master["snap_CA"] + master["snap_TX"] + master["snap_WI"]



# box_plot(x="snap_flag", y="volume", df=master[master["year"]==2015], size=(20,6), y_label="Volume", stripplot=False)
# Omitted due to memory over on kaggle

# event_name

# master[["event_name_1", "event_name_2"]] = master[["event_name_1", "event_name_2"]].fillna("no")

# master["event_flag"] = master["event_name_1"] + str("+") + master["event_name_2"]
# Omitted due to memory over on kaggle

# box_plot(x="event_flag", y="volume", df=master[master["year"]==2015], size=(20,6), y_label="Volume", stripplot=False)
gc.collect()
r_coef_plot(train, max_lag=72, size=(20,6))
# Sampling items creation

sample_num = [1, 100, 1000, 10000]

sample_id = []

for i in sample_num:

    id_name = train["item_id"].values[i]

    sample_id.append(id_name)



sample_df = master[(master["item_id"] == sample_id[0]) | (master["item_id"] == sample_id[1]) | (master["item_id"] == sample_id[2])| (master["item_id"] == sample_id[3])]

sample_df = pd.pivot_table(sample_df, index="date", columns="item_id", values="volume", aggfunc="mean")
time_series_plot(sample_df, freq=7, size=(20,12), title="item_id")
# Sample id

autocorrelation_plot(data=sample_df, lags=56)
# Normality test of sample 

normality_test(sample_df, freq=7)
normality_test(sample_df, freq=28)
del sample_df

gc.collect()
# dept_id

dept_df = pd.pivot_table(master, index="date", columns="dept_id", values="volume", aggfunc="mean")
time_series_plot(dept_df, freq=7, size=(20,12), title="dept_id")
time_series_plot(dept_df, freq=28, size=(20,12), title="dept_id")
# dept_id

autocorrelation_plot(data=dept_df, lags=56)
# dept_df

normality_test(dept_df, freq=7);
# dept_df

normality_test(dept_df, freq=28);
del dept_df

gc.collect()
# store_id

store_df = pd.pivot_table(master, index="date", columns="store_id", values="volume", aggfunc="mean")
time_series_plot(store_df, freq=28, size=(20,12), title="store_id")
time_series_plot(store_df, freq=28, size=(20,12), title="store_id")
# store_id

autocorrelation_plot(data=store_df, lags=56)
# store_df

normality_test(store_df, freq=7)
normality_test(store_df, freq=28)
del store_df

gc.collect()