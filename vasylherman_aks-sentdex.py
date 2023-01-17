import datetime as dt

import matplotlib.pyplot as plt

from matplotlib import style

import pandas as pd

import pandas_datareader.data as web

# import quandl as qdl



import numpy as np

from scipy.stats import linregress

from matplotlib.pyplot import text

from scipy.signal import argrelextrema





style.use('ggplot')

df = pd.read_csv("../input/AAPL.csv", index_col=0)

df.index = pd.to_datetime(df.index)

df = df.copy().tail(365)
def find_support_trend(df, name='High', trend='trend', slope='slope'):

    # formatting the data

    orig_df = df.copy()

    df = df.asfreq('D')

    mask = np.isnan(df[name])

    df[name][mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), df[name][~mask])



    df['date_id'] = ((df.index.date - df.index.date.min())).astype('timedelta64[D]')

    df['date_id'] = df['date_id'].dt.days + 1



    reg_df = df.copy()

    while len(reg_df)>3:

        reg = linregress(x=reg_df['date_id'], y=reg_df[name],)

        reg_df = reg_df.loc[reg_df[name] < reg[0] * reg_df['date_id'] + reg[1]]

    #     print(reg[0])



    reg = linregress(x=reg_df['date_id'], y=reg_df[name],)



    df[trend] = reg[0] * df['date_id'] + reg[1]

    plt.plot(df[trend], 'r-')

    plt.plot(df[name], 'g-')

    plt.show()

    res_df = orig_df.copy()[[name]].join(df[trend])

    res_df[slope] = reg[0]

    return res_df
def scan_df_trend_4(df, name='High', trend='trend', slope='slope'):

    result_df = pd.DataFrame()

    for idx, splited in enumerate(np.array_split(df, 12)):

        find_support_trend(splited, name, trend, slope)

#     return result_df





scan_df_trend_4(df)