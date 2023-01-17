import numpy as np 
import pandas as pd 
import json
import bq_helper
from pandas.io.json import json_normalize
import seaborn as sns 
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import TimeSeriesSplit, KFold,GroupKFold
from sklearn.metrics import mean_squared_error
import functools
from multiprocessing import Pool
import logging
import gc
import logging
import matplotlib.pyplot as plt
import time
from line_profiler import LineProfiler
from sklearn.metrics import mean_squared_error
from scipy.stats import stats
import os
import re
print(os.listdir("../input"))

def line_profiling(func,arg):
    lp=LineProfiler()
    lp_wrapper=lp(func)
    lp_wrapper(arg)
    lp.print_stats()
def process_date_time(data_df):
    
    data_df['date'] = data_df['date'].astype(str)
    data_df["date"] = data_df["date"].apply(lambda x : x[:4] + "-" + x[4:6] + "-" + x[6:])
    data_df["date"] = pd.to_datetime(data_df["date"])   
    data_df["year"] = data_df['date'].dt.year
    data_df["month"] = data_df['date'].dt.month
    data_df["day"] = data_df['date'].dt.day
    data_df["weekday"] = data_df['date'].dt.weekday
    data_df['weekofyear'] = data_df['date'].dt.weekofyear
    data_df['month_unique_user_count'] = data_df.groupby('month')['fullVisitorId'].transform('nunique')
    data_df['day_unique_user_count'] = data_df.groupby('day')['fullVisitorId'].transform('nunique')
    data_df['weekday_unique_user_count'] = data_df.groupby('weekday')['fullVisitorId'].transform('nunique')
    data_df['weekofyear_unique_user_count'] = data_df.groupby('weekofyear')['fullVisitorId'].transform('nunique')
    return data_df

def process_format(data_df):

    for col in ['visitNumber', 'totals.hits', 'totals.pageviews']:
        data_df[col] = data_df[col].astype(float)
    data_df['trafficSource.adwordsClickInfo.isVideoAd'].fillna(True, inplace=True)
    data_df['trafficSource.isTrueDirect'].fillna(False, inplace=True)
    
    return data_df
def process_totals(data_df):
    #data_df['visitNumber'] = data_df['visitNumber']
    data_df['visits_id_sum'] = data_df.groupby(['fullVisitorId'])['visitNumber'].transform('sum')
    data_df['visits_id_min'] =  data_df.groupby(['fullVisitorId'])['visitNumber'].transform('min')
    data_df['visits_id_max'] = data_df.groupby(['fullVisitorId'])['visitNumber'].transform('max')
    data_df['visits_id_mean'] = data_df.groupby(['fullVisitorId'])['visitNumber'].transform('mean')
    data_df['visits_id_nunique'] = data_df.groupby('visitNumber')['fullVisitorId'].transform('nunique')
    #data_df['totals_hits'] = data_df['totals_hits']
    data_df['hits_id_sum'] = data_df.groupby(['fullVisitorId'])['totals.hits'].transform('sum')
    data_df['hits_id_cnt'] = data_df.groupby(['fullVisitorId'])['totals.hits'].transform('count')
    data_df['totals.pageviews'] = data_df['totals.pageviews'].fillna(0)
    data_df['pageviews_id_sum'] = data_df.groupby(['fullVisitorId'])['totals.pageviews'].transform('sum')
    data_df['pageviews_id_cnt'] = data_df.groupby(['fullVisitorId'])['totals.pageviews'].transform('count')
    data_df['mean_hits_per_day'] = data_df.groupby(['day'])['totals.hits'].transform('mean')
    data_df['sum_hits_per_day'] = data_df.groupby(['day'])['totals.hits'].transform('sum')
    data_df['max_hits_per_day'] = data_df.groupby(['day'])['totals.hits'].transform('max')
    data_df['min_hits_per_day'] = data_df.groupby(['day'])['totals.hits'].transform('min')
    data_df['var_hits_per_day'] = data_df.groupby(['day'])['totals.hits'].transform('var')
    data_df['mean_pageviews_per_day'] = data_df.groupby(['day'])['totals.pageviews'].transform('mean')
    data_df['sum_pageviews_per_day'] = data_df.groupby(['day'])['totals.pageviews'].transform('sum')
    data_df['max_pageviews_per_day'] = data_df.groupby(['day'])['totals.pageviews'].transform('max')
    data_df['min_pageviews_per_day'] = data_df.groupby(['day'])['totals.pageviews'].transform('min')   
    
    return data_df
    

train_df=pd.read_hdf('../input/load-data-faster-with-hdf/train_df.h5')
test_df=pd.read_hdf('../input/load-data-faster-with-hdf/test_df.h5')
print('train loaded')
print('test_loaded')
line_profiling(process_date_time,train_df)
line_profiling(process_format,train_df)
train_df=process_format(train_df)
train_df=process_totals(train_df)
line_profiling(process_totals,train_df)
