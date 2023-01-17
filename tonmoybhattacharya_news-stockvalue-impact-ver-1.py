# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
from kaggle.competitions import twosigmanews
env = twosigmanews.make_env()

(market_train_df, news_train_df) = env.get_training_data()

#beginning data exploration

#check columns
col_list_1 = market_train_df.columns
col_list_2 = news_train_df.columns

print("market_train has {} rows and {} columns. \nnews_train has {} rows and {} columns."\
      .format(market_train_df.shape[0], market_train_df.shape[1], news_train_df.shape[0], news_train_df.shape[1]))

col_list_1 = [i for i in market_train_df.columns.values]
col_list_2 = [i for i in news_train_df.columns.values]
print("Columns of market_train_df:\n", ",\n".join(a for a in col_list_1), \
      "\n\n\nColumns of news_train_df:\n",",\n".join(a for a in col_list_2) ,"\n\n")

#check if the particular column has NaN/empty data
market_cols_hasnan = {column:market_train_df[column].isnull().any() for column in col_list_1}
news_cols_hasnan = {column:news_train_df[column].isnull().any() for column in col_list_2}
cols_null_market = [i for i in market_cols_hasnan.keys() if market_cols_hasnan.get(i)]
cols_null_news = [i for i in news_cols_hasnan.keys() if news_cols_hasnan.get(i)]
print("Columns with NaN in market_train_df:\n",cols_null_market,"\n\t There are ",len(cols_null_market)," columns with null.")
print("Columns with NaN in news_train_df:\n",cols_null_news,"\n\t There are ",len(cols_null_news)," columns with null.")

#print a single row of each for visual familiarity of the data
print("First row of market_train_df\n",market_train_df.iloc[0])
print("First row of news_train_df\n",news_train_df.iloc[0])

#makes copies of the 2 dataframes 
a_market_train_df = market_train_df.copy()
a_news_train_df = news_train_df.copy()


def list_cols_type(df, verbose=True): #list all column names with their datatypes
    s = set(zip([e for e in df.columns.values],[df[i].dtype.name for i in df.columns.values ]))
    for a in s:
        print(a)
    return s 

print("market_dataframe_follows:")
list_cols_type(a_market_train_df)
print("\nnews_dataframe_follows:")
list_cols_type(a_news_train_df)

def list_categorical_columns(df, verbose=True):
    #names of different dtypes that may be "categorical" or "categoricalish"(sorry)
    list_categorical_dtype_names = ['datetime64[ns, UTC]', 'category', 'object']
    cat_col_names = [(i, df[i].dtype.name) for i in df.columns.values if df[i].dtype.name in list_categorical_dtype_names]
    if(verbose):
        print(cat_col_names)
    return cat_col_names

print("market df follows:")
list_categorical_columns(a_market_train_df)
print("\nnews df follows:")
list_categorical_columns(a_news_train_df)
#check the null values per column in the 2 dataframes
#this is to help decide if the column should be dropped or nulls autopopulated by the proxy value

#type(a_market_train_df.groupby('time')) #>>pandas.core.groupby.groupby.DataFrameGroupBy

def list_cols_with_null(df):
    A = [a for a in df.columns.values if df[a].isnull().sum()>0]
    #TODO: Try to do the following using dictionary comprehension
    B = {} #Dictionary holding (column_name:percentage of null values)
    for a in A:
        num_null = df[a].isnull().sum()
        total_count = len(df[a])
        b = float(num_null/total_count)
        B[a] = b
    return B

A = list_cols_with_null(a_market_train_df)
B = list_cols_with_null(a_news_train_df)

print("market_df column null percentages:")
print(A)
print("\nnews_df column null percentages:")
print(B)
#Join the dataframes into a combined dataframe - 0

#perform the necessary cleaning to the market & news dataframes
#merge them into a single dataframe thereafter
def clean_data_fields(market_df, news_df, verbose=False):
    #the timestamps are string, convert them to int
    market_df['time'] = market_df.time.dt.strftime("%Y%m%d").astype(int)#convert datetime to simple int
    news_df['time'] = news_df.time.dt.strftime("%Y%m%d").astype(int)
    #news df has a list of 'assetCodes', create a column 'assetCode' with the first assetCode in assetCodes
    #drop the column assetCodes thereafter
    news_df['assetCode'] = news_df['assetCodes'].map(lambda x: list(eval(x))[0])#from: https://www.kaggle.com/rabaman/0-64-in-100-lines
    news_df.drop(['assetCodes'], axis=1, inplace=True)
    #market df has opening and closing prices, create a column with the average prices
    market_df['average'] = (market_df['close'] + market_df['open'])/2
    #columns to be dropped from the market & news dataframes
    cols_dropped_market = ['assetName','universe']
                        #['returnsClosePrevMktres1','returnsOpenPrevMktres1','returnsClosePrevMktres10',\
                        #'returnsOpenPrevMktres10','universe'] #,'returnsOpenNextMktres10'
    cols_dropped_news   = ['noveltyCount12H','noveltyCount24H','noveltyCount3D','noveltyCount5D','noveltyCount7D',\
                       'volumeCounts12H','volumeCounts24H','volumeCounts3D','volumeCounts5D','volumeCounts7D', \
                       'sourceTimestamp', 'sourceId', 'takeSequence', 'headline', 'firstCreated', 'provider', \
                       'audiences', 'firstMentionSentence', 'wordCount', 'headlineTag', 'bodySize', 'companyCount',\
                       'marketCommentary','sentenceCount', 'subjects', 'assetName']#last 2 added
    #drop columns
    if(verbose):
        print("a_market_train_df has {} columns, pre-drop".format(market_df.shape[1]))
        print("a_news_train_df has {} columns, pre-drop".format(news_df.shape[1]))
    market_df = market_df.drop(cols_dropped_market, axis=1)
    news_df = news_df.drop(cols_dropped_news, axis=1)
    if(verbose):
        print("a_market_train_df has {} columns, post-drop".format(market_df.shape[1]))
        print("a_news_train_df has {} columns, post-drop".format(news_df.shape[1]))
        print("---")
        print("a_market_train_df has {:,} rows and {} columns.".format(market_df.shape[0],market_df.shape[1]))
        print("a_news_train_df has {:,} rows and {} columns.".format(news_df.shape[0],news_df.shape[1]))
    #aggregate news_df and 
    news_df = news_df.groupby(['time', 'assetCode'], sort=False).aggregate(np.mean).reset_index()
    #merge merket & news dfs into a simgle df
    #unified_df = pd.merge(market_df, news_df, how="inner", on=['time', 'assetCode'], copy=False)
    unified_df = pd.merge(market_df, news_df, how="left", on=['time', 'assetCode'], copy=False)
    return unified_df

unified_df = clean_data_fields(a_market_train_df, a_news_train_df)
print("done.")


    
import gc
#clear mem
del a_market_train_df, market_train_df, news_train_df, a_news_train_df

gc.collect()
#check if unified df has any null fields
print(list_cols_with_null(unified_df))
#get list of columns with null
#A = [a for a in unified_df.columns.values if unified_df[a].isnull().sum()>0]
print(unified_df.shape)
print(unified_df.iloc[0])
import gc
import matplotlib.pyplot as plt
import seaborn as sns

#explore time column a bit, of the unified/merged DataFrame
time_unique = unified_df['time'].nunique()
num_rows = unified_df['time'].shape[0]
print("The Time column in unified_df has a total of {} rows which contain {} unique values".format(num_rows, time_unique))


num_rows = unified_df.shape[0]
num_rows_train = int(0.7 * num_rows) #
num_rows_test = num_rows - num_rows_train
print("There should be {} training rows and {} val. rows ".format(num_rows_train, num_rows_test))

u_train_df = unified_df.iloc[:num_rows_train,:]
u_test_df = unified_df.iloc[num_rows_train:,:]
print("Training set has {} rows while val set has {} rows".format(u_train_df.shape[0], u_test_df.shape[0] ))
import gc
#check shapes of derived dataframes to ensure all seems well
print(u_train_df.shape)
print(u_test_df.shape)
print(u_train_df.shape[0]+u_test_df.shape[0])
print(unified_df.shape)

#delete unified_df to save memory
del unified_df
gc.collect()
print("done.")
#Split the derived datasets into training and target col.sets.
#strip off y which is 'returnsOpenNextMktres10'
y_label = 'returnsOpenNextMktres10' #training target
y_train_df = u_train_df[y_label]
y_test_df = u_test_df[y_label]
u_train_df.drop(columns=y_label, inplace=True)
u_test_df.drop(columns=y_label, inplace=True)
print(" u_train_df.shape == {}, y_train_df.shape == {}".format(u_train_df.shape, y_train_df.shape))
print(" u_test_df.shape == {}, y_test_df.shape == {}".format(u_test_df.shape, y_test_df.shape))

#train a model
#import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

import matplotlib.pyplot as plt
from itertools import chain

%matplotlib inline

categorical_cols = ['assetCode']
categorical_cols_index = [1]
categorical_cols_index_str = '1'

u_train_df_ = u_train_df.copy()

#x_train_1_ = x_train_1.copy()
#x_train_2_ = x_train_2.copy()
#
#x_train_1_.drop(categorical_cols,axis=1,inplace=True)
#x_train_2_.drop(categorical_cols,axis=1,inplace=True)
#get training columns list
u_train_df_.columns.tolist()
#x_train_1_.columns.tolist()
training_columns = [
 'close',
 'open',
 'returnsClosePrevRaw1',
 'returnsOpenPrevRaw1',
 'returnsClosePrevMktres1',
 'returnsOpenPrevMktres1',
 'returnsClosePrevRaw10',
 'returnsOpenPrevRaw10',
 'returnsClosePrevMktres10',
 'returnsOpenPrevMktres10',
 'average',
 'urgency',
 'relevance',
 'sentimentClass',
 'sentimentNegative',
 'sentimentNeutral',
 'sentimentPositive',
 'sentimentWordCount']
import time
import lightgbm as lgb
#%%time
lgb_params = {
    'boosting_type': 'gbdt',
    'objective': 'binary', #'binary' 'regression'
    'metric': {'binary_logloss'}, # 'l2', 'l1', 'logloss'
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'seed': 42,
    'verbose': 50
}
time_start = time.time()

#below is without categorical data
#check if any categorical data except 'assetCode'
#print("1---")
try:
    #rerunning this cell becomes a problem as attempt is made to drop already dropped columns...
    #hence this try catch block
    u_train_df.drop(columns=['assetCode'], inplace=True)
except KeyError:
    print("There was a KeyError at u_train_df")
try:
    #rerunning this cell becomes a problem as attempt is made to drop already dropped columns...
    #hence this try catch block
    u_test_df.drop(columns=['assetCode'], inplace=True)
except KeyError:
    print("There was a KeyError at u_test_df")
list_categorical_columns(u_train_df)
#print("2---")
x_train = lgb.Dataset(u_train_df.values, y_train_df,feature_name=u_train_df.columns.tolist())
#print("3---")
x_valid = lgb.Dataset(u_test_df.values, y_test_df,feature_name=u_test_df.columns.tolist(), reference=x_train)
evals_result = {}#store evaluation results
#print("4---")
model01 = lgb.train(lgb_params, x_train, num_boost_round=1000, \
                    valid_sets=[x_train, x_valid], valid_names=['eval1','eval2'], \
                    evals_result=evals_result, verbose_eval=50 )
time_end   = time.time()
print("The work took {:2.4f} minutes".format((time_end-time_start)/60))

_ = lgb.plot_metric(evals_result)
_ = lgb.plot_importance(model01)

#Submission Process
# You can only iterate through a result from `get_prediction_days()` once
# so be careful not to lose it once you start iterating.
days = env.get_prediction_days()
print("done.")
def clean_data_fields_forPreds(market_df, news_df, verbose=False):
    #the timestamps are string, convert them to int
    #already int date only#market_df['time'] = market_df.time.dt.strftime("%Y%m%d").astype(int)#convert datetime to simple int
    #already int date only#news_df['time'] = news_df.time.dt.strftime("%Y%m%d").astype(int)
    market_df['time'] = market_df.time.astype(int)#convert datetime to simple int
    news_df['time'] = news_df.time.astype(int)#convert datetime to simple int
    
    if 'assetCodes' in news_df.columns.values:
        #news df has a list of 'assetCodes', create a column 'assetCode' with the first assetCode in assetCodes
        #drop the column assetCodes thereafter
        news_df['assetCode'] = news_df['assetCodes'].map(lambda x: list(eval(x))[0])#from: https://www.kaggle.com/rabaman/0-64-in-100-lines
        news_df.drop(['assetCodes'], axis=1, inplace=True)
    
    
    #market df has opening and closing prices, create a column with the average prices
    market_df['average'] = (market_df['close'] + market_df['open'])/2
    #columns to be dropped from the market & news dataframes
    cols_dropped_market = ['assetName'] #,'universe'
                        #['returnsClosePrevMktres1','returnsOpenPrevMktres1','returnsClosePrevMktres10',\
                        #'returnsOpenPrevMktres10','universe'] #,'returnsOpenNextMktres10'
    cols_dropped_news   = ['noveltyCount12H','noveltyCount24H','noveltyCount3D','noveltyCount5D','noveltyCount7D',\
                       'volumeCounts12H','volumeCounts24H','volumeCounts3D','volumeCounts5D','volumeCounts7D', \
                       'sourceTimestamp', 'sourceId', 'takeSequence', 'headline', 'firstCreated', 'provider', \
                       'audiences', 'firstMentionSentence', 'wordCount', 'headlineTag', 'bodySize', 'companyCount',\
                       'marketCommentary','sentenceCount', 'subjects', 'assetName']#last 2 added
    #drop columns
    if(verbose):
        print("a_market_train_df has {} columns, pre-drop".format(market_df.shape[1]))
        print("a_news_train_df has {} columns, pre-drop".format(news_df.shape[1]))
    market_df = market_df.drop(cols_dropped_market, axis=1)
    news_df = news_df.drop(cols_dropped_news, axis=1)
    if(verbose):
        print("a_market_train_df has {} columns, post-drop".format(market_df.shape[1]))
        print("a_news_train_df has {} columns, post-drop".format(news_df.shape[1]))
        print("---")
        print("a_market_train_df has {:,} rows and {} columns.".format(market_df.shape[0],market_df.shape[1]))
        print("a_news_train_df has {:,} rows and {} columns.".format(news_df.shape[0],news_df.shape[1]))
    #aggregate news_df and 
    news_df = news_df.groupby(['time', 'assetCode'], sort=False)\
                     .aggregate(np.mean)\
                     .reset_index()
    #merge merket & news dfs into a simgle df
    #unified_df = pd.merge(market_df, news_df, how="inner", on=['time', 'assetCode'], copy=False)
    unified_df = pd.merge(market_df, news_df, how="left", on=['time', 'assetCode'], copy=False)
    return unified_df
print("done.")
#print(x_train_1_.columns.values,"\n",market_obs_df.columns.values,"\n",news_obs_df.columns.values,"\n")
#list_col_allowed = x_train_1_.columns.values
list_col_allowed = u_train_df.columns.values
def cols_needing_dropping(list_cols):
    return [a for a in list_cols if a not in list_col_allowed]
target_cols = ['returnsOpenNextMktres10']
#current_df = clean_data_fields_forPreds(market_obs_df, news_obs_df, verbose=False)
print("done")
for market_obs_df, news_obs_df, predictions_template_df in days:
    current_df = clean_data_fields_forPreds(market_obs_df, news_obs_df, verbose=False)
    current_df_ = current_df[list_col_allowed]
    print(current_df.shape,"-----",current_df_.shape)
    predictions = ((model01.predict(current_df_) * 2) - 1)
    predictions_df = pd.DataFrame({'assetCode':current_df['assetCode'], '_confidence':predictions})
    predictions_template_df = predictions_template_df.merge(predictions_df, how='left')\
                                                     .drop('confidenceValue', axis=1)\
                                                     .fillna(0)\
                                                     .rename(columns={'_confidence':'confidenceValue'})
    print(predictions_df.head())
    print(predictions_template_df.head())
    env.predict(predictions_template_df)    
env.write_submission_file()


