# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

from datetime import datetime, timedelta

# Any results you write to the current directory are saved as output.
tr=pd.read_csv('/kaggle/input/save-my-data/train-flattened.csv')

test=pd.read_csv('/kaggle/input/save-my-data/test-flattened.csv',nrows = 2000)

sub=pd.read_csv('/kaggle/input/save-my-data/submit-flattened.csv',nrows=2000)

tr["date"] = pd.to_datetime(tr["date"], infer_datetime_format=True, format="%Y%m%d")

tr['totals_pageviews'] = tr['totals_pageviews'].astype(float)

tr['totals_newVisits'] = tr['totals_newVisits'].astype(float)

tr['trafficSource_isTrueDirect'] = tr['trafficSource_isTrueDirect'].astype(bool)

tr['totals_hits'] = tr['totals_hits'].astype(float)

tr['device_isMobile'] = tr['device_isMobile'].astype(bool)

tr['totals_timeOnSite'] = tr['totals_timeOnSite'].astype(float)

tr['totals_transactions'] = tr['totals_transactions'].astype(float)
test["date"] = pd.to_datetime(test["date"], infer_datetime_format=True, format="%Y%m%d")

test['totals_pageviews'] = test['totals_pageviews'].astype(float)

test['totals_newVisits'] = test['totals_newVisits'].astype(float)

test['trafficSource_isTrueDirect'] = test['trafficSource_isTrueDirect'].astype(bool)

test['totals_hits'] = test['totals_hits'].astype(float)

test['device_isMobile'] = test['device_isMobile'].astype(bool)

test['totals_timeOnSite'] = test['totals_timeOnSite'].astype(float)

test['totals_transactions'] = test['totals_transactions'].astype(float)
#replace unknown value with 0

unknownvalues=['(not set)','not available in demo dataset','(not provided)','unknown.unknown','/','Not Socially Engaged']

for unknownvalue in unknownvalues:    

    tr.replace(unknownvalue, np.nan, inplace=True)

    test.replace(unknownvalue, np.nan, inplace=True)

    
tr['totals_transactionRevenue'].replace(np.nan, 0,inplace=True)

test['totals_transactionRevenue'].replace(np.nan, 0,inplace=True)
non_constant=[i for i in tr.columns if tr[i].nunique()>1]
tr=tr[non_constant]

test=test[tr.columns]
test.columns
timedelta(days=168).days
tr['date'].max()-tr['date'].min()
def get_groupby_target(df,k=0): 

    df['fullVisitorId'] = df['fullVisitorId'].astype(str)

    start_date=(df['date'].min()+ timedelta(days=k))

    end_date=(df['date'].min() + timedelta(days=168+k))

    mask =  (df['date'] > start_date) & (df['date'] <= end_date)

    tf=df.loc[mask]

    tfg = tf.groupby('fullVisitorId').agg({

            'geoNetwork_networkDomain': {'networkDomain': lambda x: -999 if pd.isnull(x.dropna().max()) else 1},

            'geoNetwork_city': {'city':lambda x: -999 if pd.isnull(x.dropna().max()) else 1},

            'device_operatingSystem': {'operatingSystem': lambda x: -999 if pd.isnull(x.dropna().max()) else 1},

            'geoNetwork_metro': {'metro': lambda x: -999 if pd.isnull(x.dropna().max()) else 1},

            'geoNetwork_region': {'region':lambda x: -999 if pd.isnull(x.dropna().max()) else 1},

            'channelGrouping': {'channelGrouping':lambda x: -999 if pd.isnull(x.dropna().max()) else 1},

            'trafficSource_referralPath': {'referralPath': lambda x: -999 if pd.isnull(x.dropna().max()) else 1},

            'geoNetwork_country': {'country': lambda x: -999 if pd.isnull(x.dropna().max()) else 1},

            'trafficSource_source': {'source': lambda x: -999 if pd.isnull(x.dropna().max()) else 1},

            'trafficSource_medium': {'medium': lambda x: -999 if pd.isnull(x.dropna().max()) else 1},

            'trafficSource_keyword': {'keyword': lambda x: -999 if pd.isnull(x.dropna().max()) else 1},

            'device_browser':  {'browser': lambda x: -999 if pd.isnull(x.dropna().max()) else 1},

            'trafficSource_adwordsClickInfo.gclId': {'gclId': lambda x: -999 if pd.isnull(x.dropna().max()) else 1},

            'device_deviceCategory': {'deviceCategory': lambda x: -999 if pd.isnull(x.dropna().max()) else 1},

            'geoNetwork_continent': {'continent': lambda x: -999 if pd.isnull(x.dropna().max()) else 1},

            'totals_timeOnSite': {'timeOnSite_max': lambda x: x.dropna().max(),

                                  'timeOnSite_min': lambda x: x.dropna().min(), 

                                  'timeOnSite_mean': lambda x: x.dropna().mean(),

                                  'timeOnSite_sum': lambda x: x.dropna().sum()},

            'totals_pageviews': {'pageviews_max': lambda x: x.dropna().max(),

                                 'pageviews_min': lambda x: x.dropna().min(),

                                 'pageviews_mean': lambda x: x.dropna().mean(),

                                 'pageviews_sum': lambda x: x.dropna().sum()},

            'totals_hits': {'hits_max': lambda x: x.dropna().max(), 

                            'hits_min': lambda x: x.dropna().min(),

                            'hits_mean': lambda x: x.dropna().mean(),

                            'hits_sum': lambda x: x.dropna().sum()},

            'visitStartTime': {'visitStartTime_counts': lambda x: x.dropna().count()},

            'totals_sessionQualityDim': {'sessionQualityDim': lambda x: x.dropna().max()},

            'device_isMobile': {'isMobile': lambda x: x.dropna().max()},

            'visitNumber': {'visitNumber_max' : lambda x: x.dropna().max()}, 

            'totals_totalTransactionRevenue':  {'totalTransactionRevenue_sum':  lambda x:x.dropna().sum()},

            'totals_transactions' : {'transactions' : lambda x:x.dropna().sum()},

            'date':{"session":lambda x :(x.dropna().max()- x.dropna().min()).days}})

    

    target_strat=(start_date+ timedelta(days=214))

    target_end=(target_strat + timedelta(days=62))

    mask1 =  (df['date'] >= target_strat) & (df['date'] <= target_end)

    tf2=df.loc[mask1]

    tf3=tf2.groupby('fullVisitorId').agg({'totals_totalTransactionRevenue':{'target': lambda x: x.dropna().sum()}})

    tfg.columns=tfg.columns.droplevel()

    tf3.columns=['target']

    tfg['fullVisitorId'] = tfg.index

    tf3['fullVisitorId'] = tf3.index

    tfg['target']=0

    tfg['return']=0

    for i in tfg['fullVisitorId']:

        if i in tf3['fullVisitorId']:

            tfg.loc[i,'target']=tf3.loc[i,'target']

            tfg.loc[i,'return']=1

    

    return tfg

    

    

    
train_0 = get_groupby_target(tr,248)

%time

train_1 = get_groupby_target(tr,310)

%time

train_2 = get_groupby_target(tr,372)

%time

use_train = pd.concat([train_0,train_1,train_2])
use_train.to_csv("my_train_second_part.csv", index=False)
use_train['return'].hist()
use_train[use_train['return']==1]