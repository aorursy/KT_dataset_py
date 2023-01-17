
import os
import pandas as pd
import numpy as np
from pandas.io.json import json_normalize
import json
import time
import warnings

#from pycountry_convert import ( map_countries, country_name_to_country_alpha3,)
import pytz as pytz
import datetime

#Plotting
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')

#Sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

#lgm and graph viz
import graphviz 
import lightgbm as lgb

warnings.filterwarnings('ignore')

def load_df(csv_path='../input/train.csv', nrows=None):
    JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource']
      
    df = pd.read_csv(csv_path, 
                     converters={column: json.loads for column in JSON_COLUMNS}, 
                     dtype={'fullVisitorId': 'str', 'visitId':'str', 'visitStartTime':'str', 'date':'str'}, 
                     nrows=nrows)

    #Normalize JSON colunmns and drop
    for column in JSON_COLUMNS:
        column_as_df = json_normalize(df[column])
        column_as_df.columns = [f"{column}.{subcolumn}" for subcolumn in column_as_df.columns]
        df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)
    return df


def drop_constant_cols(df):
    ## Drop constant columns
    const_cols = [c for c in df.columns if df[c].nunique(dropna=False) == 1]
    df.drop(const_cols, axis=1, inplace=True)
    
    #this columnm is only in train data
    try:
        df.drop('trafficSource.campaignCode', axis=1, inplace=True)   
    except:
        None   
    

os.listdir('../input')
%%time
#Load
train_df = load_df(csv_path='../input/ga-customer-revenue-prediction/train.csv', nrows = None)
#train_df.to_pickle('train_flat_no_drop.pkl')
drop_constant_cols(train_df)

test_df = load_df(csv_path='../input/ga-customer-revenue-prediction/test.csv', nrows = None)
#train_df.to_pickle('test_flat_no_drop.pkl')
drop_constant_cols(test_df)

# Extract target values and Ids
cat_cols = ['channelGrouping','device.browser',
       'device.deviceCategory', 'device.isMobile', 'device.operatingSystem',
       'geoNetwork.city', 'geoNetwork.continent', 'geoNetwork.country',
       'geoNetwork.metro', 'geoNetwork.networkDomain', 'geoNetwork.region',
       'geoNetwork.subContinent','trafficSource.adContent',
       'trafficSource.adwordsClickInfo.adNetworkType',
       'trafficSource.adwordsClickInfo.gclId',
       'trafficSource.adwordsClickInfo.isVideoAd',
       'trafficSource.adwordsClickInfo.page',
       'trafficSource.adwordsClickInfo.slot', 'trafficSource.campaign',
       'trafficSource.isTrueDirect', 'trafficSource.keyword',
       'trafficSource.medium', 'trafficSource.referralPath',
       'trafficSource.source'  ]


num_cols = ['visitNumber', 'totals.bounces', 'totals.hits',
            'totals.newVisits', 'totals.pageviews', 
            '_local_hourofday'  ]

interaction_cols = ['totals.hits / totals.pageviews', 'totals.hits * totals.pageviews',
       'totals.hits - totals.pageviews']

visitStartTime = ['visitStartTime']

ID_cols = ['date', 'fullVisitorId', 'sessionId', 'visitId']

target_col = ['totals.transactionRevenue']


os.listdir('../input/geocodes-timezones')
#Load
geocode_df= pd.read_pickle('../input/geocodes-timezones/geocodes_timezones.pkl')

def time_zone_converter(x):
    
    try:
        return pytz.country_timezones(x)[0]
    except AttributeError:
        return np.nan
   

def time_localizer(s):
    #format of series [time,zone]
    try:
        tz =pytz.timezone(s[1])
        return pytz.utc.localize(s[0], is_dst=None).astimezone(tz)
    except:
        return np.nan
    
def remove_missing_vals(x):
    remove_list = ['(not set)', 'not available in demo dataset','unknown.unknown']
    if x in remove_list:
        return ''
    else:
        return x 
    
def map_timezone(x):   
    try:
        return timezone_dict[x]
    except KeyError:
        return 'UTC'


%%time
train_df['visitStartTime'] = pd.to_datetime(train_df['visitStartTime'], unit = 's')
test_df['visitStartTime'] = pd.to_datetime(test_df['visitStartTime'], unit = 's')

#Generate foreign key '_search_term' by concatenating city, region, country
train_df['_search_term'] = train_df['geoNetwork.city'].map(remove_missing_vals) + ' ' + train_df['geoNetwork.region'].map(remove_missing_vals) + ' ' + train_df['geoNetwork.country'].map(remove_missing_vals)
test_df['_search_term'] = test_df['geoNetwork.city'].map(remove_missing_vals) + ' ' + test_df['geoNetwork.region'].map(remove_missing_vals) + ' ' + test_df['geoNetwork.country'].map(remove_missing_vals)

#Set global variable, needed for map_timezone function
global timezone_dict
timezone_dict = dict(zip(geocode_df['search_term'], geocode_df['timeZoneId']))

#Map timezones
train_df['_timeZoneId'] = train_df['_search_term'].map(map_timezone)
test_df['_timeZoneId'] = test_df['_search_term'].map(map_timezone)
  
#Create time zone aware column
train_df['_local_time'] = train_df[['visitStartTime', '_timeZoneId']].apply(time_localizer, axis = 1).astype(str)
test_df['_local_time'] = test_df[['visitStartTime', '_timeZoneId']].apply(time_localizer, axis = 1).astype(str)  

#Localize hour time
train_df['_local_hourofday'] = train_df['_local_time'].str[11:13]
test_df['_local_hourofday'] = test_df['_local_time'].str[11:13]


%%time
def map_longitude(x):   
    try:
        return longitude_dict[x]
    except KeyError:
        return np.nan
    
def map_latitude(x):   
    try:
        return latitude_dict[x]
    except KeyError:
        return np.nan
    
global longitude_dict
longitude_dict = dict(zip(geocode_df['search_term'], geocode_df['geometry.location.lng']))

global latitude_dict
latitude_dict = dict(zip(geocode_df['search_term'], geocode_df['geometry.location.lat']))


#Map latitude
train_df['_latitude'] = train_df['_search_term'].map(map_latitude)
test_df['_latitude'] = test_df['_search_term'].map(map_latitude)

#Map longitude
train_df['_longitude'] = train_df['_search_term'].map(map_longitude)
test_df['_longitude'] = test_df['_search_term'].map(map_longitude)
%%time
train_ts = train_df[['fullVisitorId', 'sessionId', 'visitId', 'visitNumber', 'visitStartTime']].copy()
test_ts = test_df[['fullVisitorId', 'sessionId', 'visitId', 'visitNumber', 'visitStartTime']].copy()


train_df['_time_since_last_visit'] = train_ts.sort_values(['fullVisitorId', 'visitStartTime']).groupby('fullVisitorId')['visitStartTime'].diff()
train_df['_time_since_last_visit_2'] = train_ts.sort_values(['fullVisitorId', 'visitStartTime']).groupby('fullVisitorId')['visitStartTime'].diff(2)
test_df['_time_since_last_visit'] = test_ts.sort_values(['fullVisitorId', 'visitStartTime']).groupby('fullVisitorId')['visitStartTime'].diff()
test_df['_time_since_last_visit_2'] = test_ts.sort_values(['fullVisitorId', 'visitStartTime']).groupby('fullVisitorId')['visitStartTime'].diff(2)

train_df['_time_to_next_visit'] = train_ts.sort_values(['fullVisitorId', 'visitStartTime']).groupby('fullVisitorId')['visitStartTime'].diff(-1)
train_df['_time_to_next_visit_2'] = train_ts.sort_values(['fullVisitorId', 'visitStartTime']).groupby('fullVisitorId')['visitStartTime'].diff(-2)
test_df['_time_to_next_visit'] = test_ts.sort_values(['fullVisitorId', 'visitStartTime']).groupby('fullVisitorId')['visitStartTime'].diff(-1)
test_df['_time_to_next_visit_2'] = test_ts.sort_values(['fullVisitorId', 'visitStartTime']).groupby('fullVisitorId')['visitStartTime'].diff(-2)

#del train_ts
#del test_ts
%%time
for col in ['totals.bounces', 'totals.hits','totals.pageviews',  '_local_hourofday']:
    train_df['_prev_{}_1'.format(col)] = train_df.sort_values(['fullVisitorId', 'visitStartTime']).groupby('fullVisitorId')[col].shift(1)
    test_df['_prev_{}_1'.format(col)] = test_df.sort_values(['fullVisitorId', 'visitStartTime']).groupby('fullVisitorId')[col].shift(1)
    train_df['_prev_{}_2'.format(col)] = train_df.sort_values(['fullVisitorId', 'visitStartTime']).groupby('fullVisitorId')[col].shift(2)
    test_df['_prev_{}_2'.format(col)] = test_df.sort_values(['fullVisitorId', 'visitStartTime']).groupby('fullVisitorId')[col].shift(2)
    
    train_df['_next_{}_1'.format(col)] = train_df.sort_values(['fullVisitorId', 'visitStartTime']).groupby('fullVisitorId')[col].shift(-1)
    test_df['_next_{}_1'.format(col)] = test_df.sort_values(['fullVisitorId', 'visitStartTime']).groupby('fullVisitorId')[col].shift(-1)
    train_df['_next_{}_2'.format(col)] = train_df.sort_values(['fullVisitorId', 'visitStartTime']).groupby('fullVisitorId')[col].shift(-2)
    test_df['_next_{}_2'.format(col)] = test_df.sort_values(['fullVisitorId', 'visitStartTime']).groupby('fullVisitorId')[col].shift(-2)
    

%%time
train_df['_time_first_visit'] = train_df.sort_values(['fullVisitorId', 'visitStartTime']).groupby('fullVisitorId')['visitStartTime']\
.transform('first')
train_df['_time_last_visit'] = train_df.sort_values(['fullVisitorId', 'visitStartTime']).groupby('fullVisitorId')['visitStartTime']\
.transform('last')
train_df['_difference_first_last'] = train_df['_time_last_visit'] - train_df['_time_first_visit']
train_df['_time_since_first_visit'] = train_df['visitStartTime'] - train_df['_time_first_visit']
train_df.drop(['_time_first_visit', '_time_last_visit'], axis = 1,inplace = True)


test_df['_time_first_visit'] = test_df.sort_values(['fullVisitorId', 'visitStartTime']).groupby('fullVisitorId')['visitStartTime']\
.transform('first')
test_df['_time_last_visit'] = test_df.sort_values(['fullVisitorId', 'visitStartTime']).groupby('fullVisitorId')['visitStartTime']\
.transform('last')
test_df['_difference_first_last'] = test_df['_time_last_visit'] - test_df['_time_first_visit']
test_df['_difference_first_last'] = test_df['_time_last_visit'] - test_df['_time_first_visit']
test_df['_time_since_first_visit'] = test_df['visitStartTime'] - test_df['_time_first_visit']
test_df.drop(['_time_first_visit', '_time_last_visit'], axis = 1,inplace = True)


train_df.info()
%%time
#train_df['_time_since_last_visit'] = pd.to_numeric(train_df['_time_since_last_visit'])
#test_df['_time_since_last_visit'] = pd.to_numeric(test_df['_time_since_last_visit'])

#Preparation
wip_cols = ['fullVisitorId', 'sessionId', 'visitId',
       'visitNumber', 'visitStartTime','totals.bounces', 'totals.hits',
       'totals.newVisits', 'totals.pageviews', '_time_since_last_visit']

train_ts = train_df.sort_values(['fullVisitorId', 'visitStartTime']).reset_index()
train_ts['index'] = train_ts['index'].astype('str')
train_ts_grouped = train_ts.groupby('fullVisitorId')

#Calculating rolling frequency
temp_roll = train_ts_grouped.rolling('12H', on ='visitStartTime')['visitNumber'].count().reset_index().add_suffix('_12H') 
train_ts = pd.concat([train_ts, temp_roll['visitNumber_12H']], axis = 1)

temp_roll = train_ts_grouped.rolling('7D', on ='visitStartTime')['visitNumber'].count().reset_index().add_suffix('_7D') 
train_ts = pd.concat([train_ts, temp_roll['visitNumber_7D']], axis = 1)

temp_roll = train_ts_grouped.rolling('30D', on ='visitStartTime')['visitNumber'].count().reset_index().add_suffix('_30D') 
train_ts = pd.concat([train_ts, temp_roll['visitNumber_30D']], axis = 1)

train_ts['index'] = train_ts['index'].astype('int')
train_ts.set_index('index', inplace = True)
train_ts.sort_index(inplace = True)
train_df = train_ts.copy()
del train_ts

train_df.info()
%%time

test_ts = test_df.sort_values(['fullVisitorId', 'visitStartTime']).reset_index()
test_ts['index'] = test_ts['index'].astype('str')
test_ts_grouped = test_ts.groupby('fullVisitorId')

#Calculating rolling frequency
temp_roll = test_ts_grouped.rolling('12H', on ='visitStartTime')['visitNumber'].count().reset_index().add_suffix('_12H') 
test_ts = pd.concat([test_ts, temp_roll['visitNumber_12H']], axis = 1)

temp_roll = test_ts_grouped.rolling('7D', on ='visitStartTime')['visitNumber'].count().reset_index().add_suffix('_7D') 
test_ts = pd.concat([test_ts, temp_roll['visitNumber_7D']], axis = 1)

temp_roll = test_ts_grouped.rolling('30D', on ='visitStartTime')['visitNumber'].count().reset_index().add_suffix('_30D')
test_ts = pd.concat([test_ts, temp_roll['visitNumber_30D']], axis = 1)

test_ts['index'] = test_ts['index'].astype('int')
test_ts.set_index('index', inplace = True)
test_ts.sort_index(inplace = True)
test_df = test_ts.copy()
del test_ts


test_df.info()


train_df.to_pickle('train_flat_FE.pkl')
test_df.to_pickle('test_flat_FE.pkl')
%%time
#Categorical encoding
for c in cat_cols:
    #Convert NAs to unknown
    train_df[c] = train_df[c].fillna('unknown')
    test_df[c] = test_df[c].fillna('unknown')


#Rename "Other" those with less than 10
for col in cat_cols:
    #For train data
    series1 = pd.value_counts(train_df[col])
    mask1 = series1 < 10
    train_df[col] = np.where(train_df[col].isin(series1[mask1].index),'Other_{}'.format(col), train_df[col])
    
    #For test data
    series2 = pd.value_counts(test_df[col])
    mask2 = series2 < 10
    test_df[col] = np.where(test_df[col].isin(series2[mask2].index),'Other_{}'.format(col), test_df[col])
    
%%time
interact_cats = ['channelGrouping', 'device.operatingSystem',
                'geoNetwork.city', 'geoNetwork.country', 'geoNetwork.networkDomain',
                 'trafficSource.medium', 
                'trafficSource.referralPath', 'trafficSource.source']

#2-way interactions
from itertools import combinations

def categorical_interaction_terms_2(df, columns):
    for c in combinations(columns,2):
        df['{}+{}'.format(c[0], c[1]) ] = df[c[0]] + '_' + df[c[1]]
    return df

def categorical_interaction_terms_3(df, columns):
    for c in combinations(columns,3):
        df['{}+{}+{}'.format(c[0], c[1], c[2]) ] = df[c[0]] + '_' + df[c[1]] + '_' + df[c[2]]
    return df

train_df = categorical_interaction_terms_2(train_df,interact_cats )
#train_df = categorical_interaction_terms_3(train_df,interact_cats )

test_df = categorical_interaction_terms_2(test_df,interact_cats )
#test_df = categorical_interaction_terms_3(test_df,interact_cats )

interact_cats_to_keep = [ 'geoNetwork.city+geoNetwork.networkDomain',
  'device.operatingSystem+geoNetwork.networkDomain',
  'device.operatingSystem+geoNetwork.city', 
  'channelGrouping+geoNetwork.networkDomain',
  'geoNetwork.city+trafficSource.source',
 'geoNetwork.networkDomain+trafficSource.source',
 'geoNetwork.networkDomain+trafficSource.referralPath',
 'geoNetwork.networkDomain+trafficSource.medium',
 'geoNetwork.city+trafficSource.medium',
 'geoNetwork.city+geoNetwork.country']


%%time

#Factorize cats
for f in (cat_cols + interact_cats_to_keep ):
    train_df[f], indexer = pd.factorize(train_df[f])
    test_df[f] = indexer.get_indexer(test_df[f])

del indexer
train_df.to_pickle('train_flat_FE_CAT_LE.pkl')
test_df.to_pickle('test_flat_FE_CAT_LE.pkl')
train_df.info()
test_df.info()