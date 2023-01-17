import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd
from pandas.io.json import json_normalize
from datetime import datetime

import json
import os

import matplotlib.pyplot as plt
import seaborn as sns

from plotly import tools
from plotly.offline import init_notebook_mode, iplot, plot
import plotly.offline as py
import plotly.graph_objs as go
%%time

def load_df(csv_path='../input/train_v2.csv', nrows=None):
    JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource']
    df = pd.DataFrame([])
    for chunk in pd.read_csv(csv_path, 
                     converters={column: json.loads for column in JSON_COLUMNS}, 
                     dtype={'fullVisitorId': 'str'},
                     chunksize=nrows):

        chunk = chunk.drop(['customDimensions','hits'], axis=1)
        df = df.append(chunk, ignore_index=True)
        print(df.shape[0])
    print("Dataframe assembled; Working on JSON columns...")
    for column in JSON_COLUMNS:
        column_as_df = json_normalize(df[column])
        column_as_df.columns = [f"{column}.{subcolumn}" for subcolumn in column_as_df.columns]
        df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)
    print("Done!")
    return df

train = load_df('../input/train_v2.csv', 100000)
train["totals.transactionRevenue"] = train["totals.transactionRevenue"].astype('float')
print("Colunms in dataset: " + str(train.shape[1]))
print("Rows in dataset: " + str(train.shape[0]))
train[:10]
na_counts = pd.DataFrame(train.isnull().sum())
na_counts.columns = ['na_counts']
na_counts['%na'] = round(na_counts['na_counts'] / train.shape[0], 4) * 100
uniq_counts = pd.DataFrame(train.nunique(dropna=False))
uniq_counts.columns = ['uniq_counts']
na_uniq_counts = na_counts.join(uniq_counts, how='outer').sort_values(by='na_counts', ascending=False)

print("There are "+str(len(na_uniq_counts.loc[na_uniq_counts["na_counts"] > 0]))+" columns with na values")
const_cols = [c for c in train.columns if train[c].nunique(dropna=False)==1 ]
print("There are "+str(len(const_cols))+" colunms with only one unique value")

na_uniq_counts
# train = train.drop(const_cols, axis=1)
plt.figure(figsize=[40,20])

fontdict = {'fontsize': 30, 'fontweight' : 'bold'}

plt.subplot(231)
plt.scatter(range(train.shape[0]), np.sort(train['totals.transactionRevenue'].values))
plt.title("Revenue Range", fontdict=fontdict)
plt.tick_params(axis='both', which='major', labelsize=10)
plt.xlabel('')
plt.ylabel('')

plt.subplot(232)
sns.distplot( np.log(train[train['totals.transactionRevenue'] > 0]["totals.transactionRevenue"]), bins=60, kde=True)
plt.title("Distribution of Revenue", fontdict=fontdict)
plt.tick_params(axis='both', which='major', labelsize=10)
plt.xlabel('')
plt.ylabel('')
plt.show()
print("In the dataset of "+str(train.shape[0])+" visits there are...")
print("Number of unique visitors: "+str(train["fullVisitorId"].nunique()))
print("Presentage of visits that bought something: %"+str(train.loc[train["totals.transactionRevenue"] > 0].shape[0]/train.shape[0]*100))

train_first_visit = train.loc[train["visitNumber"] == 1]
print("Percentage of visits that were for the first time: %"+str(train_first_visit.shape[0]/train.shape[0]*100))
print("Presentage of first time visitors that bought something: %"+str(train_first_visit.loc[train_first_visit["totals.transactionRevenue"] > 0].shape[0]/train.shape[0]*100))
plt.figure(figsize=[20,16])

def create_countplot(column, title, plt_n, df=train, mean_ch = False):
    labels = list(df[column].value_counts().index.unique())

    fontdict = {'fontsize': 10, 'fontweight' : 'bold'}
    plt.subplot(plt_n)
    if mean_ch:
        sns.barplot(x = column, y="totals.transactionRevenue", data=df, order = labels[:10])
    else:
        sns.countplot(x = column, data=df, order = labels[:10])
    plt.xticks(np.arange(0, len(labels[:10]), 1), labels, rotation=-80)
    plt.title(title, fontdict=fontdict)
    plt.tick_params(axis='both', which='major', labelsize=10)
    plt.xlabel('')
    plt.ylabel('')
    
create_countplot( 'device.deviceCategory', 'Device Used in Visits', 231)
create_countplot( 'device.deviceCategory', 'Device Used in Visits with Purchase', 232, train.loc[train["totals.transactionRevenue"] > 0])
create_countplot( 'device.deviceCategory', 'Device Used in Visits with Purchase (Mean)', 233, train.loc[train["totals.transactionRevenue"] > 0], True)
plt.figure(figsize=[20,16])

create_countplot( 'device.browser', 'Browser Used in Visits', 231)
create_countplot( 'device.browser', 'Browser Used in Visits with Purchase', 232, train.loc[train["totals.transactionRevenue"] > 0])
create_countplot( 'device.browser', 'Browser Used in Visits with Purchase (Mean)', 233, train.loc[train["totals.transactionRevenue"] > 0], True)
plt.figure(figsize=[20,16])

create_countplot( 'device.operatingSystem', 'Operating System Used in Visits', 231)
create_countplot( 'device.operatingSystem', 'Operating System Used in Visits with Purchase', 232, train.loc[train["totals.transactionRevenue"] > 0])
create_countplot( 'device.operatingSystem', 'Operating System Used in Visits with Purchase (Mean)', 233, train.loc[train["totals.transactionRevenue"] > 0], True)
plt.figure(figsize=[20,16])

create_countplot( 'geoNetwork.networkDomain', 'Network Domain in Visits', 231)
create_countplot( 'geoNetwork.networkDomain', 'Network Domain Used in Visits with Purchase', 232, train.loc[train["totals.transactionRevenue"] > 0])
create_countplot( 'geoNetwork.networkDomain', 'Network Domain in Visits with Purchase (Mean)', 233, train.loc[train["totals.transactionRevenue"] > 0], True)
plt.figure(figsize=[20,16])

create_countplot( 'geoNetwork.continent', 'Continent in Visits', 231)
create_countplot( 'geoNetwork.continent', 'Continent in Visits with Purchase', 232, train.loc[train["totals.transactionRevenue"] > 0])
create_countplot( 'geoNetwork.continent', 'Continent in Visits with Purchase (Mean)', 233, train.loc[train["totals.transactionRevenue"] > 0], True)
plt.figure(figsize=[20,16])

create_countplot( 'geoNetwork.subContinent', 'Sub Continent in Visits', 231)
create_countplot( 'geoNetwork.subContinent', 'Sub Continent in Visits with Purchase', 232, train.loc[train["totals.transactionRevenue"] > 0])
create_countplot( 'geoNetwork.subContinent', 'Sub Continent in Visits with Purchase (Mean)', 233, train.loc[train["totals.transactionRevenue"] > 0], True)
plt.figure(figsize=[20,16])

create_countplot( 'trafficSource.source', 'Traffic Source in Visits', 231)
create_countplot( 'trafficSource.source', 'Traffic Source in Visits with Purchase', 232, train.loc[train["totals.transactionRevenue"] > 0])
create_countplot( 'trafficSource.source', 'Traffic Source in Visits with Purchase (Mean)', 233, train.loc[train["totals.transactionRevenue"] > 0], True)
plt.figure(figsize=[20,16])

create_countplot( 'trafficSource.medium', 'Traffic Medium in Visits', 231)
create_countplot( 'trafficSource.medium', 'Traffic Medium in Visits with Purchase', 232, train.loc[train["totals.transactionRevenue"] > 0])
create_countplot( 'trafficSource.medium', 'Traffic Medium in Visits with Purchase (Mean)', 233, train.loc[train["totals.transactionRevenue"] > 0], True)
def time_stamp_features(df):
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d") # seting the column as pandas datetime
    df["weekday"] = df['date'].dt.weekday #extracting week day
    df["day"] = df['date'].dt.day # extracting day
    df["month"] = df['date'].dt.month # extracting day
    df["year"] = df['date'].dt.year # extracting day
    df["hour"] = (df['visitStartTime'].apply(lambda x: str(datetime.fromtimestamp(x).hour))).astype(int)
    return df
time_train = time_stamp_features(train)
p_time_train = time_train.loc[time_train["totals.transactionRevenue"] > 0]

def create_timeplot(var, title, xlabel, ylabel, labels = []):
    plt.figure(figsize=[40,16])
    
    df_count = time_train[var].value_counts().to_frame().reset_index().sort_values('index') 
    df_sum = df_count[var].sum()

    sns.lineplot(x = "index", y = var, data = df_count, estimator=lambda x: x / df_sum * 100, label="%Visits")

    df_count = p_time_train[var].value_counts().to_frame().reset_index().sort_values('index')
    df_sum = df_count[var].sum()

    sns.lineplot(x = "index", y = var, data = df_count, color="Orange", estimator=lambda x: x / df_sum * 100, label="%Visits with Purchase")

    fontdict = {'fontsize': 40, 'fontweight' : 'bold'}

    plt.title(title, fontdict=fontdict)
    plt.tick_params(axis='both', which='major', labelsize=30)
    if labels != []:
        plt.xticks(np.arange(0, len(labels), 1), labels)
    plt.xlabel(xlabel, fontdict=fontdict)
    plt.ylabel(ylabel, fontdict=fontdict)
    plt.legend(fontsize=30)
    plt.show()
create_timeplot("hour","Visits by Hour of the Day","Hour of the Day", "% of Visits")
create_timeplot("date","Visits by Date","Date", "% of Visits")
create_timeplot("weekday","Visits by Weekday","Weekday", "% of Visits", ["Monday","Tuesday","Wendsday", "Thursday", "Friday", "Saturday", "Sunday"])
train['totals.pageviews'] = train['totals.pageviews'].astype("float")
train['totals.hits'] = train['totals.hits'].astype("float")

plt.figure(figsize=[20,16])
def create_viewplot(column, title, plt_n, df=train, mean_ch = False):
    df = df.loc[df[column] <= 60].sort_values(column) 
    labels = np.arange(0, 60, step=5)
    fontdict = {'fontsize': 10, 'fontweight' : 'bold'}
    plt.subplot(plt_n)
    if mean_ch:
        sns.barplot(x = column, y="totals.transactionRevenue", data=df)
    else:
        sns.countplot(x = column, data=df)
    plt.xticks(np.arange(-1, 60, step=5),labels, rotation=-80)
    plt.title(title, fontdict=fontdict)
    plt.tick_params(axis='both', which='major', labelsize=10)
    plt.xlabel('')
    plt.ylabel('')

create_viewplot( 'totals.pageviews', '# of Page Views in Visit', 231)
create_viewplot( 'totals.pageviews', '# of Page Views in Visit with Purchase', 232, train.loc[train["totals.transactionRevenue"] > 0])
create_viewplot( 'totals.pageviews', '# of Page Views in Visit with Purchase (Mean)', 233, train.loc[train["totals.transactionRevenue"] > 0], True)

create_viewplot( 'totals.hits', '# of hits in Visit', 234)
create_viewplot( 'totals.hits', '# of hits in Visit with Purchase', 235, train.loc[train["totals.transactionRevenue"] > 0])
create_viewplot( 'totals.hits', '# of hits in Visit with Purchase (Mean)', 236, train.loc[train["totals.transactionRevenue"] > 0], True)

plt.show()