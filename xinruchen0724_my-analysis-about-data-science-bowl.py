# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns # data visualization

import matplotlib.style as style

style.use('fivethirtyeight')

import calendar

import matplotlib.pylab as plt # pyplot + numpy

from IPython.display import HTML # display HTML

import warnings

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import datetime

from time import time

from collections import Counter

from scipy import stats



from sklearn.model_selection import GroupKFold # K fold 

from typing import Any

from numba import jit

import lightgbm as lgb

import xgboost as xgb

from catboost import CatBoostRegressor, CatBoostClassifier

from sklearn import metrics

from itertools import product

import copy

import time



import random

seed = 1234

random.seed(seed)

np.random.seed(seed)





import os

import json

import matplotlib.pyplot as plt2

from tqdm import tqdm_notebook as tqdm

%matplotlib inline



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

pd.set_option('max_columns', 100)

warnings.filterwarnings('ignore')

sns.set_style('darkgrid')

my_pal = sns.color_palette(n_colors=10)
!ls -GFlash ../input/data-science-bowl-2019/
# original variable for multiple analysis

train_original = pd.read_csv('../input/data-science-bowl-2019/train.csv')

train_labels_original = pd.read_csv('../input/data-science-bowl-2019/train_labels.csv')

test_original = pd.read_csv('../input/data-science-bowl-2019/test.csv')

specs_original = pd.read_csv('../input/data-science-bowl-2019/specs.csv')

sample_submission_original = pd.read_csv('../input/data-science-bowl-2019/sample_submission.csv')
import copy

train = copy.deepcopy(train_original)

train_labels = copy.deepcopy(train_labels_original)

test = copy.deepcopy(test_original)

specs = copy.deepcopy(specs_original)

sample_submission = copy.deepcopy(sample_submission_original)
import copy

train_df = copy.deepcopy(train_original)

train_labels_df = copy.deepcopy(train_labels_original)

test_df = copy.deepcopy(test_original)

specs_df = copy.deepcopy(specs_original)

sample_submission_df = copy.deepcopy(sample_submission_original)
import copy

train_df2 = copy.deepcopy(train_original)

train_labels_df2 = copy.deepcopy(train_labels_original)

test_df2 = copy.deepcopy(test_original)

specs_df2 = copy.deepcopy(specs_original)

sample_submission_df2 = copy.deepcopy(sample_submission_original)
train_df2.shape
keep_id = train_df2[train.type == 'Assessment'][['installation_id']].drop_duplicates()

train_df2 = pd.merge(train, keep_id, on='installation_id', how='inner')
pd.set_option('max_colwidth', 800)

specs_df.head()
def missing_data(data):

    total = data.isnull().sum()

    percent = (data.isnull().sum() / data.isnull().count() * 100)

    tt = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

    types = []

    for col in data.columns:

        dtype = str(data[col].dtype)

        types.append(dtype)

    tt['Types'] = types

    return np.transpose(tt)
display(missing_data(train_df))

display(missing_data(test_df))

display(missing_data(train_labels_df))

display(missing_data(specs_df))
def unique_values(data):

    total = data.count()

    tt = pd.DataFrame(total)

    tt.columns = ['Total']

    uniques = []

    for col in data.columns:

        unique = data[col].nunique()

        uniques.append(unique)

    tt['Uniques'] = uniques

    return np.transpose(tt)
display(unique_values(train_df))

display(unique_values(test_df))

display(unique_values(train_labels_df))

display(unique_values(specs_df))
def most_frequent_values(data):

    total = data.count()

    tt = pd.DataFrame(total)

    tt.columns = ['Total']

    most_frequent_items = []

    freqs = []

    for col in data.columns:

        most_frequent_item = data[col].value_counts().index[0]

        freq = data[col].value_counts().values[0]

        most_frequent_items.append(most_frequent_item)

        freqs.append(freq)

    tt['Most frequenct item'] = most_frequent_items

    tt['Frequency'] = freqs

    tt['Percent from total'] = np.round(freqs / total * 100, 3)

    return (np.transpose(tt))

    
display(most_frequent_values(train_df))

display(most_frequent_values(test_df))

display(most_frequent_values(train_labels_df))

display(most_frequent_values(specs_df))
def plot_count(feature, title, df, size=1):

    f, ax = plt.subplots(1, 1, figsize=(4*size, 4))

    total = float(len(df))

    g = sns.countplot(df[feature], order = df[feature].value_counts().index[:20], palette='Set3')

    g.set_title("Number and percentage of {}".format(title))

    if size > 2:

        plt.xticks(rotation=90, size=8)

    for p in ax.patches:

        height = p.get_height()

        ax.text(p.get_x() + p.get_width() / 2., height + 3, '{:1.2f}%'.format(100*height/total), ha='center')

    plt.show()

    

# this function is used to count the number of a specified feature
display(plot_count('title', 'title (first most frequent 20 values - train)', train_df, size=4))

display(plot_count('title', 'title (first most frequent 20 values - test)', test_df, size=4))
plot_count('title', 'title - train_labels', train_labels_df, size=3)
plot_count('accuracy', 'accuracy - train_labels', train_labels_df, size=4)
plot_count('accuracy_group', 'accuracy_group - train_labels', train_labels_df, size=2)
plot_count('num_correct', 'num_correct - train_labels', train_labels_df, size=1)
plot_count('num_incorrect', 'num_incorrect - train_labels', train_labels_df, size=4)
sample_train_df = train_df.sample(100000)
sample_train_df.head(5)
sample_train_df.iloc[0]['event_data']
%%time

extracted_event_data = pd.io.json.json_normalize(sample_train_df['event_data'].apply(json.loads)) # normalize json
extracted_event_data.head(3)
missing_data(extracted_event_data)
def existing_data(data):

    total = data.isnull().count() - data.isnull().sum()

    percent = 100 - (data.isnull().sum() / data.isnull().count() * 100)

    tt = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

    tt = pd.DataFrame(tt.reset_index())

    

    return tt.sort_values(['Total'], ascending=False)
stat_event_data = existing_data(extracted_event_data)
stat_event_data.head(5)
plt.figure(figsize=(10, 10))

sns.set(style='darkgrid')

ax = sns.barplot(x='Percent', y='index', data=stat_event_data.head(40), color='gold')

plt.title('Most frequent features in event_data')

plt.ylabel('Features')
specs_df.head(3)
specs_df.iloc[0]['args']
specs_args_extracted = pd.DataFrame()

for i in range(0, specs_df.shape[0]):

    for item in json.loads(specs_df.args[i]):

        new_df = pd.DataFrame({'event_id': specs_df['event_id'][i],\

                               'info': specs_df['info'],\

                               'args_name': item['name'],\

                               'args_type': item['type'],\

                               'args_info': item['info']}, index=[i]

                              )

        specs_args_extracted = specs_args_extracted.append(new_df)

print(f'Extracted args from specs: {specs_args_extracted.shape}')
specs_args_extracted.head(10)
tmp = specs_args_extracted.groupby(['event_id'])['event_id'].count() # event - number of arguments per event

df = pd.DataFrame({'event_id': tmp.index, 'count': tmp.values})

plt.figure(figsize=(6, 4))

sns.set(style='darkgrid')

ax = sns.distplot(df['count'], kde=True, hist=True, bins=30)

plt.title('Distribution of number of arguments per event_id')

plt.xlabel('Number of arguments'); plt.ylabel('Density'); plt.show()

# it is a plot about the density of arguments per event
plot_count('args_name', 'args_name (first 20 most frequent values) - specs', specs_args_extracted, size=4)
plot_count('args_type', 'args_type (first 20 most frequent values) - specs', specs_args_extracted, size=3)
plot_count('args_info', 'args_info (first 20 most frequent values) - specs', specs_args_extracted, size=3)
def extract_time_features(df):

    df['timestamp'] = pd.to_datetime(df['timestamp'])

    df['date'] = df['timestamp'].dt.date

    df['month'] = df['timestamp'].dt.month

    df['year'] = df['timestamp'].dt.year

    df['hour'] = df['timestamp'].dt.hour

    df['dayofweek'] = df['timestamp'].dt.dayofweek

    df['weekofyear'] = df['timestamp'].dt.weekofyear

    df['dayofyear'] = df['timestamp'].dt.dayofyear

    df['quarter'] = df['timestamp'].dt.quarter

    df['is_month_start'] = df['timestamp'].dt.is_month_start

    return df
train_df = extract_time_features(train_df)

test_df = extract_time_features(test_df)
plot_count('year', 'year - train', train_df, size=2)
plot_count('is_month_start', 'is_month_start - train', train_df, size=2)
plot_count('month', 'month - train', train_df, size=2)

# do the same with rest of the date info
numerical_columns = ['game_time', 'month', 'dayofweek', 'hour']

categorical_columns = ['type', 'world']



comp_train_df = pd.DataFrame({'installation_id': train_df['installation_id'].unique()})

comp_train_df.set_index('installation_id', inplace=True)
comp_train_df.head(10)
def get_numeric_columns(df, column):

    df = df.groupby('installation_id').agg({f'{column}': ['mean', 'sum', 'min', 'max', 'std', 'skew']})

    df[column].fillna(df[column].mean(), inplace=True)

    df.columns = [f'{column}_mean', f'{column}_sum', f'{column}_min', f'{column}_max', f'{column}_std', f'{column}_skew']

    return df
for numerical_column in numerical_columns:

    comp_train_df = comp_train_df.merge(get_numeric_columns(train_df, numerical_column), left_index=True, right_index=True)
print(f'comp_train shape: {comp_train_df.shape}')
comp_train_df.columns
# for some reason the dataframe I produces has game_time_xx_y, droping it and renaming game_time_xx_x to game_time_x

# comp_train_df.drop(columns=['game_time_mean_y', 'game_time_sum_y', 'game_time_min_y',

#        'game_time_max_y', 'game_time_std_y', 'game_time_skew_y'], inplace=True)

comp_train_df.rename({'game_time_mean_x': 'game_time_mean', 'game_time_sum_x': 'game_time_sum', 'game_time_min_x': 'game_time_min',

       'game_time_max_x': 'game_time_max', 'game_time_std_x': 'game_time_std', 'game_time_skew_x': 'game_time_skew'}, axis=1, inplace=True)
comp_train_df.head()
comp_train_df.shape
train_labels_df.head()
train_labels_df.groupby('title')['accuracy_group'].head()
# this part I don't understand

labels_map = dict(train_labels_df.groupby('title')['accuracy_group'].agg(lambda x: x.value_counts().index[0])) 

# value_counts gets frequencies for different accuracy_group, and get the first value??

labels = train_labels_df[['installation_id', 'title', 'accuracy_group']]

labels['title'] = labels['title'].map(labels_map)

comp_train_df = labels.merge(comp_train_df, on='installation_id', how='left')

print('We have {} training rows'.format(comp_train_df.shape[0]))

comp_train_df.head()
print(f'comp_train_df shape: {comp_train_df.shape}')

for feature in comp_train_df.columns.values[3: 20]:

    print(f'{feature} unique values: {comp_train_df[feature].nunique()}') # dataframe.nunique => series, series => integer
plot_count('title', 'title - compound train', comp_train_df)
plot_count('accuracy_group', 'accuracy_group - compound train', comp_train_df)
plt.figure(figsize=(16, 6))

_titles = comp_train_df['title'].unique() # pandas.series.unique => array of unique values

plt.title('Distribution of log(game time mean) values (grouped by title) in the comp train')

for _title in _titles:

    red_comp_train_df = comp_train_df.loc[comp_train_df.title == _title] # find rows where the titles are the same

    sns.distplot(np.log(red_comp_train_df['game_time_mean']), kde=True, label=f'title: {_title}')

plt.legend()

plt.show()



# do the same with different stats features such as game_time_skew, hour_mean, hour_std etc
train_labels.head()
train_labels.groupby('accuracy_group')['game_session'].count().plot(kind='barh', figsize=(15, 5), title='Target (accuracy group)')

plt.show()
# don't really understand how pairplot works

sns.pairplot(train_labels, hue='accuracy_group')

plt.show()
train.head()
# change event_id and game_session from hex into integer

train['event_id_as_int'] = train['event_id'].apply(lambda x : int(x, 16))

train['game_session_as_int'] = train['game_session'].apply(lambda x : int(x, 16))
type(train['timestamp'][0]) # timestamp is string
train['timestamp'], test['timestamp'] = pd.to_datetime(train['timestamp']), pd.to_datetime(test['timestamp']) # convert string into datetime

train['date'], train['hour'], train['weekday_name'] = train['timestamp'].dt.date, train['timestamp'].dt.hour, train['timestamp'].dt.weekday_name

test['date'], test['hour'], test['weekday_name'] = test['timestamp'].dt.date, test['timestamp'].dt.hour, test['timestamp'].dt.weekday_name
print(f'Train data has shape: {train.shape}')

print(f'Test data has shape: {test.shape}')
train.columns # train has two more columns - "game_session_as_int" and "event_id_as_int"
train.groupby('date')['event_id'].agg('count').plot(figsize=(15, 3), title='Number of Event Observations by Date', color='blue')

plt.show()

train.groupby('hour')['event_id'].agg('count').plot(figsize=(15, 3), title='Number of Event Observations by Date', color='blue')

plt.show()

# can see that during 15 - 20 more activities than 5 - 10 
train.groupby('weekday_name')['event_id'].agg('count').plot(figsize=(15, 3), title='Number of Event Observations by Date', color='blue')

plt.show()

# can see that Friday has most activities
print(train['event_data'][4])

print(train['event_data'][5])
train['installation_id'].nunique() # number of unique installation ids 
train.groupby('installation_id').count()['event_id'].plot(kind='hist', bins=40, color='pink', figsize=(15, 5), title='Count of Observations by installation_id')

plt.show()
train.groupby('installation_id').count()['event_id'].apply(np.log1p).plot(kind='hist', bins=40, color='pink', figsize=(15, 5), title='Log(Count) of Observations by installation_id')

plt.show()



# why log(count)? 
train.groupby('installation_id').count()['event_id'].sort_values(ascending=False).head(5)
train.query('installation_id == "f1c21eda"').set_index('timestamp')['event_code'].plot(figsize=(15, 5), title='installation_id #f1c21eda event Id - event code vs time', style='.', color='orange')

plt.show()



# this graph is not natural, the interface could be installed by a bot
train.groupby('event_code').count()['event_id'].sort_values().plot(kind='bar', figsize=(15, 5), title='Count of different event codes.')

plt.show()
train['game_time'].apply(np.log1p).plot(kind='hist', bins=100, title='Log Transform of game_time', color='gold')

plt.show()
train.groupby('title')['event_id'].count().sort_values().plot(kind='barh', title='Count of Observation by Game/Video title', figsize=(15, 15))

plt.show()

train.groupby('type')['event_id'].count().sort_values().plot(kind='barh', figsize=(15, 4), title='Count by Type', color='turquoise')

plt.show()
train.groupby('world')['event_id'].count().sort_values().plot(kind='bar', figsize=(15, 4), title='Count by World', color='gold')

plt.show()
train['log_game_time'] = train['game_time'].apply(np.log1p)
fig, ax = plt.subplots(figsize=(15, 5))

sns.catplot(x='type', y='log_game_time', data=train.sample(10000), alpha=0.5, ax=ax)

ax.set_title('Distribution of log(game_time) by Type')

plt.close()

plt.show()



fig, ax = plt.subplots(figsize=(15, 5))

sns.catplot(x='world', y='log_game_time', data=train.sample(10000), alpha=0.5, ax=ax)

ax.set_title('Distribution of log(game_time) by World')

plt.close()

plt.show()
specs.head()
specs.describe()
specs.iloc[0][-2]
from sklearn.model_selection import train_test_split



train['cleared'] = True

train.loc[train['event_data'].str.contains('false') & train['event_code'].isin([4100, 4110]), 'cleared'] = False # why 4100-4110



test['cleared'] = True

test.loc[test['event_data'].str.contains('false') & test['event_code'].isin([4100, 4110]), 'cleared'] = False # why 4100-4110



aggs = {'hour':['max', 'min', 'mean'], 'cleared': ['mean']}



train_aggs = train.groupby('installation_id').agg(aggs)

test_aggs = test.groupby('installation_id').agg(aggs)



train_aggs = train_aggs.reset_index()

test_aggs = test_aggs.reset_index()



train_aggs.columns = ['_'.join(col).strip() for col in train_aggs.columns.values]

test_aggs.columns = ['_'.join(col).strip() for col in test_aggs.columns.values]



train_aggs = train_aggs.rename(columns={'installation_id_' : 'installation_id'})

train_aggs.head(5)
train_aggs.merge(train_labels[['installation_id', 'accuracy_group']], how='left')