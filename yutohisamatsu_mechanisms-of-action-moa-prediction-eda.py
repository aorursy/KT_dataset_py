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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px

import matplotlib.style as style



import random

import plotly.graph_objects as go

from plotly.subplots import make_subplots

import time



pd.options.display.max_columns = None
train = pd.read_csv('/kaggle/input/lish-moa/train_features.csv')

test = pd.read_csv('/kaggle/input/lish-moa/test_features.csv')



targets = pd.read_csv('/kaggle/input/lish-moa/train_targets_scored.csv')

targets_non = pd.read_csv('/kaggle/input/lish-moa/train_targets_nonscored.csv')



train['dataset'] = 'train'

test['dataset'] = 'test'



sample_submit = pd.read_csv('/kaggle/input/lish-moa/sample_submission.csv')



all_data = pd.concat([train, test])
train.head()
test.head()
print('Number of rows in training set: ', train.shape[0])

print('Number of columns in training set: ', train.shape[1] - 1) # 追加したdatasetの分抜いておく



print('')



print('Number of rows in test set: ', test.shape[0])

print('Number of columns in test set: ', test.shape[1] - 1) # 追加したdatasetの分抜いておく
all_data.info()
targets.head()
targets_non.head()
# 提出するサンプル

sample_submit.head()
print('Number of rows in targets-data set: ', targets.shape[0])

print('Number of columns in targets-data set: ', targets.shape[1])
# 欠損値



# ----train-----

missing_train = train.isnull().sum()

missing_train = missing_train[missing_train > 0]

missing_train.sort_values(inplace=True)



# ----test-----

missing_test = test.isnull().sum()

missing_test = missing_test[missing_test > 0]

missing_test.sort_values(inplace=True)



# ----targets-----

missing_targets = targets.isnull().sum()

missing_targets = missing_targets[missing_targets > 0]

missing_targets.sort_values(inplace=True)
missing_train
missing_test
missing_targets
dataSet = all_data.groupby(['cp_type', 'dataset'])['sig_id'].count().reset_index()

dataSet
dataSet.columns = ['cp_type', 'dataset', 'count']

dataSet
fig = px.bar(

    dataSet, 

    x='cp_type', 

    y="count", 

    color = 'dataset',

    barmode='group',

    orientation='v', 

    title='cp_type train/test counts', 

    width=500,

    height=400

)



fig.show()
dataSet = all_data.groupby(['cp_time', 'dataset'])['sig_id'].count().reset_index()

dataSet
dataSet.columns = ['cp_time', 'dataset', 'count']

dataSet
fig = px.bar(

    dataSet, 

    x='cp_time', 

    y="count", 

    color = 'dataset',

    barmode='group',

    orientation='v', 

    title='cp_time train/test counts', 

    width=500,

    height=400

)



fig.show()
dataSet = all_data.groupby(['cp_dose', 'dataset'])['sig_id'].count().reset_index()

dataSet.columns = ['cp_dose', 'dataset', 'count']



fig = px.bar(

    dataSet, 

    x='cp_dose', 

    y="count", 

    color = 'dataset',

    barmode='group',

    orientation='v', 

    title='cp_dose train/test counts', 

    width=500,

    height=400

)



fig.show()
train_columns = train.columns.to_list()

g_list = [column for column in train.columns if column.startswith('g-')]

c_list = [column for column in train.columns if column.startswith('c-')]
def plot_set_histograms(plot_list, title):

    fig = make_subplots(rows=4, cols=3)

    traces = [go.Histogram(x=train[col], nbinsx=20, name=col) for col in plot_list]



    for i in range(len(traces)):

        fig.append_trace(traces[i], (i // 3) + 1, (i % 3) + 1)



    fig.update_layout(

        title_text=title,

        height=1000,

        width=1000

    )

    fig.show()
# gene features

plot_list = [g_list[random.randint(0, len(g_list)-1)] for i in range(50)]

plot_list = list(set(plot_list))[:12]

plot_set_histograms(plot_list, 'Randomly selected gene expression features distributions')
# cell feature

plot_list = [c_list[random.randint(0, len(c_list)-1)] for i in range(50)]

plot_list = list(set(plot_list))[:12]

plot_set_histograms(plot_list, 'Randomly selected cell expression features distributions')
columns = g_list + c_list
corrmat = train[columns].corr()

f, ax = plt.subplots(figsize=(14,14))

sns.heatmap(corrmat, square=True, vmax=.8);
# ランダム

for_correlation = list(set([columns[random.randint(0, len(columns)-1)] for i in range(200)]))[:40]

data = all_data[for_correlation]



f = plt.figure(figsize=(19, 17))

plt.matshow(data.corr(), fignum=f.number)

plt.xticks(range(data.shape[1]), data.columns, fontsize=14, rotation=50)

plt.yticks(range(data.shape[1]), data.columns, fontsize=14)

cb = plt.colorbar()

cb.ax.tick_params(labelsize=13)
cols = ['cp_time'] + columns # columns = g_list + c_list

all_columns = []

for i in range(0, len(cols)):

    for j in range(i+1, len(cols)):

        if abs(train[cols[i]].corr(train[cols[j]])) > 0.9:

            all_columns.append(cols[i])

            all_columns.append(cols[j])
all_columns = list(set(all_columns)) # 重複する要素が除外されて一意な値のみが要素となるsetオブジェクトが生成

print('Number of columns: ', len(all_columns))
all_columns
# 相関の高いペア同士のヒートマップ図

data = all_data[all_columns]



f = plt.figure(figsize=(19, 15))

plt.matshow(data.corr(), fignum=f.number)

plt.xticks(range(data.shape[1]), data.columns, fontsize=14, rotation=50)

plt.yticks(range(data.shape[1]), data.columns, fontsize=14)

cb = plt.colorbar()

cb.ax.tick_params(labelsize=14)
fig = make_subplots(rows=12, cols=3)

traces = [go.Histogram(x=train[col], nbinsx=20, name=col) for col in all_columns]



for i in range(len(traces)):

    fig.append_trace(traces[i], (i // 3) + 1, (i % 3) + 1)



fig.update_layout(

    title_text='Highly correlated features',

    height=1200

)

fig.show()
print('Number of rows : ', targets.shape[0])

print('Number of cols : ', targets.shape[1])



targets.head()
# 降順

x = targets.drop(['sig_id'], axis=1).sum(axis=0).sort_values().reset_index()

x.columns = ['column', 'nonzero_records']

x
# 出現率の高い順

fig = px.bar(

    x.tail(50), 

    x='nonzero_records', 

    y='column', 

    orientation='h', 

    title='Columns with the higher number of positive samples (top 50)', 

    height=1000, 

    width=800

)



fig.show();
# 昇順

x = targets.drop(['sig_id'], axis=1).sum(axis=0).sort_values(ascending=False).reset_index()

x.columns = ['column', 'nonzero_records']



# 出現率の低い順

fig = px.bar(

    x.tail(50), 

    x='nonzero_records', 

    y='column', 

    orientation='h', 

    title='Columns with the lowest number of positive samples (top 50)', 

    height=1000, 

    width=800

)



fig.show();
x = targets.drop(['sig_id'], axis=1).sum(axis=0).sort_values(ascending=False).reset_index()

x.columns = ['column', 'count']

x['count'] = x['count'] * 100 / len(targets)



fig = px.bar(

    x, 

    x='column', 

    y='count', 

    orientation='v', 

    title='Percent of positive records for every column in target', 

    height=800, 

    width=1200

)



fig.show()
# data　207のターゲットのうち、何個1があるか

data = targets.drop(['sig_id'], axis=1).astype(bool).sum(axis=1).reset_index()

data.columns = ['row', 'count']

data = data.groupby(['count'])['row'].count().reset_index()
data
fig = px.bar(

    data, 

    y=data['row'], 

    x="count", 

    title='Number of activations in targets for every sample', 

    width=800, 

    height=500

)



fig.show()
targets.describe()
correlation_matrix = pd.DataFrame()

for t_col in targets.columns:

    corr_list = list()

    if t_col == 'sig_id':

        continue

        

        # columns = g_list + c_list

    for col in columns:

        res = train[col].corr(targets[t_col])

        corr_list.append(res)

    correlation_matrix[t_col] = corr_list
correlation_matrix['train_features'] = columns

correlation_matrix = correlation_matrix.set_index('train_features')

correlation_matrix
maxCol=lambda x: max(x.min(), x.max(), key=abs)

high_scores = correlation_matrix.apply(maxCol, axis=0).reset_index()

high_scores.columns = ['column', 'best_correlation']



fig = px.bar(

    high_scores, 

    x='column', 

    y="best_correlation", 

    orientation='v', 

    title='Best correlation with train columns for every target column', 

    width=1200,

    height=800

)



fig.show()
col_df = pd.DataFrame()

tr_cols = list()

tar_cols = list()

for col in correlation_matrix.columns:

    tar_cols.append(col)

    tr_cols.append(correlation_matrix[col].abs().sort_values(ascending=False).reset_index()['train_features'].head(1).values[0])



col_df['column'] = tar_cols

col_df['train_best_column'] = tr_cols



total_scores = pd.merge(high_scores, col_df)

total_scores
count_features = total_scores['train_best_column'].value_counts().reset_index().sort_values('train_best_column')

count_features.columns = ['column', 'count']

fig = px.bar(

    count_features.tail(33), 

    x='count', 

    y="column", 

    orientation='h', 

    title='Columns from training set with number of high correlations with target columns', 

    width=800,

    height=700

)



fig.show()