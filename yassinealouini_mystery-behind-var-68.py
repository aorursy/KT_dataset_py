import pandas as pd

from datetime import datetime

import matplotlib.pyplot as plt

import numpy as np

%matplotlib inline
train_df = pd.read_csv('../input/train.csv')
train_df.drop(['ID_code', 'target'], axis=1).nunique().sort_values()
f"Min: {train_df['var_68'].min()} and max: {train_df['var_68'].max()}"
epoch_datetime = pd.datetime(1900, 1, 1)

trf_var_68_s = (train_df['var_68']*10000 - 7000 + epoch_datetime.toordinal()).astype(int)

date_s = trf_var_68_s.map(datetime.fromordinal)

train_df['date'] = date_s

sorted_train_df = train_df.drop('var_68', axis=1).sort_values('date')
fig, ax = plt.subplots(1, 1, figsize=(20, 8))

sorted_train_df.set_index('date')['var_0'].plot(ax=ax)
fig, ax = plt.subplots(1, 1, figsize=(20, 8))

sorted_train_df.set_index('date')['var_1'].plot(ax=ax)
fig, ax = plt.subplots(1, 1, figsize=(20, 8))

sorted_train_df.set_index('date')['var_2'].plot(ax=ax)
fig, ax = plt.subplots(1, 1, figsize=(20, 8))

sorted_train_df.set_index('date')['target'].plot(ax=ax)
date_s.nunique()
f"Train starts: {date_s.min()}, ends: {date_s.max()}"
sorted_train_df['date'].dt.month.value_counts()
sorted_train_df['date'].dt.month.value_counts().plot(kind='bar')
sorted_train_df['date'].dt.year.value_counts()
sorted_train_df['date'].dt.year.value_counts().plot(kind='bar')
sorted_train_df['date'].dt.dayofweek.value_counts()
sorted_train_df['date'].dt.dayofweek.value_counts().plot(kind='bar')
fig, ax = plt.subplots(1, 1, figsize=(20, 8))

sorted_train_df.groupby('date')['target'].agg(['std', 'mean', 'max', 'min']).plot(ax=ax)
# In another cell signs the count is much bigger than the other statistics

fig, ax = plt.subplots(1, 1, figsize=(20, 8))

sorted_train_df.groupby('date')['target'].agg(['count']).plot(ax=ax)
test_df = pd.read_csv('../input/test.csv')

epoch_datetime = pd.datetime(1900, 1, 1)

s = (test_df['var_68']*10000 - 7000 + epoch_datetime.toordinal()).astype(int)

test_df['date'] = s.map(datetime.fromordinal)

sorted_test_df = test_df.drop('var_68', axis=1).sort_values('date')
f"Test starts: {test_df['date'].min()} and ends: {test_df['date'].max()}"
test_df['date'].dt.year.value_counts().plot(kind='bar')
test_df['date'].dt.month.value_counts().plot(kind='bar')
test_df['date'].dt.dayofweek.value_counts().plot(kind='bar')
fig, ax = plt.subplots(1, 1, figsize=(20, 8))

sorted_test_df.groupby('date')['var_1'].agg(['count']).plot(ax=ax)
len(set(sorted_train_df['date']))
len(set(sorted_test_df['date']))
len(set(sorted_train_df['date']) & set(sorted_test_df['date']))
len(set(sorted_train_df['date']) - set(sorted_test_df['date']))
len(set(sorted_test_df['date']) - set(sorted_train_df['date']))
set(sorted_test_df['date']) - set(sorted_train_df['date'])
fig, ax = plt.subplots(1, 1, figsize=(12, 8))

sorted_train_df.groupby('date')['var_91'].count().plot(ax=ax, label="train")

sorted_test_df.groupby('date')['var_91'].count().plot(ax=ax, label="test")

ax.legend()
# Zoom on 2018

fig, ax = plt.subplots(1, 1, figsize=(12, 8))

(sorted_train_df.loc[lambda df: df.date.dt.year == 2018]

               .groupby('date')['var_91']

               .count()

               .plot(ax=ax, label="train"))

(sorted_test_df.loc[lambda df: df.date.dt.year == 2018]

               .groupby('date')['var_91']

               .count()

               .plot(ax=ax, label="test"))

ax.legend()
# Zoom on 2018-1

fig, ax = plt.subplots(1, 1, figsize=(12, 8))

(sorted_train_df.loc[lambda df: (df.date.dt.year == 2018) & (df.date.dt.month == 1)]

               .groupby('date')['var_91']

               .count()

               .plot(ax=ax, label="train"))

(sorted_test_df.loc[lambda df: (df.date.dt.year == 2018) & (df.date.dt.month == 1)]

               .groupby('date')['var_91']

               .count()

               .plot(ax=ax, label="test"))

ax.legend()
# Zoom on 2018-1

fig, ax = plt.subplots(1, 1, figsize=(12, 8))

(sorted_train_df.loc[lambda df: (df.date.dt.year == 2018) & (df.date.dt.month == 1)]

               .groupby('date')['var_91']

               .mean()

               .plot(ax=ax, label="train"))

(sorted_test_df.loc[lambda df: (df.date.dt.year == 2018) & (df.date.dt.month == 1)]

               .groupby('date')['var_91']

               .mean()

               .plot(ax=ax, label="test"))

ax.legend()
overlapping_dates = set(sorted_train_df['date']) & set(sorted_test_df['date'])
grouped_df = (sorted_train_df.loc[lambda df: df.date.isin(overlapping_dates)]

                             .groupby('date')['target']

                             .mean())
grouped_df.plot(kind='hist', bins=100)
grouped_df.to_csv('grouped_df.csv', index=False)