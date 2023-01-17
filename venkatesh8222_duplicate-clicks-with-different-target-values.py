import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib
%matplotlib inline 
import time
import gc
from datetime import datetime
import os

os.listdir("../input")
dtypes = {
        'ip'            : 'uint32',
        'app'           : 'uint16',
        'device'        : 'uint16',
        'os'            : 'uint16',
        'channel'       : 'uint16',
        'is_attributed' : 'uint8',
        'click_id'      : 'uint32'
        }
print('loading train data...')
train_df = pd.read_csv("../input/train.csv", dtype=dtypes,skiprows=range(1,131886954),
                       usecols=['ip','app','device','os', 'channel', 'click_time', 'is_attributed'])
len_train = len(train_df)
train_df['click_time'] = pd.to_datetime(train_df['click_time'])
dup = train_df[train_df.duplicated(["ip", 'app', 'channel', 'device', 'os', 'click_time'], False)]
group = dup.groupby(['ip', 'app', 'channel', 'device', 'os', 'click_time']).is_attributed.mean().reset_index().rename(index=str, columns={'is_attributed': 'mean'})
dup = dup.merge(group, on=['ip', 'app', 'channel', 'device', 'os', 'click_time'], how='left')
del train_df
del group
gc.collect()
len_dup = len(dup)
print('Number of Duplicate clicks in train data: ', len_dup)
dup_diff_target = dup[(dup['mean']!=0.0) & (dup['mean'] !=1.0)]
len_dup_diff_target = len(dup_diff_target)
print('NUmber of duplicate clicks with different target values in train data: ', len_dup_diff_target)
dup_diff_target.head()
len_dup_diff_target/len_dup
print('loading test supplement data...')
test_sup_df = pd.read_csv("../input/test_supplement.csv", dtype=dtypes, usecols=['ip','app','device','os', 'channel', 'click_time', 'click_id'])
test_sup_df['click_time'] = pd.to_datetime(test_sup_df['click_time'])
test_sup_dup = test_sup_df[test_sup_df.duplicated(["ip", 'app', 'channel', 'device', 'os', 'click_time'], False)]
len_test_sup = len(test_sup_df)
del test_sup_df
gc.collect()
len_test_sup_dup = len(test_sup_dup)
print('Number of Duplicate clicks in test supplement data: ', len_test_sup_dup)
print('loading test data...')
test_df = pd.read_csv("../input/test.csv", dtype=dtypes, usecols=['ip','app','device','os', 'channel', 'click_time', 'click_id'])
test_df['click_time'] = pd.to_datetime(test_df['click_time'])
test_dup = test_df[test_df.duplicated(["ip", 'app', 'channel', 'device', 'os', 'click_time'], False)]
del test_df
gc.collect()
len_test_dup = len(test_dup)
print('Number of Duplicate clicks in test(For submission) data: ', len_test_dup)
percent_train_dup = len_dup/len_train
print("percent of dupplicate in train data: ", percent_train_dup)

percent_test_dup = len_test_sup_dup/len_test_sup
print("percent of dupplicate in test data: ", percent_test_dup)

percent_of_test_csv_dup = len_test_dup/len_test_sup_dup
print("Percent of test.csv duplicates in test_supplement.csv: ", percent_of_test_csv_dup)

maybe_worng_preds = (len_test_sup_dup*len_dup_diff_target)/len_dup
print("Number of wrong(may be) predictions for test supplement: ", maybe_worng_preds)

maybe_worng_preds_for_submission = percent_of_test_csv_dup*maybe_worng_preds
print("Number of wrong(may be) predictions for submission: ", maybe_worng_preds_for_submission)
group = dup_diff_target.groupby(["ip", 'app', 'channel', 'device', 'os', 'click_time'])
first = group.nth(0).is_attributed.value_counts()
second = group.nth(1).is_attributed.value_counts()
third = group.nth(2).is_attributed.value_counts()
print('First click target value counts:\n', first)
print('Second click target value counts:\n', second)
print('Third click target value counts:\n', third)
