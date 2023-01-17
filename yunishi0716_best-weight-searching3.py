import gc

import os

from pathlib import Path

import random

import sys



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



from sklearn import preprocessing

from sklearn.model_selection import KFold

import lightgbm as lgb

import xgboost as xgb

import catboost as cb



from sklearn.metrics import mean_squared_error
#reduce the memory usage

from pandas.api.types import is_datetime64_any_dtype as is_datetime

from pandas.api.types import is_categorical_dtype

def reduce_mem_usage(df):

    """ iterate through all the columns of a dataframe and modify the data type

        to reduce memory usage.        

    """

    start_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    

    for col in df.columns:

        if is_datetime(df[col]) or is_categorical_dtype(df[col]):

            # skip datetime type or categorical type

            continue

        col_type = df[col].dtype

        

        if col_type != object:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)  

            else:

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)

        else:

            df[col] = df[col].astype('category')



    end_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))

    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    

    return df
# using original data

root = Path('../input/ashrae-feather-format-for-fast-loading')

# train_df = pd.read_feather(root/'train.feather')

test_df = pd.read_feather(root/'test.feather')

building_meta_df = pd.read_feather(root/"building_metadata.feather")
# using LK data

leak_df = pd.read_feather('../input/ashrae-leak-data-station/leak.feather')

leak_df.fillna(0, inplace=True)

leak_df = leak_df[(leak_df.timestamp.dt.year > 2016) & (leak_df.timestamp.dt.year < 2019)]

leak_df.loc[leak_df.meter_reading < 0, 'meter_reading'] = 0 # remove large negative values

leak_df = leak_df[leak_df.building_id!=245]
print(leak_df.duplicated().sum())
print(leak_df.isnull().sum())
gc.collect()
!ls ../input/
kfold_submission = pd.read_csv('../input/ashrae-kfold-lightgbm-without-leak-1-08/submission.csv', index_col=0)

halfhalf_submission = pd.read_csv('../input/ashrae-half-and-half/submission.csv', index_col=0)

highway_submission = pd.read_csv('../input/ashrae-highway-kernel-route4/submission.csv', index_col=0)

gep3_submission = pd.read_csv("../input/ashrae-great-energy-predictor-iii-eda-model/Normal.csv", index_col=0)

simple_submission = pd.read_csv("../input/ashrae-simple-data-cleanup-lb-1-08-no-leaks/submission.csv", index_col=0)
test_df["pred1"] = kfold_submission.meter_reading

test_df["pred2"] = halfhalf_submission.meter_reading

test_df["pred3"] = highway_submission.meter_reading

test_df["pred4"] = gep3_submission.meter_reading

test_df["pred5"] = simple_submission.meter_reading



del kfold_submission, halfhalf_submission, highway_submission, gep3_submission

del simple_submission

gc.collect()
test_df.loc[test_df.pred3 < 0, "pred3"] = 0

test_df.loc[test_df.pred4 < 0, "pred4"] = 0

test_df.loc[test_df.pred5 < 0, "pred5"] = 0

test_df = reduce_mem_usage(test_df)

leak_df = reduce_mem_usage(leak_df)
test_df.head(10)
leak_df = leak_df.merge(test_df[['building_id', 'meter', 'timestamp', 'pred1', 'pred2', 'pred3', 'pred4', 'pred5', 'row_id']],

                        left_on = ['building_id','meter', 'timestamp'], right_on = ['building_id', 'meter', 'timestamp'],

                        how = "left")

leak_df = leak_df.merge(building_meta_df[['building_id', 'site_id']], on='building_id', how='left')
leak_df["pred1_l1p"] = np.log1p(leak_df.pred1)

leak_df["pred2_l1p"] = np.log1p(leak_df.pred2)

leak_df["pred3_l1p"] = np.log1p(leak_df.pred3)

leak_df["pred4_l1p"] = np.log1p(leak_df.pred4)

leak_df["pred5_l1p"] = np.log1p(leak_df.pred5)

leak_df["meter_reading_l1p"] = np.log1p(leak_df.meter_reading)
leak_df.head(10)
leak_df.isnull().sum()
print(leak_df[leak_df.pred1_l1p.isnull()])

print(leak_df[leak_df.pred4_l1p.isnull()])

print(leak_df[leak_df.pred5_l1p.isnull()])
del building_meta_df

gc.collect()
# kfold

sns.distplot(leak_df.pred1_l1p)

sns.distplot(leak_df.meter_reading_l1p)



leak_score = np.sqrt(mean_squared_error(leak_df.pred1_l1p, leak_df.meter_reading_l1p))

print ('score1=', leak_score)
# half and half

sns.distplot(leak_df.pred2_l1p)

sns.distplot(leak_df.meter_reading_l1p)



leak_score = np.sqrt(mean_squared_error(leak_df.pred2_l1p, leak_df.meter_reading_l1p))

print ('score2=', leak_score)
#highway route4

sns.distplot(leak_df.pred3_l1p)

sns.distplot(leak_df.meter_reading_l1p)



leak_score = np.sqrt(mean_squared_error(leak_df.pred3_l1p, leak_df.meter_reading_l1p))

print ('score3=', leak_score)
#gep3

sns.distplot(leak_df.pred4_l1p)

sns.distplot(leak_df.meter_reading_l1p)



leak_score = np.sqrt(mean_squared_error(leak_df.pred4_l1p, leak_df.meter_reading_l1p))

print ('score4=', leak_score)
#simple_clean

sns.distplot(leak_df.pred5_l1p)

sns.distplot(leak_df.meter_reading_l1p)



leak_score = np.sqrt(mean_squared_error(leak_df.pred5_l1p, leak_df.meter_reading_l1p))

print ('score5=', leak_score)
all_combinations = list(np.linspace(0.15,0.4,15))

all_combinations
import itertools
l = [all_combinations, all_combinations, all_combinations, all_combinations, all_combinations]

# all_l = list(itertools.product(*l)) + list(itertools.product(*reversed(l)))

all_l = list(itertools.product(*l))
gc.collect()
filtered_combis = [l for l in all_l if l[0] + l[1] + l[2] + l[3] + l[4] > 0.95 and l[0] + l[1] + l[2] + l[3] + l[4] < 1.03]
print(len(filtered_combis))
best_combi = [] # of the form (i, score)

for i, combi in enumerate(filtered_combis):

    #print("Now at: " + str(i) + " out of " + str(len(filtered_combis))) # uncomment to view iterations

    score1 = combi[0]

    score2 = combi[1]

    score3 = combi[2]

    score4 = combi[3]

    score5 = combi[4]

    v = score1 * leak_df['pred1'].values + score2 * leak_df['pred3'].values + score3 * leak_df['pred2'].values + score4 * leak_df['pred4'] + score5 * leak_df['pred5']

    vl1p = np.log1p(v)

    curr_score = np.sqrt(mean_squared_error(vl1p, leak_df.meter_reading_l1p))

    

    if best_combi:

        prev_score = best_combi[0][1]

        if curr_score < prev_score:

            best_combi[:] = []

            best_combi += [(i, curr_score)]

    else:

        best_combi += [(i, curr_score)]

            

score = best_combi[0][1]

print(score)
# test_df = pd.read_feather(root/'test.feather')

# kfold_submission = pd.read_csv('../input/ashrae-kfold-lightgbm-without-leak-1-08/submission.csv', index_col=0)

# halfhalf_submission = pd.read_csv('../input/ashrae-half-and-half/submission.csv', index_col=0)

# highway_submission = pd.read_csv('../input/ashrae-highway-kernel-route4/submission.csv', index_col=0)



# test_df["pred1"] = kfold_submission.meter_reading

# test_df["pred2"] = halfhalf_submission.meter_reading

# test_df["pred3"] = highway_submission.meter_reading



# del kfold_submission, halfhalf_submission, highway_submission

# gc.collect()



# test_df.loc[test_df.pred3 < 0, "pred3"] = 0

# test_df = reduce_mem_usage(test_df)
submission_form = pd.read_feather(os.path.join(root, 'sample_submission.feather'))

final_combi = filtered_combis[best_combi[0][0]]

w1 = final_combi[0]

w2 = final_combi[1]

w3 = final_combi[2]

w4 = final_combi[3]

w5 = final_combi[4]

print("Best weight is w1(kfold):{}, w2(halfhalf):{}, w3(highway):{}, w4(gep3):{}, w5(simple):{}".format(w1, w2, w3, w4, w5))



submission_form['meter_reading'] = w1 * test_df.pred1 +  w2 * test_df.pred3  + w3 * test_df.pred2 + w4 * test_df.pred4 + w5 * test_df.pred5

submission_form.loc[submission_form.meter_reading < 0, 'meter_reading'] = 0
sns.distplot(np.log1p(submission_form.meter_reading))
gc.collect()
leak_df = leak_df[['meter_reading', 'row_id']].set_index('row_id').dropna()

print(len(leak_df))
submission_form.loc[leak_df.index, 'meter_reading'] = leak_df['meter_reading']
sns.distplot(np.log1p(submission_form.meter_reading))
gc.collect()
submission_form.to_csv('submission.csv', index=False, float_format='%.4f')