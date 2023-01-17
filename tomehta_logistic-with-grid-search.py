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
import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV

import lightgbm as lgb

from sklearn.model_selection import train_test_split

#sns.set_palette("bright")

%matplotlib inline



train_df = pd.read_csv("../input/train.csv")

test_df = pd.read_csv("../input/test.csv")



# this function reduces the memory print for dataset. it helps since we are using gridsearch

def reduce_mem_usage(df, verbose=True):

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    start_mem = df.memory_usage().sum() / 1024**2

    for col in df.columns:

        col_type = df[col].dtypes

        if col_type in numerics:

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



    end_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))

    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))



    return df
train_df=reduce_mem_usage(train_df)

test_df=reduce_mem_usage(test_df)
train_df.describe()
train_df.head()

# drop the columns which are not required for modelling like ID. 

target = train_df["target"]

train_df = train_df.drop(["target","id"],axis=1)

test_id = test_df["id"]

test_df = test_df.drop(["id"],axis=1)



plt.figure(figsize=(8,6))

sns.distplot(target,kde=False,color='b')

plt.title("Target Distribution",fontsize=16)

plt.xlabel("Target Freq",fontsize=12)

plt.ylabel("Target",fontsize=12)



#looks none is correlated well with each other

plt.figure(figsize=(15,15))

sns.heatmap(train_df.corr(),cmap='viridis')

plt.title("Cor between target features")

plt.show()
# split the training into 0.75 train and 0.25 test for cross validation

from sklearn.model_selection import StratifiedShuffleSplit

shuffle_split = StratifiedShuffleSplit(test_size=0.25,train_size=0.75,n_splits=25)
param_grid = {'C'     : [1,0.01,0.1,10,100],

              'penalty' : ["l1", "l2"],

              'class_weight' : [{1:0.5, 0:0.5}, {1:0.4, 0:0.6}, {1:0.6, 0:0.4}, {1:0.7, 0:0.3}]

          }

grid_search = GridSearchCV(

    estimator = LogisticRegression(random_state=12,solver="liblinear"),

    param_grid = param_grid, 

     scoring='roc_auc',

    cv = shuffle_split

   )
grid_search.fit(train_df,target)
print("Best parameters : {}".format(grid_search.best_params_))

print("Best cross validation score: {:.2f}".format(grid_search.best_score_))

print("Best estimator: {}".format(grid_search.best_estimator_))

results = pd.DataFrame(grid_search.cv_results_)

results.head(10)
scores_mean = np.array(results.mean_test_score).reshape(-1)

scores_std = np.array(results.std_test_score).reshape(-1)



print("mean CV scores for each fold {} ".format(scores_mean))

print("std CV scores for each fold {} ".format(scores_std))

import gc

gc.collect()
#print( "Predictions on test set {}".format(grid_search.predict(test_df)))

pred_lr = grid_search.predict_proba(test_df)[:,1]

pred_lr[:-5]
#print test file 

sub_df = pd.DataFrame()

sub_df["id"] = test_id 

sub_df["target"] = pred_lr

sub_df.to_csv("baseline_lr.csv", index=False)
sub_df.head()