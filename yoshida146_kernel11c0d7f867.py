import pandas as pd

import numpy as np

import os

import joblib

import matplotlib.pyplot as plt

import seaborn as sns

from tqdm.notebook import tqdm

sns.set()

sns.set_context('poster')
_input_path = os.path.join('..', 'input', '1056lab-brain-cancer-classification')

os.listdir(_input_path)
target_col = 'type'
df_train = pd.read_csv(os.path.join(_input_path, 'train.csv'), index_col=0)

df_test = pd.read_csv(os.path.join(_input_path, 'test.csv'), index_col=0)
df_train.shape
target = df_train[target_col].copy()

df_train.drop(target_col, axis=1, inplace=True)
rep_target = {

    'normal': 0,

    'ependymoma': 1,

    'glioblastoma': 2,

    'medulloblastoma': 3,

    'pilocytic_astrocytoma': 4

}



target.replace(rep_target, inplace=True)

target = target.astype(np.int8)
df_train.shape
def reduce_mem_usage(df, verbose=True):

    """ iterate through all the columns of a dataframe and modify the data type

        to reduce memory usage.        

    """

    start_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    

    for col in tqdm(df.columns, disable=not verbose):

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
df_train.info()
df_test.info()
# df_train_ = reduce_mem_usage(df_train)

# print('\n\n')

# df_test_ = reduce_mem_usage(df_test)
df_train.head()
df_test.head()
sns.countplot(target)
train_var = df_train.var()
train_var.median()
df_train.columns[df_train.var() >= 0.3]
df_train.isnull().sum().sort_values()
df_test.isnull().sum().sort_values()
import lightgbm as lgb

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

from sklearn.model_selection import cross_validate



lgb_pipe = Pipeline([

    ('sc', StandardScaler()),

    ('lgb', lgb.LGBMClassifier())

])
print('LightGBM Crossvalidation: ', end='')

cross_validate(lgb_pipe, df_train, target, scoring='f1_micro', cv=5)['test_score'].mean()
from sklearn.naive_bayes import GaussianNB

from sklearn.multiclass import OneVsRestClassifier



gnb_pipe = Pipeline([

    ('sc', StandardScaler()),

    ('clf', OneVsRestClassifier(GaussianNB()))

])



print('Naive Bayes Crossvalidation: ', end='')

cross_validate(lgb_pipe, df_train, target, scoring='f1_micro', cv=5)['test_score'].mean()
# lgb_pipe.fit(df_train, target)
import lightgbm as lgb

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

from sklearn.model_selection import cross_validate



lgb_pipe = Pipeline([

    ('sc', StandardScaler()),

    ('lgb', lgb.LGBMClassifier())

])
lgb_pipe.fit(df_train, target)
predict = lgb_pipe.predict(df_test)
df_submit = pd.read_csv(os.path.join(_input_path, 'sampleSubmission.csv'), index_col=0)

df_submit[target_col] = predict

df_submit.to_csv('lgb_submit.csv')
from sklearn.naive_bayes import GaussianNB

from sklearn.multiclass import OneVsRestClassifier



gnb_pipe = Pipeline([

    ('sc', StandardScaler()),

    ('clf', OneVsRestClassifier(GaussianNB()))

])
gnb_pipe.fit(df_train.values, target)

gnb_predict = gnb_pipe.predict(df_test)
df_submit = pd.read_csv(os.path.join(_input_path, 'sampleSubmission.csv'), index_col=0)

df_submit[target_col] = gnb_predict

df_submit.to_csv('gnb_submit.csv')