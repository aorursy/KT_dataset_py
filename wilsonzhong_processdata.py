import pandas as pd

import numpy as np

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

import lightgbm as lgb

%matplotlib inline

import matplotlib.pyplot as plt

import os

from sklearn.preprocessing import LabelEncoder

from scipy.sparse import csr_matrix, hstack

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import log_loss

import warnings

from collections import Counter

from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")
from pandarallel import pandarallel

pandarallel.initialize()
age_train = pd.read_csv('../input/huawei/age_train.csv', header=None)

age_test = pd.read_csv('../input/huawei/age_test.csv', header=None)

userInfo = pd.read_csv('../input/huawei/user_basic_info.csv', header=None)

userBehavior = pd.read_csv('../input/huawei/user_behavior_info.csv', header=None)

userAppUsed = pd.read_csv('../input/processedinput/user_app_class_record.csv')

userAppDuration = pd.read_csv('../input/huaweiprocessed/user_app_duration.csv')

userAppTimes = pd.read_csv('../input/huaweiprocessed/user_app_times.csv')
userInfo.columns=['uId', 'gender', 'city', 'prodName', 'ramCapacity', 'ramLeftRation', 'romCapacity', 'romLeftRation', 'color', 'fontSize', 'ct', 'carrier', 'os']

age_train.columns=['uId', 'age_group']

age_test.columns=['uId']

userBehavior.columns=['uId', 'bootTimes', 'AFuncTimes', 'BFuncTimes', 'CFuncTimes', 'DFuncTimes', 'EFuncTimes', 'FFuncTimes', 'FFuncSum']
#将user_basic_info.csv 和 user_behavior_info.csv中的字符值编码成可以训练的数值类型，合并

class2id = {}

id2class = {}

def mergeBasicTables(baseTable):

    resTable = baseTable.merge(userInfo, how='left', on='uId', suffixes=('_base0', '_ubaf'))

    resTable = resTable.merge(userBehavior, how='left', on='uId', suffixes=('_base1', '_ubef'))

    cat_columns = ['city','prodName','color','carrier','os','ct']

    for c in cat_columns:

        resTable[c] = resTable[c].apply(lambda x: x if type(x)==str else str(x))

        sort_temp = sorted(list(set(resTable[c])))  

        class2id[c+'2id'] = dict(zip(sort_temp, range(1, len(sort_temp)+1)))

        id2class['id2'+c] = dict(zip(range(1,len(sort_temp)+1), sort_temp))

        resTable[c] = resTable[c].apply(lambda x: class2id[c+'2id'][x])

        

    return resTable
trainData = mergeBasicTables(age_train)

testData = mergeBasicTables(age_test)

trainLabel = age_train.age_group
for column in list(trainData.columns[trainData.isnull().sum() > 0]):

    mean_val = trainData[column].mean()

    trainData[column].fillna(mean_val, inplace=True)

for column in list(testData.columns[testData.isnull().sum() > 0]):

    mean_val = testData[column].mean()

    testData[column].fillna(mean_val, inplace=True)
trainData.head()
trainData = trainData.merge(userAppUsed, on='uId', how='left')

print(trainData.isnull().sum())

testData = testData.merge(userAppUsed, on='uId', how='left')

print(testData.isnull().sum())

for column in list(trainData.columns[trainData.isnull().sum() > 0]):

    trainData[column].fillna(0, inplace=True)

for column in list(testData.columns[testData.isnull().sum() > 0]):

    testData[column].fillna(0, inplace=True)
trainData = trainData.merge(userAppDuration, on='uId', how='left')

testData = testData.merge(userAppDuration, on='uId', how='left')

for column in list(trainData.columns[trainData.isnull().sum() > 0]):

    median_val = trainData[column].median()

    trainData[column].fillna(median_val, inplace=True)

for column in list(testData.columns[testData.isnull().sum() > 0]):

    median_val = testData[column].median()

    testData[column].fillna(median_val, inplace=True)
trainData = trainData.merge(userAppTimes, on='uId', how='left')

testData = testData.merge(userAppTimes, on='uId', how='left')

for column in list(trainData.columns[trainData.isnull().sum() > 0]):

    median_val = trainData[column].median()

    trainData[column].fillna(median_val, inplace=True)

for column in list(testData.columns[testData.isnull().sum() > 0]):

    median_val = testData[column].median()

    testData[column].fillna(median_val, inplace=True)
trainData.head()
trainData.drop(['age_group'], axis=1, inplace=True)
print(trainData.shape)

print(testData.shape)
trainData.head()
testData.head()
trainData.to_csv('trainFeatures.csv', index=False)

testData.to_csv('testFeatures.csv', index=False)