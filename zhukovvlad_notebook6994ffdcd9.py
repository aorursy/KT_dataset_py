# Load in our libraries

import pandas as pd

import numpy as np

import re

import sklearn

import xgboost as xgb

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls



import warnings

warnings.filterwarnings('ignore')



# Going to use these 5 base models for the stacking

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier

from sklearn.svm import SVC

from sklearn.cross_validation import KFold;
# Load in the train and test datasets

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

full_data = [train,test]



# Store our passenger ID for easy access

PassengerId = test['PassengerId']
train.head(3)
train.info()
train[['Pclass','Survived']].groupby(['Pclass'], as_index=False).mean()
train[['Sex','Survived']].groupby(['Sex'],as_index=False).mean()
for dataset in full_data:

    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

print (train[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean())
train['FamilySize'].value_counts()