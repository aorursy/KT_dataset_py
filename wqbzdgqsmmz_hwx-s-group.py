# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import matplotlib.pyplot as plt

import seaborn as sns

import matplotlib as mpl

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.linear_model import LinearRegression

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/pubg-finish-placement-prediction"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/pubg-finish-placement-prediction/train_V2.csv")

test = pd.read_csv("../input/pubg-finish-placement-prediction/test_V2.csv")
train.columns
train.head()
feature = ["boosts","damageDealt","heals","kills","killStreaks","longestKill","walkDistance","weaponsAcquired"]

label = "winPlacePerc"
X_train = train[feature]
Y = train[label]
X_train.isnull().sum()
Y.isnull().sum()
Y = Y.fillna(0)
lm = LinearRegression()
lm.fit(X_train,Y)
lm.score(X_train,Y)
X_test = test[feature]
lm.predict(X_test)
result= lm.predict(X_test)
pd.DataFrame({'Id':test['Id'],'winPlacePerc':result}).to_csv('submission.csv',index=False)