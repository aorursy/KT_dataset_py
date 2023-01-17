import numpy as np

import pandas as pd

import random

import missingno as msno 

from sklearn.model_selection import KFold

import lightgbm as lgb

from sklearn.metrics import classification_report

from sklearn.ensemble import RandomForestClassifier

from sklearn import preprocessing



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv('../input/dollar-prices-and-infos/database_15min.csv')

df.head()
df = df.drop(['21-11-19 09:03'], axis=1) # remove columns useless to the model

df.head(5)
df.corr()
import matplotlib.pyplot as plt

import itertools

columns=df.columns[:8]

plt.subplots(figsize=(18,15))

length=len(columns)

for i,j in itertools.zip_longest(columns,range(length)):

    plt.subplot((length/2),3,j+1)

    plt.subplots_adjust(wspace=0.2,hspace=0.5)

    df[i].hist(bins=20,edgecolor='black')

    plt.title(i)

plt.show()
import seaborn as sns

sns.heatmap(df[df.columns[:]].corr(),annot=True,cmap='RdYlGn')

fig=plt.gcf()

fig.set_size_inches(10,8)

plt.show()