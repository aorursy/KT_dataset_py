# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



# pandas

import pandas as pd

from pandas import Series,DataFrame



# numpy, matplotlib, seaborn

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')

%matplotlib inline



# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')



train_df = train_df[['Id', 'LotArea', 'SalePrice', 'TotalBsmtSF', 'GrLivArea','GarageCars', 'YearBuilt','PoolArea', 'TotRmsAbvGrd', 'FullBath', 'HalfBath', 'KitchenAbvGr', 'BedroomAbvGr']]

train_df['TotalBuildArea'] = train_df['GrLivArea'] + train_df['TotalBsmtSF']

train_df['TotalBathAbvGr'] = train_df['FullBath'] + train_df['HalfBath']

train_df.drop(['FullBath', 'HalfBath'], axis=1, inplace=True)

#train_df['TotalBathBsmt'] = train_df['BsmtFullBath'] + train_df['BsmtHalfBath']

train_df.info()


fig, (axis1) = plt.subplots(1,1,figsize=(100,10))



#train_df['newRooms'] = train_df['TotRmsAbvGrd']-train_df['TotalBathAbvGr']+train_df['TotalBathBsmt']-train_df['BedroomAbvGr']

sns.pointplot(x='GrLivArea', y='SalePrice', data=train_df[['SalePrice', 'GrLivArea']], ax=axis1)

#train_df['newRooms'].plot(kind='line', ax=axis1, xlim=(0, 10))

#train_df['KitchenAbvGr'].plot(kind='line', ax=axis1, xlim=(0, 10))