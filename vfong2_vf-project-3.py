# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

df = pd.read_csv("../input/train.csv") 

df_test = pd.read_csv("../input/test.csv")



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
#Print all rows and columns. Dont hide any

##df.drop = ('id', axis = 1)

pd.set_option('display.max_rows', None)

pd.set_option('display.max_columns', None)

df.head(10)
##identify the columns with the most parameters and apply factorization to them (code from Alexandru)

df.iloc[:,1:116].apply(lambda x: pd.unique(x).shape[0]).sort_values(ascending = False).head(10)
train_cat_large = df[["cat116", "cat110","cat109","cat113", "cat112", "cat115"]]

df_frac = df.drop(train_cat_large, axis = 1)

X_int_cat = train_cat_large.apply(lambda x: pd.factorize(x)[0])

df = pd.concat([df_frac, X_int_cat], axis = 1)

#new dataframe factorizes the columns with most variables
df.head()
df = pd.get_dummies(df, dummy_na = False, drop_first = True)

df.tail()

print(df.shape)
col = list(df.columns)

#numcol = col[:15]

df['log_loss'] = np.log1p(df['loss'])

df.head()
import seaborn as sns

sns.lmplot('cont3' , 'log_loss', data = df)
from sklearn import datasets

from sklearn.cross_validation import train_test_split

from sklearn import linear_model

import statsmodels.api as sm
##Split data and log transform

local_train, local_test = train_test_split(df,test_size=0.2,random_state=123)

df_y_train = np.logp1(local_train['loss'])

df_x_train = local_train.drop(["loss"],axis=1)

df_y_test = np.logp1(local_test['loss'])

df_x_test = local_test.drop(["loss"],axis=1)
##Training on linear Regression

clf = sm.OLS(df_y_train, df_x_train)

result = clf.fit()

preds = result.predict(df_x_test)

preds