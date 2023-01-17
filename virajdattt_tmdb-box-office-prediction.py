# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np

from scipy.stats import norm

from scipy import stats

import warnings

warnings.filterwarnings(action="ignore")



#print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from sklearn.model_selection import train_test_split
#Reading data

df = pd.read_csv("../input/train.csv")
df.shape
#Let us explore the dataset 

df.head(4)


sns.distplot((df['revenue']))

plt.show()


sns.distplot((np.log(df['revenue'])))

plt.show()
#skewness and kurtosis

print("Skewness: %f" % df['revenue'].skew())

print("Kurtosis: %f" % df['revenue'].kurt())
#scatter plot budget/revenue

var = 'budget'

data = pd.concat([df['revenue'],df[var]], axis=1)

data.plot.scatter(x=var, y='revenue');
#scatter plot budget/revenue

var = 'budget'

data = pd.concat([np.log(df['revenue']), np.log(df[var])], axis=1)

data.plot.scatter(x=var, y='revenue');
#df['budget'].replace({0:df['budget'].mean()}, inplace=True)
#histogram and normal probability plot

sns.distplot(df['revenue'], fit=norm);

fig = plt.figure()

res = stats.probplot(df['revenue'], plot=plt)
#histogram and normal probability plot

sns.distplot(np.log(df['revenue']), fit=norm);

fig = plt.figure()

res = stats.probplot(df['revenue'], plot=plt)


X_train, X_test, y_train, y_test = train_test_split(df['budget'], df['revenue'], test_size=0.2, random_state=101)
from sklearn.linear_model import LinearRegression

lm = LinearRegression()

lm.fit(X_train.values.reshape(-1, 1),y_train.values.reshape(-1, 1))

#lm.fit(np.log(df['budget'].values.reshape(-1, 1)),np.log(df['revenue'].values.reshape(-1, 1)))
lm.score(X_test.values.reshape(-1, 1), y_test.values.reshape(-1, 1))
test = pd.read_csv("../input/test.csv")

predctions = lm.predict(test['budget'].values.reshape(-1, 1))
#lm.fit(np.log(X_train.values.reshape(-1, 1)),np.log(y_train.values.reshape(-1, 1)))

#lm.score(np.log(X_test.values.reshape(-1, 1)), np.log(y_test.values.reshape(-1, 1)))
X_train, X_test, y_train, y_test = train_test_split(df[['popularity','budget']], df['revenue'], test_size=0.2, random_state=101)


lm_2 = LinearRegression()

lm_2.fit(X_train,y_train)
lm_2.score(X_test,y_test)