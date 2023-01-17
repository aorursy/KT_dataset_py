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
df = pd.read_csv("../input/testdataset/bikeshare.csv")



print (df.isna().sum())

print (df.isna().any())



print (df[['registered', 'casual', 'count']].corr())
import matplotlib as mpl

import matplotlib.pyplot as plt



for i in df.columns[1:-1]:

    plt.scatter(df[i], df['count'])

    plt.show()

    print(i)

for i in df:

    print (i)
from sklearn.linear_model import LinearRegression as lr

from sklearn.model_selection import train_test_split as tts



x = df[['casual', 'registered']]

y = df['count'].values



x_train, x_test, y_train, y_test = tts(x, y, random_state = 2)



print (y_test)

from sklearn import metrics



regressor = lr()

regressor.fit(x_train, y_train)



predicted_values = regressor.predict(x_test)



mse = metrics.mean_squared_error(y_test, predicted_values)

print (predicted_values, y_test)



print (mse)



plt.scatter(predicted_values, y_test)

plt.show()



import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



df1 = pd.read_csv('../input/testdata1/clean_df.csv')



df1.corr()



sns.regplot(df1['peak-rpm'], df1['price'])



sns.boxplot(df1['body-style'], df1['price'])
for i in df1:

    if i != 'price':

        if len(df1[i].unique()) < 10 or df1[i].dtype == 'object':

            sns.boxplot(df1[i], df1['price'])

            plt.show()

        else:

            sns.regplot(df1[i], df1['price'])

            plt.show()

        