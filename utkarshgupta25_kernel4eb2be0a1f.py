# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split as tts

from sklearn.linear_model import LogisticRegression as lr

from sklearn.linear_model import LinearRegression as lir



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/glass.data', header = None)



df.columns = ['id', 'ri', 'na', 'mg', 'al', 'si', 'k', 'ca', 'ba', 'fe', 'glass_type']



df.head()
df['glass_type'].unique()
df.isna().sum()
df.corr()
df.describe() #We observe the mean values and the std dev tells us that how spread it is. Hence very helpful
for i in df.drop(columns = ['id', 'glass_type']):

    sns.regplot(df['glass_type'], df[i])

    plt.show()
for i in df.drop(columns = ['id', 'glass_type']):

    sns.boxplot(df['glass_type'], df[i])

    plt.show()
def plotBarChart(data,col,label):

    g = sns.FacetGrid(data, col=col)

    g.map(plt.hist, label, bins=10)



for val in df.drop(columns = ['id', 'glass_type']):

    plotBarChart(df,'glass_type',val)   
df['glass_type_final'] = df['glass_type'].apply(lambda x: 0 if (x >= 1 and x < 4) else 1)
def plotBarChart(data,col,label):

    g = sns.FacetGrid(data, col=col)

    g.map(plt.hist, label, bins=10)



for val in df.drop(columns = ['id', 'glass_type', 'glass_type_final']):

    plotBarChart(df,'glass_type_final',val)   
for i in df.drop(columns = ['id', 'glass_type', 'glass_type_final']):

    sns.boxplot(df['glass_type_final'], df[i])

    plt.show()
x = df.drop(columns = ['id', 'glass_type', 'glass_type_final'])

y = df['glass_type']



#x_train, x_test, y_train, y_test = tts(x,y, random_state = 17)



regressor = lr()



regressor.fit(x,y)



df['predicted'] = regressor.predict(x)



regressor.score(x,y)
sns.regplot(df['glass_type_final'], df['predicted'])
from sklearn.tree import DecisionTreeClassifier as dtc



regressor = dtc()



x_train, x_test, y_train, y_test = tts(x,y,random_state = 16)



regressor.fit(x_train,y_train)



regressor.score(x_test ,y_test)