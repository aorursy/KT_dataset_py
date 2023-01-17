# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import MeanShift
from sklearn.naive_bayes import GaussianNB
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
pd.options.mode.chained_assignment=None
pd.set_option('display.max_columns', None)
df=pd.read_csv('../input/ks-projects-201801.csv')
print(df.head(5))

# Any results you write to the current directory are saved as output.
df.info()  #to see columns and datetypes.
df.corr() #to see correlations
print(df.groupby('category').size())
categories = df.category.value_counts()
plt.figure(figsize=(15, 5))
sns.barplot(x=categories[:15].index, y=categories[:15].values)
plt.ylabel('number of projects')
plt.xlabel('categories')
plt.title('main categories')
print(df.groupby('state').size())
categories2=df.state.value_counts()
plt.figure(figsize=(15, 5))
sns.barplot(x=categories2[:6].index, y=categories2[:6].values)
plt.ylabel('number')
plt.xlabel('state')
plt.title('state of projects')
print(df.groupby('country').size())
categories3=df.country.value_counts()
plt.figure(figsize=(15, 5))
sns.barplot(x=categories3[:23].index, y=categories3[:23].values)
plt.ylabel('number')
plt.xlabel('countries')
plt.title('number of countries in data')
lab_en=LabelEncoder()#turns object to number.  it is easier to deal with number for machine learning.
df['main_category']=lab_en.fit_transform(df['main_category'].fillna('0'))
X=df['main_category']
df['state']=lab_en.fit_transform(df['state'].fillna('0'))
Y=df['state']
X=np.array(X).reshape(len(X), 1)
Y=np.array(Y).reshape(len(Y), 1)
x_train, x_test, y_train, y_test=train_test_split(X, Y, test_size=0.2)
knc=KNeighborsClassifier()
knc.fit(x_train, y_train)
score=knc.score(x_train, y_train)
print(score)
#X2=knc.predict(x_test)
#print("real: ", x_test,"predicted:", X2)


gaus=GaussianNB()
gaus.fit(x_train, y_train)
scr=gaus.score(x_train, y_train)
print(scr)
