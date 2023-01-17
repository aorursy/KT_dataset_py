# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df=pd.read_csv('../input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')
df.head()
df['quality'].unique()
df.info()
df.describe()
df.corr()
bins = [2,4,6,9]

labels= ['bad','medium','good']

df['quality']=pd.cut(df['quality'],bins=bins, labels=labels)

df.head()
df['quality'].replace('medium',1,inplace=True)

df['quality'].replace('good',2,inplace=True)

df['quality'].replace('bad',0,inplace=True)
df['quality'].head()
fig, ax = plt.subplots(figsize=(15,7))

sns.heatmap(df.corr(),annot=True)
x = df.drop('quality',axis=1)

y=df['quality']
x.head()

y.head()
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3)
from sklearn.linear_model import LogisticRegression
LR = LogisticRegression()

LR.fit(x_train,y_train)
predictions = LR.predict(x_test)
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,predictions))

confusion_matrix(y_test,predictions)
from sklearn.neighbors import KNeighborsClassifier
error_rate = []

for i in range(1,40):    

    knn = KNeighborsClassifier(n_neighbors=i)

    knn.fit(x_train,y_train)

    pred_i = knn.predict(x_test)

    error_rate.append(np.mean(pred_i != y_test))

    #print(error_rate)                                                  



plt.figure(figsize=(10,6))

plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', 

         marker='o',  markerfacecolor='red', markersize=10)

plt.title('Error Rate vs. K Value')

plt.xlabel('K')

plt.ylabel('Error Rate')
knn = KNeighborsClassifier(n_neighbors=12)
knn.fit(x_train,y_train)

pred = knn.predict(x_test)
print(classification_report(y_test,pred))

print(confusion_matrix(y_test,pred))