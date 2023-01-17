# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



print(os.listdir("../input/"))
win = pd.read_csv("../input/red-wine-quality-cortez-et-al-2009/winequality-red.csv")
win.head()
win.describe()
win.dtypes
plt.figure(figsize=(20,16))

i=1

for col in win.columns[:-1] :

    plt.subplot(4,3,i)

    b=sns.barplot(x='quality',y=col,data=win)

    b.set_xlabel("quality",fontsize=18)

    b.set_ylabel(col,fontsize=20)

    i = i+1
plt.figure(figsize=(20,16))

i=1

for col in win.columns[:-1] :

    plt.subplot(4,3,i)

    k=sns.boxplot(x='quality',y=col,data=win)

    k.set_xlabel("quality",fontsize=18)

    k.set_ylabel(col,fontsize=20)

    i = i+1
quality = win["quality"].values

category = []

for num in quality:

    if num<=5:

        category.append(0)

    elif num>=6:

        category.append(1)
category = pd.DataFrame(data=category, columns=["category"])

data = pd.concat([win,category],axis=1)

data.drop(columns="quality",axis=1,inplace=True)

data.head()
plt.figure(figsize=(10,6))

c=sns.countplot(data["category"],palette="muted")

c.set_xlabel("Category",fontsize=18)

c.set_ylabel("Count",fontsize=20)

data["category"].value_counts()
x=data.drop('category',axis=1)

y=data['category']
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3,random_state=53)
from sklearn.ensemble import RandomForestClassifier

clf=RandomForestClassifier(oob_score=True,n_estimators=400,random_state=53)

clf.fit(x_train,y_train)
clf.score(x_test,y_test)
clf.oob_score_
for w in range(400,600,30):

    clf=RandomForestClassifier(oob_score=True,n_estimators=w,random_state=53)

    clf.fit(x_train,y_train)

    oob=clf.oob_score_

    print ('For n_estimators = '+str(w))

    print ('oob score is'+str(oob))

    print ('********************')
clf=RandomForestClassifier(oob_score=True,n_estimators=490,random_state=53)

clf.fit(x_train,y_train)

clf.score(x_test,y_test)
clf.oob_score_
clf.feature_importances_
imp=pd.Series(clf.feature_importances_,index=x.columns.tolist())

imp.sort_values(ascending=False)
imp.sort_values(ascending=False).plot(kind='bar')

plt.rc('xtick', labelsize=20) 

plt.rc('ytick', labelsize=20)