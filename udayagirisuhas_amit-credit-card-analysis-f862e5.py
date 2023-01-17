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
### Q1) Read the dataset into the notebook

cfd=pd.read_csv("../input/creditcard.csv")

cfd.head()
cfd.info()
### Q2) Print the shape of the data

cfd.shape
### Q3) List out the feature variables and their data-types

cfd.iloc[:,:30].info()
import seaborn as sns

import matplotlib.pyplot as plt

% matplotlib inline
### Q7) Check for outliers in the feature variables

fig= plt.figure(figsize=(15,100))

cols = cfd.columns



fig.subplots_adjust(hspace=0.5, wspace=0.4)

for i in range(1,30):

    ax = fig.add_subplot(10,3,i)

    sns.boxplot(x=cfd[cols[i-1]])

    plt.title(cols[i-1])
cfd.iloc[:,:30].describe()
q1 = cfd.iloc[:,:30].quantile(0.25)

q3 = cfd.iloc[:,:30].quantile(0.75)

q1
q3
iqr = q3 - q1
iqr
lower_bound = q1 -(1.5 * iqr) 

upper_bound = q3 +(1.5 * iqr) 
lower_bound
upper_bound
print(((cfd< (q1 - 1.5 * iqr)) |(cfd > (q3 + 1.5 * iqr))))
cfd1 = cfd[~((cfd< (q1 - 1.5 * iqr)) |(cfd > (q3 + 1.5 * iqr))).any(axis=1)]

cfd1.shape
cfd1.info()
cfd1.hist(figsize=(20,20))

plt.show()
fig= plt.figure(figsize=(15,100))

cols = cfd1.columns



fig.subplots_adjust(hspace=0.5, wspace=0.4)

for i in range(1,30):

    ax = fig.add_subplot(10,3,i)

    sns.boxplot(x=cfd1[cols[i-1]])

    plt.title(cols[i-1])
cfd2=cfd1.corr()

cfd2
plt.figure(figsize=(20,20))

sns.heatmap(cfd2)
from sklearn.model_selection import train_test_split
x = cfd1.iloc[:,:30]

y = cfd1['Class']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()

logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))