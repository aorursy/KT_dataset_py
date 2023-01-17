# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

from warnings import filterwarnings

filterwarnings('ignore')

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
iris=pd.read_csv('../input/iris/Iris.csv',index_col='Id')

iris.head()
iris.info()
iris.Species.value_counts()
plt.pie(iris.Species.value_counts(),labels=iris.Species.unique(),autopct = '%1.2f%%')
y = iris.pop('Species') #Target

X = iris  #DataFrame with features



X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=10)

print(y_train.value_counts())

print(y_test.value_counts())
plt.figure(figsize=(10,15))

plt.subplot('121')

plt.pie(y_train.value_counts(),labels=y_train.unique(),autopct = '%1.2f%%',shadow = True)

plt.title('Training Dataset')



plt.subplot('122')

plt.pie(y_test.value_counts(),labels=y_test.unique(),autopct = '%1.2f%%', shadow =True)

plt.title('Test Dataset')



plt.tight_layout()
from sklearn.model_selection import StratifiedShuffleSplit 



splitter=StratifiedShuffleSplit(n_splits=1,random_state=12) #we can make a number of combinations of split

#But we are interested in only one.



for train,test in splitter.split(X,y):     #this will splits the index

    X_train_SS = X.iloc[train]

    y_train_SS = y.iloc[train]

    X_test_SS = X.iloc[test]

    y_test_SS = y.iloc[test]

print(y_train_SS.value_counts())  

print(y_test_SS.value_counts())
plt.figure(figsize=(10,15))



plt.subplot('121')

plt.pie(y_train_SS.value_counts(),labels=y_train_SS.unique(),autopct = '%1.2f%%')

plt.title('Training Dataset')



plt.subplot('122')

plt.pie(y_test_SS.value_counts(),labels=y_test_SS.unique(),autopct = '%1.2f%%')

plt.title('Test Dataset')



plt.tight_layout()
# Model 1 with stratified sample

model1=LogisticRegression()

model1.fit(X_train,y_train)

y_pred_m1=model1.predict(X_test)

acc_m1=accuracy_score(y_pred_m1,y_test)



print(acc_m1)
# Model 2 with stratified sample

model2=LogisticRegression()

model2.fit(X_train_SS,y_train_SS)

y_pred_m2=model1.predict(X_test_SS)

acc_m2=accuracy_score(y_pred_m2,y_test_SS)



print(acc_m2)
#visualizing result

plt.bar(['Random Split','Stratified split'],[acc_m1,acc_m2])

plt.title('Random vs Stratified split')