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
my_data = pd.read_csv("../input/diabetes/diabetes.csv")
my_data.head()
my_data.shape
outcome_true = len(my_data.loc[my_data['Outcome'] == 1])

outcome_true
outcome_false = len(my_data.loc[my_data['Outcome']==0])

outcome_false
# Checking for Null values

my_data.apply(lambda x: x.isnull().sum())
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
corrmat = my_data.corr()

top_corr_features = corrmat.index

plt.figure(figsize=(20,10))

g = sns.heatmap(my_data[top_corr_features].corr(),annot=True)
x = my_data.iloc[:,:-1].values

y = my_data.iloc[:,8].values
my_data.hist(figsize=(10,10))
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

x = sc.fit(x).transform(x)

x
from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression,LinearRegression

from sklearn import metrics
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)
regressor1 = DecisionTreeRegressor(random_state=0)

regressor1.fit(x_train,y_train)
y1_pred = regressor1.predict(x_test)

y1_pred
#Accuracy score for decision tree regressor

print("Accuracy_score: ",metrics.accuracy_score(y_test,y1_pred))
regressor2 = LogisticRegression(C=0.01,solver='liblinear')

regressor2.fit(x_train,y_train)
y2_pred = regressor2.predict(x_test)

y2_pred
#Predict Proba

y2_pred_proba = regressor2.predict_proba(x_test)

y2_pred_proba
#Accuracy score

print("Accuracy Score: ",metrics.accuracy_score(y_test,y2_pred))
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=21,random_state=40,bootstrap=True)

rf.fit(x_train,y_train)
y3_pred = rf.predict(x_test)

y3_pred
#Accuracy score

print("Accuracy score: ",metrics.accuracy_score(y_test,y3_pred))