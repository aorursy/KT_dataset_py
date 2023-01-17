# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn import preprocessing
%matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/Social_Network_Ads.csv")
data.head()
#encode gender in numeric data
le = preprocessing.LabelEncoder()
data['Gender'] = le.fit_transform(data['Gender'])
#seperating features and labels
X = data.iloc[:, [2, 3]].values
y = data.iloc[:, 4].values
X_features = ['Gender','Age','EstimatedSalary']
y_features = ['Purchased']
X = data[X_features]
y = data[y_features]
#plotting the distribution of feature AGE.
sns.distplot(X['Age']);
sns.distplot(X['EstimatedSalary'], bins=10, kde=False, rug=True);
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_pred, y_test)
print(accuracy)
