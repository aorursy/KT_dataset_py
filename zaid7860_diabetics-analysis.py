# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/diabetes2.csv')
df
sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')
sns.heatmap(df.corr(),annot=True)
sns.scatterplot('Glucose','BMI',hue='Outcome',data=df)
sns.pairplot(df,hue='Outcome')
X = df.iloc[:,0:8]
X
y=df.iloc[:,-1:]
y
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.33, random_state=42)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train,y_train)
predict = lr.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,predict))
cm = confusion_matrix(y_test,predict)
sns.heatmap(cm,annot=True,cmap='viridis')