import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
data=pd.read_csv('../input/advertising/advertising.csv')
data.head()
data.info()
data.describe()
sns.set_style('whitegrid')

data['Age'].hist(bins=30)

plt.xlabel('Age')
sns.jointplot('Age','Area Income',data=data)
sns.jointplot('Age','Daily Time Spent on Site',data=data,kind='kde',color='red')
sns.jointplot(x='Daily Time Spent on Site',y='Daily Internet Usage',data=data,color='green')
sns.pairplot(data,hue='Clicked on Ad',palette='bwr')
from sklearn.model_selection import train_test_split
X = data[['Daily Time Spent on Site', 'Age', 'Area Income','Daily Internet Usage', 'Male']]

y = data['Clicked on Ad']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
from sklearn.linear_model import LogisticRegression
lm=LogisticRegression()

lm.fit(X_train,y_train)
predict=lm.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_test,predict))