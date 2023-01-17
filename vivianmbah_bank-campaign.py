import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline 

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix

df = pd.read_csv("../input/bank-marketing-dataset/bank.csv")
df.info()
## to encode categorical data

df_sklearn = df.copy()

lb_make = LabelEncoder()



df_sklearn['job'] = lb_make.fit_transform(df['job'])

df_sklearn['marital'] = lb_make.fit_transform(df['marital'])

df_sklearn['education'] = lb_make.fit_transform(df['education'])

df_sklearn['default'] = lb_make.fit_transform(df['default'])

df_sklearn['housing'] = lb_make.fit_transform(df['housing'])

df_sklearn['loan'] = lb_make.fit_transform(df['loan'])

df_sklearn['contact'] = lb_make.fit_transform(df['contact'])

df_sklearn['month'] = lb_make.fit_transform(df['month'])

df_sklearn['poutcome'] = lb_make.fit_transform(df['poutcome'])

df_sklearn['deposit'] = lb_make.fit_transform(df['deposit'])



df_sklearn #Results in appending a new column to df

df = df_sklearn

df
#check for missing values

df.isnull().sum()
df.describe()
#to pick the best variables

import seaborn as sns

plt.figure(figsize=(20,10))

sns.heatmap(df_sklearn.corr(),annot=True,linewidths=.05);
X = df_sklearn.iloc[:,[11,13]].values  # predictor

y = df_sklearn.iloc[:,-1].values  #target attribute
# splitting into train and test data set

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state = 0)

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.fit_transform(X_test)
from sklearn.neighbors import KNeighborsClassifier

knn= KNeighborsClassifier(n_neighbors=5 , metric= 'minkowski', p=2)  #minkowski= distance formula and p=2 power 2 of distance formula

#fitting the knn model

knn.fit(X_train, y_train)



y_pred = knn.predict(X_test)

y_pred
from sklearn.metrics import accuracy_score

ac=accuracy_score(y_test,y_pred)

ac*100