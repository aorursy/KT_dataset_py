%matplotlib inline

import matplotlib.pyplot as plt

import numpy as np

import pandas as  pd

import seaborn as sns
#load data

df =pd.read_csv("../input/heart.csv")

df.info()
df.head()
df.describe()
#detect outlier (maximum heart rate = 220 - age)

max_thalach = 220 - 29

df['thalach'].loc[df['thalach'] > max_thalach]
#drop outlier

df.drop([72,103,125,248],inplace=True)
df.reset_index(inplace=True)

df.drop(['index'],axis = 1,inplace=True)
target = df['target']

df= df.drop(['target'],axis=1)
corr_matrix = df.corr()

plt.figure(figsize=(10, 8))

sns.heatmap(corr_matrix,cmap="PuBuGn",annot=True)
#normalize data

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X = np.array(df)

scaler.fit(X)
new_df = scaler.transform(X)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(new_df, target, test_size = 0.3)

train, _ = X_train.shape 

test,  _ = X_test.shape 

print (train, test)
# import warnings filter

from warnings import simplefilter

# ignore all future warnings

simplefilter(action='ignore', category=FutureWarning)
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import classification_report

knn = KNeighborsClassifier()

knn.fit(X_train, y_train)

print(classification_report(y_test,knn.predict(X_test)))

from sklearn.svm import SVC

clf = SVC(gamma='auto')

clf.fit(X_train, y_train)

print(classification_report(y_test,clf.predict(X_test)))

from sklearn.linear_model import LogisticRegression

logcl= LogisticRegression(penalty='l2')

logcl.fit(X_train, y_train)

print(classification_report(y_test,logcl.predict(X_test)))
import xgboost as xgb

from xgboost import XGBClassifier

model = XGBClassifier()

model=XGBClassifier(learning_rate=0.1,n_estimators=100)

model.fit(X_train, y_train)

print(classification_report(y_test,model.predict(X_test)))
from sklearn import ensemble

from sklearn.model_selection import GridSearchCV 

rf = ensemble.RandomForestClassifier()

rf.fit(X_train, y_train)

print(classification_report(y_test,rf.predict(X_test)))
importances = rf.feature_importances_

importances

plt.figure(figsize=(13, 5))

feature_names = df.columns

plt.title("Feature importances")

plt.bar(range(13), importances, align='center')

plt.xticks(range(13), np.array(feature_names), rotation=90)

plt.xlim([0, 13]);