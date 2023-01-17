import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
df = pd.read_table('../input/adult.data.txt',delimiter=',', names=('age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country','class'))
import numpy as np
sns.heatmap(df.isnull())
df['workclass_num'] = df.workclass.map({' Private':0, ' State-gov':1, ' Federal-gov':2, ' Self-emp-not-inc':3, ' Self-emp-inc':4, ' Local-gov':5, ' Without-pay':6})

df['marital_num'] = df['marital-status'].map({' Widowed':0, ' Divorced':1, ' Separated':2, ' Never-married':3, ' Married-civ-spouse':4, ' Married-AF-spouse':4, ' Married-spouse-absent':5})

df['race_num'] = df.race.map({' White':0, ' Black':1, ' Asian-Pac-Islander':2, ' Amer-Indian-Eskimo':3, ' Other':4})

df['sex_num'] = np.where(df.sex == ' Female', 0, 1)

df['rel_num'] = df.relationship.map({' Not-in-family':0, ' Unmarried':0, ' Own-child':0, ' Other-relative':0, ' Husband':1, ' Wife':1})

df.head()
pd.get_dummies(df['class']).head()
df['class'] = pd.get_dummies(df['class'],drop_first=True)
sns.heatmap(df.isnull())
df.dropna(inplace=True)
X = df[['age','fnlwgt','education-num','workclass_num', 'marital_num', 'race_num', 'sex_num', 'rel_num', 'capital-gain', 'capital-loss','hours-per-week']]
y= df['class']
sns.heatmap(X.isnull())
X.head(3) #our features
from sklearn.linear_model import LogisticRegression

from sklearn.feature_selection import RFE #recursive feature elimination
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.33, random_state=5)
logreg = LogisticRegression()
rfe = RFE(logreg,3)

rfe = rfe.fit(X_train,y_train)
print(rfe.support_)

print(rfe.ranking_)
#lets drop second column with 9th rank

X.drop('fnlwgt',axis=1,inplace=True)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.33, random_state=5)
logreg.fit(X_train,y_train)
from sklearn.metrics import accuracy_score
logreg.score(X_train,y_train)
logreg.score(X_test,y_test)
y_pred = logreg.predict(X_test)
print(accuracy_score(y_test,y_pred))
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report,confusion_matrix
kf = KFold(n_splits=10)



print(cross_val_score(logreg, X, y, cv=kf).mean())
skf = StratifiedKFold(n_splits=5)
for train_index, test_index in skf.split(X, y):

    X_train, X_test = X.iloc[train_index], X.iloc[test_index]

    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    logmodel = LogisticRegression()

    logmodel.fit(X_train, y_train)

    predictions = logmodel.predict(X_test)

    print(classification_report(y_test, predictions))
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.33, random_state=5)

print(confusion_matrix(y_test,y_pred))
df['class'].value_counts()
23068/7650
logreg_p = LogisticRegression(class_weight={0:1,1:3})
logreg_p.fit(X_train,y_train)
y_pred2 = logreg_p.predict(X_test)
confusion_matrix(y_test,y_pred2)