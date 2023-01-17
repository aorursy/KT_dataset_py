%matplotlib inline

import matplotlib.pyplot as plt

import numpy as np

import pandas as  pd

import seaborn as sns
train = pd.read_csv("../input/train.csv")

train.shape
test = pd.read_csv("../input/test.csv")

test.shape
data = train.append(test, sort=False)
data.info()
data.head()
data.describe()
corr_matrix = data.corr()

plt.figure(figsize=(6, 4))

sns.heatmap(corr_matrix,cmap="PuBuGn",annot=True)
data.loc[data['Pclass'].isna() == True].index.shape
data[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().plot.bar('Pclass')
data.loc[data['Age'].isna() == True].index.shape
data[['Age', 'Survived']].groupby(['Age'],as_index=False).mean().plot.scatter('Age','Survived')

plt.show()
data['Age'].fillna(data['Age'].mean(),inplace = True)
data['Family'] = data['Parch'] + data['SibSp'] + 1
data.loc[data['Family'].isna() == True].index.shape
data[['Family', 'Survived']].groupby(['Family'], as_index=False).mean().plot.bar('Family')
data.loc[data['Fare'].isna() == True]
value = data.Fare.loc[ (data.Pclass == 3) & (data.Embarked == 'S')].mean()

data['Fare'].fillna(value,inplace = True)
data[['Fare', 'Survived']].groupby(['Fare'],as_index=False).mean().plot.scatter('Fare','Survived')
data.describe(include='object')
data.loc[data['Sex'].isna() == True].index.shape
data[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().plot.bar('Sex')
data.loc[data['Name'] == 'Connolly, Miss. Kate']
data.loc[data['Embarked'].isna() == True]
data['Embarked'].fillna('S',inplace = True)
data[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().plot.bar('Embarked')
data.loc[data['Cabin'].isna() == True].index.shape
data.drop(labels='Cabin', axis=1,inplace=True)
data.loc[data['Ticket'].isna() == True].index.shape
data['Ticket'].head(15)
data.Ticket = data.Ticket.map(lambda x: x[0])

data.Ticket.unique()
data[['Ticket', 'Survived']].groupby(['Ticket'], as_index=False).mean()
data['Ticket'].value_counts()
data['Ticket'].replace(['W','F','A','7','4','6','L','5','9','8'], '4',inplace=True)
data.Ticket.unique()
data.columns
data.drop(labels=['PassengerId','Name','SibSp','Parch'], axis=1,inplace=True)
target = data['Survived']

data.drop(['Survived'],axis = 1,inplace = True)
categ_features =  [f for f in data.columns if data[f].dtype.name == 'object']

categ_features
data['Sex'].replace({'female': 1, 'male': 0},inplace=True)
data['Embarked'].unique()
data['Embarked'].replace({'S': 1, 'C': 2, 'Q' : 3},inplace=True)
data = pd.get_dummies(data,columns=['Ticket'])
data.info()
data.shape
X_train = data[0:891]

X_test  = data[891:]

y_train = target[0:891]

X_train.shape
score=[]
from lightgbm import LGBMClassifier

from sklearn.metrics import classification_report,accuracy_score



lgbm_clf = LGBMClassifier()

lgbm_clf.fit(X_train, y_train)

score.append(accuracy_score(y_train,lgbm_clf.predict(X_train)))

print(classification_report(y_train,lgbm_clf.predict(X_train)))
import xgboost as xgb

from xgboost import XGBClassifier

from sklearn.metrics import classification_report,accuracy_score

xgb=XGBClassifier(learning_rate=0.1,n_estimators=100)

xgb.fit(X_train, y_train)

score.append(accuracy_score(y_train,xgb.predict(X_train)))

print(classification_report(y_train,xgb.predict(X_train)))
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()

rf.fit(X_train, y_train)

score.append(accuracy_score(y_train,rf.predict(X_train)))

print(classification_report(y_train,rf.predict(X_train)))
from sklearn.svm import SVC

svc = SVC(gamma='auto')

svc.fit(X_train, y_train)

score.append(accuracy_score(y_train,svc.predict(X_train)))

print(classification_report(y_train,svc.predict(X_train)))
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import classification_report

knn = KNeighborsClassifier()

knn.fit(X_train, y_train)

score.append(accuracy_score(y_train,knn.predict(X_train)))

print(classification_report(y_train,knn.predict(X_train)))
from sklearn.metrics import f1_score

from sklearn.model_selection import GridSearchCV 



params_grid = {'algorithm': ['auto','ball_tree','kd_tree','brute'], 'weights': ['uniform', 'distance'],'n_neighbors': [3,5,10]}



knn_grid = GridSearchCV(estimator = knn, param_grid = params_grid, scoring='f1',cv = 5, n_jobs=-1)



knn_grid.fit(X_train,y_train)



knn_grid.best_params_
from sklearn.ensemble import BaggingClassifier

bag_clf = BaggingClassifier(base_estimator=rf, n_estimators=100,

                            bootstrap=True, n_jobs=-1,

                            random_state=42)



bag_clf.fit(X_train, y_train)

y_pred = bag_clf.predict(X_test)

score.append(accuracy_score(y_train,bag_clf.predict(X_train)))

print(classification_report(y_train,bag_clf.predict(X_train)))
from sklearn.ensemble import VotingClassifier

clf1 = RandomForestClassifier()

clf2 = KNeighborsClassifier(algorithm= 'ball_tree', n_neighbors= 3, weights ='distance')

clf3 = XGBClassifier()

clf4 = SVC()

clf5 = bag_clf

voter = VotingClassifier(estimators=[ ('rf', clf1), ('knn', clf2),('xgb', clf3),('svc',clf4),('bag_clf',clf5)], voting='hard', n_jobs=-1)

voter.fit(X_train, y_train)

score.append(accuracy_score(y_train,voter.predict(X_train)))

print(classification_report(y_train,voter.predict(X_train).astype(int)))
clf = ['lgbm','xgb','rf','svc','knn','bag_clf','voter']
fig = plt.figure()

plt.bar(clf, score)

plt.title('Accuracy score')

plt.grid(True)
y_pred = voter.predict(X_test)



output = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': y_pred})

output.to_csv('submission.csv', index=False)