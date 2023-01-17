import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



%matplotlib inline



from sklearn import preprocessing
train = pd.read_csv("../input/train.csv")

test  = pd.read_csv("../input/test.csv")
vars = ['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']
X_train = train[vars].copy()

y_train = train['Survived'].copy()



X_test  = test[vars].copy()
le = preprocessing.LabelEncoder()



X_train.Sex = le.fit_transform(X_train.Sex)

X_test.Sex = le.transform(X_test.Sex) # use same encoder as for the training set
X_train = pd.concat([X_train, pd.get_dummies(X_train.Embarked)], axis=1).drop('Embarked', axis=1)

X_test  = pd.concat([X_test,  pd.get_dummies(X_test.Embarked)],  axis=1).drop('Embarked', axis=1)
imp = preprocessing.Imputer()



X_train.loc[:,'Age'] = imp.fit_transform(X_train.Age.reshape(-1,1))

X_test.loc[:,'Age']  = imp.transform(X_test.Age.reshape(-1,1))
X_train.loc[:,'Fare'] = imp.fit_transform(X_train.Fare.reshape(-1,1))

X_test.loc[:,'Fare']  = imp.transform(X_test.Fare.reshape(-1,1))
X_train.head()
scaler = preprocessing.StandardScaler()



X_train_norm = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)

X_test_norm  = pd.DataFrame(scaler.transform(X_test),      columns=X_test.columns)
X_train_norm.head()
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()



from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()



from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier()



from sklearn.svm import SVC

svc = SVC()



from sklearn.ensemble import AdaBoostClassifier

boost = AdaBoostClassifier()
models = [knn, logreg, rfc, svc, boost]



from sklearn.model_selection import cross_val_score



for model in models:

    scores = cross_val_score(model, X_train_norm, y_train, cv=10, scoring='accuracy')

    print(str(model)[:5], scores.mean(), scores.std())
from sklearn.model_selection import cross_val_predict



predictions = {}



for model in models:

    predictions[str(model)[:5]] = cross_val_predict(model, X_train_norm, y_train, cv=10)
for k1, v1 in predictions.items():

    for k2, v2 in predictions.items():

        print(k1,"-",k2,":", (v1==v2).mean())
from sklearn.ensemble import VotingClassifier



estims = [(str(model)[:5], model) for model in models]



ensemble = VotingClassifier(estimators=estims)
scores = cross_val_score(ensemble, X_train_norm, y_train, cv=10, scoring='accuracy')

print(scores.mean(), scores.std())
ensemble.fit(X_train_norm, y_train) # fit the ensemble on all training data



y_pred = ensemble.predict(X_test_norm)
submission = pd.DataFrame({'PassengerId': test.iloc[:,0], 'Survived': y_pred})

submission.to_csv('submission.csv', index=False)