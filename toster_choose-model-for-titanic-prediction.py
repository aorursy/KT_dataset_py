import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



%matplotlib inline



from sklearn import preprocessing
train = pd.read_csv("../input/train.csv")

test  = pd.read_csv("../input/test.csv")
print(train.shape, test.shape)
train.head()
# var for building the model



vars = ['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']
X_train = train[vars].copy()

y_train = train['Survived'].copy()



X_test  = test[vars].copy()
le = preprocessing.LabelEncoder()



X_train.Sex = le.fit_transform(X_train.Sex)

X_test.Sex = le.transform(X_test.Sex) # use same encoder as for the training set
X_train = pd.concat([X_train, pd.get_dummies(X_train.Embarked)], axis=1).drop('Embarked', axis=1)

X_test  = pd.concat([X_test,  pd.get_dummies(X_test.Embarked)],  axis=1).drop('Embarked', axis=1)
X_train.head()
imp = preprocessing.Imputer()



X_train.loc[:,'Age'] = imp.fit_transform(X_train.Age.reshape(-1,1))

X_test.loc[:,'Age']  = imp.transform(X_test.Age.reshape(-1,1))
X_train.loc[:,'Fare'] = imp.fit_transform(X_train.Fare.reshape(-1,1))

X_test.loc[:,'Fare']  = imp.transform(X_test.Fare.reshape(-1,1))
pd.crosstab(X_train.Pclass, y_train)
pd.crosstab(X_train.Sex, y_train)
temp = pd.DataFrame({'Age':X_train.Age, 'Survived':y_train})

temp.boxplot('Age', by='Survived')

plt.show()
pd.crosstab(X_train.Parch, y_train)
pd.crosstab(X_train.SibSp, y_train)
temp = pd.DataFrame({'Fare':X_train.Fare, 'Survived':y_train})

temp.boxplot('Fare', by='Survived')

plt.ylim([0,100])

plt.show()
pd.crosstab(train.Embarked, y_train)
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
models = [knn, logreg, rfc, svc]



from sklearn.cross_validation import cross_val_score



for model in models:

    scores = cross_val_score(model, X_train_norm, y_train, cv=10, scoring='accuracy')

    print(str(model)[:5], scores.mean(), scores.std())
from sklearn.grid_search import GridSearchCV



param_grid = dict(C = [1, 2, 4, 10, 20],

                 gamma = [0.2, 0.1, 0.05, 0.02])



grid = GridSearchCV(svc, param_grid, cv=10, scoring='accuracy')



grid.fit(X_train_norm, y_train)
print(grid.best_params_, grid.best_score_)
y_pred = grid.predict(X_test_norm)
submission = pd.DataFrame({'PassengerId': test.iloc[:,0], 'Survived': y_pred})

submission.to_csv('submission.csv', index=False)