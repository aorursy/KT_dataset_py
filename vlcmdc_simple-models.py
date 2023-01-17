from sklearn import metrics, cross_validation, grid_search, linear_model



import warnings



import numpy as np

import pandas as pd

import seaborn as sns

from matplotlib import pyplot as plt



warnings.filterwarnings('ignore')
%pylab inline
data = pd.read_csv("../input/train.csv", header = 0, sep = ',')
data.info()
data.head()
data.isnull().sum()
data.shape
data.describe()
sns.set(font_scale=1)

pd.options.display.mpl_style = 'default'

data.drop(['PassengerId', 'Survived', 'Pclass'], axis=1).hist(figsize=(10, 7), grid=False)

plt.show()
plt.figure()



plt.subplot(221)

data.Pclass.value_counts().plot(kind='bar', figsize=(10, 10))

plt.xlabel("Passenger class")

plt.ylabel("Count")

plt.title("Passenger class distribution")



plt.subplot(222)

data.Embarked.value_counts().plot(kind='bar', figsize=(10, 10))

plt.xlabel("Emabarked")

plt.ylabel("Count")

plt.title("Embarked distribution")

plt.show()
plt.figure(1)



plt.subplots(1, 1, figsize=(10, 10))

plt.subplot(221)

sns.barplot(y='Survived', x='Pclass', data=data)

plt.title("Survived by passenger class")



plt.subplot(222)

sns.barplot(y='Survived', x='Embarked', data=data)

plt.title("Survived by Embarked")

plt.show()
sns.barplot(y='Survived', x="Sex", data=data)

plt.title("Male/female survived distribution")

plt.ylabel("Survived")

plt.show()
plt.figure(1)



plt.subplots(1, 1, figsize=(10, 10))



plt.subplot(221)

ax = data[data.Survived == 1].Age.plot(kind='hist', alpha=0.5)

ax = data[data.Survived == 0].Age.plot(kind='hist', alpha=0.5)

plt.title("Age distribution")

plt.xlabel("Age")

plt.legend(("survived", "not survived"), loc='best')



plt.subplot(222)

data.Age.plot(kind='kde', grid=False)

plt.title("Age distribution")

plt.xlabel("Age")

plt.xlim((0,80))

plt.show()
corr = data.corr()



plt.figure(figsize=(10, 8))



sns.heatmap(corr, square=True)

plt.title("Feature correlations")
t_data = data.drop(['Cabin', 'Ticket', 'PassengerId', 'Survived'], axis=1)

t_labels = data['Survived']
t_data.head()
t_data['Name_pred'] = data.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
pd.crosstab(t_data['Name_pred'], t_data['Sex'])
t_data['Name_pred'] = t_data['Name_pred'].replace("Mlle", "Miss")

t_data['Name_pred'] = t_data['Name_pred'].replace("Ms", "Miss")

t_data['Name_pred'] = t_data['Name_pred'].replace("Mme", "Mrs")
t_data['Name_pred'] = t_data['Name_pred'].replace(['Capt', 'Col', 'Countess', 'Don', 'Dr', 'Jonkheer',\

                                                  'Lady', 'Major', 'Rev', 'Sir'], 'Other')
preds = {'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4, 'Other': 5}



t_data['Name_pred'] = t_data['Name_pred'].map(preds)
t_data = t_data.drop('Name', axis=1)
t_data.head()
t_data['Sex'] = t_data['Sex'].apply(lambda x: int(x == 'male'))
t_data.Embarked = t_data.Embarked.fillna(value='S')
emb = { 'S': 1, 'C': 2, 'Q': 3}
t_data.Embarked = t_data.Embarked.map(emb)
# zeros as first try

t_data.Age = t_data.Age.fillna(value=0)
t_data.head()
real_cols = ['Age', 'SibSp', 'Parch', 'Fare']

cat_cols = list(set(t_data.columns.values.tolist()) - set(real_cols))
X_real = t_data[real_cols]

X_cat = t_data[cat_cols]
from sklearn.preprocessing import OneHotEncoder

from sklearn.feature_extraction import DictVectorizer
encoder = OneHotEncoder(categorical_features='all', sparse=True, n_values='auto')
X_cat.head()
X_cat_oh = encoder.fit_transform(X_cat).toarray()
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()



X_real_scaled = scaler.fit_transform(X_real)
X = np.hstack((X_real_scaled, X_cat_oh))
(X_train, X_test, y_train, y_test) = cross_validation.train_test_split(X, t_labels,

                                                                      test_size=0.3,

                                                                      stratify=t_labels)
clf = linear_model.SGDClassifier(class_weight='balanced')
clf.fit(X_train, y_train)
print(metrics.roc_auc_score(y_test, clf.predict(X_test)))
param_grid = {

    'loss': ['hinge', 'log', 'squared_hinge', 'squared_loss'],

    'penalty': ['l1', 'l2'],

    'n_iter': list(range(3, 10)),

    'alpha': np.linspace(0.0001, 0.01, num=10)

}
grid_cv = grid_search.GridSearchCV(clf, param_grid, scoring='accuracy', cv=3)
grid_cv.fit(X_train, y_train)
print(grid_cv.best_params_)
print(metrics.roc_auc_score(y_test, grid_cv.best_estimator_.predict(X_test)))
from sklearn import tree
clf = tree.DecisionTreeClassifier(max_depth=3, class_weight='balanced')
clf.get_params().keys()
params_grid = {

    'max_depth': list(range(1, 10)),

    'min_samples_leaf': list(range(2, 10))

}

grid_cv = grid_search.GridSearchCV(clf, params_grid, scoring='accuracy', cv=4)
grid_cv.fit(X_train, y_train)
print(grid_cv.best_params_)
print(metrics.roc_auc_score(y_test, grid_cv.best_estimator_.predict_proba(X_test)[:,1]))
from sklearn import ensemble
rf_clf = ensemble.RandomForestClassifier()
rf_clf.get_params().keys()
rf_clf.fit(X_train, y_train)
print(metrics.roc_auc_score(y_test, rf_clf.predict_proba(X_test)[:,1]))
params_grid = {

    'min_samples_leaf': list(range(1, 10)),

    'n_estimators': [10, 50, 100, 250, 500, 1000],

    'max_depth': list(range(1, 10))

}



rand_cv = grid_search.RandomizedSearchCV(rf_clf, params_grid, scoring='accuracy', cv=4, n_iter=40)



rand_cv.fit(X_train, y_train)
print(metrics.roc_auc_score(y_test, rand_cv.predict_proba(X_test)[:,1]))
test = pd.read_csv("../input/test.csv", header=0, sep=',')
test.head()
test.isnull().sum()
test_data = test.drop(['Cabin', 'Ticket', 'PassengerId'], axis=1)
test_data['Name_pred'] = test.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
test_data['Name_pred'] = test_data['Name_pred'].replace("Mlle", "Miss")

test_data['Name_pred'] = test_data['Name_pred'].replace("Ms", "Miss")

test_data['Name_pred'] = test_data['Name_pred'].replace("Mme", "Mrs")
test_data['Name_pred'] = test_data['Name_pred'].replace(['Capt', 'Col', 'Countess', 'Don', 'Dr', 'Jonkheer',\

                                              'Lady', 'Major', 'Rev', 'Sir'], 'Other')
preds = {'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4, 'Other': 5}

test_data['Name_pred'] = test_data['Name_pred'].map(preds)

test_data = test_data.drop('Name', axis=1)
test_data.Name_pred = test_data.Name_pred.fillna(value=5)
test_data.Name_pred = test_data.Name_pred.apply(int)
test_data['Sex'] = test_data['Sex'].apply(lambda x: int(x == 'male'))
test_data.Embarked = test_data.Embarked.fillna(value='S')

emb = { 'S': 1, 'C': 2, 'Q': 3}

test_data.Embarked = test_data.Embarked.map(emb)
test_data.Age = test_data.Age.fillna(value=0)
real_cols = ['Age', 'SibSp', 'Parch', 'Fare']

cat_cols = list(set(test_data.columns.values.tolist()) - set(real_cols))
Test_real = test_data[real_cols]

Test_cat = test_data[cat_cols]
encoder = OneHotEncoder(categorical_features='all', sparse=True, n_values='auto')

Test_cat_oh = encoder.fit_transform(Test_cat).toarray()
Test_real.Fare = Test_real.Fare.fillna(value=0)
scaler = StandardScaler()

X_real_scaled = scaler.fit_transform(Test_real)
X = np.hstack((Test_real, Test_cat_oh))
predict = rand_cv.predict(X)
submission = pd.DataFrame({

        "PassengerId": test.PassengerId,

        "Survived": predict

    })

submission.to_csv("predict.csv", index=False)
rand_cv.score(X_train, y_train)
print(rand_cv.best_estimator_)