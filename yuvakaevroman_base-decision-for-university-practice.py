from sklearn import tree
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
titanic_data = pd.read_csv('../input/titanic/train.csv')
titanic_data.head()
X = titanic_data.drop(['PassengerId', 'Survived', 'Name', 'Ticket', 'Cabin'], axis = 1)
y = titanic_data.Survived
X = pd.get_dummies(X)
X = X.fillna({'Age' : X.Age.median()})
clf = tree.DecisionTreeClassifier(criterion = 'entropy')
clf.fit(X, y)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)
clf.fit(X_train, y_train)
clf.score(X_train, y_train)
clf.score(X_test, y_test)
clf = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth = 5)
clf.fit(X_train, y_train)
clf.score(X_train, y_train)
clf.score(X_test, y_test)
from sklearn.model_selection import cross_val_score
best_clf = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth = 10)
cross_val_score(clf, X_test, y_test, cv = 5).mean()
from sklearn.model_selection import GridSearchCV
clf = tree.DecisionTreeClassifier()
clf
parameters = {'criterion' : ['gini', 'entropy'], 'max_depth' : range(1, 30)}
gcv_clf = GridSearchCV(clf, parameters, cv = 5)
gcv_clf
gcv_clf.fit(X_train, y_train)
gcv_clf.best_params_
best_clf = gcv_clf.best_estimator_
best_clf.score(X_test, y_test)
from sklearn.metrics import precision_score, recall_score
y_pred = best_clf.predict(X_test)
precision_score(y_test, y_pred)
recall_score(y_test, y_pred)
y_predicted_prob = best_clf.predict_proba(X_test)
y_predicted_prob
pd.Series(y_predicted_prob[:, 1]).hist()
y_pred = np.where(y_predicted_prob[:, 1] > 0.8, 1, 0)
precision_score(y_test, y_pred)
recall_score(y_test, y_pred)
clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=3, min_samples_split=100, min_samples_leaf=10)
clf.fit(X_train, y_train)
plt.figure(figsize=(40, 20),dpi=80)
p = tree.plot_tree(clf, fontsize=30,filled=True,feature_names=list(X))
from sklearn.ensemble import RandomForestClassifier
clf_rf = RandomForestClassifier()
parametrs = {'n_estimators' : [10, 20, 30], 'max_depth' : [2, 5, 7, 10]}
gcv_clf = GridSearchCV(clf_rf, parametrs, cv=5)
gcv_clf.fit(X_train, y_train)
gcv_clf.best_params_
best_rf = gcv_clf.best_estimator_
best_rf.fit(X_train, y_train)
y_pred = best_rf.predict(X_test)
recall_score(y_test, y_pred)
precision_score(y_test, y_pred)
best_rf.score(X_test, y_test)
features_importances = best_rf.feature_importances_
features_importances_df = pd.DataFrame({'features' : list(X_train),
                                        'features_importances' : features_importances})
features_importances_df
features_importances_df.sort_values('features_importances', ascending=False)
test_data = pd.read_csv('../input/titanic/test.csv')
X = test_data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis = 1)
X = pd.get_dummies(X)
X = X.fillna({'Age' : X.Age.median(), 'Fare' : X.Fare.median()})
test_predictions = best_rf.predict(X)
output = pd.DataFrame({'PassengerId': test_data.PassengerId.astype(int), 'Survived': test_predictions.astype(int)})
output.to_csv('submission.csv', index=False)
