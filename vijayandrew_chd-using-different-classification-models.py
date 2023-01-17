import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
df = pd.read_csv('../input/US_Heart_Patients.csv')

df.head()
df.shape
df.isnull().sum()
df = df.fillna(method='ffill')
df.info()
from sklearn.model_selection import train_test_split



X = df.drop('TenYearCHD', axis=1)

y = df['TenYearCHD']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
from sklearn.tree import DecisionTreeClassifier



dt = DecisionTreeClassifier(max_depth=3, min_samples_leaf=10 )

dt.fit(X, y)
from sklearn.tree import DecisionTreeClassifier



dt = DecisionTreeClassifier()

dt.fit(X_train, y_train)



y_pred_train = dt.predict(X_train)

y_pred = dt.predict(X_test)

y_prob = dt.predict_proba(X_test)
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve



print('Accuracy of Decision Tree-Train: ', accuracy_score(y_pred_train, y_train))

print('Accuracy of Decision Tree-Test: ', accuracy_score(y_pred, y_test))
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
dt = DecisionTreeClassifier()



params = {'max_depth' : [2,3,4,5,6,7,8],

        'min_samples_split': [2,3,4,5,6,7,8,9,10],

        'min_samples_leaf': [1,2,3,4,5,6,7,8,9,10]}



gsearch = GridSearchCV(dt, param_grid=params, cv=3)



gsearch.fit(X,y)



gsearch.best_params_
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(

                                    X, y, test_size=0.3, random_state=1)
dt = DecisionTreeClassifier(**gsearch.best_params_)



dt.fit(X_train, y_train)



y_pred_train = dt.predict(X_train)

y_prob_train = dt.predict_proba(X_train)[:,1]



y_pred = dt.predict(X_test)

y_prob = dt.predict_proba(X_test)[:,1]



from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve



print('Accuracy of Decision Tree-Train: ', accuracy_score(y_pred_train, y_train))

print('Accuracy of Decision Tree-Test: ', accuracy_score(y_pred, y_test))
print('AUC of Decision Tree-Train: ', roc_auc_score(y_train, y_prob_train))

print('AUC of Decision Tree-Test: ', roc_auc_score(y_test, y_prob))
from scipy.stats import randint as sp_randint



dt = DecisionTreeClassifier(random_state=1)



params = {'max_depth' : sp_randint(2,10),

        'min_samples_split': sp_randint(2,50),

        'min_samples_leaf': sp_randint(1,20),

         'criterion':['gini', 'entropy']}



rand_search = RandomizedSearchCV(dt, param_distributions=params, cv=3, 

                                 random_state=1)



rand_search.fit(X, y)

print(rand_search.best_params_)
dt = DecisionTreeClassifier(**rand_search.best_params_)



dt.fit(X_train, y_train)



y_pred_train = dt.predict(X_train)

y_prob_train = dt.predict_proba(X_train)[:,1]



y_pred = dt.predict(X_test)

y_prob = dt.predict_proba(X_test)[:,1]



from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve



print('Accuracy of Decision Tree-Train: ', accuracy_score(y_pred_train, y_train))

print('Accuracy of Decision Tree-Test: ', accuracy_score(y_pred, y_test))
print('AUC of Decision Tree-Train: ', roc_auc_score(y_train, y_prob_train))

print('AUC of Decision Tree-Test: ', roc_auc_score(y_test, y_prob))
from sklearn.ensemble import RandomForestClassifier



rfc = RandomForestClassifier(n_estimators=10, random_state=1)



rfc.fit(X_train, y_train)



y_pred_train = rfc.predict(X_train)

y_prob_train = rfc.predict_proba(X_train)[:,1]



y_pred = rfc.predict(X_test)

y_prob = rfc.predict_proba(X_test)[:,1]



from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve



print('Accuracy of Random Forest-Train: ', accuracy_score(y_pred_train, y_train))

print('Accuracy of Random Forest-Test: ', accuracy_score(y_pred, y_test))

print('AUC of Random Forest-Train: ', roc_auc_score(y_train, y_prob_train))

print('AUC of Random Forest-Test: ', roc_auc_score(y_test, y_prob))
from scipy.stats import randint as sp_randint



rfc = RandomForestClassifier(random_state=1)



params = {'n_estimators': sp_randint(5,25),

    'criterion': ['gini', 'entropy'],

    'max_depth': sp_randint(2, 10),

    'min_samples_split': sp_randint(2,20),

    'min_samples_leaf': sp_randint(1, 20),

    'max_features': sp_randint(2,15)}



rand_search_rfc = RandomizedSearchCV(rfc, param_distributions=params,

                                 cv=3, random_state=1)



rand_search_rfc.fit(X, y)

print(rand_search_rfc.best_params_)
rfc = RandomForestClassifier(**rand_search_rfc.best_params_)



rfc.fit(X_train, y_train)



y_pred_train = rfc.predict(X_train)

y_prob_train = rfc.predict_proba(X_train)[:,1]



y_pred = rfc.predict(X_test)

y_prob = rfc.predict_proba(X_test)[:,1]



from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve



print('Accuracy of Random Forest-Train: ', accuracy_score(y_pred_train, y_train))

print('Accuracy of Random Forest-Test: ', accuracy_score(y_pred, y_test))
print('AUC of Random Forest-Train: ', roc_auc_score(y_train, y_prob_train))

print('AUC of Random Forest-Test: ', roc_auc_score(y_test, y_prob))
fpr, tpr, thresholds = roc_curve(y_test, y_prob)



plt.plot(fpr, tpr)

plt.plot(fpr, fpr, 'r-')

plt.xlabel('FPR')

plt.ylabel('TPR')

plt.show()
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
from sklearn.preprocessing import StandardScaler

ss = StandardScaler()



Xs = ss.fit_transform(X)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)



X_trains = ss.fit_transform(X_train)

X_tests = ss.transform(X_test)
knn.fit(X_trains, y_train)



y_pred_train = knn.predict(X_trains)

y_prob_train = knn.predict_proba(X_trains)[:,1]



y_pred = knn.predict(X_tests)

y_prob = knn.predict_proba(X_tests)[:,1]



from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve



print('Accuracy of kNN-Train: ', accuracy_score(y_pred_train, y_train))

print('Accuracy of kNN-Test: ', accuracy_score(y_pred, y_test))
print('AUC of kNN-Train: ', roc_auc_score(y_train, y_prob_train))

print('AUC of kNN-Test: ', roc_auc_score(y_test, y_prob))
knn = KNeighborsClassifier()



params = {'n_neighbors': sp_randint(1,20),

        'p': sp_randint(1,5)}



rand_search_knn = RandomizedSearchCV(knn, param_distributions=params,

                                 cv=3, random_state=1)

rand_search_knn.fit(Xs, y)

print(rand_search_knn.best_params_)
knn = KNeighborsClassifier(**rand_search_knn.best_params_)



knn.fit(X_trains, y_train)



y_pred_train = knn.predict(X_trains)

y_prob_train = knn.predict_proba(X_trains)[:,1]



y_pred = knn.predict(X_tests)

y_prob = knn.predict_proba(X_tests)[:,1]



from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve



print('Accuracy of Random Forest-Train: ', accuracy_score(y_pred_train, y_train))

print('Accuracy of Random Forest-Test: ', accuracy_score(y_pred, y_test))
print('AUC of kNN-Train: ', roc_auc_score(y_train, y_prob_train))

print('AUC of kNN-Test: ', roc_auc_score(y_test, y_prob))
from sklearn.ensemble import VotingClassifier

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver='liblinear')

rfc = RandomForestClassifier(**rand_search_rfc.best_params_)

knn = KNeighborsClassifier(**rand_search_knn.best_params_)
clf = VotingClassifier(estimators=[('lr',lr), ('rfc',rfc), ('knn',knn)], 

                       voting='soft')

clf.fit(X_train, y_train)

y_pred_train = clf.predict(X_train)

y_prob_train = clf.predict_proba(X_train)[:,1]



y_pred = clf.predict(X_test)

y_prob = clf.predict_proba(X_test)[:,1]
print('Accuracy of Stacked Algos-Train: ', accuracy_score(y_pred_train, y_train))

print('Accuracy of Stacked Algos-Test: ', accuracy_score(y_pred, y_test))
print('AUC of Stacked Algos: ', roc_auc_score(y_train, y_prob_train))

print('AUC of Stacked Algos: ', roc_auc_score(y_test, y_prob))
clf = VotingClassifier(estimators=[('lr',lr), ('rfc',rfc), ('knn',knn)], 

                       voting='soft', weights=[2,3,1])

clf.fit(X_train, y_train)

y_pred_train = clf.predict(X_train)

y_prob_train = clf.predict_proba(X_train)[:,1]



y_pred = clf.predict(X_test)

y_prob = clf.predict_proba(X_test)[:,1]
print('Accuracy of Stacked Algos-Train: ', accuracy_score(y_pred_train, y_train))

print('Accuracy of Stacked Algos-Test: ', accuracy_score(y_pred, y_test))
print('AUC of Stacked Algos: ', roc_auc_score(y_train, y_prob_train))

print('AUC of Stacked Algos: ', roc_auc_score(y_test, y_prob))