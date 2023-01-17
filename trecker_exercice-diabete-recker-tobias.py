# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from matplotlib import pyplot as plt
import seaborn as sns
t = pd.read_csv("../input/pima-indians-diabetes-database/diabetes.csv")
t.head(10)
t.count()
t.BloodPressure = t.BloodPressure.replace([0],t.BloodPressure.mean())
t.head(20)
t.SkinThickness = t.SkinThickness.replace([0],t.SkinThickness.mean())
t.Insulin = t.Insulin.replace([0],t.Insulin.mean())
t.BMI = t.BMI.replace([0],t.BMI.mean())
t.head(20)
t.Outcome.value_counts()
nb_malade = (t.Outcome == 1)
t[nb_malade].head(10)
prob = (t[nb_malade].Outcome.count())/(t.Outcome.count())
print(prob)
data_train = t.sample(frac=0.8, random_state=1)
data_test = t.drop(data_train.index)
X_train = data_train.drop(['Outcome'], axis=1)
y_train = data_train['Outcome']
X_test = data_test.drop(['Outcome'], axis=1)
y_test = data_test['Outcome']
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train,y_train)
y_lr = lr.predict(X_test)
from sklearn.metrics import accuracy_score, confusion_matrix
lr_score = accuracy_score(y_test, y_lr)
print(lr_score)
cm = confusion_matrix(y_test, y_lr)
print(cm)
sns.pairplot(t, hue="Outcome")
sns.kdeplot(t.Insulin, color='blue')
t['log_Insulin'] = np.log(t.Insulin+1)
t.head()
sns.kdeplot(t.log_Insulin, color='blue')
t = t.drop(['Insulin'], axis = 1)
t.head()
data_train = t.sample(frac=0.8, random_state=1)
data_test = t.drop(data_train.index)
X_train = data_train.drop(['Outcome'], axis=1)
y_train = data_train['Outcome']
X_test = data_test.drop(['Outcome'], axis=1)
y_test = data_test['Outcome']
lr = LogisticRegression()
lr.fit(X_train,y_train)
y_lr = lr.predict(X_test)
lr_score = accuracy_score(y_test, y_lr)
print(lr_score)
cm = confusion_matrix(y_test, y_lr)
print(cm)
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score,auc, accuracy_score
print(classification_report(y_test, y_lr))
probas = lr.predict_proba(X_test)
print(probas)
tprobas = pd.DataFrame(probas,columns=['proba_0','proba_1'])
tprobas['y'] = np.array(y_test)
tprobas
plt.figure(figsize=(10,10))
sns.distplot(1-tprobas.proba_0[tprobas.y==0], bins=50)
sns.distplot(tprobas.proba_1[tprobas.y==1], bins=50)
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test,probas[:, 1])
roc_auc = auc(false_positive_rate, true_positive_rate)
print (roc_auc)
plt.figure(figsize=(12,12))
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate, true_positive_rate, 'b', label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')        # plus mauvaise courbe
plt.plot([0,0,1],[0,1,1],'g:')     # meilleure courbe
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
from sklearn import ensemble
rf = ensemble.RandomForestClassifier()
rf.fit(X_train, y_train)
y_rf = rf.predict(X_test)
print(classification_report(y_test, y_rf))
cm = confusion_matrix(y_test, y_rf)
print(cm)
rf1 = ensemble.RandomForestClassifier(n_estimators=10, min_samples_leaf=10, max_features=3)
rf1.fit(X_train, y_train)
y_rf1 = rf.predict(X_test)
print(classification_report(y_test, y_rf1))
X = t.drop(['Outcome'], axis=1)
y = t.Outcome
from sklearn.model_selection import validation_curve
params = np.arange(1, 300,step=30)
train_score, val_score = validation_curve(rf, X, y, 'n_estimators', params, cv=7)
plt.figure(figsize=(12,12))
plt.plot(params, np.median(train_score, 1), color='blue', label='training score')
plt.plot(params, np.median(val_score, 1), color='red', label='validation score')
plt.legend(loc='best')
plt.ylim(0, 2)
plt.xlabel('n_estimators')
plt.ylabel('score');
from sklearn import model_selection
param_grid = {
              'n_estimators': [10, 100, 500],
              'min_samples_leaf': [1, 20, 50]
             }
estimator = ensemble.RandomForestClassifier()
rf_gs = model_selection.GridSearchCV(estimator, param_grid)
rf_gs.fit(X_train, y_train)
print(rf_gs.best_params_)
rf2 = rf_gs.best_estimator_
y_rf2 = rf2.predict(X_test)
print(classification_report(y_test, y_rf2))
importances = rf2.feature_importances_
indices = np.argsort(importances)
plt.figure(figsize=(8,5))
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), X_train.columns[indices])
plt.title('Importance des caracteristiques')
import xgboost as XGB
xgb  = XGB.XGBClassifier()
xgb.fit(X_train, y_train)
y_xgb = xgb.predict(X_test)
cm = confusion_matrix(y_test, y_xgb)
print(cm)
print(classification_report(y_test, y_xgb))