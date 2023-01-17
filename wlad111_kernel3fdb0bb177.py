# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import numpy as np

import pandas as pd



from pandas_profiling import ProfileReport



from collections import OrderedDict



import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.preprocessing import StandardScaler



import os



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
def get_roc_auc(estimator, y_test):

    probs = estimator.predict_proba(X_test)

    preds = probs[:,1]

    fpr, tpr, threshold = roc_curve(y_test, preds)

    roc_auc = auc(fpr, tpr)



    plt.title('Receiver Operating Characteristic')

    plt.plot(fpr, tpr, 'b', label = 'AUC = %f' % roc_auc)

    plt.legend(loc = 'lower right')

    plt.plot([0, 1], [0, 1],'r--')

    plt.xlim([0, 1])

    plt.ylim([0, 1])

    plt.ylabel('True Positive Rate')

    plt.xlabel('False Positive Rate')

    plt.show()

    

def get_roc_auc_inv(estimator, y_test):

    probs = estimator.predict_proba(X_test)

    preds = probs[:,0]

    fpr, tpr, threshold = roc_curve(y_test, preds)

    roc_auc = auc(fpr, tpr)



    plt.title('Receiver Operating Characteristic')

    plt.plot(fpr, tpr, 'b', label = 'AUC = %f' % roc_auc)

    plt.legend(loc = 'lower right')

    plt.plot([0, 1], [0, 1],'r--')

    plt.xlim([0, 1])

    plt.ylim([0, 1])

    plt.ylabel('True Positive Rate')

    plt.xlabel('False Positive Rate')

    plt.show()
heart = pd.read_csv('/kaggle/input/heart-disease-uci/heart.csv')
heart
heart['target'].value_counts(normalize=True)
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV  

from sklearn import metrics

from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, auc
y = heart.target

X = heart.drop(['target'], axis=1)
cols_to_draw = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'target']

sns.pairplot(heart[cols_to_draw], hue="target")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, stratify=y, shuffle=True)
from sklearn.linear_model import LogisticRegression
log_regr_clf = LogisticRegression(random_state=0)

log_regr_clf.fit(X_train, y_train)
y_pred_test = log_regr_clf.predict(X_test)

print(metrics.classification_report(y_test, y_pred_test))
get_roc_auc_inv(log_regr_clf, y_test)
get_roc_auc(log_regr_clf, y_test)
coin_toss_preds = np.random.uniform(0, 1,303)
preds = coin_toss_preds

fpr, tpr, threshold = roc_curve(y, preds)

roc_auc = auc(fpr, tpr)



plt.title('Receiver Operating Characteristic')

plt.plot(fpr, tpr, 'b', label = 'AUC = %f' % roc_auc)

plt.legend(loc = 'lower right')

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0, 1])

plt.ylim([0, 1])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.show()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, stratify=y, shuffle=True, random_state=42)
y_pred_test = log_regr_clf.predict(X_test)

print(metrics.classification_report(y_test, y_pred_test))
from sklearn.model_selection import cross_val_score
logreg = LogisticRegression()
cv = cross_val_score(logreg, X, y, cv=5, scoring='recall')

print(cv)
print(np.mean(cv))
print(np.std(cv))