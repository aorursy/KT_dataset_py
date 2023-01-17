# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
from tqdm import tqdm
from matplotlib import pyplot as plt # show graph

from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample
from imblearn.over_sampling import RandomOverSampler,SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek

import xgboost as xgb

from scipy.stats import ks_2samp

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
# Read data
inputDF = pd.read_csv('/kaggle/input/creditcardfraud/creditcard.csv')
inputDF.head(5)
inputDF.shape
# Remove any duplications
inputDF = inputDF.drop_duplicates(subset=None)
inputDF.shape
inputDF.describe()
# Graph distribution
inputDF.hist(bins=50, figsize=(20,15), color = 'deepskyblue')
plt.show()
len(inputDF[inputDF['Class']==0]), len(inputDF[inputDF['Class']==1])
#correlation matrix 
f, (ax1, ax2, ax3) = plt.subplots(1,3,figsize =( 30, 16))

sns.heatmap(inputDF.query('Class==1').drop(['Class'],1).corr(), vmax = .8, square=True, ax = ax1)
ax1.set_title('Fraud')

sns.heatmap(inputDF.query('Class==0').drop(['Class'],1).corr(), vmax = .8, square=True, ax = ax2);
ax2.set_title('Normal')

sns.heatmap(inputDF.corr(), vmax = .8, square=True, ax = ax3);
ax3.set_title('All')

plt.show()
y = inputDF.Class
X = inputDF.drop('Class', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, stratify=y, shuffle=True)
X_train.shape, X_test.shape
def checkTest(clf, X_test, y_test):
    y_pred = clf.predict(X_test)
    reportTest(y_pred, y_test)
    
def reportTest(y_pred, y_test):
    print("Classification report:")
    print(classification_report(y_test, y_pred))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("The accuracy is {}".format(accuracy_score(y_test, y_pred))) 
    print("The precision is {}".format(precision_score(y_test, y_pred))) 
    print("The recall is {}".format(recall_score(y_test, y_pred))) 
    print("The F1-Score is {}".format(f1_score(y_test, y_pred))) 
    print("The AUC is {} ".format(roc_auc_score(y_test, y_pred)))
    
#     importances = clf.feature_importances_
#     std = np.std([tree.feature_importances_ for tree in clf.estimators_], axis=0)
#     indices = np.argsort(importances)[::-1]

#     # Print the feature ranking
#     print("\nFeature ranking:")
#     for f in range(X_train.shape[1]):
#         print("%d. feature %s (%f)" % (f + 1, X_train.columns[indices[f]], importances[indices[f]]))
rfc = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
xgbc = xgb.XGBClassifier(n_jobs = -1)

for clf, clf_name in [(rfc, 'rf'), (xgbc, 'xgb')]:
    print("Training for " + clf_name)
    clf.fit(X_train, y_train)
    checkTest(clf, X_test, y_test)
    print("-------------\n")

# scores = cross_validate(clf, X_train, y_train, cv=5, n_jobs=-1, scoring=('accuracy', 'precision', 'recall', 'f1'))
# print('Accuracy: %0.5f (+/- %0.5f)' % scores['test_accuracy'].mean(), scores['test_accuracy'].std() * 2)
# print('Precision: %0.5f (+/- %0.5f)' % scores['test_precision'].mean(), scores['test_precision'].std() * 2)
# print('Recall: %0.5f (+/- %0.5f)' % scores['test_recall'].mean(), scores['test_recall'].std() * 2)
# print('F1: %0.5f (+/- %0.5f)' % scores['test_f1'].mean(), scores['test_f1'].std() * 2)
# sampler = RandomOverSampler()
# sampler = RandomUnderSampler()
# sampler = SMOTETomek(sampling_strategy='auto', n_jobs=-1, random_state=42)
sampler = SMOTE(sampling_strategy='minority', n_jobs=-1, random_state=42)
X_train_resampled, y_train_resampled = sampler.fit_sample(X_train, y_train)

rfc_SMOTE = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
xgbc_SMOTE = xgb.XGBClassifier(n_jobs = -1)

for clf, clf_name in [(rfc_SMOTE, 'rf_SMOTE'), (xgbc_SMOTE, 'xgb_SMOTE')]:
    print("Training for " + clf_name)
    clf.fit(X_train_resampled, y_train_resampled)
    checkTest(clf, X_test, y_test)
    print("-------------\n")
totalWeight = 0
totalPred = 0
for clf, weight in [(rfc, 0.77), (xgbc, 0.75), (rfc_SMOTE, 0.79), (xgbc_SMOTE, 0.8)]:
    totalWeight += weight
    pred = clf.predict_proba(X_test)
    totalPred += weight * pred
totalPred /= totalWeight
y_pred = []
for prob0, prob1 in totalPred:
    y_pred.append(0 if prob0 > 0.5 else 1)
reportTest(y_pred, y_test)