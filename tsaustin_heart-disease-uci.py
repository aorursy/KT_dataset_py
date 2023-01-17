import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# import data into dataframe

df = pd.read_csv('../input/heart.csv')
# quick peek

df.head()
# check for null values

df.isnull().sum()
df.describe()
# split into inputs and targets

X = df.drop('target',axis=1)

y = df['target']
# Feature normalization

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().fit(X)

X_scaled = scaler.transform(X)
# split into train and test

from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(

    X_scaled, y)
# Logistic Regression

from sklearn.linear_model import LogisticRegression

import numpy as np



clf = LogisticRegression(solver="lbfgs")

from sklearn.model_selection import validation_curve

import matplotlib.pyplot as plt

import seaborn as sns



param_range = [0.01,0.025,0.05,0.1,0.2]

train_scores, valid_scores = validation_curve(clf, X_train, y_train,

                                              "C",param_range,

                                              cv=5,scoring="f1")



train_mean = np.mean(train_scores,axis=1)

valid_mean = np.mean(valid_scores,axis=1)



plt.plot(param_range, train_mean, label="Training")

plt.plot(param_range, valid_mean, label="Validation")

plt.tight_layout()

plt.legend(loc="best")

plt.show()
from sklearn.metrics import roc_curve



clf = LogisticRegression(solver="lbfgs", C=0.1).fit(X_train, y_train)

y_pred_quant = clf.predict_proba(X_test)[:, 1]



fpr, tpr, thresholds = roc_curve(y_test, y_pred_quant)



fig, ax = plt.subplots()

ax.plot(fpr, tpr)

ax.plot([0, 1], [0, 1], transform=ax.transAxes, ls="--", c=".3")

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.0])

plt.rcParams['font.size'] = 12

plt.title('ROC curve for Logistic Classifier')

plt.xlabel('False Positive Rate (1 - Specificity)')

plt.ylabel('True Positive Rate (Sensitivity)')

plt.grid(True)
from sklearn.metrics import auc

print(auc(fpr, tpr))
from sklearn.metrics import confusion_matrix

predict = clf.predict(X_test)

cnf_matrix = confusion_matrix(y_test,predict)



class_names = [0,1]

fig,ax = plt.subplots()

tick_marks = np.arange(len(class_names))

plt.xticks(tick_marks,class_names)

plt.yticks(tick_marks,class_names)



#create a heat map

sns.heatmap(pd.DataFrame(cnf_matrix), annot = True, cmap = 'YlGnBu',

           fmt = 'g')

ax.xaxis.set_label_position('top')

plt.tight_layout()

plt.title('Confusion matrix for Logistic Regression', y = 1.1)

plt.ylabel('Actual label')

plt.xlabel('Predicted label')

plt.show()
from sklearn.ensemble import GradientBoostingClassifier



clf = GradientBoostingClassifier()



param_range = [2,3,5,7,10]

train_scores, valid_scores = validation_curve(clf, X_train, y_train,

                                              "n_estimators",param_range,

                                              cv=5,scoring="f1")



train_mean = np.mean(train_scores,axis=1)

valid_mean = np.mean(valid_scores,axis=1)



plt.plot(param_range, train_mean, label="Training")

plt.plot(param_range, valid_mean, label="Validation")

plt.tight_layout()

plt.legend(loc="best")

plt.show()
clf = GradientBoostingClassifier(n_estimators=5)



param_range = [0.01,0.02,0.05,0.075,0.1,0.2]

train_scores, valid_scores = validation_curve(clf, X_train, y_train,

                                              "learning_rate",param_range,

                                              cv=5,scoring="f1")



train_mean = np.mean(train_scores,axis=1)

valid_mean = np.mean(valid_scores,axis=1)



plt.plot(param_range, train_mean, label="Training")

plt.plot(param_range, valid_mean, label="Validation")

plt.tight_layout()

plt.legend(loc="best")

plt.show()
clf = GradientBoostingClassifier(n_estimators=3, learning_rate=0.075)



param_range = [2,3,4,5,6]

train_scores, valid_scores = validation_curve(clf, X_train, y_train,

                                              "max_depth",param_range,

                                              cv=5,scoring="f1")



train_mean = np.mean(train_scores,axis=1)

valid_mean = np.mean(valid_scores,axis=1)



plt.plot(param_range, train_mean, label="Training")

plt.plot(param_range, valid_mean, label="Validation")

plt.tight_layout()

plt.legend(loc="best")

plt.show()
clf = GradientBoostingClassifier(n_estimators=3, learning_rate=0.05, max_depth=4).fit(X_train, y_train)

y_pred_quant = clf.predict_proba(X_test)[:, 1]



fpr, tpr, thresholds = roc_curve(y_test, y_pred_quant)



fig, ax = plt.subplots()

ax.plot(fpr, tpr)

ax.plot([0, 1], [0, 1], transform=ax.transAxes, ls="--", c=".3")

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.0])

plt.rcParams['font.size'] = 12

plt.title('ROC curve for GBDT')

plt.xlabel('False Positive Rate (1 - Specificity)')

plt.ylabel('True Positive Rate (Sensitivity)')

plt.grid(True)
from sklearn.metrics import auc

print(auc(fpr, tpr))
predict = clf.predict(X_test)

cnf_matrix = confusion_matrix(y_test,predict)



class_names = [0,1]

fig,ax = plt.subplots()

tick_marks = np.arange(len(class_names))

plt.xticks(tick_marks,class_names)

plt.yticks(tick_marks,class_names)



#create a heat map

sns.heatmap(pd.DataFrame(cnf_matrix), annot = True, cmap = 'YlGnBu',

           fmt = 'g')

ax.xaxis.set_label_position('top')

plt.tight_layout()

plt.title('Confusion matrix for GBDT', y = 1.1)

plt.ylabel('Actual label')

plt.xlabel('Predicted label')

plt.show()