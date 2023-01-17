# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn import linear_model, datasets

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV

from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, f1_score, cohen_kappa_score

from sklearn import preprocessing

from sklearn.model_selection import KFold
data = pd.read_csv("../input/creditcard.csv")

data.head()

print("Non fraud rate:")

len(data.loc[data.loc[:, 'Class'] == 0, :]) / len(data.loc[:, 'Class'])
# under sample non fraud

len_fraud = len(data.loc[data.loc[:, 'Class'] == 1, :])

print("number of fraud: ", len_fraud)

# sample from non fraud to have 50/50 propotion

sub_non_fraud = data.loc[data.loc[:, 'Class'] == 0, :].sample(len_fraud)

sub_non_fraud.head()
# combine resample fraud and non fraud data

data_resample = pd.concat([sub_non_fraud, data.loc[data.loc[:, 'Class'] == 1, :]])

print("Non fraud rate:")

len(data_resample.loc[data_resample.loc[:, 'Class'] == 0, :]) / len(data_resample.loc[:, 'Class'])
# use k fold to find the highest recall rate parameter

X = data_resample.iloc[:, 1:29]

Y = data_resample.loc[:, "Class"]

kf = KFold(n_splits=5, shuffle=True)

kf.get_n_splits(X)

# test on different regularation power

cs = [0.001, 0.01, 0.1, 1, 10, 100]

for c in cs:

    accuracy = []

    recall = []

    f1 = []

    kappa = []

    for train_index, test_index in kf.split(X):

        X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]

        y_train, y_test = Y.iloc[train_index], Y.iloc[test_index]

        logreg = LogisticRegression(penalty='l1', solver='liblinear', C=c, max_iter=100)

        logreg.fit(X_train, y_train)

        y_test_pred = logreg.predict(X_test)

        recall.append(recall_score(y_test, y_test_pred))

        accuracy.append(accuracy_score(y_test, y_test_pred))

        f1.append(f1_score(y_test, y_test_pred))

        kappa.append(cohen_kappa_score(y_test, y_test_pred))

    print("For c=", c, "\t recall rate is", np.mean(recall), " and accuracy is", np.mean(accuracy), " f1 =", np.mean(f1), " kappa =", np.mean(kappa))
# use all data after resample to train the model

X = data_resample.iloc[:, 1:29]

Y = data_resample.loc[:, "Class"]

logreg = LogisticRegression(penalty='l1', solver='liblinear', C=1, max_iter=100)

logreg.fit(X, Y)

# apply to original dataset

ori_data = pd.read_csv("../input/creditcard.csv")

X_test = ori_data.iloc[:, 1:29]

Y_test = ori_data.loc[:, 'Class']

Y_test_predict = logreg.predict(X_test)

print("accuracy rate is ", accuracy_score(Y_test, Y_test_predict))

print("recall rate is ", recall_score(Y_test, Y_test_predict))

# AUC score

roc_auc_score(Y_test, Y_test_predict)
# function to try to use different ratio and output the result

def result_by_ratio(size_non_fraud):

    # generate different ratio of data

    sub_non_fraud = data.loc[data.loc[:, 'Class'] == 0, :].sample(int(len_fraud*size_non_fraud))

    data_resample = pd.concat([sub_non_fraud, data.loc[data.loc[:, 'Class'] == 1, :]])

    print("-----------------------------------------------")

    print("Non fraud rate:", len(data_resample.loc[data_resample.loc[:, 'Class'] == 0, :]) / len(data_resample.loc[:, 'Class']))

    # use k fold to find the highest recall rate parameter

    X = data_resample.iloc[:, 1:29]

    Y = data_resample.loc[:, "Class"]

    kf = KFold(n_splits=5, shuffle=True)

    kf.get_n_splits(X)

    # test on different regularation power

    cs = [0.001, 0.01, 0.1, 1, 10, 100]

    for c in cs:

        accuracy = []

        recall = []

        f1 = []

        kappa = []

        for train_index, test_index in kf.split(X):

            X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]

            y_train, y_test = Y.iloc[train_index], Y.iloc[test_index]

            logreg = LogisticRegression(penalty='l1', solver='liblinear', C=c, max_iter=100)

            logreg.fit(X_train, y_train)

            y_test_pred = logreg.predict(X_test)

            recall.append(recall_score(y_test, y_test_pred))

            accuracy.append(accuracy_score(y_test, y_test_pred))

            f1.append(f1_score(y_test, y_test_pred))

            kappa.append(cohen_kappa_score(y_test, y_test_pred))

    chosed_c = cs[kappa.index(max(kappa))]

    # use all data after resample to train the model

    X = data_resample.iloc[:, 1:29]

    Y = data_resample.loc[:, "Class"]

    logreg = LogisticRegression(penalty='l1', solver='liblinear', C=chosed_c, max_iter=100)

    logreg.fit(X, Y)

    # apply to original dataset

    ori_data = pd.read_csv("../input/creditcard.csv")

    X_test = ori_data.iloc[:, 1:29]

    Y_test = ori_data.loc[:, 'Class']

    Y_test_predict = logreg.predict(X_test)

    print("when ratio is 1:", size_non_fraud, "accuracy is ", \

              accuracy_score(Y_test, Y_test_predict), "and recall is",recall_score(Y_test, Y_test_predict), \

          " and ROC score is", roc_auc_score(Y_test, Y_test_predict))
ratio_list = [1, 1.5, 2, 2.5, 3, 3.5, 4]

for ratio in ratio_list:

    result_by_ratio(ratio)
from imblearn.over_sampling import SMOTE

sub_non_fraud = data.loc[data.loc[:, 'Class'] == 0, :].sample(int(len(data.loc[:, 'Class']) / 2))

data_resample = pd.concat([sub_non_fraud, data.loc[data.loc[:, 'Class'] == 1, :]])

X = data_resample.iloc[:, 1:29]

y = data_resample.loc[:, "Class"]

sm = SMOTE(kind='regular')

X_resampled, y_resampled = sm.fit_sample(X, y)
# size of X and y after SMOTE

print("Size of X", X_resampled.shape)

print("Size of y", y_resampled.shape)

print("Size of fraud", y_resampled[y_resampled == 1].shape)
from sklearn.model_selection import train_test_split

X_resampled = pd.DataFrame(X_resampled)

y_resampled = pd.DataFrame(y_resampled)

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size = 0.3, random_state = 42)

c = [0.01, 0.1, 1, 10]

logreg = LogisticRegressionCV(penalty='l2', solver='sag', Cs=c, refit=True, cv=10, max_iter=100)

logreg.fit(X_train, y_train)

y_test_predict = logreg.predict(X_test)
print("accuracy is ", accuracy_score(y_test, y_test_predict), "and recall is",recall_score(y_test, y_test_predict))

print("AUC score is ", roc_auc_score(y_test, y_test_predict))
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score

rf = RandomForestClassifier(n_estimators=100)

rf.fit(X_train, y_train)

y_test_predict = rf.predict(X_test)

print("accuracy is ", accuracy_score(y_test, y_test_predict), "and recall is",recall_score(y_test, y_test_predict))

print("AUC score is ", roc_auc_score(y_test, y_test_predict))
# under sample non fraud

len_fraud = len(data.loc[data.loc[:, 'Class'] == 1, :])

# sample from non fraud to have 50/50 propotion

sub_non_fraud = data.loc[data.loc[:, 'Class'] == 0, :].sample(len_fraud)

data_resample = pd.concat([sub_non_fraud, data.loc[data.loc[:, 'Class'] == 1, :]])

X = data_resample.iloc[:, 1:29]

y = data_resample.loc[:, "Class"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

rf2 = RandomForestClassifier(n_estimators=100)

rf2.fit(X_train, y_train)



ori_data = pd.read_csv("../input/creditcard.csv")

X_test = ori_data.iloc[:, 1:29]

y_test = ori_data.loc[:, 'Class']

y_test_predict = rf2.predict(X_test)

print("accuracy rate is ", accuracy_score(y_test, y_test_predict))

print("recall rate is ", recall_score(y_test, y_test_predict))

print("AUC score is ", roc_auc_score(y_test, y_test_predict))