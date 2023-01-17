import warnings

warnings.filterwarnings('ignore')
!pip install xgboost

!pip install imbalanced-learn
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()
df = pd.read_csv('../input/paysim1/PS_20174392719_1491204439457_log.csv')

df.head()
df.info()
frauds = df[df.isFraud == 1]

non_frauds = df[df.isFraud == 0]



frauds['balanceChange'] = frauds['newbalanceOrig'] - frauds['oldbalanceOrg']

non_frauds['balanceChange'] = non_frauds['newbalanceOrig'] - non_frauds['oldbalanceOrg']



frauds_mean_balancechange = frauds['balanceChange'].mean()

nfrauds_mean_balancechange = non_frauds['balanceChange'].mean()



width = 0.8

fig, ax = plt.subplots(1,1, figsize = (10, 6))

ax.bar(0.5, nfrauds_mean_balancechange, width, label='Avg. Non Fraud Account Balance Change', align='center')

ax.bar(1.5, frauds_mean_balancechange, width, label='Avg. Fraud Account Balance Change')

fig.legend(loc='best')

plt.axis([0, 2, -1500000, 200000])
num_positive_frauds = len(frauds[frauds['balanceChange'] > 0])

num_positive_frauds
paysim_negative = pd.concat([frauds[frauds['balanceChange'] <= 0], non_frauds[non_frauds['balanceChange'] <= 0]])
num_frauds = len(paysim_negative[paysim_negative['isFraud'] == 1])

num_frauds / len(paysim_negative)
paysim_negative.corr()
paysim_byClient = paysim_negative[['nameDest', 'oldbalanceOrg', 'oldbalanceDest', 'balanceChange']].groupby(['nameDest']).mean()

frauds_byClient = paysim_negative[['nameDest', 'isFraud']].groupby(['nameDest']).sum()

clientData = pd.concat([paysim_byClient, frauds_byClient], axis=1)

clientData['numTrans'] = paysim_negative[['nameDest', 'isFraud']].groupby(['nameDest']).count()['isFraud']

clientData = clientData.sort_values(by='isFraud', ascending=False)

clientData.head(20)
from scipy import stats



oldBalance = clientData[['oldbalanceOrg', 'isFraud']].groupby(['oldbalanceOrg']).sum()

kde = stats.gaussian_kde(oldBalance.index)

xx = np.linspace(0, 9, 1000)

plt.plot(xx, kde(xx))

oldBalance[1:].hist(bins=10)
newBalanceOrig = paysim_negative[['newbalanceOrig', 'isFraud']].groupby(['newbalanceOrig']).sum()

newBalanceOrig[1:].hist(bins=10)
oldBalanceDest = paysim_negative[['oldbalanceDest', 'isFraud']].groupby(['oldbalanceDest']).sum()

oldBalanceDest[1:].hist(bins=10)
newBalanceDest = paysim_negative[['newbalanceDest', 'isFraud']].groupby(['newbalanceDest']).sum()

newBalanceDest[1:].hist(bins=10)
numTrans = clientData[['numTrans', 'isFraud']].groupby(['numTrans']).sum()

kde = stats.gaussian_kde(numTrans.index)

xx = np.linspace(0, max(numTrans.isFraud), 1000)

plt.plot(xx, kde(xx))

numTrans.hist(bins=10)
ax = df.type.value_counts().plot.bar(figsize = (12, 6))

ax.set_title("Distribution of transactions for different payment types", fontsize = 20)

plt.xticks(rotation = 45)
df.drop(columns = ['step', 'isFlaggedFraud', 'nameOrig', 'nameDest'], inplace = True)

df.isFraud.value_counts()
X = df[['type', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']].copy()

y = df['isFraud'].copy()

X.type = X.type.astype('category')
from sklearn.compose import make_column_transformer

from sklearn.preprocessing import OneHotEncoder



col_transform = make_column_transformer((['type'], OneHotEncoder(sparse = False, drop = 'first')),

                                        remainder = 'passthrough')

X = pd.DataFrame(col_transform.fit_transform(X))
from sklearn.model_selection import StratifiedKFold

from sklearn.ensemble import RandomForestClassifier

from datetime import datetime

from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score



cv_pr_aucs = []

recalls = []

precisions = []

thresholds_list = []

skf = StratifiedKFold(n_splits = 5, random_state = 42, shuffle = True)



for train_idx, test_idx in skf.split(X, y):

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]

    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    print("Started training the model for the current fold at", datetime.now())

    rf_model = RandomForestClassifier(random_state = 42)

    rf_model.fit(X_train, y_train)

    y_pred = list(rf_model.predict_proba(X_test)[:, 1])

    precision, recall, thresholds = precision_recall_curve(y_test.tolist(), y_pred)

    recalls.append(recall)

    precisions.append(precision)

    thresholds_list.append(thresholds)

    cv_pr_aucs.append(auc(recall, precision))

    print("Completed training the model for the current fold at", datetime.now(), "\n")
rf_model = RandomForestClassifier(random_state = 42)

rf_model.fit(X, y)



fig, axis = plt.subplots(figsize = (10, 6))

feature_list = ['CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']

axis.bar(feature_list, (rf_model.feature_importances_ * 100))

plt.xticks(rotation = 45)

axis.set_title('Feature Importance from Random Forests', fontsize = 17)

axis.set_xlabel('Feature', fontsize = 17)

axis.set_ylabel('Importance (%)', fontsize = 17)
print("Area under PR curve:", np.mean(cv_pr_aucs))



precision = np.mean(precisions, axis = 0)

recall = np.mean(recalls, axis = 0)

threshold = np.mean(thresholds_list, axis = 0)



fig = plt.figure(figsize = (10, 6))

plt.plot(recall, precision, marker = '.')

plt.plot([0, 1], [sum(y == 1)/len(y), sum(y == 1)/len(y)], linestyle = '--')

ax = plt.gca()

ax.axvline(x = recall[int(np.where(threshold == 0.5)[0])], color = 'r', linestyle = '--', ymin = 0.05, ymax = 1)

ax.annotate('50% Threshold', (recall[int(np.where(threshold == 0.5)[0])] - 0.085, recall[int(np.where(threshold == 0.5)[0])] - 0.3), fontsize = 15)

ax.set_title("Precision-Recall Curve", fontsize = 17)

ax.set_xlabel('Recall', fontsize = 17)

ax.set_ylabel('Precision', fontsize = 17)
from xgboost import XGBClassifier



cv_pr_aucs = []

recalls = []

precisions = []

skf = StratifiedKFold(n_splits = 5, random_state = 42, shuffle = True)



for train_idx, test_idx in skf.split(X, y):

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]

    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    print("Started training the model for the current fold at", datetime.now())

    xgb_model = XGBClassifier(random_state = 42)

    xgb_model.fit(X_train, y_train)

    y_pred = list(xgb_model.predict_proba(X_test)[:, 1])

    precision, recall, thresholds = precision_recall_curve(y_test.tolist(), y_pred)

    recalls.append(recall)

    precisions.append(precision)

    cv_pr_aucs.append(auc(recall, precision))

    print("Completed training the model for the current fold at", datetime.now(), "\n")

    

print("Area under PR curve:", np.mean(cv_pr_aucs))
from sklearn.model_selection import GridSearchCV



parameters = {'n_estimators': [10, 50, 100],

              'criterion': ['gini', 'entropy'],

              'max_depth': [None, 3, 5]

             }
from sklearn.utils import resample



cv_pr_aucs = []

recalls = []

precisions = []

thresholds_list = []

skf = StratifiedKFold(n_splits = 5, random_state = 42, shuffle = True)



for train_idx, test_idx in skf.split(X, y):

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]

    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]    

    

    X_frauds_df = X_train[y_train == 1]

    y_frauds_df = y_train[y_train == 1]

    

    X_not_frauds_df = X_train[y_train == 0]

    y_not_frauds_df = y_train[y_train == 0]

    

    X_not_frauds_df, y_not_frauds_df = resample(X_not_frauds_df, y_not_frauds_df, replace = False, n_samples = len(X_frauds_df), random_state = 42)

    

    X_train = pd.concat([X_frauds_df, X_not_frauds_df])

    y_train = pd.concat([y_frauds_df, y_not_frauds_df])

    

    X_train = X_train.reset_index().drop(columns = ['index'])

    y_train = y_train.reset_index().drop(columns = ['index'])

    

    print("Started training the model for the current fold at", datetime.now())

    rf_model = RandomForestClassifier(random_state = 42)

    rf_model = GridSearchCV(rf_model, parameters, cv = 5, scoring = 'average_precision')

    rf_model.fit(X_train, y_train)

    y_pred = list(rf_model.predict_proba(X_test)[:, 1])

    precision, recall, thresholds = precision_recall_curve(y_test.tolist(), y_pred)

    recalls.append(recall)

    precisions.append(precision)

    thresholds_list.append(thresholds)

    cv_pr_aucs.append(auc(recall, precision))

    print("Completed training the model for the current fold at", datetime.now(), "\n")
print("Area under PR curve:", np.mean(cv_pr_aucs))
parameters = {'n_estimators': [10, 50, 100],

              'criterion': ['gini', 'entropy'],

              'max_depth': [2, 3, 6]

             }
cv_pr_aucs = []

recalls = []

precisions = []

skf = StratifiedKFold(n_splits = 5, random_state = 42, shuffle = True)



for train_idx, test_idx in skf.split(X, y):

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]

    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]    

    

    X_frauds_df = X_train[y_train == 1]

    y_frauds_df = y_train[y_train == 1]

    

    X_not_frauds_df = X_train[y_train == 0]

    y_not_frauds_df = y_train[y_train == 0]

    

    X_not_frauds_df, y_not_frauds_df = resample(X_not_frauds_df, y_not_frauds_df, replace = False, n_samples = len(X_frauds_df), random_state = 42)

    

    X_train = pd.concat([X_frauds_df, X_not_frauds_df])

    y_train = pd.concat([y_frauds_df, y_not_frauds_df])

    

    X_train = X_train.reset_index().drop(columns = ['index'])

    y_train = y_train.reset_index().drop(columns = ['index'])

    

    print("Started training the model for the current fold at", datetime.now())

    xgb_model = XGBClassifier(random_state = 42)

    xgb_model = GridSearchCV(xgb_model, parameters, cv = 5, scoring = 'average_precision')

    xgb_model.fit(X_train, y_train)

    y_pred = list(xgb_model.predict_proba(X_test)[:, 1])

    precision, recall, thresholds = precision_recall_curve(y_test.tolist(), y_pred)

    recalls.append(recall)

    precisions.append(precision)

    cv_pr_aucs.append(auc(recall, precision))

    print("Completed training the model for the current fold at", datetime.now(), "\n")
print("Area under PR curve:", np.mean(cv_pr_aucs))
from keras.models import Sequential

from keras.layers import Dense, Dropout
model = Sequential()

model.add(Dense(12, input_dim = 9, activation = 'relu'))

model.add(Dropout(0.3, seed = 42))

model.add(Dense(8, activation = 'relu'))

model.add(Dense(1, activation = 'sigmoid'))

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15, stratify = y, random_state = 42)



X_frauds_df = X_train[y_train == 1]

y_frauds_df = y_train[y_train == 1]



X_not_frauds_df = X_train[y_train == 0]

y_not_frauds_df = y_train[y_train == 0]



X_not_frauds_df, y_not_frauds_df = resample(X_not_frauds_df, y_not_frauds_df, replace = False, n_samples = len(X_frauds_df), random_state = 42)



X_train = pd.concat([X_frauds_df, X_not_frauds_df])

y_train = pd.concat([y_frauds_df, y_not_frauds_df])



X_train = X_train.reset_index().drop(columns = ['index'])

y_train = y_train.reset_index().drop(columns = ['index'])



std_scaler = StandardScaler()

X_train = pd.DataFrame(std_scaler.fit_transform(X_train))
model.fit(X_train, y_train, epochs = 150, batch_size = 10)
losses = model.history.history['loss']

accs = model.history.history['acc']
fig = plt.figure(figsize = (10, 6))

plt.plot(losses)

ax = plt.gca()

ax.set_title("Loss (Cross-Entropy) vs Epochs", fontsize = 17)

ax.set_xlabel('Epoch', fontsize = 17)

ax.set_ylabel('Loss', fontsize = 17)
fig = plt.figure(figsize = (10, 6))

plt.plot(np.array(accs)*100)

ax = plt.gca()

ax.set_title("Accuracy vs Epochs", fontsize = 17)

ax.set_xlabel('Epoch', fontsize = 17)

ax.set_ylabel('Accuracy (%)', fontsize = 17)
y_pred = model.predict_proba(std_scaler.transform(X_test))[:, 0]

precision, recall, thresholds = precision_recall_curve(y_test.tolist(), y_pred)

print("Area under PR curve: ", auc(recall, precision))
fig = plt.figure(figsize = (10, 6))

plt.plot(recall, precision, marker = '.')

plt.plot([0, 1], [sum(y == 1)/len(y), sum(y == 1)/len(y)], linestyle = '--')

ax = plt.gca()

ax.axvline(x = recall[int(np.where((thresholds > 0.49) & (thresholds < 0.51))[0][0])], color = 'r', linestyle = '--', ymin = 0.05, ymax = 0.96)

ax.annotate('50% Threshold', (recall[int(np.where((thresholds > 0.49) & (thresholds < 0.51))[0][0])] - 0.15, recall[int(np.where((thresholds > 0.49) & (thresholds < 0.51))[0][0])] - 0.3), fontsize = 15)

ax.set_title("Precision-Recall Curve", fontsize = 17)

ax.set_xlabel('Recall', fontsize = 17)

ax.set_ylabel('Precision', fontsize = 17)
from imblearn.over_sampling import SMOTE

from datetime import datetime



cv_pr_aucs = []

recalls = []

precisions = []

thresholds_list = []

skf = StratifiedKFold(n_splits = 5, random_state = 42, shuffle = True)



for train_idx, test_idx in skf.split(X, y):

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]

    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]   

    smote = SMOTE(random_state = 42)

    X_train, y_train = smote.fit_resample(X_train, y_train)

    X_train = pd.DataFrame(X_train)

    y_train = pd.Series(y_train)

    

    print("Started training the model for the current fold at", datetime.now())

    rf_model = RandomForestClassifier(random_state = 42)

    rf_model.fit(X_train, y_train)

    y_pred = list(rf_model.predict_proba(X_test)[:, 1])

    precision, recall, thresholds = precision_recall_curve(y_test.tolist(), y_pred)

    recalls.append(recall)

    precisions.append(precision)

    thresholds_list.append(thresholds)

    cv_pr_aucs.append(auc(recall, precision))

    print("Completed training the model for the current fold at", datetime.now(), "\n")
print("Area under PR curve:", np.mean(cv_pr_aucs))



precision = np.mean(precisions, axis = 0)

recall = np.mean(recalls, axis = 0)

threshold = np.mean(thresholds_list, axis = 0)



fig = plt.figure(figsize = (10, 6))

plt.plot(recall, precision, marker = '.')

plt.plot([0, 1], [sum(y == 1)/len(y), sum(y == 1)/len(y)], linestyle = '--')

ax = plt.gca()

ax.axvline(x = recall[int(np.where(threshold == 0.5)[0])], color = 'r', linestyle = '--', ymin = 0.05, ymax = 1)

ax.annotate('50% Threshold', (recall[int(np.where(threshold == 0.5)[0])] - 0.085, recall[int(np.where(threshold == 0.5)[0])] - 0.3), fontsize = 15)

ax.set_title("Precision-Recall Curve", fontsize = 17)

ax.set_xlabel('Recall', fontsize = 17)

ax.set_ylabel('Precision', fontsize = 17)
cv_pr_aucs = []

recalls = []

precisions = []

thresholds_list = []

skf = StratifiedKFold(n_splits = 5, random_state = 42, shuffle = True)



for train_idx, test_idx in skf.split(X, y):

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]

    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]   

    smote = SMOTE(random_state = 42)

    X_train, y_train = smote.fit_resample(X_train, y_train)

    X_train = pd.DataFrame(X_train)

    y_train = pd.Series(y_train)

    

    print("Started training the model for the current fold at", datetime.now())

    xgb_model = XGBClassifier(random_state = 42)

    xgb_model.fit(X_train, y_train)

    y_pred = list(xgb_model.predict_proba(X_test)[:, 1])

    precision, recall, thresholds = precision_recall_curve(y_test.tolist(), y_pred)

    recalls.append(recall)

    precisions.append(precision)

    thresholds_list.append(thresholds)

    cv_pr_aucs.append(auc(recall, precision))

    print("Completed training the model for the current fold at", datetime.now(), "\n")
print("Area under PR curve:", np.mean(cv_pr_aucs))