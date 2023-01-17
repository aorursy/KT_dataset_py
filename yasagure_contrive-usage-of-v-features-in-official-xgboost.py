import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

print(os.listdir("../input"))

from sklearn import preprocessing

import matplotlib.pylab as plt

%matplotlib inline

import xgboost as xgb
train_transaction = pd.read_csv('../input/ieee-fraud-detection/train_transaction.csv', index_col='TransactionID')

test_transaction = pd.read_csv('../input/ieee-fraud-detection/test_transaction.csv', index_col='TransactionID')



train_identity = pd.read_csv('../input/ieee-fraud-detection/train_identity.csv', index_col='TransactionID')

test_identity = pd.read_csv('../input/ieee-fraud-detection/test_identity.csv', index_col='TransactionID')



sample_submission = pd.read_csv('../input/ieee-fraud-detection/sample_submission.csv', index_col='TransactionID')
train = train_transaction.merge(train_identity, how='left', left_index=True, right_index=True)

test = test_transaction.merge(test_identity, how='left', left_index=True, right_index=True)
train.columns[54:393]
train.iloc[:,54:393].corr()
train.loc[:,["V319","V320"]].corr()
train.loc[:,["V109","V110"]].corr()
train.loc[:,["V329","V330"]].corr()
train.loc[:,["V316","V331"]].corr()
train.loc[:,["V4","V5"]].corr()
train["diff_V319_V320"] = np.zeros(train.shape[0])



train.loc[train["V319"]!=train["V320"],"diff_V319_V320"] = 1



test["diff_V319_V320"] = np.zeros(test.shape[0])



test.loc[test["V319"]!=test["V320"],"diff_V319_V320"] = 1



train["diff_V109_V110"] = np.zeros(train.shape[0])



train.loc[train["V109"]!=train["V110"],"diff_V109_V110"] = 1



test["diff_V109_V110"] = np.zeros(test.shape[0])



test.loc[test["V109"]!=test["V110"],"diff_V109_V110"] = 1



train["diff_V329_V330"] = np.zeros(train.shape[0])



train.loc[train["V329"]!=train["V330"],"diff_V329_V330"] = 1



test["diff_V329_V330"] = np.zeros(test.shape[0])



test.loc[test["V329"]!=test["V330"],"diff_V329_V330"] = 1





train["diff_V316_V331"] = np.zeros(train.shape[0])



train.loc[train["V331"]!=train["V316"],"diff_V316_V331"] = 1



test["diff_V316_V331"] = np.zeros(test.shape[0])



test.loc[test["V316"]!=test["V331"],"diff_V316_V331"] = 1





train["diff_V4_V5"] = np.zeros(train.shape[0])



train.loc[train["V4"]!=train["V5"],"diff_V4_V5"] = 1



test["diff_V4_V5"] = np.zeros(test.shape[0])



test.loc[test["V4"]!=test["V5"],"diff_V4_V5"] = 1
plt.bar(train.groupby("diff_V319_V320").mean().isFraud.index,train.groupby("diff_V319_V320").mean().isFraud.values)

plt.bar(train.groupby("diff_V109_V110").mean().isFraud.index,train.groupby("diff_V109_V110").mean().isFraud.values)
plt.bar(train.groupby("diff_V329_V330").mean().isFraud.index,train.groupby("diff_V329_V330").mean().isFraud.values)
plt.bar(train.groupby("diff_V316_V331").mean().isFraud.index,train.groupby("diff_V316_V331").mean().isFraud.values)
plt.bar(train.groupby("diff_V4_V5").mean().isFraud.index,train.groupby("diff_V4_V5").mean().isFraud.values)
train = train.drop("diff_V109_V110",axis=1)

test = test.drop("diff_V109_V110",axis=1)



train = train.drop("diff_V329_V330",axis=1)

test = test.drop("diff_V329_V330",axis=1)



train = train.drop("diff_V316_V331",axis=1)

test = test.drop("diff_V316_V331",axis=1)





train = train.drop("diff_V4_V5",axis=1)

test = test.drop("diff_V4_V5",axis=1)

print(train.shape)

print(test.shape)



y_train = train['isFraud'].copy()



# Drop target, fill in NaNs

X_train = train.drop('isFraud', axis=1)

X_test = test.copy()

X_train = X_train.fillna(-999)

X_test = X_test.fillna(-999)
del train, test, train_transaction, train_identity, test_transaction, test_identity

# Label Encoding

for f in X_train.columns:

    if X_train[f].dtype=='object' or X_test[f].dtype=='object': 

        lbl = preprocessing.LabelEncoder()

        lbl.fit(list(X_train[f].values) + list(X_test[f].values))

        X_train[f] = lbl.transform(list(X_train[f].values))

        X_test[f] = lbl.transform(list(X_test[f].values))   
clf = xgb.XGBClassifier(n_estimators=500,

                        n_jobs=4,

                        max_depth=9,

                        learning_rate=0.05,

                        subsample=0.9,

                        colsample_bytree=0.9,

                        missing=-999)



clf.fit(X_train, y_train)
sample_submission['isFraud'] = clf.predict_proba(X_test)[:,1]

sample_submission.to_csv('simple_xgboost.csv')