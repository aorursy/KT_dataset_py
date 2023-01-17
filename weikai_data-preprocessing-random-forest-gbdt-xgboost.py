import os

import pandas as pd

import random

from sklearn import preprocessing

import matplotlib.pyplot as plt

import category_encoders

from sklearn.decomposition import PCA
os.chdir("/kaggle/input/ieee-fraud-detection")

os.listdir()
path = ''

# import the data

train_identity = pd.read_csv(path + 'train_identity.csv')

train_transaction = pd.read_csv(path + 'train_transaction.csv')

test_identity = pd.read_csv(path + 'test_identity.csv')

test_transaction = pd.read_csv(path + 'test_transaction.csv')



# merge identity and transaction to one dataframe

train = pd.merge(train_transaction, train_identity, on='TransactionID', how='left')

test = pd.merge(test_transaction, test_identity, on='TransactionID', how='left')

# train_raw, test_raw = train.copy(), test.copy()



del train_identity, train_transaction, test_identity, test_transaction
sampleIdx = random.sample([i for i in range(train.shape[0])], k=5000)

train = train.iloc[sampleIdx, :]

train.head()
print(f'Training dataset has {train.shape[0]} observations and {train.shape[1]} features.')

print(f'Test dataset has {test.shape[0]} observations and {test.shape[1]} features.')
Ytr = train["isFraud"]

X = train.drop(["isFraud","TransactionID", "TransactionDT"], axis=1).append(test.drop(["TransactionID", "TransactionDT"], axis=1))

# X_raw, Ytr_raw = X.copy(), Ytr.copy()
# proporation of missing values

missPropor = [X[col].isnull().sum() / X.shape[0] for col in X.columns]

plt.hist(missPropor, bins=30)

plt.ylabel("Frequency")

plt.xlabel("Proportion of missing values")

plt.show()
# delete features with high proporation of missing values

many_null_cols = [X.columns[i] for i in range(X.shape[1]) if missPropor[i] > 0.7]

X = X.drop(many_null_cols, axis=1)

print(f"After deleting features with high proporation of missing values, there are {X.shape[1]} features.")
# fill missing values in categorical variables with their mode.

# fill missing values in numerical variables with their mean.

for i in range(X.shape[1]):

    if missPropor[i] > 0:

        if X.iloc[:, i].dtype == "object":

            X.iloc[:, i] = X.iloc[:, i].fillna(X.iloc[:, i].mode()[0])

        elif X.iloc[:, i].dtype in ['int64', 'float64']:

            X.iloc[:, i] = X.iloc[:, i].fillna(X.iloc[:, i].mean())
# numeric encoding (label encoding)

X_le = X.copy()

for f in X.columns:

    if X_le[f].dtype == 'object': 

        le = preprocessing.LabelEncoder()

        le.fit(list(X_le[f].values))

        X_le[f] = le.transform(list(X_le[f].values))
# binary encoding

X_be = X.copy()

for f in X.columns:

    if X_be[f].dtype == 'object': 

        if X_be[f].nunique() <= 2:

            le = preprocessing.LabelEncoder()

            le.fit(list(X_be[f].values))

            X_be[f] = le.transform(list(X_be[f].values))

        else:

            be = category_encoders.BinaryEncoder(cols=f)

            X_be = be.fit_transform(X_be)
X_le_pca = X_le.copy()
# standardize the data

scaler = preprocessing.StandardScaler()

scaler.fit(X_le_pca)

X_le_pca = scaler.transform(X_le_pca)
# apply PCA

# choose the minimum number of principal components 

# such that 99% of the variance is retained.

pca = PCA(0.99)

pca.fit(X_le_pca)

X_le_pca = pca.transform(X_le_pca)

X_le_pca = pd.DataFrame(X_le_pca)
print(f"Number of features after PCA is {X_le_pca.shape[1]}.")
Xtr_le = X_le.iloc[:train.shape[0], :]

Xte_le = X_le.iloc[train.shape[0]:, :]

Xtr_be = X_be.iloc[:train.shape[0], :]

Xte_be = X_be.iloc[train.shape[0]:, :]

Xtr_le_pca = X_le_pca.iloc[:train.shape[0], :]

Xte_le_pca = X_le_pca.iloc[train.shape[0]:, :]
# Xtr_le.to_csv(f"{path}X_train_labelencoding.csv", index=False)

# Xte_le.to_csv(f"{path}X_test_labelencoding.csv", index=False)

# Xtr_be.to_csv(f"{path}X_train_binaryencoding.csv", index=False)

# Xte_be.to_csv(f"{path}X_test_binaryencoding.csv", index=False)

# Xtr_le_pca.to_csv(f"{path}X_train_labelencoding_pca.csv", index=False)

# Xte_le_pca.to_csv(f"{path}X_test_labelencoding_pca.csv", index=False)

# Ytr.to_csv(f"{path}Y_train.csv", header="isFraud", index=False)
import numpy as np

import pandas as pd

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.model_selection import GridSearchCV

import random

import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression

import xgboost as xgb
# path = '/home/wkm/Documents/Data set/ieee-fraud-detection/'

# Xtr = pd.read_csv(f"{path}X_train_binaryencoding.csv")

# Xte = pd.read_csv(f"{path}X_test_binaryencoding.csv")

# Ytr = pd.read_csv(f"{path}Y_train.csv")
Xtr = Xtr_be

Xte = Xte_be
test_transaction = pd.read_csv(path + 'test_transaction.csv')

submission = pd.DataFrame(test_transaction["TransactionID"])

del test_transaction
lr = LogisticRegression(penalty='l2', max_iter=500, n_jobs=6, tol=1e-6, solver="sag")

lr.fit(Xtr, np.ravel(Ytr))

Yhat_lr = lr.predict_proba(Xte)

submission["isFraud"] = Yhat_lr[:, 1]

# submission.to_csv(f"{path}Y_hat_logistic.csv", index=False)
treeCount = 100



bagging = RandomForestClassifier(max_features="auto", min_samples_leaf=1, n_estimators=treeCount)

bagging.fit(Xtr, np.ravel(Ytr))

Yhat_bagging = bagging.predict_proba(Xte)

submission["isFraud"] = Yhat_bagging[:, 1]

# submission.to_csv(f"{path}Y_hat_bagging.csv", index=False)
# use oob error to find the best max_features

nFeatures = Xtr.shape[1]

oobErrList = list()

mList = [m for m in range(10, nFeatures+1, 30)]



for m in mList:

    rf = RandomForestClassifier(max_features=m, min_samples_leaf=1,\

                                oob_score=True, n_estimators=50)

    rf.fit(Xtr, np.ravel(Ytr))

    oobErrList.append(1-rf.oob_score_)

    print(m, 1-rf.oob_score_)
print(oobErrList)

plt.plot([m for m in range(10, nFeatures+1, 30)], oobErrList)

plt.ylabel('OOB error with (n_estimators=50)')

plt.xlabel('m, the number of variables considered at each split')

plt.show()
treeCount = 1000

m = 15



rf = RandomForestClassifier(max_features=m, min_samples_leaf=1, n_estimators=treeCount)

rf.fit(Xtr, np.ravel(Ytr))

Yhat_rf = rf.predict_proba(Xte)
submission["isFraud"] = Yhat_rf[:, 1]

# submission.to_csv(f"{path}Y_hat_rf_m{m}_t{treeCount}.csv", index=False)
# use default parameters

gbm0 = GradientBoostingClassifier()

gbm0.fit(Xtr, np.ravel(Ytr))

submission["isFraud"] = gbm0.predict_proba(Xte)[:, 1]

# submission.to_csv(f"{path}Y_hat_gbm_default.csv", index=False)
param_test1 = {'n_estimators':range(100, 1200, 100), 'learning_rate':[0.01, 0.1, 1]}

gbm_tune1 = GradientBoostingClassifier(max_features='sqrt', min_samples_leaf=0.001, max_depth=4)



gs1 = GridSearchCV(estimator=gbm_tune1, param_grid=param_test1, iid=False, scoring='roc_auc', n_jobs=6, cv=5)

gs1.fit(Xtr, np.ravel(Ytr))
print(f"The best parameters: {gs1.best_params_}, and the highest mean_test_score is {gs1.best_score_}")
param_test2 = {'max_depth':range(2, 16, 2), 'min_samples_leaf':[10**i for i in range(-5,0)]}

gbm_tune2 = GradientBoostingClassifier(max_features='sqrt', n_estimators=100, learning_rate=0.1)



gs2 = GridSearchCV(estimator=gbm_tune2, param_grid=param_test2, iid=False, scoring='roc_auc', n_jobs=6, cv=5)

gs2.fit(Xtr, np.ravel(Ytr))
print(f"The best parameters: {gs2.best_params_}, and the highest mean_test_score is {gs2.best_score_}")
# use tuned parameters

gbm1 = GradientBoostingClassifier(max_depth=10, min_samples_leaf=0.001, 

                                  learning_rate=0.1, n_estimators=100)

gbm1.fit(Xtr, np.ravel(Ytr))

submission["isFraud"] = gbm1.predict_proba(Xte)[:, 1]

# submission.to_csv(f"{path}Y_hat_gbm_tuned1.csv", index=False)
xgbc = xgb.XGBClassifier(n_jobs=4, max_depth=10, min_samples_leaf=0.001, 

                         learning_rate=0.1, n_estimators=100, eval_metric="auc")

xgbc.fit(Xtr, np.ravel(Ytr))

submission["isFraud"] = xgbc.predict_proba(Xte)[:, 1]

# submission.to_csv(f"{path}Y_hat_xgb_tuned5.csv", index=False)