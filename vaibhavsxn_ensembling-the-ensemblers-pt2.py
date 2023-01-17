import numpy as np

import pandas as pd



from sklearn.model_selection import KFold, StratifiedKFold, StratifiedShuffleSplit

from sklearn.metrics import roc_auc_score

from sklearn.preprocessing import MinMaxScaler



import gc

import os

import time



import lightgbm as lgb

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
seed_val = 42
print(os.listdir("../input"))
# Original data for holdout and test:

x_hold = pd.read_csv('../input/creditensemble/x_hold.csv', index_col=0)

test_x = pd.read_csv('../input/creditensemble/test_x.csv', index_col=0)



# Predictions and target for holdout:

df_holdout = pd.read_csv('../input/creditensemble/holdout_res.csv', index_col=0).clip(0, None)

df_holdout.set_index('SK_ID_CURR', inplace=True, drop=True)



df_tst = pd.read_csv('../input/creditensemble/test_res.csv', index_col=0).clip(0, None)

df_tst.set_index('SK_ID_CURR', inplace=True, drop=True)



# Target for holdout:

y_hold = pd.read_csv('../input/creditensemble/y_hold.csv', index_col=0, header=None)
for column in df_holdout.columns:

    print(column, roc_auc_score(y_hold, df_holdout[column]))
df_corr = df_holdout.corr()

df_corr.style.background_gradient().set_precision(2)
df_holdout.drop(labels='pred_lgb', axis=1, inplace = True)

df_tst.drop(labels='pred_lgb', axis=1, inplace = True)



df_corr = df_holdout.corr()

df_corr.style.background_gradient().set_precision(2)
# Hold out score:

ensemble_holdout = df_holdout.mean(axis=1)

roc_auc_score(y_hold, ensemble_holdout)
# Calculate test submission:

ensemble_sub = df_tst.mean(axis=1)
# Save:

sub_train = pd.DataFrame(df_tst.index)

sub_train['TARGET'] = ensemble_sub.values

sub_train[['SK_ID_CURR', 'TARGET']].to_csv('sub_average.csv', index=False)
from scipy.optimize import nnls
# Find weigths by solving linear regression with constraint:

weights = nnls(df_holdout.values, y_hold.values.ravel())[0]

weights
# Hold out score: multiply, sum, clip values out of range.

ensemble_holdout = (df_holdout.values*weights).sum(axis=1).clip(0,1)

roc_auc_score(y_hold, ensemble_holdout)
from sklearn.linear_model import LogisticRegression
# Put test and hold in one frame:

frames = [df_holdout, df_tst]

hold_test = pd.concat(frames)



# Compute ranking:

ranked_hold_test = hold_test.rank(axis=0)/hold_test.shape[0]



# Split the frames:

ranked_hold = ranked_hold_test.loc[df_holdout.index,:]

ranked_test = ranked_hold_test.loc[df_tst.index,:]
# Match test values to holdout ranks: (this will take some time)

from tqdm import tqdm, trange



def historical_ranking(df_tst, ranked_holdout):

    ranked_test = df_tst.copy()

    for c in df_tst.columns:

        for i in tqdm(df_tst.index):

            value_to_find = df_tst.loc[i,c]

            ranked_test.loc[i,c] = ranked_holdout[c].iloc[(df_holdout[c]-value_to_find).abs().values.argmin()]

    return ranked_test
# Train logistic regression on holdout:

clf = LogisticRegression()

clf.fit(ranked_hold.values, y_hold.values.ravel())
# Hold out score: multiply, sum, stretch values within range.

ensemble_holdout = clf.predict_proba(ranked_hold.values)[:, 1]

ensemble_holdout = (ensemble_holdout - ensemble_holdout.min()) / (ensemble_holdout.max() - ensemble_holdout.min())

roc_auc_score(y_hold, ensemble_holdout)
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PolynomialFeatures



def preprocessing(X, degree):



    poly = PolynomialFeatures(degree)

    scaler = MinMaxScaler()  

    lin_scaler = StandardScaler()

    poly_df = pd.DataFrame(lin_scaler.fit_transform(poly.fit_transform(scaler.fit_transform(X))))

    poly_df['SK_ID_CURR'] = X.index

    poly_df.set_index('SK_ID_CURR', inplace=True, drop=True)

    return poly_df
# Compute poly features:

degree = 2

poly_hold_test = preprocessing(hold_test, degree)



# Split the frames:

poly_hold = poly_hold_test.loc[df_holdout.index,:]

poly_test = poly_hold_test.loc[df_tst.index,:]
# Train logistic regression on holdout:

clf = LogisticRegression()

clf.fit(poly_hold.values, y_hold.values.ravel())
# Hold out score: multiply, sum, clip values out of range.

ensemble_holdout = clf.predict_proba(poly_hold.values)[:, 1]

roc_auc_score(y_hold, ensemble_holdout)
frames = [x_hold, test_x]

x_hold_test = pd.concat(frames)
from sklearn.decomposition import PCA

pca = PCA(n_components=100, random_state=seed_val)

pca.fit(x_hold_test)

pca.explained_variance_ratio_.sum()
# Reduce dimensions add ranks:

pca_hold_test = pd.DataFrame(pca.transform(x_hold_test))

pca_hold_test.set_index(x_hold_test.index, inplace=True)



ranks_pca = pd.concat([ranked_hold_test, pca_hold_test], axis=1)
# Split the frames:

ranks_pca_hold = ranks_pca.loc[df_holdout.index,:]

ranks_pca_test = ranks_pca.loc[df_tst.index,:]
# Train logistic regression:

clf = LogisticRegression()

clf.fit(ranks_pca_hold.values, y_hold.values.ravel())

ensemble_holdout = clf.predict_proba(ranks_pca_hold.values)[:, 1]



#Linear stretch:

ensemble_holdout = (ensemble_holdout - ensemble_holdout.min()) / (ensemble_holdout.max() - ensemble_holdout.min())

roc_auc_score(y_hold, ensemble_holdout)
# Train on test and ave:

sub_train = clf.predict_proba(ranks_pca_test.values)[:, 1]



sub_train = pd.DataFrame(test_x.index)

sub_train['TARGET'] = ensemble_sub.values

sub_train[['SK_ID_CURR', 'TARGET']].to_csv('sub_log_pca_rank.csv', index=False)
wf_hold_test = pd.DataFrame(pca_hold_test.index)

for feature in pca_hold_test.columns:

    for predictor in hold_test.columns:

        col_name = str(predictor)+str(feature)

        wf_hold_test[col_name] = (pca_hold_test[feature]*hold_test[predictor]).values



wf_hold_test.set_index('SK_ID_CURR', inplace=True, drop=True)

wf_hold_test.head()
# Split the frames:

wf_hold = wf_hold_test.loc[df_holdout.index,:]

wf_test = wf_hold_test.loc[df_tst.index,:]
# Train logistic regression:

clf = LogisticRegression()

clf.fit(wf_hold.values, y_hold.values.ravel())

ensemble_holdout = clf.predict_proba(wf_hold.values)[:, 1]



#Linear stretch:

ensemble_holdout = (ensemble_holdout - ensemble_holdout.min()) / (ensemble_holdout.max() - ensemble_holdout.min())

roc_auc_score(y_hold, ensemble_holdout)
from sklearn.linear_model import LinearRegression
lr = LinearRegression()

lr.fit(wf_hold.values, y_hold.values.ravel())

ensemble_holdout = lr.predict(wf_hold.values)



#Linear stretch:

ensemble_holdout = (ensemble_holdout - ensemble_holdout.min()) / (ensemble_holdout.max() - ensemble_holdout.min())

roc_auc_score(y_hold, ensemble_holdout)
import lightgbm as lgb



def kfold_lightgbm(trn_x, trn_y, num_folds=3):

       

    # Cross validation model

    in_folds = StratifiedShuffleSplit(n_splits= num_folds, random_state=seed_val)

        

    # Create arrays and dataframes to store results

    for train_idx, valid_idx in in_folds.split(trn_x, trn_y):

        dtrain = lgb.Dataset(data=trn_x[train_idx], 

                             label=trn_y[train_idx], 

                             free_raw_data=False, silent=True)

        dvalid = lgb.Dataset(data=trn_x[valid_idx], 

                             label=trn_y[valid_idx], 

                             free_raw_data=False, silent=True)



        # LightGBM parameters found by Bayesian optimization

        params = {

            'objective': 'binary',

            'boosting_type': 'gbdt',

            'nthread': 4,

            'learning_rate': 0.02,  # 02,

            'num_leaves': 20,

            'colsample_bytree': 0.9497036,

            'subsample': 0.8715623,

            'subsample_freq': 1,

            'max_depth': 8,

            'reg_alpha': 0.041545473,

            'reg_lambda': 0.0735294,

            'min_split_gain': 0.0222415,

            'min_child_weight': 60, # 39.3259775,

            'seed': seed_val,

            'verbose': -1,

            'metric': 'auc',

        }

        

        clf = lgb.train(

            params=params,

            train_set=dtrain,

            num_boost_round=1000,

            valid_sets=[dtrain, dvalid],

            early_stopping_rounds=200,

            verbose_eval=False

        )



        del dtrain, dvalid

        gc.collect()

    

    return clf
# Train lightGBM:

model = kfold_lightgbm(ranks_pca_hold.values, y_hold.values.ravel())

ensemble_holdout = model.predict(ranks_pca_hold.values)



#Linear stretch:

#ensemble_holdout = (ensemble_holdout - ensemble_holdout.min()) / (ensemble_holdout.max() - ensemble_holdout.min())

roc_auc_score(y_hold, ensemble_holdout)
# Train on test and ave:

sub_train = model.predict(ranks_pca_test.values)



sub_train = pd.DataFrame(test_x.index)

sub_train['TARGET'] = ensemble_sub.values

sub_train[['SK_ID_CURR', 'TARGET']].to_csv('sub_lgb_pca_rank.csv', index=False)
estimators = ['lgb']

estimator = estimators[0] #, 'xgb','ridge','f10_dnn']

j=0

train_x = ranks_pca_hold

train_y = y_hold



#train_len = train_x.shape[0]
folds = StratifiedShuffleSplit(n_splits= 3,

                                random_state=seed_val,

                                test_size = 1/3,

                                train_size = 2/3)



half_folds = StratifiedShuffleSplit(n_splits= 1,

                                random_state=seed_val,

                                test_size = 0.5,

                                train_size = 0.5)
test_len = ranks_pca_test.shape[0]

test_probas = np.zeros((test_len, len(estimators)*3))

test_proba = np.zeros(test_len)



# For every fold:

for i, (train, test) in enumerate(folds.split(train_x, train_y)):

    trn_x = train_x.iloc[train, :]

    trn_y = train_y.iloc[train].values.ravel()

    val_x = train_x.iloc[test, :]

    val_y = train_y.iloc[test].values.ravel()  

    

    val_len = val_x.shape[0]

    estimators_probas = np.zeros((val_len, len(estimators)*3))

    

    for i, (half_train, half_test) in enumerate(half_folds.split(trn_x, trn_y)):

        half_trn_x = trn_x.iloc[half_train, :].values

        half_trn_y = trn_y[half_train].ravel()

        half_val_x = trn_x.iloc[half_test, :].values

        half_val_y = trn_y[half_test].ravel()



        #Train on one part, predict the other:

        if estimator == 'lgb':

            #Train on halves and on the whole set:

            model_a = kfold_lightgbm(half_trn_x, half_trn_y)

            model_b = kfold_lightgbm(half_val_x, half_val_y)

            model = kfold_lightgbm(ranks_pca_hold.values, y_hold.values.ravel())

            

            #Predict val set:

            estimators_probas[:, j*3] = model_a.predict(val_x)

            estimators_probas[:, j*3+1] = model_b.predict(val_x) 

            estimators_probas[:, j*3+2] = model.predict(val_x)

            

            #Predict test set:

            test_probas[:, j*3] = model_a.predict(ranks_pca_test)

            test_probas[:, j*3+1] = model_b.predict(ranks_pca_test) 

            test_probas[:, j*3+2] = model.predict(ranks_pca_test)            

    

    # Train logistic regression on holdout:

    clf = LogisticRegression()

    clf.fit(estimators_probas, val_y.ravel())

    

    # Hold out score: multiply, sum, stretch values within range.

    ensemble_holdout = clf.predict_proba(estimators_probas)[:, 1]

    ensemble_holdout = (ensemble_holdout - ensemble_holdout.min()) / (ensemble_holdout.max() - ensemble_holdout.min())

    print(roc_auc_score( val_y.ravel(), ensemble_holdout))

    

    ensemble_test = clf.predict_proba(test_probas)[:, 1]

    ensemble_test = (ensemble_test - ensemble_test.min()) / (ensemble_test.max() - ensemble_test.min())    

    test_proba += ensemble_test*1/3
# Train on test and ave:

sub_train = test_proba



sub_train = pd.DataFrame(test_x.index)

sub_train['TARGET'] = ensemble_sub.values

sub_train[['SK_ID_CURR', 'TARGET']].to_csv('dragons.csv', index=False)
import pandas as pd

holdout_res = pd.read_csv("../input/creditensemble/holdout_res.csv")

test_res = pd.read_csv("../input/creditensemble/test_res.csv")

test_x = pd.read_csv("../input/creditensemble/test_x.csv")

x_hold = pd.read_csv("../input/creditensemble/x_hold.csv")

y_hold = pd.read_csv("../input/creditensemble/y_hold.csv")