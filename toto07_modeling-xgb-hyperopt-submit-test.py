import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import scipy as sp

from scipy import stats

import matplotlib.pyplot as plt

import seaborn as sns



# Standard plotly imports

#import plotly.plotly as py

import plotly.graph_objs as go

import plotly.tools as tls

from plotly.offline import iplot, init_notebook_mode

#import cufflinks

#import cufflinks as cf

import plotly.figure_factory as ff



# Using plotly + cufflinks in offline mode

init_notebook_mode(connected=True)

#cufflinks.go_offline(connected=True)



# Preprocessing, modelling and evaluating

from sklearn import preprocessing

from sklearn.metrics import confusion_matrix, roc_auc_score

from sklearn.model_selection import StratifiedKFold, cross_val_score, KFold

from xgboost import XGBClassifier

import xgboost as xgb



## Hyperopt modules

from hyperopt import fmin, hp, tpe, Trials, space_eval, STATUS_OK, STATUS_RUNNING

from functools import partial



import os

import gc

print(os.listdir("../input/ieee-fraud-detection"))
def resumetable(df):

    print(f"Dataset Shape: {df.shape}")

    summary = pd.DataFrame(df.dtypes,columns=['dtypes'])

    summary = summary.reset_index()

    summary['Name'] = summary['index']

    summary = summary[['Name','dtypes']]

    summary['Missing'] = df.isnull().sum().values    

    summary['Uniques'] = df.nunique().values

    summary['First Value'] = df.loc[0].values

    summary['Second Value'] = df.loc[1].values

    summary['Third Value'] = df.loc[2].values



    for name in summary['Name'].value_counts().index:

        summary.loc[summary['Name'] == name, 'Entropy'] = round(stats.entropy(df[name].value_counts(normalize=True), base=2),2) 



    return summary



## Function to reduce the DF size

# https://www.kaggle.com/gemartin/load-data-reduce-memory-usage

def reduce_mem_usage(df, verbose=True):

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    start_mem = df.memory_usage().sum() / 1024**2    

    for col in df.columns:

        col_type = df[col].dtypes

        if col_type in numerics:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)  

            else:

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)    

    end_mem = df.memory_usage().sum() / 1024**2

    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))

    return df



def CalcOutliers(df_num): 



    # calculating mean and std of the array

    data_mean, data_std = np.mean(df_num), np.std(df_num)



    # seting the cut line to both higher and lower values

    # You can change this value

    cut = data_std * 3



    #Calculating the higher and lower cut values

    lower, upper = data_mean - cut, data_mean + cut



    # creating an array of lower, higher and total outlier values 

    outliers_lower = [x for x in df_num if x < lower]

    outliers_higher = [x for x in df_num if x > upper]

    outliers_total = [x for x in df_num if x < lower or x > upper]



    # array without outlier values

    outliers_removed = [x for x in df_num if x > lower and x < upper]

    

    print('Identified lowest outliers: %d' % len(outliers_lower)) # printing total number of values in lower cut of outliers

    print('Identified upper outliers: %d' % len(outliers_higher)) # printing total number of values in higher cut of outliers

    print('Total outlier observations: %d' % len(outliers_total)) # printing total number of values outliers of both sides

    print('Non-outlier observations: %d' % len(outliers_removed)) # printing total number of non outlier values

    print("Total percentual of Outliers: ", round((len(outliers_total) / len(outliers_removed) )*100, 4)) # Percentual of outliers in points

    

    return
df_trans = pd.read_csv('../input/ieee-fraud-detection/train_transaction.csv')

df_test_trans = pd.read_csv('../input/ieee-fraud-detection/test_transaction.csv')



df_id = pd.read_csv('../input/ieee-fraud-detection/train_identity.csv')

df_test_id = pd.read_csv('../input/ieee-fraud-detection/test_identity.csv')



sample_submission = pd.read_csv('../input/ieee-fraud-detection/sample_submission.csv', index_col='TransactionID')



df_train = df_trans.merge(df_id, how='left', left_index=True, right_index=True, on='TransactionID')

df_test = df_test_trans.merge(df_test_id, how='left', left_index=True, right_index=True, on='TransactionID')



print(df_train.shape)

print(df_test.shape)



# y_train = df_train['isFraud'].copy()

del df_trans, df_id, df_test_trans, df_test_id

df_train = reduce_mem_usage(df_train)

df_test = reduce_mem_usage(df_test)
useful_features = ['TransactionAmt', 'ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6', 'dist1',

                   'P_emaildomain', 'R_emaildomain', 'C1', 'C2', 'C3','C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13',

                   'C14', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D8', 'D9', 'D10', 'D11', 'D12', 'D13', 'D14', 'D15', 'M1', 'M2', 'M3',

                   'M4', 'M5', 'M6', 'M7', 'M8', 'M9',  'id_01', 'id_02', 'id_03', 'id_04', 'id_05', 'id_06', 'id_09', 'id_10',

                   'id_11', 'id_12', 'id_13', 'id_14', 'id_15', 'id_16', 'id_17', 'id_18', 'id_19', 'id_20', 'id_28', 'id_29', 'id_30', 

                   'id_31', 'id_32', 'id_34', 'id_35', 'id_36', 'id_37', 'id_38', 'DeviceType', 'DeviceInfo']+['V%d'%i for i in range(1,338,1)]
cols_to_drop = [col for col in df_train.columns if col not in useful_features]

cols_to_drop.remove('isFraud')

cols_to_drop.remove('TransactionDT')



train = df_train.drop(cols_to_drop, axis=1)

test = df_test.drop(cols_to_drop, axis=1)
# Some arbitrary features interaction

for feature in ['id_02__id_20', 'id_02__D8', 'D11__DeviceInfo', 'DeviceInfo__P_emaildomain', 'P_emaildomain__C2', 

                'card2__dist1', 'card1__card5', 'card2__id_20', 'card5__P_emaildomain', 'addr1__card1']:



    f1, f2 = feature.split('__')

    train[feature] = train[f1].astype(str) + '_' + train[f2].astype(str)

    test[feature] = test[f1].astype(str) + '_' + test[f2].astype(str)



    le = preprocessing.LabelEncoder()

    le.fit(list(train[feature].astype(str).values) + list(test[feature].astype(str).values))

    train[feature] = le.transform(list(train[feature].astype(str).values))

    test[feature] = le.transform(list(test[feature].astype(str).values))



# Encoding - count encoding separately for train and test

for feature in ['id_01', 'id_31', 'id_33', 'id_36']:

    train[feature + '_count_dist'] = train[feature].map(train[feature].value_counts(dropna=False))

    test[feature + '_count_dist'] = test[feature].map(test[feature].value_counts(dropna=False))
test['isFraud'] = 'test'

df = pd.concat([train, test], axis=0, sort=False)

df = df.reset_index()

df = df.drop('index', axis=1)

del train,test
columns_a = ['TransactionAmt', 'id_02', 'D15']

columns_b = ['card1', 'card4', 'addr1']



for col_a in columns_a:

    for col_b in columns_b:

        df[f'{col_a}_to_mean_{col_b}'] = df[col_a] / df.groupby([col_b])[col_a].transform('mean')

        df[f'{col_a}_to_std_{col_b}'] = df[col_a] / df.groupby([col_b])[col_a].transform('std')



# New feature - decimal part of the transaction amount.

df['TransactionAmt_decimal'] = ((df['TransactionAmt'] - df['TransactionAmt'].astype(int)) * 1000).astype(int)



# New feature - log of transaction amount.

df['TransactionAmt'] = np.log(df['TransactionAmt'])

df['TransactionAmt'] = np.log(df['TransactionAmt'])



# New feature - day of week in which a transaction happened.

import datetime

START_DATE = '2017-12-01'

startdate = datetime.datetime.strptime(START_DATE,'%Y-%m-%d')

df['Date'] = df['TransactionDT'].apply(\

    lambda x:(startdate+datetime.timedelta(seconds=x)))

df['Transaction_day_of_week'] = df['Date'].dt.dayofweek

df['Transaction_hour'] = df['Date'].dt.hour

df['Transaction_days'] = df['Date'].dt.day

del df['Date']



for col in ['Transaction_day_of_week','Transaction_hour','Transaction_days']:

    from sklearn.preprocessing import minmax_scale

    df[col] = (minmax_scale(df[col], feature_range=(0,1)))

    

# Encoding - count encoding for both train and test

for feature in ['card1', 'card2', 'card3', 'card4', 'card5', 'card6', 'id_36']:

    df[feature + '_count_full'] = df[feature].map(df[feature].value_counts(dropna=False))
df_train, df_test = df[df['isFraud'] != 'test'], df[df['isFraud'] == 'test'].drop('isFraud', axis=1)
# Label Encoding

for f in df_train.drop('isFraud', axis=1).columns:

    if df_train[f].dtype=='object' or df_test[f].dtype=='object': 

        print(f)

        try:

            lbl = preprocessing.LabelEncoder()

            lbl.fit(list(df_train[f].values) + list(df_test[f].values))

            df_train[f] = lbl.transform(list(df_train[f].values))

            df_test[f] = lbl.transform(list(df_test[f].values))   



        except:

            print('drop %s'%f)

            df_train = df_train.drop(f, axis=1)

            df_test = df_test.drop(f, axis=1)
df_test['isFraud'] = 'test'

df = pd.concat([df_train, df_test], axis=0, sort=False )

df = df.reset_index()

df = df.drop('index', axis=1)
def PCA_change(df, cols, n_components, prefix='PCA_', rand_seed=4):

    pca = PCA(n_components=n_components, random_state=rand_seed)



    principalComponents = pca.fit_transform(df[cols])



    principalDf = pd.DataFrame(principalComponents)



    df.drop(cols, axis=1, inplace=True)



    principalDf.rename(columns=lambda x: str(prefix)+str(x), inplace=True)



    df = pd.concat([df, principalDf], axis=1)

    

    return df
mas_v = df.columns[df.columns.str.startswith('V')]

mas_v
from sklearn.preprocessing import minmax_scale

from sklearn.decomposition import PCA

# from sklearn.cluster import KMeans



for col in mas_v:

    df[col] = df[col].fillna((df[col].min() - 2))

    df[col] = (minmax_scale(df[col], feature_range=(0,1)))



    

df = PCA_change(df, mas_v, prefix='PCA_V_', n_components=30)
df = reduce_mem_usage(df)
df_train, df_test = df[df['isFraud'] != 'test'], df[df['isFraud'] == 'test'].drop('isFraud', axis=1)
df_train.shape
X_train = df_train.sort_values('TransactionDT').drop(['isFraud', 

                                                      'TransactionDT', 

                                                      #'Card_ID'

                                                     ],

                                                     axis=1)

y_train = df_train.sort_values('TransactionDT')['isFraud'].astype(bool)



X_test = df_test.sort_values('TransactionDT').drop(['TransactionDT',

                                                    #'Card_ID'

                                                   ], 

                                                   axis=1)

del df_train

df_test = df_test[["TransactionDT"]]
column_names = X_train.columns

'TransactionID' in column_names
from sklearn.model_selection import KFold,TimeSeriesSplit

from sklearn.metrics import roc_auc_score

from xgboost import plot_importance

from sklearn.metrics import make_scorer



import time

def objective(params):

    time1 = time.time()

    params = {

        'max_depth': int(params['max_depth']),

        'gamma': "{:.3f}".format(params['gamma']),

        'subsample': "{:.2f}".format(params['subsample']),

        'reg_alpha': "{:.3f}".format(params['reg_alpha']),

        'reg_lambda': "{:.3f}".format(params['reg_lambda']),

        'learning_rate': "{:.3f}".format(params['learning_rate']),

        'num_leaves': '{:.3f}'.format(params['num_leaves']),

        'colsample_bytree': '{:.3f}'.format(params['colsample_bytree']),

        'min_child_samples': '{:.3f}'.format(params['min_child_samples']),

        'feature_fraction': '{:.3f}'.format(params['feature_fraction']),

        'bagging_fraction': '{:.3f}'.format(params['bagging_fraction'])

    }



    print("\n############## New Run ################")

    print(f"params = {params}")

    FOLDS = 7

    count=1

    skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)



    tss = TimeSeriesSplit(n_splits=FOLDS)

    y_preds = np.zeros(sample_submission.shape[0])

    y_oof = np.zeros(X_train.shape[0])

    score_mean = 0

    for tr_idx, val_idx in tss.split(X_train, y_train):#0829去除ID的分组[column_names[1:]]

        clf = xgb.XGBClassifier(

            n_estimators=600, random_state=4, verbose=True, 

            tree_method='gpu_hist', 

            **params

        )



        X_tr, X_vl = X_train.iloc[tr_idx, :], X_train.iloc[val_idx, :]#X_train.iloc[tr_idx, 1:], X_train.iloc[val_idx, 1:]

        y_tr, y_vl = y_train.iloc[tr_idx], y_train.iloc[val_idx]

        

        clf.fit(X_tr, y_tr)

        #y_pred_train = clf.predict_proba(X_vl)[:,1]

        #print(y_pred_train)

        score = make_scorer(roc_auc_score, needs_proba=True)(clf, X_vl, y_vl)

        # plt.show()

        score_mean += score

        print(f'{count} CV - score: {round(score, 4)}')

        count += 1

    time2 = time.time() - time1

    print(f"Total Time Run: {round(time2 / 60,2)}")

    gc.collect()

    print(f'Mean ROC_AUC: {score_mean / FOLDS}')

    del X_tr, X_vl, y_tr, y_vl, clf, score

    return -(score_mean / FOLDS)





space = {

    # The maximum depth of a tree, same as GBM.

    # Used to control over-fitting as higher depth will allow model 

    # to learn relations very specific to a particular sample.

    # Should be tuned using CV.

    # Typical values: 3-10

    'max_depth': hp.quniform('max_depth', 7, 23, 1),

    

    # reg_alpha: L1 regularization term. L1 regularization encourages sparsity 

    # (meaning pulling weights to 0). It can be more useful when the objective

    # is logistic regression since you might need help with feature selection.

    'reg_alpha':  hp.uniform('reg_alpha', 0.01, 0.4),

    

    # reg_lambda: L2 regularization term. L2 encourages smaller weights, this

    # approach can be more useful in tree-models where zeroing 

    # features might not make much sense.

    'reg_lambda': hp.uniform('reg_lambda', 0.01, .4),

    

    # eta: Analogous to learning rate in GBM

    # Makes the model more robust by shrinking the weights on each step

    # Typical final values to be used: 0.01-0.2

    'learning_rate': hp.uniform('learning_rate', 0.01, 0.2),

    

    # colsample_bytree: Similar to max_features in GBM. Denotes the 

    # fraction of columns to be randomly samples for each tree.

    # Typical values: 0.5-1

    'colsample_bytree': hp.uniform('colsample_bytree', 0.3, .9),

    

    # A node is split only when the resulting split gives a positive

    # reduction in the loss function. Gamma specifies the 

    # minimum loss reduction required to make a split.

    # Makes the algorithm conservative. The values can vary depending on the loss function and should be tuned.

    'gamma': hp.uniform('gamma', 0.01, .7),

    

    # more increases accuracy, but may lead to overfitting.

    # num_leaves: the number of leaf nodes to use. Having a large number 

    # of leaves will improve accuracy, but will also lead to overfitting.

    'num_leaves': hp.choice('num_leaves', list(range(20, 250, 10))),

    

    # specifies the minimum samples per leaf node.

    # the minimum number of samples (data) to group into a leaf. 

    # The parameter can greatly assist with overfitting: larger sample

    # sizes per leaf will reduce overfitting (but may lead to under-fitting).

    'min_child_samples': hp.choice('min_child_samples', list(range(100, 250, 10))),

    

    # subsample: represents a fraction of the rows (observations) to be 

    # considered when building each subtree. Tianqi Chen and Carlos Guestrin

    # in their paper A Scalable Tree Boosting System recommend 

    'subsample': hp.choice('subsample', [0.2, 0.4, 0.5, 0.6, 0.7, .8, .9]),

    

    # randomly select a fraction of the features.

    # feature_fraction: controls the subsampling of features used

    # for training (as opposed to subsampling the actual training data in 

    # the case of bagging). Smaller fractions reduce overfitting.

    'feature_fraction': hp.uniform('feature_fraction', 0.4, .8),

    

    # randomly bag or subsample training data.

    'bagging_fraction': hp.uniform('bagging_fraction', 0.4, .9)

    

    # bagging_fraction and bagging_freq: enables bagging (subsampling) 

    # of the training data. Both values need to be set for bagging to be used.

    # The frequency controls how often (iteration) bagging is used. Smaller

    # fractions and frequencies reduce overfitting.

}

# Set algoritm parameters

best = fmin(fn=objective,

            space=space,

            algo=tpe.suggest,

            max_evals=27)



# Print best parameters

best_params = space_eval(space, best)
print("BEST PARAMS: ", best_params)



best_params['max_depth'] = int(best_params['max_depth'])
clf = xgb.XGBClassifier(

    n_estimators=300,

    **best_params,

    tree_method='gpu_hist'

)



clf.fit(X_train[column_names[1:]], y_train)



y_preds = clf.predict_proba(X_test[column_names[1:]])[:,1] 
feature_important = clf.get_booster().get_score(importance_type="weight")

keys = list(feature_important.keys())

values = list(feature_important.values())



data = pd.DataFrame(data=values, index=keys, columns=["score"]).sort_values(by = "score", ascending=False)



# Top 10 features

data.head(20)


sample_submission['isFraud'] = y_preds

sample_submission.to_csv('submission.csv')