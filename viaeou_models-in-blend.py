# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import gc
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
def merge_data():
    print('merge data ...'.center(50, '*'))
    gc.enable()
    bur_bal = pd.read_csv('../input/bureau_balance.csv')
    print('bureau_balance shape:', bur_bal.shape)
    #bur_bal.head()
    bur_bal = pd.concat([bur_bal, pd.get_dummies(bur_bal.STATUS, prefix='bur_bal_status')],
                       axis=1).drop('STATUS', axis=1)
    bur_cnts = bur_bal[['SK_ID_BUREAU', 'MONTHS_BALANCE']].groupby('SK_ID_BUREAU').count()
    bur_bal['bur_cnt'] = bur_bal['SK_ID_BUREAU'].map(bur_cnts['MONTHS_BALANCE'])
    avg_bur_bal = bur_bal.groupby('SK_ID_BUREAU').mean()
    avg_bur_bal.columns = ['bur_bal_' + f_ for f_ in avg_bur_bal.columns]
    del bur_bal
    gc.collect()

    bur = pd.read_csv('../input/bureau.csv')
    print('bureau shape:', bur.shape)
    #bur.head()
    bur_credit_active_dum = pd.get_dummies(bur.CREDIT_ACTIVE, prefix='ca')
    bur_credit_currency_dum = pd.get_dummies(bur.CREDIT_CURRENCY, prefix='cc')
    bur_credit_type_dum = pd.get_dummies(bur.CREDIT_TYPE, prefix='ct')

    bur_full = pd.concat([bur, bur_credit_active_dum, bur_credit_currency_dum, bur_credit_type_dum], axis=1).drop(['CREDIT_ACTIVE', 'CREDIT_CURRENCY', 'CREDIT_TYPE'], axis=1)
    del bur_credit_active_dum, bur_credit_currency_dum, bur_credit_type_dum
    gc.collect()
    bur_full = bur_full.merge(right=avg_bur_bal.reset_index(), how='left', on='SK_ID_BUREAU',suffixes=('', '_bur_bal'))
    nb_bureau_per_curr = bur_full[['SK_ID_CURR', 'SK_ID_BUREAU']].groupby('SK_ID_CURR').count()
    bur_full['SK_ID_BUREAU'] = bur_full['SK_ID_CURR'].map(nb_bureau_per_curr['SK_ID_BUREAU'])
    avg_bur = bur_full.groupby('SK_ID_CURR').mean()
    avg_bur.columns = ['bur_' + f_ for f_ in avg_bur.columns]
    del bur, bur_full, avg_bur_bal
    gc.collect()

    prev = pd.read_csv('../input/previous_application.csv')
    print('previous_application shape:', prev.shape)
    #prev.head()
    prev_cat_features = [f_ for f_ in prev.columns if prev[f_].dtype == 'object']
    prev_dum = pd.DataFrame()
    for f_ in prev_cat_features:
        prev_dum = pd.concat([prev_dum, pd.get_dummies(prev[f_], prefix=f_)], axis=1)
    prev = pd.concat([prev, prev_dum],axis=1)
    del prev_dum
    gc.collect()
    nb_prev_per_curr = prev[['SK_ID_CURR', 'SK_ID_PREV']].groupby('SK_ID_CURR').count()
    prev['SK_ID_PREV'] = prev['SK_ID_CURR'].map(nb_prev_per_curr['SK_ID_PREV'])
    avg_prev = prev.groupby('SK_ID_CURR').mean()
    avg_prev.columns = ['prev_' + f_ for f_ in avg_prev.columns]
    del prev
    gc.collect()

    pos = pd.read_csv('../input/POS_CASH_balance.csv')
    print('pos_cash_balance shape:', pos.shape)
    #pos.head()
    pos = pd.concat([pos, pd.get_dummies(pos['NAME_CONTRACT_STATUS'], prefix='ncs')], axis=1)
    nb_prevs = pos[['SK_ID_CURR', 'SK_ID_PREV']].groupby('SK_ID_CURR').count()
    pos['SK_ID_PREV'] = pos['SK_ID_CURR'].map(nb_prevs['SK_ID_PREV'])
    avg_pos = pos.groupby('SK_ID_CURR').mean()
    avg_pos.columns = ['pos_' + f_ for f_ in avg_pos.columns]
    del pos, nb_prevs
    gc.collect()

    cc_bal = pd.read_csv('../input/credit_card_balance.csv')
    print('credit_card_balance shape:', cc_bal.shape)
    cc_bal = pd.concat([cc_bal, pd.get_dummies(cc_bal['NAME_CONTRACT_STATUS'], prefix='ncs')], axis=1)
    nb_prevs = cc_bal[['SK_ID_CURR', 'SK_ID_PREV']].groupby('SK_ID_CURR').count()
    cc_bal['SK_ID_PREV'] = cc_bal['SK_ID_CURR'].map(nb_prevs['SK_ID_PREV'])
    avg_cc_bal = cc_bal.groupby('SK_ID_CURR').mean()
    avg_cc_bal.columns = ['cc_bal_' + f_ for f_ in avg_cc_bal.columns]
    del cc_bal, nb_prevs
    gc.collect()

    inst = pd.read_csv('../input/installments_payments.csv')
    print('installment_payment shape:', inst.shape)
    nb_prevs = inst[['SK_ID_CURR', 'SK_ID_PREV']].groupby('SK_ID_CURR').count()
    inst['SK_ID_PREV'] = inst['SK_ID_CURR'].map(nb_prevs['SK_ID_PREV'])
    avg_inst = inst.groupby('SK_ID_CURR').mean()
    avg_inst.columns = ['inst_' + f_ for f_ in avg_inst.columns]
    del inst, nb_prevs
    gc.collect()

    train = pd.read_csv('../input/application_train.csv')
    test = pd.read_csv('../input/application_test.csv')
    print('train shape:', train.shape)
    print('test shape:', test.shape)
    y = train['TARGET']
    del train['TARGET']
    cat_feats = [f_ for f_ in train.columns if train[f_].dtype == 'object']
    for f_ in cat_feats:
        train[f_], indexer = pd.factorize(train[f_])#类似于类似于类似于label encoder
        test[f_] = indexer.get_indexer(test[f_])
    train = train.merge(right = avg_bur.reset_index(), how='left', on='SK_ID_CURR')
    test = test.merge(right = avg_bur.reset_index(), how='left', on='SK_ID_CURR')
    train = train.merge(right = avg_prev.reset_index(), how='left', on='SK_ID_CURR')
    test = test.merge(right = avg_prev.reset_index(), how='left', on='SK_ID_CURR')
    train = train.merge(right = avg_pos.reset_index(), how='left', on='SK_ID_CURR')
    test = test.merge(right = avg_pos.reset_index(), how='left', on='SK_ID_CURR')
    train = train.merge(right = avg_cc_bal.reset_index(), how='left', on='SK_ID_CURR')
    test = test.merge(right = avg_cc_bal.reset_index(), how='left', on='SK_ID_CURR')
    train = train.merge(right = avg_inst.reset_index(), how='left', on='SK_ID_CURR')
    test = test.merge(right = avg_inst.reset_index(), how='left', on='SK_ID_CURR')
    del avg_bur, avg_prev, avg_pos, avg_cc_bal, avg_inst
    gc.collect()
    return train, test, y
train, test, y = merge_data()
def feat_ext_source(df):
    x1 = df['EXT_SOURCE_1'].fillna(-1) + 1e-1
    x2 = df['EXT_SOURCE_2'].fillna(-1) + 1e-1
    x3 = df['EXT_SOURCE_3'].fillna(-1) + 1e-1
    
    df['EXT_SOURCE_1over2_NAminus1_Add0.1'] = x1/x2
    df['EXT_SOURCE_2over1_NAminus1_Add0.1'] = x2/x1
    df['EXT_SOURCE_1over3_NAminus1_Add0.1'] = x1/x3
    df['EXT_SOURCE_3over1_NAminus1_Add0.1'] = x3/x1
    df['EXT_SOURCE_2over3_NAminus1_Add0.1'] = x2/x3
    df['EXT_SOURCE_3over2_NAminus1_Add0.1'] = x3/x2
    df['EXT_SOURCE_1_log'] = np.log(df['EXT_SOURCE_1'] + 1)
    df['EXT_SOURCE_2_log'] = np.log(df['EXT_SOURCE_2'] + 1)
    df['EXT_SOURCE_3_log'] = np.log(df['EXT_SOURCE_3'] + 1) 
    return df
train = feat_ext_source(train)
test = feat_ext_source(test)
def nan_process(train, test):
    print('NaN process ...'.center(50, '*'))
    print(train.shape, test.shape)
    train = train.fillna(-1)
    test = test.fillna(-1)
    return train, test, y

train, test, y = nan_process(train, test)
def smote(train, test, y):
    print('smote ...'.center(50, '*'))
    train, val_x, y, val_y = train_test_split(train, y, test_size=0.1, random_state = 14)
    features = train.columns
    #sm = SMOTE(random_state=42, kind='borderline2', n_jobs=-1)
    #train, y = sm.fit_sample(train, y)
    return train, test, y, val_x, val_y, features

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
train, test, y, val_x, val_y, features = smote(train, test, y)
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
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
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df

def reduce(train, test, y, val_x, val_y, features):
    print('reduce memory usage...'.center(50, '*'))
    train = pd.DataFrame(train)
    val_x = pd.DataFrame(val_x)
    test = pd.DataFrame(test)
    train = reduce_mem_usage(train)
    val_x = reduce_mem_usage(val_x)
    test = reduce_mem_usage(test)
    feats = [f_ for f_ in features if f_ not in ['SK_ID_CURR']]
    y = pd.DataFrame(y)
    val_y = pd.DataFrame(val_y)
    train.columns = features
    train = train[feats]
    test.columns = features
    try:
        train.to_csv('train_.csv', index=False)
        y.to_csv('y_.csv', index=False)
        test.to_csv('test_.csv', index=False)
        val_x.to_csv('val_x_.csv', index=False)
        val_y.to_csv('val_y_.csv', index=False)
    except IOError:
        print('write to csv failed!')
    return train, test, y,val_x, val_y, features, feats

train, test, y,val_x, val_y, features, feats = reduce(train, test, y, val_x, val_y, features)
train_x = pd.read_csv('train_.csv')
train_y = pd.read_csv('y_.csv')
val_x = pd.read_csv('val_x_.csv')
val_y = pd.read_csv('val_y_.csv')
train_x.head()
val_x.head()
val_x = val_x.drop(columns=['SK_ID_CURR'])
val_x.head()
train_x, train_y, val_x, val_y = train_x.values, train_y.values, val_x.values, val_y.values
def train_model(train_x, train_y, val_x_, val_y_, folds_):
    
    oof_preds = np.zeros(train_x.shape[0])
    val_preds = np.zeros(val_x_.shape[0])
    
    for n_fold, (trn_idx, val_idx) in enumerate(folds_.split(train_x)):
        #print(train_.type)
        trn_x, trn_y = pd.DataFrame(train_x).iloc[trn_idx], pd.DataFrame(train_y).iloc[trn_idx]
        val_x, val_y = pd.DataFrame(train_x).iloc[val_idx], pd.DataFrame(train_y).iloc[val_idx]
        
        clf = LGBMClassifier(
            n_estimators = 4000,
            learning_rate = 0.03,
            num_leaves = 30,
            colsample_bytree = .8,
            subsample = .9,
            max_depth = 7,
            reg_alpha = .1,
            min_split_gain = .01,
            min_child_weight = 2,
            silent = -1,
            verbose = -1
            )
        clf.fit(trn_x, trn_y, 
            eval_set = [(trn_x, trn_y), (val_x, val_y)],
            eval_metric = 'auc', verbose = 100, early_stopping_rounds = 100)
        
        oof_preds[val_idx] = clf.predict_proba(val_x, num_iteration = clf.best_iteration_)[:, 1]
        val_preds += clf.predict_proba(pd.DataFrame(val_x_))[:, 1] / folds_.n_splits
        print('fold %2d validate AUC score %.6f'%(n_fold + 1,roc_auc_score(val_y_, val_preds * folds_.n_splits)))
        print('fold %2d AUC %.6f'%(n_fold+1, roc_auc_score(val_y, oof_preds[val_idx])))
        del clf, trn_x, trn_y, val_x, val_y
        gc.collect()
    print('validate AUC score %.6f'%roc_auc_score(val_y_, val_preds))
    print('full AUC score %.6f'%roc_auc_score(train_y, oof_preds))
    
    return oof_preds, val_preds
from sklearn.model_selection import KFold
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
folds = KFold(n_splits=5, shuffle=True, random_state=0)
oof_preds, val_preds = train_model(train_x, train_y, val_x, val_y, folds)
def rf_model(train_x, train_y, val_x_, val_y_):
    
    val_preds = np.zeros(val_x_.shape[0])  
    clf = RandomForestClassifier(
        max_features = 'sqrt',
        n_estimators = 400,
        min_samples_leaf = 10,
        n_jobs = -1,
        random_state = 14,
        oob_score = True
        )
    clf.fit(train_x, train_y)
    val_preds = clf.predict_proba(pd.DataFrame(val_x_))[:, 1]
    print('validate AUC score %.6f'%roc_auc_score(val_y_, val_preds))
    print('full AUC score %.6f'%roc_auc_score(train_y, clf.predict_proba(train_x)[:,1]))
    
    return oof_preds, val_preds
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
oof_preds, val_preds = rf_model(train_x, train_y, val_x, val_y)
def et_model(train_x, train_y, val_x_, val_y_):
    
    val_preds = np.zeros(val_x_.shape[0])  
    clf = ExtraTreesClassifier(
        max_features = 'sqrt',
        n_estimators = 400,
        min_samples_leaf = 10,
        n_jobs = -1,
        random_state = 14,
        oob_score = True,
        bootstrap = True
        )
    clf.fit(train_x, train_y)
    val_preds = clf.predict_proba(pd.DataFrame(val_x_))[:, 1]
    print('validate AUC score %.6f'%roc_auc_score(val_y_, val_preds))
    try:
        print('full AUC score %.6f'%roc_auc_score(train_y, clf.predict_proba(train_x)[:,1]))
    except:
        print('oob_score error')
    return oof_preds, val_preds
oof_preds, val_preds = et_model(train_x, train_y, val_x, val_y)
def gbdt_model(train_x, train_y, val_x_, val_y_):
    
    val_preds = np.zeros(val_x_.shape[0])  
    clf = GradientBoostingClassifier(
        max_features = 'sqrt',
        n_estimators = 500,
        learning_rate = 0.3,
        subsample = 0.8,
        min_samples_leaf = 5,
        random_state = 14
        )
    clf.fit(train_x, train_y)
    val_preds = clf.predict_proba(pd.DataFrame(val_x_))[:, 1]
    
    print('validate AUC score %.6f'%roc_auc_score(val_y_, val_preds))
    try:
        print('full AUC score %.6f'%roc_auc_score(train_y, clf.predict_proba(train_x)[:,1]))
    except:
        print('error!')
    return oof_preds, val_preds
oof_preds, val_preds = gbdt_model(train_x, train_y, val_x, val_y)
def xgb_model(train_x, train_y, val_x_, val_y_):
    
    val_preds = np.zeros(val_x_.shape[0])  
    clf = XGBClassifier(
        min_child_weight = 0.01,
        n_jobs = -1,
        learning_rate = 0.3,
        n_estimators = 500,
        gamma = 0.5,
        subsample = 0.8,
        colsample_bytree = 0.8,
        booster = 'gbtree',
        scale_pos_weight = 2,
        reg_alpha = 1,
        random_state = 14
        )
    clf.fit(train_x, train_y)
    val_preds = clf.predict_proba(pd.DataFrame(val_x_))[:, 1]
    
    print('validate AUC score %.6f'%roc_auc_score(val_y_, val_preds))
    try:
        print('full AUC score %.6f'%roc_auc_score(train_y, clf.predict_proba(train_x)[:,1]))
    except:
        print('error!')
    return oof_preds, val_preds
oof_preds, val_preds = xgb_model(train_x, train_y, val_x, val_y)
