import pandas as pd

import numpy as np

import datetime as dt

import matplotlib.pyplot as plt

import seaborn as sns

import lightgbm

import os

import gc

import scipy.stats as stats

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import KFold, StratifiedKFold

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from boruta import BorutaPy

from joblib import Parallel, delayed

!ls ../input/dsa-junho
N_SPLITS = 5

RANDOM_SEED = 1234



FEATS_EXCLUDED = ['first_active_month', 'target', 'card_id',

                  'hist_purchase_date_max', 'hist_purchase_date_min', 'hist_card_id_size',

                 'new_purchase_date_max', 'new_purchase_date_min', 'new_card_id_size']



np.random.seed(RANDOM_SEED)
%%time

train_data = pd.read_csv('../input/competicao-dsa-machine-learning-jun-2019/dataset_treino.csv')

test_data = pd.read_csv('../input/competicao-dsa-machine-learning-jun-2019/dataset_teste.csv')

history = pd.read_csv('../input/dsa-junho/transacoes_historicas.csv/transacoes_historicas.csv', 

                    dtype={'city_id': np.int16, 'installments': np.int16, 'merchant_category_id': np.int16, 

                   'month_lag': np.int8, 'purchace_amount': np.float32, 'category_2': np.float32,

                   'state_id': np.int8, 'subsector_id': np.int8})

new_transact = pd.read_csv('../input/dsa-junho/novas_transacoes_comerciantes.csv/novas_transacoes_comerciantes.csv',

                   dtype={'city_id': np.int16, 'installments': np.int16, 'merchant_category_id': np.int16, 

                   'month_lag': np.int8, 'purchace_amount': np.float32, 'category_2': np.float32,

                   'state_id': np.int8, 'subsector_id': np.int8})
# Missing values

def fill_na(data):

    data['category_2'].fillna(1.0,inplace=True) #Moda

    data['category_3'].fillna('A',inplace=True) #Moda

    data['merchant_id'].fillna('M_ID_00a6ca8a8a',inplace=True) #Moda

    data['installments'].replace(-1, np.nan,inplace=True)

    data['installments'].replace(999, np.nan,inplace=True)



    data['authorized_flag'] = data['authorized_flag'].map({'Y': 1, 'N': 0}).astype(np.uint8)

    data['category_1'] = data['category_1'].map({'Y': 1, 'N': 0}).astype(np.uint8)

    data['category_3'] = data['category_3'].map({'A':0, 'B':1, 'C':2}).astype(np.uint8)

    

    data['merchant_id'] = data['merchant_id'].astype('category').cat.codes

    data['merchant_category_id'] = data['merchant_category_id'].astype('category').cat.codes
fill_na(history)

fill_na(new_transact)

gc.collect()
def get_features(data, prefix, num_rows=None):

    # Colunas com tipo Datetime

    print('Feature datetime')

    data['purchase_date'] = pd.to_datetime(data['purchase_date'])

    data['month'] = data['purchase_date'].dt.month.astype(np.uint8)

    data['day'] = data['purchase_date'].dt.day.astype(np.uint8)

    data['hour'] = data['purchase_date'].dt.hour.astype(np.uint8)

    data['weekofyear'] = data['purchase_date'].dt.weekofyear.astype(np.uint8)

    data['weekday'] = data['purchase_date'].dt.weekday.astype(np.uint8)

    data['weekend'] = (data['purchase_date'].dt.weekday >=5).astype(np.uint8)

    

    



    data['parcela'] = (data['purchase_amount'] / data['installments']).astype(np.float16)



    print('Features holidays')

    data['natal_2017']=(pd.to_datetime('2017-12-25')-data['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0).astype(np.uint8)

    data['dia_maes_2017']=(pd.to_datetime('2017-05-13')-data['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0).astype(np.uint8)

    data['dia_pais_2017']=(pd.to_datetime('2017-08-13')-data['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0).astype(np.uint8)

    data['dia_criancas_2017']=(pd.to_datetime('2017-10-12')-data['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0).astype(np.uint8)

    data['dia_namorados_2017']=(pd.to_datetime('2017-06-12')-data['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0).astype(np.uint8)

    data['black_friday_2017']=(pd.to_datetime('2017-11-24') - data['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0).astype(np.uint8)

    data['dia_maes_2018']=(pd.to_datetime('2018-05-13')-data['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0).astype(np.uint8)



    data['month_diff'] = (((dt.datetime.today() - data['purchase_date']).dt.days)//30).astype(np.uint8)

    data['month_diff'] += data['month_lag']



    data['prazo'] = (data['purchase_amount']*data['month_diff']).astype(np.float16)

    data['amount_month_ratio'] = (data['purchase_amount']/data['month_diff']).astype(np.float16)



    col_unique =['subsector_id', 'merchant_id', 'merchant_category_id']

    col_seas = ['month', 'hour', 'weekofyear', 'weekday', 'day']

    aggs = {}

    for col in col_unique:

        aggs[col] = ['nunique']



    for col in col_seas:

        aggs[col] = ['nunique', 'mean', 'min', 'max'] 



    def interval(fun):

        def diff_interval_(x):

            if len(x)>2:

                diff = np.diff(x.sort_values()).astype(np.int64)*1e-9                

                return fun(diff)

            else:

                return np.nan

        diff_interval_.__name__ = 'interval_{}'.format(fun.__name__)

        return diff_interval_



    aggs['purchase_amount'] = ['sum','max','min','mean','var','skew']

    aggs['installments'] = ['sum','max','mean','var','skew']

    aggs['purchase_date'] = ['max','min', interval(np.mean), interval(np.min), interval(np.max), interval(np.std)]

    aggs['month_lag'] = ['max','min','mean','var','skew']

    aggs['month_diff'] = ['max','min','mean','var','skew']

    aggs['authorized_flag'] = ['mean']

    aggs['weekend'] = ['mean'] 

    aggs['weekday'] = ['mean'] 

    aggs['day'] = ['nunique', 'mean', 'min'] 

    aggs['category_1'] = ['mean', lambda x: stats.mode(x)[0][0]]

    aggs['category_2'] = ['mean']

    aggs['category_3'] = ['mean']

    aggs['card_id'] = ['size','count']

    aggs['parcela'] = ['sum','mean','max','min','var']

    aggs['natal_2017'] = ['mean']

    aggs['dia_maes_2017'] = ['mean']

    aggs['dia_pais_2017'] = ['mean']

    aggs['dia_criancas_2017'] = ['mean']

    aggs['dia_namorados_2017'] = ['mean']

    aggs['black_friday_2017'] = ['mean']

    aggs['dia_maes_2018'] = ['mean']

    aggs['prazo']=['mean','min','max','var','skew']

    aggs['amount_month_ratio']=['mean','min','max','var','skew']



    print('Agregacoes iniciais')

    for col in ['category_2','category_3','merchant_id']:

        data[col+'_mean'] = data.groupby([col])['purchase_amount'].transform('mean').astype(np.float16)

        data[col+'_std'] = data.groupby([col])['purchase_amount'].transform('std').astype(np.float16)

        data[col+'_min'] = data.groupby([col])['purchase_amount'].transform('min').astype(np.float16)

        data[col+'_max'] = data.groupby([col])['purchase_amount'].transform('max').astype(np.float16)

        data[col+'_sum'] = data.groupby([col])['purchase_amount'].transform('sum').astype(np.float16)

        aggs[col+'_mean'] = ['mean']

        aggs[col+'_std'] = ['mean']



    print('Iniciando Agregacao...')

    # S2 paralelismo ;)

    def aggregate(df):

        return df.groupby('card_id').agg(aggs)

    data['reduced_card_id'] = data.reset_index()['card_id'].str[:8]

    data_parts = Parallel(n_jobs=-1, verbose=1)(delayed(aggregate)(group) for _,group in data.groupby('reduced_card_id'))

    data = pd.concat(data_parts, axis=0)

    print('Agregado.')



    # Padroniza nomes

    data.columns = pd.Index([e[0] + "_" + e[1] for e in data.columns.tolist()])

    data.columns = [prefix + c for c in data.columns]



    data[prefix + 'purchase_date_diff'] = (data[prefix + 'purchase_date_max']-data[prefix + 'purchase_date_min']).dt.days

    data[prefix + 'purchase_date_average'] = data[prefix + 'purchase_date_diff']/data[prefix + 'card_id_size']

    data[prefix + 'purchase_date_uptonow'] = (dt.datetime.today()-data[prefix + 'purchase_date_max']).dt.days

    data[prefix + 'purchase_date_uptomin'] = (dt.datetime.today()-data[prefix + 'purchase_date_min']).dt.days



    #Colunas float16 geram problemas no modelo

    cols = [c for c,d in zip(data.columns, data.dtypes) if d == np.dtype('float16')]

    data[cols] = data[cols].astype(np.float32)



    return data
history = get_features(history, 'hist_')

gc.collect()
new_transact = get_features(new_transact, 'new_')

gc.collect()
features = history.merge(new_transact, how='left', left_index=True, right_index=True)

del new_transact

del history

gc.collect()



train_features = train_data.merge(features.reset_index(), how='left', on='card_id')

test_features = test_data.merge(features.reset_index(), how='left', on='card_id')
reg_idx = train_features['target']>-50

reg_X = train_features.loc[reg_idx].drop(FEATS_EXCLUDED, axis=1)

reg_y = train_features.loc[reg_idx, 'target']

test_reg_X = test_features.drop([x for x in FEATS_EXCLUDED if x!='target'], axis=1)



clf_X = train_features.drop(FEATS_EXCLUDED, axis=1)

clf_y = (train_features['target']<-33).astype(np.uint8)

test_clf_X = test_reg_X
stratifications = pd.cut(reg_y, 5, labels=False)

splits = list(StratifiedKFold(n_splits=N_SPLITS, random_state=RANDOM_SEED, shuffle=True).split(reg_X, stratifications))



reg_test_pred = []

reg_metrics = []

reg_feature_importances = []



for fold_id, (train_idx, val_idx) in enumerate(splits):

    print('Training Fold {}'.format(fold_id))

    

    train_X, val_X = reg_X.iloc[train_idx], reg_X.iloc[val_idx]

    train_y, val_y = reg_y.iloc[train_idx], reg_y.iloc[val_idx]

    

    model = lightgbm.LGBMRegressor(**{

                'task': 'train',

                'objective': 'regression',

                'metric': 'rmse',

                'learning_rate': 0.01,

                'subsample': 0.9,

                'max_depth': -1,

                'num_leaves': 64,

                'reg_alpha': 5.0,

                'colsample_bytree': 0.7,

                'min_split_gain': 1.0,

                'reg_lambda': 5.0,

                'min_data_in_leaf': 50,

                'early_stopping_rounds': 100,

                'num_iterations': 300,

                'nthreads': -1,

                })

    model.fit(train_X, train_y, eval_set=(val_X, val_y), verbose=100)

    

    reg_test_pred.append(model.predict(test_reg_X))

    reg_feature_importances.append(model.feature_importances_)

    reg_metrics.append(model.best_score_['valid_0']['rmse'])

    print('Fold RMSE: {}'.format(reg_metrics[-1]))

print('Validation RMSE: {} +- {}'.format(np.mean(reg_metrics), np.std(reg_metrics)))

reg_feature_importances = pd.DataFrame(np.stack(reg_feature_importances, axis=1), columns=[i for i in range(N_SPLITS)], index=reg_X.columns)

reg_feature_importances = reg_feature_importances.loc[reg_feature_importances.mean(axis=1).sort_values(ascending=False).index]
_, ax = plt.subplots(1, figsize=(12,8))

lightgbm.plot_importance(model, max_num_features=25, height=0.5, ax=ax)

plt.tight_layout()
splits = list(StratifiedKFold(n_splits=N_SPLITS, random_state=RANDOM_SEED, shuffle=True).split(clf_X, clf_y))



clf_test_pred = []

clf_metrics = []

clf_feature_importances = []



for fold_id, (train_idx, val_idx) in enumerate(splits):

    print('Training Fold {}'.format(fold_id))

    

    train_X, val_X = clf_X.iloc[train_idx], clf_X.iloc[val_idx]

    train_y, val_y = clf_y.iloc[train_idx], clf_y.iloc[val_idx]

    

    model = lightgbm.LGBMClassifier(**{

                'objective': 'binary',

                'boosting': 'gbdt',

                'metric': 'auc',

                'learning_rate': 0.01,

                'subsample': 0.99,

                'max_depth': 7,

                'num_leaves': 128,

                'reg_alpha': 1.0,

                'colsample_bytree': 0.6,

                'reg_lambda': 1.0,

                'min_data_in_leaf': 10,

                'pos_scale_weight': 1.0,

                'early_stopping_rounds': 200,

                'num_iterations': 10000,

                'nthreads': -1,

                })

    model.fit(train_X, train_y, eval_set=(val_X, val_y), verbose=200)

    

    clf_test_pred.append(model.predict_proba(test_clf_X)[:, 1])

    clf_feature_importances.append(model.feature_importances_)

    clf_metrics.append(roc_auc_score(val_y, model.predict_proba(val_X)[:, 1]))

    print('Fold AUC: {}'.format(clf_metrics[-1]))

print('Validation AUC: {} +- {}'.format(np.mean(clf_metrics), np.std(clf_metrics)))

clf_feature_importances = pd.DataFrame(np.stack(clf_feature_importances, axis=1), columns=[i for i in range(N_SPLITS)], index=clf_X.columns)

clf_feature_importances = clf_feature_importances.loc[clf_feature_importances.mean(axis=1).sort_values(ascending=False).index]
_, ax = plt.subplots(1, figsize=(12,8))

lightgbm.plot_importance(model, max_num_features=25, height=0.5, ax=ax)

plt.tight_layout()
sub = pd.read_csv('../input/competicao-dsa-machine-learning-jun-2019/sample_submission.csv')

test_predictions = np.mean(reg_test_pred, axis=0)

test_outliers = np.mean(clf_test_pred, axis=0)

sub['target'] = (1-test_outliers)*test_predictions + test_outliers*-9

sub['target'] *= 0.9
sub['target'].describe()
plt.figure(figsize=(9,7))

plt.hist(sub['target'], bins=30)

_ = plt.title('Test Set: Target Distribution', fontsize=15)