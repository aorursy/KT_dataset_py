import gc

import datetime

import warnings

import numpy as np

import pandas as pd

import seaborn as sns

import lightgbm as lgb

import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings('ignore')

np.random.seed(2020)
def reduzirMemoria(df):

    reduzirType = ['int16', 'float16', 'int32', 'float32', 'int64', 'float64']

    print('Memoria Inicial:', df.memory_usage().sum() / 1024 ** 2)

    

    for item in df.columns:

        itemType = df[item].dtypes

        

        if itemType in reduzirType:

            vmin, vmax = df[item].min(), df[item].max()

           

            if str(itemType).find('int') >= 0:

                if vmin >= np.iinfo(np.int8).min and vmax <= np.iinfo(np.int8).max:

                    df[item] = df[item].astype(np.int8)

                elif vmin >= np.iinfo(np.int16).min and vmax <= np.iinfo(np.int16).max:

                    df[item] = df[item].astype(np.int16)

                elif vmin >= np.iinfo(np.int32).min and vmax <= np.iinfo(np.int32).max:

                    df[item] = df[item].astype(np.int32)

                elif vmin >= np.iinfo(np.int64).min and vmax <= np.iinfo(np.int64).max:

                    df[item] = df[item].astype(np.int64)  

            

            else:

                if vmin >= np.finfo(np.float16).min and vmax <= np.finfo(np.float16).max:

                    df[item] = df[item].astype(np.float16)

                elif vmin >= np.finfo(np.float32).min and vmax <= np.finfo(np.float32).max:

                    df[item] = df[item].astype(np.float32)

                else:

                    df[item] = df[item].astype(np.float64)    

                    

    print('Memoria Final:', df.memory_usage().sum() / 1024 ** 2)
def showCorr(df):

    fig = plt.subplots(figsize = (20, 20))

    sns.set(font_scale=1.5)

    sns.heatmap(df.corr(),square = True,cbar=True,annot=True,annot_kws={'size': 10})

    plt.show()
def create_new_columns(name, aggs):

    return [name + '_' + k + '_' + agg for k in aggs.keys() for agg in aggs[k]]
def set_map_df(df):

    df['authorized_flag'] = df['authorized_flag'].map({'N':0, 'Y':1})

    df['category_1'] = df['category_1'].map({'N':0, 'Y':1}) 

    df['category_3'] = df['category_3'].map({'A':0, 'B':1, 'C':2})   
def set_fillna_df(df):    

    df['installments'].replace({-1: np.nan, 999: np.nan}, inplace=True)

    df['category_2'].fillna(1.0, inplace=True)

    df['category_3'].fillna('A', inplace=True)

    df['merchant_id'].fillna('M_ID_00a6ca8a8a', inplace=True)

    print('Percentile Purchase_Amount:',np.percentile(df['purchase_amount'], [1, 5, 50, 95, 99]))

    df['purchase_amount'] = df['purchase_amount'].apply(lambda x: min(x, 0.9))
def set_date_df(df):

    pDate = pd.to_datetime(df['purchase_date'])

    df['purchase_date'] = pDate

    df['year'] = pDate.dt.year

    df['weekofyear'] = pDate.dt.weekofyear

    df['month'] = pDate.dt.month

    df['dayofweek'] = pDate.dt.dayofweek

    df['weekend'] = (pDate.dt.weekday >=5).astype(int)

    df['hour'] = pDate.dt.hour

    df['month_diff'] = ((datetime.datetime.today() - pDate).dt.days)//30

    df['month_diff'] += df['month_lag']

    

    df['month_rank'] = df.groupby(['card_id'])['month_lag'].rank(method='dense',ascending=False)

    df['date_rank'] = df.groupby(['card_id'])['purchase_date'].rank(method='dense',ascending=False)
def get_list_aggregation():

    aggs = {}

    aggs['card_id'] = ['size', 'count']

    aggs['year'] = ['nunique', 'mean']

    aggs['purchase_date'] = ['max','min', 'nunique']

    

    for col in ['purchase_amount', 'installments']:

        aggs[col] = ['min', 'max', 'mean', 'sum', 'std', 'var']

        

    for col in ['month', 'hour', 'weekofyear', 'dayofweek', 'month_lag', 'month_diff']:

        aggs[col] = ['nunique', 'min', 'max', 'mean', 'var']

    

    for col in ['authorized_flag', 'weekend']:

        aggs[col] = ['sum', 'mean', 'nunique']

        

    for col in ['category_1', 'category_2', 'category_3']:

        aggs[col] = ['nunique', 'min', 'max', 'mean', 'sum', 'std']

        

    for col in ['month_rank', 'date_rank']:

        aggs[col] = ['sum', 'mean', 'min', 'nunique', 'std']

        

    for col in ['subsector_id','merchant_id','merchant_category_id', 'state_id', 'city_id']:

        aggs[col] = ['nunique']

    

    return aggs
def set_purchase_amount_by_category(df):

    items = ['min', 'max', 'mean', 'sum', 'std']

    for col in ['category_2','category_3']:

        for item in items:

            df[col+item] = df.groupby([col])['purchase_amount'].transform(item)
def set_purchase_date_group(df, pred):

    col = pred + '_purchase_date_'

    df[col+'diff'] = (df[col+'max'] - df[col+'min']).dt.days

    df[col+'average'] = df[col+'diff'] / df[pred+'_card_id_size']

    df[col+'uptonow'] = (datetime.datetime.today() - df[col+'max']).dt.days

    df[col+'uptomin'] = (datetime.datetime.today() - df[col+'min']).dt.days 
dir_local = '../input/dsa-jun2019/'

dir_dsa = '../input/competicao-dsa-machine-learning-jun-2019/'
train_df = pd.read_csv(dir_dsa + 'dataset_treino.csv')

test_df = pd.read_csv(dir_dsa + 'dataset_teste.csv')
novas_vendas = pd.read_csv(dir_local + 'novas_transacoes_comerciantes.csv')

reduzirMemoria(novas_vendas)
vendas = pd.read_csv(dir_local + 'transacoes_historicas.csv')

reduzirMemoria(vendas)
train_df.describe().T
test_df.describe().T
plt.figure(figsize=(20, 5))

sns.distplot(train_df.target)

plt.show()
set_fillna_df(vendas)

set_map_df(vendas)

set_date_df(vendas)

set_purchase_amount_by_category(vendas)

vendas.describe().T
aggs = get_list_aggregation()

aggs
new_columns = create_new_columns('hist', aggs)

new_columns
vendas_group = vendas.groupby('card_id').agg(aggs)
del vendas; gc.collect()
vendas_group.columns = new_columns

vendas_group.reset_index(drop=False,inplace=True)
set_purchase_date_group(vendas_group, 'hist')
train_df = train_df.merge(vendas_group,on='card_id',how='left')

test_df = test_df.merge(vendas_group,on='card_id',how='left')
del vendas_group; gc.collect()
set_fillna_df(novas_vendas)

set_map_df(novas_vendas)

set_date_df(novas_vendas)

set_purchase_amount_by_category(novas_vendas)

novas_vendas.describe().T
aggs = get_list_aggregation()

aggs
new_columns = create_new_columns('new', aggs)

new_columns
novas_vendas_group = novas_vendas.groupby('card_id').agg(aggs)
del novas_vendas; gc.collect()
novas_vendas_group.columns = new_columns

novas_vendas_group.reset_index(drop=False,inplace=True)
set_purchase_date_group(novas_vendas_group, 'new')
train_df = train_df.merge(novas_vendas_group,on='card_id',how='left')

test_df = test_df.merge(novas_vendas_group,on='card_id',how='left')
del novas_vendas_group; gc.collect()
train_df['outliers'] = 0

train_df.loc[train_df['target'] < -30, 'outliers'] = 1

outls = train_df['outliers'].value_counts()

print("Outliers: {}".format(outls))
## process both train and test

for df in [train_df, test_df]:

    df['first_active_month'] = pd.to_datetime(df['first_active_month'])

    df['dayofweek'] = df['first_active_month'].dt.dayofweek

    df['weekofyear'] = df['first_active_month'].dt.weekofyear

    df['dayofyear'] = df['first_active_month'].dt.dayofyear

    df['quarter'] = df['first_active_month'].dt.quarter

    df['is_month_start'] = df['first_active_month'].dt.is_month_start

    df['month'] = df['first_active_month'].dt.month

    df['elapsed_time'] = (datetime.datetime.today() - df['first_active_month']).dt.days

    df['hist_first_buy'] = (df['hist_purchase_date_min'] - df['first_active_month']).dt.days

    df['hist_last_buy'] = (df['hist_purchase_date_max'] - df['first_active_month']).dt.days

    df['new_first_buy'] = (df['new_purchase_date_min'] - df['first_active_month']).dt.days

    df['new_last_buy'] = (df['new_purchase_date_max'] - df['first_active_month']).dt.days

    for f in ['hist_purchase_date_max', 'hist_purchase_date_min', 'new_purchase_date_max', 'new_purchase_date_min']:

        df[f] = df[f].astype(np.int64) * 1e-9

    df['card_id_total'] = df['new_card_id_size']+df['hist_card_id_size']

    df['card_id_cnt_total'] = df['new_card_id_count']+df['hist_card_id_count']

    df['purchase_amount_total'] = df['new_purchase_amount_sum']+df['hist_purchase_amount_sum']

    df['purchase_amount_mean'] = df['new_purchase_amount_mean']+df['hist_purchase_amount_mean']

    df['purchase_amount_max'] = df['new_purchase_amount_max']+df['hist_purchase_amount_max']
for f in ['feature_1','feature_2','feature_3']:

    order_label = train_df.groupby([f])['outliers'].mean()

    train_df[f] = train_df[f].map(order_label)

    test_df[f] = test_df[f].map(order_label)
train_df.describe().T
test_df.describe().T
train_columns = [c for c in train_df.columns if c not in ['card_id', 'first_active_month','target','outliers']]

target = train_df['target']

del train_df['target']
remove_columns = ['hist_month_min',

                 'hist_month_max',

                 'hist_purchase_amount_max', 

                 'hist_purchase_amount_var',

                 'hist_purchase_amount_std',

                 'hist_purchase_date_nunique',

                 'hist_month_lag_nunique',

                 'hist_month_lag_var',

                 'hist_merchant_category_id_nunique',

                 'hist_month_diff_min',

                 'hist_month_diff_max',

                 'hist_month_diff_mean',

                 'hist_category_1_nunique',

                 'hist_month_rank_min',

                 'hist_date_rank_min',

                 'hist_card_id_size',

                 'hist_card_id_count',

                 'hist_date_rank_nunique',

                 'hist_date_rank_std',

                 'hist_month_rank_nunique',

                 'new_merchant_category_id_nunique',

                 'new_month_mean',

                 'new_hour_nunique',

                 'new_weekofyear_mean',

                 'new_month_diff_max',

                 'new_month_diff_min',

                 'new_month_diff_mean',

                 'new_card_id_size',

                 'new_card_id_count',

                 'is_month_start',

                 'dayofyear',

                 'quarter',

                 'purchase_amount_max',

                 'elapsed_time',

                 'hist_last_buy',

                 'new_first_buy',

                 'card_id_cnt_total',

                 'new_purchase_date_uptomin',

                 'new_purchase_date_uptonow',

                 'new_purchase_date_min',

                 'purchase_amount_mean',

                 'hist_weekofyear_mean',

                 'hist_installments_std',

                 'hist_month_rank_sum',

                 'hist_date_rank_mean',

                 'new_weekofyear_min',

                 'new_weekofyear_max']



for item in remove_columns:

    if item in train_columns:

        train_columns.remove(item)

        print('train_columns.remove:', item)
train_columns
showCorr(train_df[train_columns[0:40]])
showCorr(train_df[train_columns[30:70]])
showCorr(train_df[train_columns[60:100]])
showCorr(train_df[train_columns[90:]])
len(train_columns)
param = {'num_leaves': 60,

         'min_data_in_leaf': 20, 

         'objective':'regression',

         'max_depth': -1,

         'learning_rate': 0.007,

         "boosting": "gbdt",

         "feature_fraction": 0.85,

         "bagging_freq": 1,

         "bagging_fraction": 0.85,

         "bagging_seed": 40,

         "metric": 'rmse',

         "lambda_l1": 0.1,

         "verbosity": -1,

         "nthread": 4,

         "random_state": 2020}
#prepare fit model with cross-validation

folds = StratifiedKFold(n_splits=20, shuffle=True, random_state=2020)

oof = np.zeros(len(train_df))

predictions = np.zeros(len(test_df))

feature_importance_df = pd.DataFrame()
#run model

for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_df,train_df['outliers'].values)):

    strLog = "fold {}".format(fold_)

    print(strLog)

    trn_data = lgb.Dataset(train_df.iloc[trn_idx][train_columns], label=target.iloc[trn_idx])

    val_data = lgb.Dataset(train_df.iloc[val_idx][train_columns], label=target.iloc[val_idx])



    num_round = 10000

    clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=200, early_stopping_rounds = 150)

    oof[val_idx] = clf.predict(train_df.iloc[val_idx][train_columns], num_iteration=clf.best_iteration)

    #feature importance

    fold_importance_df = pd.DataFrame()

    fold_importance_df["Feature"] = train_columns

    fold_importance_df["importance"] = clf.feature_importance()

    fold_importance_df["fold"] = fold_ + 1

    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    #predictions

    predictions += clf.predict(test_df[train_columns], num_iteration=clf.best_iteration) / folds.n_splits
np.sqrt(mean_squared_error(oof, target))
##plot the feature importance

cols = (feature_importance_df[["Feature", "importance"]].groupby("Feature").mean().sort_values(by="importance", ascending=False)[:1000].index)

best_features = feature_importance_df.loc[feature_importance_df.Feature.isin(cols)]
plt.figure(figsize=(15,30))

sns.barplot(x="importance", y="Feature", data=best_features.sort_values(by="importance",ascending=False))

plt.title('LightGBM Features (Averaged Over Folds)')

plt.tight_layout()
##submission

sub_df = pd.DataFrame({"card_id":test_df["card_id"].values})

sub_df["target"] = predictions

sub_df.to_csv("DSA_Jun2019_Final.csv", index=False)