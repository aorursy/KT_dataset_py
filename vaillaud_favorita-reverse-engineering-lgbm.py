"""

This is an upgraded version of Ceshine's LGBM starter script, simply adding more

average features and weekly average features on it.

"""

from datetime import date, timedelta

import gc # garbage collector

import pandas as pd

import numpy as np

from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import LabelEncoder



import lightgbm as lgb



import os

print(os.listdir("../input"))
# Memory saving function credit to https://www.kaggle.com/gemartin/load-data-reduce-memory-usage

def reduce_mem_usage(df):

    """ iterate through all the columns of a dataframe and modify the data type

        to reduce memory usage.        

    """

    #start_mem = df.memory_usage().sum() / 1024**2

    #print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))



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

            elif str(col_type)[:3] == 'float':

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)



    #end_mem = df.memory_usage().sum() / 1024**2

    #print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))

    #print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))



    return df
import sys

def sizeof_fmt(num, suffix='B'):

    ''' By Fred Cirera, after https://stackoverflow.com/a/1094933/1870254'''

    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:

        if abs(num) < 1024.0:

            return "%3.1f%s%s" % (num, unit, suffix)

        num /= 1024.0

    return "%.1f%s%s" % (num, 'Yi', suffix)



for name, size in sorted(((name, sys.getsizeof(value)) for name,value in locals().items()), key= lambda x: -x[1])[:10]:

    print("{:>30}: {:>8}".format(name,sizeof_fmt(size)))
#del _48

gc.collect()
items = pd.read_csv(

    "../input/items.csv",

).set_index("item_nbr")



#items = reduce_mem_usage(items)



items.head()
stores = pd.read_csv(

    "../input/stores.csv",

).set_index("store_nbr")



#stores = reduce_mem_usage(stores)



stores.head()
df_test = pd.read_csv(

    "../input/test.csv", usecols=[0, 1, 2, 3, 4],

    dtype= {'onpromotion': bool},

    parse_dates=["date"]  # , date_parser=parser

).set_index(

    ['store_nbr', 'item_nbr', 'date']

)



#df_test = reduce_mem_usage(df_test)



df_test.head()
df_test.dtypes
df_train = pd.read_csv(

    '../input/train.csv', usecols=[1, 2, 3, 4, 5],

    dtype={'onpromotion': bool},

    converters={'unit_sales': lambda u: np.log1p(

        float(u)) if float(u) > 0 else 0},

    parse_dates=["date"],

    skiprows=range(1, 66458909)  # 2016-01-01

)

#df_train = reduce_mem_usage(df_train)
#df_train.head()

df_train.dtypes

#df_train['item_nbr'].max()
le = LabelEncoder()
items['family'] = le.fit_transform(items['family'].values)

#items = reduce_mem_usage(items)

items.head()
stores['city'] = le.fit_transform(stores['city'].values)

stores['state'] = le.fit_transform(stores['state'].values)

stores['type'] = le.fit_transform(stores['type'].values)

#stores = reduce_mem_usage(stores)

stores.head()
df_2017 = df_train.loc[df_train.date>=pd.datetime(2017,1,1)]

del df_train

gc.collect()



df_2017.head()
promo_2017_train = df_2017.set_index(

    ["store_nbr", "item_nbr", "date"])[["onpromotion"]].unstack(

        level=-1).fillna(False)





promo_2017_train.head()
promo_2017_train.columns = promo_2017_train.columns.get_level_values(1)

promo_2017_train.head()
promo_2017_test = df_test[["onpromotion"]].unstack(level=-1).fillna(False)

promo_2017_test.columns = promo_2017_test.columns.get_level_values(1)

promo_2017_test.head()
promo_2017_test = promo_2017_test.reindex(promo_2017_train.index).fillna(False)

promo_2017_test.head()
promo_2017 = pd.concat([promo_2017_train, promo_2017_test], axis=1)

del promo_2017_test, promo_2017_train

gc.collect()
df_2017 = df_2017.set_index(

    ["store_nbr", "item_nbr", "date"])[["unit_sales"]].unstack(

        level=-1).fillna(0)

df_2017.columns = df_2017.columns.get_level_values(1)

df_2017.head()
items = items.reindex(df_2017.index.get_level_values(1))

stores = stores.reindex(df_2017.index.get_level_values(0))

items.head()
df_2017_item = df_2017.groupby('item_nbr')[df_2017.columns].sum()

df_2017_item.head()
promo_2017_item = promo_2017.groupby('item_nbr')[promo_2017.columns].sum()
df_2017_store_class = df_2017.reset_index()

df_2017_store_class['class'] = items['class'].values # par quelle sorcellerie est-ce que ça fonctionne ?

df_2017_store_class.head()
df_2017_store_class_index = df_2017_store_class[['class', 'store_nbr']]

df_2017_store_class = df_2017_store_class.groupby(['class', 'store_nbr'])[df_2017.columns].sum()

df_2017_store_class.head()
df_2017_promo_store_class = promo_2017.reset_index()

df_2017_promo_store_class['class'] = items['class'].values

df_2017_promo_store_class_index = df_2017_promo_store_class[['class', 'store_nbr']]

df_2017_promo_store_class = df_2017_promo_store_class.groupby(['class', 'store_nbr'])[promo_2017.columns].sum()

df_2017_promo_store_class.head()
# Selects df columns which correspond to the "periods" days after the "minus" day before the date "dt"

def get_timespan(df, dt, minus, periods, freq='D'):

    return df[pd.date_range(dt - timedelta(days=minus), periods=periods, freq=freq)]



def prepare_dataset(df, promo_df, t2017, is_train=True, name_prefix=None):

    # Création de 6 fenêtres. 3 dans le passé et 3 dans le futur

    # Le nombre de promotions qu'il y a dans le passé et le futur, à différents intervalles de temps

    X = {

        "promo_14_2017": get_timespan(promo_df, t2017, 14, 14).sum(axis=1).values,

        "promo_60_2017": get_timespan(promo_df, t2017, 60, 60).sum(axis=1).values,

        "promo_140_2017": get_timespan(promo_df, t2017, 140, 140).sum(axis=1).values,

        "promo_3_2017_aft": get_timespan(promo_df, t2017 + timedelta(days=16), 15, 3).sum(axis=1).values,

        "promo_7_2017_aft": get_timespan(promo_df, t2017 + timedelta(days=16), 15, 7).sum(axis=1).values,

        "promo_14_2017_aft": get_timespan(promo_df, t2017 + timedelta(days=16), 15, 14).sum(axis=1).values,

    }

    #print("ça démarre!!")

    

    # Moyenne d'items vendus avec ou sans promotion, normale ou avec poids exponentiel

    for i in [3, 7, 14, 30, 60, 140]:

        # number of items sold i days before t2017 to t2017

        tmp1 = get_timespan(df, t2017, i, i)

        # promotions on items i days before t2017 to t2017 (1 if promotion, else 0)

        tmp2 = (get_timespan(promo_df, t2017, i, i) > 0) * 1

        

        # mean of items sold on promotion in [t2017 - 1; t2017]

        X['has_promo_mean_%s' % i] = (tmp1 * tmp2.replace(0, np.nan)).mean(axis=1).values

        # same but exponentially weighted

        X['has_promo_mean_%s_decay' % i] = (tmp1 * tmp2.replace(0, np.nan) * np.power(0.9, np.arange(i)[::-1])).sum(axis=1).values

        

        # mean of items sold without promotion in [t2017 - 1; t2017]

        X['no_promo_mean_%s' % i] = (tmp1 * (1 - tmp2).replace(0, np.nan)).mean(axis=1).values

        # same but exponentially weighted

        X['no_promo_mean_%s_decay' % i] = (tmp1 * (1 - tmp2).replace(0, np.nan) * np.power(0.9, np.arange(i)[::-1])).sum(axis=1).values

    

    #print("une étape")

    # stats d'items vendus

    for i in [3, 7, 14, 30, 60, 140]:

        tmp = get_timespan(df, t2017, i, i)

        # mean of variation of units sold

        X['diff_%s_mean' % i] = tmp.diff(axis=1).mean(axis=1).values

        # sum of exponentially weighted units sold

        X['mean_%s_decay' % i] = (tmp * np.power(0.9, np.arange(i)[::-1])).sum(axis=1).values

        # stats of units sold

        X['mean_%s' % i] = tmp.mean(axis=1).values

        X['median_%s' % i] = tmp.median(axis=1).values

        X['min_%s' % i] = tmp.min(axis=1).values

        X['max_%s' % i] = tmp.max(axis=1).values

        X['std_%s' % i] = tmp.std(axis=1).values

    

    #print("2 étapes")

    # Memes stats avec un décallage d'une semaine dans le passé

    for i in [3, 7, 14, 30, 60, 140]:

        tmp = get_timespan(df, t2017 + timedelta(days=-7), i, i)

        X['diff_%s_mean_2' % i] = tmp.diff(axis=1).mean(axis=1).values

        X['mean_%s_decay_2' % i] = (tmp * np.power(0.9, np.arange(i)[::-1])).sum(axis=1).values

        X['mean_%s_2' % i] = tmp.mean(axis=1).values

        X['median_%s_2' % i] = tmp.median(axis=1).values

        X['min_%s_2' % i] = tmp.min(axis=1).values

        X['max_%s_2' % i] = tmp.max(axis=1).values

        X['std_%s_2' % i] = tmp.std(axis=1).values

    

    #print("3 étapes")

    # Nombre de jours où a eu lieu une vente/promotion dans la fenêtre temporelle, et jours écoulés depuis première/dernière vente/promotion

    for i in [7, 14, 30, 60, 140]:

        tmp = get_timespan(df, t2017, i, i)

        # Number of days a sale has been made

        X['has_sales_days_in_last_%s' % i] = (tmp > 0).sum(axis=1).values

        # Number of days since last sales in period

        X['last_has_sales_day_in_last_%s' % i] = i - ((tmp > 0) * np.arange(i)).max(axis=1).values

        # Number of days since first sales in period

        X['first_has_sales_day_in_last_%s' % i] = ((tmp > 0) * np.arange(i, 0, -1)).max(axis=1).values



        tmp = get_timespan(promo_df, t2017, i, i)

        # Number of days where there was a promotion

        X['has_promo_days_in_last_%s' % i] = (tmp > 0).sum(axis=1).values

        # Number of days since there was a promotion in period

        X['last_has_promo_day_in_last_%s' % i] = i - ((tmp > 0) * np.arange(i)).max(axis=1).values

        # Number of days since first promotion in period

        X['first_has_promo_day_in_last_%s' % i] = ((tmp > 0) * np.arange(i, 0, -1)).max(axis=1).values

    

    #print("4 étapes")

    # Nombre de promotions dans les deux semaines à venir, temps avant première et dernière promotion dans la même fenêtre de temps

    tmp = get_timespan(promo_df, t2017 + timedelta(days=16), 15, 15)

    X['has_promo_days_in_after_15_days'] = (tmp > 0).sum(axis=1).values

    X['last_has_promo_day_in_after_15_days'] = i - ((tmp > 0) * np.arange(15)).max(axis=1).values

    X['first_has_promo_day_in_after_15_days'] = ((tmp > 0) * np.arange(15, 0, -1)).max(axis=1).values

    

    # Nombre de ventes le jour i avant aujourd'hui

    for i in range(1, 16):

        X['day_%s_2017' % i] = get_timespan(df, t2017, i, 1).values.ravel()

    

    #print("Presque fini...")

    for i in range(7):

        # mean of sales every same day of week during the month before today

        X['mean_4_dow{}_2017'.format(i)] = get_timespan(df, t2017, 28-i, 4, freq='7D').mean(axis=1).values

        # mean of sales every same day of week during the twenty weeks before today

        X['mean_20_dow{}_2017'.format(i)] = get_timespan(df, t2017, 140-i, 20, freq='7D').mean(axis=1).values

    

    # détecte si il y a eu une promotion i jours avant puis après aujourd'hui

    for i in range(-16, 16):

        X["promo_{}".format(i)] = promo_df[t2017 + timedelta(days=i)].values.astype(np.uint8)

    X = pd.DataFrame(X)



    if is_train:

        # Si il y a entraînement, y devient les 16 jours suivants

        y = df[

            pd.date_range(t2017, periods=16)

        ].values

        return X, y

    if name_prefix is not None:

        X.columns = ['%s_%s' % (name_prefix, c) for c in X.columns]

    return X

print("Preparing dataset...")

t2017 = date(2017, 6, 14)

num_days = 6

X_l, y_l = [], []

for i in range(num_days):

    print('-'*50)

    print("ROUND {} / {}:".format(i+1, num_days))

    # décallage d'une semaine à chaque fois (pendant 6 semaines)

    delta = timedelta(days=7 * i)

    # Préparation du dataset en distinguant les boutiques (prend du temps!!!)

    # stats par item par boutique

    print("1/3")

    X_tmp, y_tmp = prepare_dataset(df_2017, promo_2017, t2017 + delta)

    

    # Préparation du dataset sans distinguer les boutiques (prend peu de temps)

    # stats générales des items

    print("2/3")

    X_tmp2 = prepare_dataset(df_2017_item, promo_2017_item, t2017 + delta, is_train=False, name_prefix='item')

    X_tmp2.index = df_2017_item.index

    X_tmp2 = X_tmp2.reindex(df_2017.index.get_level_values(1)).reset_index(drop=True)

    

    # stats par type d'objet par boutique

    print("3/3")

    X_tmp3 = prepare_dataset(df_2017_store_class, df_2017_promo_store_class, t2017 + delta, is_train=False, name_prefix='store_class')

    X_tmp3.index = df_2017_store_class.index

    X_tmp3 = X_tmp3.reindex(df_2017_store_class_index).reset_index(drop=True)

    

    #concaténation horizontale des trois X

    X_tmp = pd.concat([X_tmp, X_tmp2, X_tmp3, items.reset_index(), stores.reset_index()], axis=1)

    X_l.append(X_tmp)

    y_l.append(y_tmp)

    

    del X_tmp

    del X_tmp2

    del X_tmp3

    del y_tmp

    gc.collect()
X_train = pd.concat(X_l, axis=0)

y_train = np.concatenate(y_l, axis=0)
del X_l, y_l
X_train.head()
X_val, y_val = prepare_dataset(df_2017, promo_2017, date(2017, 7, 26))



X_val2 = prepare_dataset(df_2017_item, promo_2017_item, date(2017, 7, 26), is_train=False, name_prefix='item')

X_val2.index = df_2017_item.index

X_val2 = X_val2.reindex(df_2017.index.get_level_values(1)).reset_index(drop=True)



X_val3 = prepare_dataset(df_2017_store_class, df_2017_promo_store_class, date(2017, 7, 26), is_train=False, name_prefix='store_class')

X_val3.index = df_2017_store_class.index

X_val3 = X_val3.reindex(df_2017_store_class_index).reset_index(drop=True)



X_val = pd.concat([X_val, X_val2, X_val3, items.reset_index(), stores.reset_index()], axis=1)



X_test = prepare_dataset(df_2017, promo_2017, date(2017, 8, 16), is_train=False)



X_test2 = prepare_dataset(df_2017_item, promo_2017_item, date(2017, 8, 16), is_train=False, name_prefix='item')

X_test2.index = df_2017_item.index

X_test2 = X_test2.reindex(df_2017.index.get_level_values(1)).reset_index(drop=True)



X_test3 = prepare_dataset(df_2017_store_class, df_2017_promo_store_class, date(2017, 8, 16), is_train=False, name_prefix='store_class')

X_test3.index = df_2017_store_class.index

X_test3 = X_test3.reindex(df_2017_store_class_index).reset_index(drop=True)



X_test = pd.concat([X_test, X_test2, X_test3, items.reset_index(), stores.reset_index()], axis=1)



del X_test2, X_val2, df_2017_item, promo_2017_item, df_2017_store_class, df_2017_promo_store_class, df_2017_store_class_index

gc.collect()
cate_vars = ['family', 'perishable', 'city', 'state', 'type', 'cluster']

X_val.columns[(X_val.dtypes != 'int16') & (X_val.dtypes != 'int32') & (X_val.dtypes != 'int64') & (X_val.dtypes != 'int8') & (X_val.dtypes != np.float) ]

#X_train['promo_-16'].dtypes

print("Training and predicting models...")

params = {

    'num_leaves': 80,

    'objective': 'regression',

    'min_data_in_leaf': 200,

    'learning_rate': 0.02,

    'feature_fraction': 0.8,

    'bagging_fraction': 0.7,

    'bagging_freq': 1,

    'metric': 'l2',

    'num_threads': 16

}



MAX_ROUNDS = 5000

val_pred = []

test_pred = []

cate_vars = []

for i in range(16):

    print("=" * 50)

    print("Step %d" % (i+1))

    print("=" * 50)

    dtrain = lgb.Dataset(

        X_train, label=y_train[:, i],

        categorical_feature=cate_vars,

        weight=pd.concat([items["perishable"]] * num_days) * 0.25 + 1

    )

    dval = lgb.Dataset(

        X_val, label=y_val[:, i], reference=dtrain,

        weight=items["perishable"] * 0.25 + 1,

        categorical_feature=cate_vars)

    bst = lgb.train(

        params, dtrain, num_boost_round=MAX_ROUNDS,

        valid_sets=[dtrain, dval], early_stopping_rounds=125, verbose_eval=50

    )

    print("\n".join(("%s: %.2f" % x) for x in sorted(

        zip(X_train.columns, bst.feature_importance("gain")),

        key=lambda x: x[1], reverse=True

    )))

    val_pred.append(bst.predict(

        X_val, num_iteration=bst.best_iteration or MAX_ROUNDS))

    test_pred.append(bst.predict(

        X_test, num_iteration=bst.best_iteration or MAX_ROUNDS))



print("Validation mse:", mean_squared_error(

    y_val, np.array(val_pred).transpose()))



weight = items["perishable"] * 0.25 + 1

err = (y_val - np.array(val_pred).transpose())**2

err = err.sum(axis=1) * weight

err = np.sqrt(err.sum() / weight.sum() / 16)

print('nwrmsle = {}'.format(err))



y_val = np.array(val_pred).transpose()

df_preds = pd.DataFrame(

    y_val, index=df_2017.index,

    columns=pd.date_range("2017-07-26", periods=16)

).stack().to_frame("unit_sales")

df_preds.index.set_names(["store_nbr", "item_nbr", "date"], inplace=True)

df_preds["unit_sales"] = np.clip(np.expm1(df_preds["unit_sales"]), 0, 1000)

df_preds.reset_index().to_csv('lgb_cv.csv', index=False)



print("Making submission...")

y_test = np.array(test_pred).transpose()

df_preds = pd.DataFrame(

    y_test, index=df_2017.index,

    columns=pd.date_range("2017-08-16", periods=16)

).stack().to_frame("unit_sales")

df_preds.index.set_names(["store_nbr", "item_nbr", "date"], inplace=True)



submission = df_test[["id"]].join(df_preds, how="left").fillna(0)

submission["unit_sales"] = np.clip(np.expm1(submission["unit_sales"]), 0, 1000)

submission.to_csv('lgb_sub.csv', float_format='%.4f', index=None)