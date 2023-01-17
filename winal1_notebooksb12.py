import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import model_selection, preprocessing

import xgboost as xgb

import datetime



#load files

train = pd.read_csv('../input/train.csv', parse_dates=['timestamp'])

test = pd.read_csv('../input/test.csv', parse_dates=['timestamp'])

macro = pd.read_csv('../input/macro.csv', parse_dates=['timestamp'])

id_test = test.id
train_f=train.columns

test_f=test.columns

#Clean Data (JasonBenner Code)

bad_index = train[train.life_sq > train.full_sq].index

train.ix[bad_index, "life_sq"] = np.NaN

equal_index = [601,1896,2791]

test.ix[equal_index, "life_sq"] = test.ix[equal_index, "full_sq"]

bad_index = test[test.life_sq > test.full_sq].index

test.ix[bad_index, "life_sq"] = np.NaN

bad_index = train[train.life_sq < 5].index

train.ix[bad_index, "life_sq"] = np.NaN

bad_index = test[test.life_sq < 5].index

test.ix[bad_index, "life_sq"] = np.NaN

bad_index = train[train.full_sq < 5].index

train.ix[bad_index, "full_sq"] = np.NaN

bad_index = test[test.full_sq < 5].index

test.ix[bad_index, "full_sq"] = np.NaN

kitch_is_build_year = [13117]

train.ix[kitch_is_build_year, "build_year"] = train.ix[kitch_is_build_year, "kitch_sq"]

bad_index = train[train.kitch_sq >= train.life_sq].index

train.ix[bad_index, "kitch_sq"] = np.NaN

bad_index = test[test.kitch_sq >= test.life_sq].index

test.ix[bad_index, "kitch_sq"] = np.NaN

bad_index = train[(train.kitch_sq == 0).values + (train.kitch_sq == 1).values].index

train.ix[bad_index, "kitch_sq"] = np.NaN

bad_index = test[(test.kitch_sq == 0).values + (test.kitch_sq == 1).values].index

test.ix[bad_index, "kitch_sq"] = np.NaN

bad_index = train[(train.full_sq > 210) & (train.life_sq / train.full_sq < 0.3)].index

train.ix[bad_index, "full_sq"] = np.NaN

bad_index = test[(test.full_sq > 150) & (test.life_sq / test.full_sq < 0.3)].index

test.ix[bad_index, "full_sq"] = np.NaN

bad_index = train[train.life_sq > 300].index

train.ix[bad_index, ["life_sq", "full_sq"]] = np.NaN

bad_index = test[test.life_sq > 200].index

test.ix[bad_index, ["life_sq", "full_sq"]] = np.NaN

train.product_type.value_counts(normalize= True)

test.product_type.value_counts(normalize= True)

bad_index = train[train.build_year < 1500].index

train.ix[bad_index, "build_year"] = np.NaN

bad_index = test[test.build_year < 1500].index

test.ix[bad_index, "build_year"] = np.NaN

bad_index = train[train.num_room == 0].index 

train.ix[bad_index, "num_room"] = np.NaN

bad_index = test[test.num_room == 0].index 

test.ix[bad_index, "num_room"] = np.NaN

bad_index = [10076, 11621, 17764, 19390, 24007, 26713, 29172]

train.ix[bad_index, "num_room"] = np.NaN

bad_index = [3174, 7313]

test.ix[bad_index, "num_room"] = np.NaN

bad_index = train[(train.floor == 0).values * (train.max_floor == 0).values].index

train.ix[bad_index, ["max_floor", "floor"]] = np.NaN

bad_index = train[train.floor == 0].index

train.ix[bad_index, "floor"] = np.NaN

bad_index = train[train.max_floor == 0].index

train.ix[bad_index, "max_floor"] = np.NaN

bad_index = test[test.max_floor == 0].index

test.ix[bad_index, "max_floor"] = np.NaN

bad_index = train[train.floor > train.max_floor].index

train.ix[bad_index, "max_floor"] = np.NaN

bad_index = test[test.floor > test.max_floor].index

test.ix[bad_index, "max_floor"] = np.NaN

train.floor.describe(percentiles= [0.9999])

bad_index = [23584]

train.ix[bad_index, "floor"] = np.NaN

train.material.value_counts()

test.material.value_counts()

train.state.value_counts()

bad_index = train[train.state == 33].index

train.ix[bad_index, "state"] = np.NaN

test.state.value_counts()



macro_f=macro.columns

train_mac = train

test_mac = test

train_mac["price_doc"]

hi = np.percentile(train_mac.price_doc.values, 99)

lo = np.percentile(train_mac.price_doc.values, 1)

train_mac["price_doc"].ix[train_mac["price_doc"]>hi] = hi

train_mac["price_doc"].ix[train_mac["price_doc"]<lo] = lo

train_mac["price_for_sq"]=train_mac["price_doc"]/train_mac["full_sq"]

for f in train_mac.columns:

    if train_mac[f].dtype=='object':

        lbl = preprocessing.LabelEncoder()

        lbl.fit(list(train_mac[f].values.astype('str')) + list(test_mac[f].values.astype('str')))

        train_mac[f] = lbl.transform(list(train_mac[f].values.astype('str')))

        test_mac[f] = lbl.transform(list(test_mac[f].values.astype('str')))

train_mac=train_mac[train_mac["price_for_sq"]>=10000]

train_mac=train_mac[train_mac["price_for_sq"]<=600000]

train_mac.drop("price_for_sq",axis=1,inplace=True)
#missing data

total = train_mac.isnull().sum().sort_values(ascending=False)

percent = (train_mac.isnull().sum()/train_mac.isnull().count()).sort_values(ascending=False)

missing_data_train = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data_train.head(20)

total = test_mac.isnull().sum().sort_values(ascending=False)

percent = (test_mac.isnull().sum()/test_mac.isnull().count()).sort_values(ascending=False)

missing_data_test = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data_test.head(100)

miss=missing_data_train[missing_data_train['Percent'] > 0.4]

#delete Percent>0.4

train_mac.drop(miss.index,axis=1)

test_mac.drop(miss.index, axis=1)

miss
train_p=train_mac["timestamp"].dt.year

test_p=test_mac["timestamp"].dt.year

train_mac["year"]=train_p

test_mac["year"]=test_p



train_p=train_mac["timestamp"].dt.month

test_p=test_mac["timestamp"].dt.month

train_mac["month"]=train_p

test_mac["month"]=test_p



train_p=train_mac["timestamp"].dt.dayofweek

test_p=test_mac["timestamp"].dt.dayofweek

train_mac["dow"]=train_p

test_mac["dow"]=test_p



train_p=train_mac["timestamp"].dt.weekofyear

test_p=test_mac["timestamp"].dt.weekofyear

train_mac["woy"]=train_p

test_mac["woy"]=test_p



train_p=train_mac["timestamp"].dt.year*100+train_mac["timestamp"].dt.weekofyear

test_p=test_mac["timestamp"].dt.year*100+test_mac["timestamp"].dt.weekofyear

train_mac["yearweek"]=train_p

test_mac["yearweek"]=test_p



train_p=train_mac["timestamp"].dt.year*100+train_mac["timestamp"].dt.month

test_p=test_mac["timestamp"].dt.year*100+test_mac["timestamp"].dt.month

train_mac["yearmonth"]=train_p

test_mac["yearmonth"]=test_p
train_mac.groupby("year")["price_doc"].agg(np.median).plot(kind="bar")
train_mac.groupby("month")["price_doc"].agg(np.median).plot(kind="bar")
train_mac.groupby("dow")["price_doc"].agg(np.median).plot(kind="bar")
train_mac.groupby("woy")["price_doc"].agg(np.median).plot(kind="bar")
train_mac["ratio_livsq_fullsq"]= train_mac["life_sq"] / train_mac["full_sq"].astype(float)

test_mac["ratio_livsq_fullsq"]= test_mac["life_sq"] / test_mac["full_sq"].astype(float)
pl=train_mac

correlations = pl.corr()

corrwithprice=correlations[["price_doc"]]

corrwithprice.sort_values(by="price_doc",inplace=True)

unimportant_features=[]

for i in corrwithprice["price_doc"].index:

    if abs(corrwithprice.loc[i,"price_doc"])<0.1:

        unimportant_features.append(i)

unimportant_features
train_X = train_mac.drop(["id", "timestamp", "price_doc"], axis=1)

test_X = test_mac.drop(["id", "timestamp"], axis=1)

train_Y = np.log1p(train_mac["price_doc"].values)
val_time = 201407

dev_indices = np.where(train_X["yearmonth"]<val_time)

val_indices = np.where(train_X["yearmonth"]>=val_time)

dev_X = train_X.ix[dev_indices]

val_X = train_X.ix[val_indices]

dev_Y = train_Y[dev_indices]

val_Y = train_Y[val_indices]

print(dev_X.shape, val_X.shape)
xgb_params = {

    'eta': 0.05,

    'max_depth': 5,

    'subsample': 0.75,

    'colsample_bytree': 0.7,

    'objective': 'reg:linear',

    'eval_metric': 'rmse',

    'min_child_weight':1,

    'silent': 1,

    'seed':0

}



xgtrain = xgb.DMatrix(train_X, train_Y, feature_names=train_X.columns)





watchlist = [ (xgtrain,'train')]

num_rounds = 400 # Increase the number of rounds while running in local

model = xgb.train(xgb_params, xgtrain, num_rounds, watchlist, early_stopping_rounds=50, verbose_eval=5)
fig, ax = plt.subplots(figsize=(12,66))

xgb.plot_importance(model, max_num_features=200, height=0.8,ax=ax)


importance = model.get_fscore()

limp=[]

for i in importance:

    if importance[i]<10:

        limp.append(i)

len(limp)

test_X
test_mac_matrix = xgb.DMatrix(test_X)

predict = model.predict(test_mac_matrix)

output = pd.DataFrame({'id': id_test, 'price_doc': np.exp(predict)})

output.to_csv("out2.csv",index=False)