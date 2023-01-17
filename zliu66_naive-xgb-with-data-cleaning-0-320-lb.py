import numpy as np

import pandas as pd

import xgboost as xgb

import matplotlib.pyplot as plt

from sklearn import metrics

from sklearn.model_selection import GridSearchCV

import seaborn as sns
# Deal with Multicollinearity class

# From here: https://www.kaggle.com/ffisegydd/sklearn-multicollinearity-class

# From here: https://www.kaggle.com/robertoruiz/sberbank-russian-housing-market/dealing-with-multicollinearity/notebook



realty_drop_cols = ['cafe_count_5000_price_2500', 'cafe_count_5000_price_1500', 'church_count_5000',

                    'cafe_count_5000_price_4000', 'leisure_count_5000', 'sport_count_5000', 'big_church_count_5000']

macro_cols = ['balance_trade', 'balance_trade_growth', 'eurrub', 'average_provision_of_build_contract', 

'micex_rgbi_tr', 'micex_cbi_tr', 'deposits_rate', 'mortgage_value', 'mortgage_rate',

'income_per_cap', 'rent_price_4+room_bus', 'museum_visitis_per_100_cap', 'apartment_build']
df_train = pd.read_csv('../input/train.csv', parse_dates=['timestamp'])

df_test = pd.read_csv('../input/test.csv', parse_dates=['timestamp'])

df_macro = pd.read_csv('../input/macro.csv', parse_dates=['timestamp'], usecols=['timestamp'] + macro_cols)
# drop some data with irregular price

price_ulimit = 1E8

price_sq_llimit = 40000

df_train =  df_train.loc[df_train['price_doc'] < price_ulimit] 

# df_train =  df_train.loc[df_train['price_doc'] / df_train['full_sq']  > price_sq_llimit] 

# df_train.head()

# ax = df_train['price_doc'].hist(bins=50)
# Build df_all = (df_train+df_test).join(df_macro)

num_train = len(df_train)

df_all = pd.concat([df_train, df_test])

df_all = pd.merge_ordered(df_all, df_macro, on='timestamp', how='left')

df_all.drop(realty_drop_cols, axis=1, inplace=True)

# print(df_all.shape)
full_sq_ulimit = 250

life_sq_ulimit = 250

full_sq_llimit = 10

life_sq_llimit = 5

df_all.loc[df_all['full_sq']>full_sq_ulimit, 'full_sq'] = np.nan

df_all.loc[df_all['full_sq']<full_sq_llimit, 'full_sq'] = np.nan

df_all.loc[df_all['life_sq']>life_sq_ulimit, 'life_sq'] = np.nan

df_all.loc[df_all['life_sq']<life_sq_llimit, 'life_sq'] = np.nan



df_all['life_full_ratio'] = df_all['life_sq'] / df_all['full_sq']



df_all.loc[df_all['life_full_ratio'] > 0.85, 'life_sq'] = np.nan



df_all.loc[df_all['floor'] == 0, 'floor'] = np.nan

df_all.loc[df_all['max_floor'] == 0, 'max_floor'] = np.nan

df_all.loc[df_all['max_floor'] < df_all['floor'], ['floor', 'max_floor']] = np.nan

df_all['floor_ratio'] = df_all['floor'] / df_all['max_floor']



df_all.loc[df_all['build_year'] > 2017, 'build_year'] = np.nan

df_all.loc[df_all['build_year'] < 1900, 'build_year'] = np.nan





df_all.loc[df_all['num_room'] == 0, 'num_room'] = np.nan

df_all.loc[df_all['num_room'] >= 10, 'num_room'] = np.nan



df_all.loc[df_all['kitch_sq'] <= 3.0 , 'kitch_sq'] = np.nan

df_all.loc[df_all['full_sq'] - df_all['kitch_sq'] <= 5.0 , 'kitch_sq'] = np.nan



df_all.loc[df_all['state'] == 33 , 'state'] = 3
# Add month-year

month_year = (df_all.timestamp.dt.month + df_all.timestamp.dt.year * 100)

month_year_cnt_map = month_year.value_counts().to_dict()

df_all['month_year_cnt'] = month_year.map(month_year_cnt_map)



# Add week-year count

week_year = (df_all.timestamp.dt.weekofyear + df_all.timestamp.dt.year * 100)

week_year_cnt_map = week_year.value_counts().to_dict()

df_all['week_year_cnt'] = week_year.map(week_year_cnt_map)



# Add month and day-of-week

df_all['month'] = df_all.timestamp.dt.month

df_all['dow'] = df_all.timestamp.dt.dayofweek



# Other feature engineering

df_all['rel_floor'] = df_all['floor'] / df_all['max_floor'].astype(float)

df_all['rel_kitch_sq'] = df_all['kitch_sq'] / df_all['full_sq'].astype(float)

# Please check https://www.kaggle.com/c/sberbank-russian-housing-market/discussion/32717

# There is difference between The CV - LB score, 

# Trick is simple - undersample specific Investment type rows in train set - 

# there has been steep decline of <=1M Investment purchases in last months; 

# Below is some working code how to do it - {10, 3, 2} are hand picked and can be tuned.





df_train = df_all.iloc[:num_train, :]

df_test = df_all.iloc[num_train:,:]



print('Before under sampling, df_train shape is', df_train.shape)

print('Before under sampling, df_test shape is', df_test.shape)



df_sub = df_train[df_train.timestamp < '2015-01-01']

df_sub = df_sub[df_sub.product_type.values == 'Investment']



ind_1m = df_sub[df_sub.price_doc <= 1000000].index

ind_2m = df_sub[df_sub.price_doc == 2000000].index

ind_3m = df_sub[df_sub.price_doc == 3000000].index



train_index = set(df_train.index.copy())



for ind, gap in zip([ind_1m, ind_2m, ind_3m], [10, 3, 2]):

    ind_set = set(ind)

    ind_set_cut = ind.difference(set(ind[::gap]))



    train_index = train_index.difference(ind_set_cut)



df_train = df_train.loc[train_index, :]



print('After under sampling, df_train shape is', df_train.shape)

print('After under sampling, df_test shape is', df_test.shape)
# Deal with categorical values

df_all = pd.concat([df_train, df_test])



df_numeric = df_all.select_dtypes(exclude=['object'])

df_obj = df_all.select_dtypes(include=['object']).copy()



for c in df_obj:

    df_obj[c] = pd.factorize(df_obj[c])[0]



df_all = pd.concat([df_numeric, df_obj], axis=1)

num_train = len(df_train)

df_train_fac = df_all.iloc[:num_train, :]

df_test_fac = df_all.iloc[num_train:,:]



print('df_train_fac shape is', df_train_fac.shape)

print('df_test_fac shape is', df_test_fac.shape)
# ylog will be log(1+y), as suggested by https://github.com/dmlc/xgboost/issues/446#issuecomment-135555130

ylog_train_all = np.log1p(df_train_fac['price_doc'].values)

id_test = df_test_fac['id']

id_train = df_train_fac['id']



df_train_fac.drop(['id', 'price_doc', 'timestamp'], axis=1, inplace=True)

df_test_fac.drop(['id', 'price_doc', 'timestamp'], axis=1, inplace=True)



# Remove timestamp column (may overfit the model in train)

df_all.drop(['id', 'price_doc', 'timestamp'], axis=1, inplace=True)



print('df_train_fac shape is', df_train_fac.shape)

print('df_test_fac shape is', df_test_fac.shape)

print('ylog_train_all shape is', ylog_train_all.shape)

print('df_all shape is', df_all.shape)
# Convert to numpy values

# X_all = df_all.values

# print(X_all.shape)



# Create a validation set, with last 20% of data

num_train = len(df_train_fac)

num_val = int(num_train * 0.2)



X_train_all = df_train_fac.values

X_train = X_train_all[:num_train-num_val]

X_val = X_train_all[num_train-num_val:num_train]

ylog_train = ylog_train_all[:-num_val]

ylog_val = ylog_train_all[-num_val:]



X_test = df_test_fac.values



df_columns = df_all.columns



print('X_train_all shape is', X_train_all.shape)

print('ylog_train_all shape is', ylog_train_all.shape)

print('X_train shape is', X_train.shape)

print('y_train shape is', ylog_train.shape)

print('X_val shape is', X_val.shape)

print('y_val shape is', ylog_val.shape)

print('X_test shape is', X_test.shape)
dtrain_all = xgb.DMatrix(X_train_all, ylog_train_all, feature_names=df_columns)

dtrain = xgb.DMatrix(X_train, ylog_train, feature_names=df_columns)

dval = xgb.DMatrix(X_val, ylog_val, feature_names=df_columns)

dtest = xgb.DMatrix(X_test, feature_names=df_columns)
xgb_params = {

    'eta': 0.05,

    'max_depth': 3,

    'subsample': 0.7,

    'colsample_bytree': 0.7,

    'objective': 'reg:linear',

    'eval_metric': 'rmse',

    'silent': 1,

    'min_child_weight': 3,

    'gamma': 0.3

}



# # Uncomment to tune XGB `num_boost_rounds`

# partial_model = xgb.train(xgb_params, dtrain, num_boost_round=1000, evals=[(dval, 'val')],

#                        early_stopping_rounds=20, verbose_eval=20)



# num_boost_round = partial_model.best_iteration
# Uncomment to tune XGB `num_boost_rounds`



cv_result = xgb.cv(xgb_params, dtrain, num_boost_round=1000, early_stopping_rounds=20,

   verbose_eval=20, show_stdv=True)

cv_result[['train-rmse-mean', 'test-rmse-mean']].plot()

num_boost_rounds = len(cv_result)



model = xgb.train(xgb_params, dtrain_all, num_boost_round=num_boost_rounds, verbose_eval=20)



ylog_pred = model.predict(dtest)

y_pred = np.exp(ylog_pred) - 1



df_sub = pd.DataFrame({'id': id_test, 'price_doc': y_pred})



df_sub.to_csv('Predict_xgb.csv', index=False)

plt.show()
# diviation exploration



predict_y = model.predict(dtrain_all)

plt.scatter(ylog_train_all, predict_y)

plt.plot([13, 19], [13, 19])

plt.show()
dev = predict_y - ylog_train_all

df_train['price_sq'] = df_train['price_doc'] / df_train['full_sq']

large_dev_index = np.abs(dev) > 0.5

train_large_dev = df_train[large_dev_index]

large_dev = dev[large_dev_index]
plt.scatter(df_train.loc[df_train['product_type'] == 'Investment', 'price_sq'], 

            dev[df_train['product_type'] == 'Investment'], 

            color = 'r', marker  = 'o', label = 'Investment')

plt.scatter(df_train.loc[df_train['product_type'] == 'OwnerOccupier', 'price_sq'], 

            dev[df_train['product_type'] == 'OwnerOccupier'], 

            color = 'b', marker = 's', label = 'OwnerOccupier')

plt.legend()

plt.xlim((0, 600000))

plt.xlabel('Price per square meter')

plt.ylabel('Prediction - ground truth (log1p)')

plt.show()
plt.scatter(df_train.loc[df_train['product_type'] == 'Investment', 'price_doc'], 

            dev[df_train['product_type'] == 'Investment'], 

            color = 'r', marker  = 'o', label = 'Investment')

plt.scatter(df_train.loc[df_train['product_type'] == 'OwnerOccupier', 'price_doc'], 

            dev[df_train['product_type'] == 'OwnerOccupier'], 

            color = 'b', marker = 's', label = 'OwnerOccupier')

plt.legend()

plt.xlim((0, 1E8))

plt.xlabel('Price')

plt.ylabel('Prediction - ground truth (log1p)')

plt.show()