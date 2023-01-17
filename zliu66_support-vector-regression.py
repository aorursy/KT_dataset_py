import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import linear_model

from sklearn import neural_network

from sklearn.svm import SVR

from sklearn.ensemble import RandomForestRegressor



from sklearn.metrics import mean_squared_error





import matplotlib.pyplot as plt

# from sklearn.utils import shuffle

# from sklearn.model_selection import GridSearchCV

import seaborn as sns

from sklearn import preprocessing





# Input data files are available in the 'input/' directory.





%matplotlib inline

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots

plt.rcParams['image.interpolation'] = 'nearest'

plt.rcParams['image.cmap'] = 'gray'



%load_ext autoreload

%autoreload 2

realty_drop_cols = ['male_f', 'female_f', 'young_female', 'work_all', 'work_female', 

                   'railroad_station_walk_min', 'railroad_station_avto_km', 'railroad_station_avto_min',

                  'sadovoe_km', 'bulvar_ring_km', 'kremlin_km']

macro_cols = ["balance_trade", "balance_trade_growth", "eurrub", "average_provision_of_build_contract", 'cpi', 'brent', 

"micex_rgbi_tr", "micex_cbi_tr", "deposits_rate", "mortgage_value", "mortgage_rate",

"income_per_cap", "rent_price_4+room_bus", "museum_visitis_per_100_cap", "apartment_build"]



df_train = pd.read_csv("../input/train.csv", parse_dates=['timestamp'])



df_test = pd.read_csv("../input/test.csv", parse_dates=['timestamp'])

df_macro = pd.read_csv("../input/macro.csv", parse_dates=['timestamp'], usecols=['timestamp'] + macro_cols)

df_macro = pd.read_csv("../input/macro.csv", parse_dates=['timestamp'])
# ylog will be log(1+y), as suggested by https://github.com/dmlc/xgboost/issues/446#issuecomment-135555130

ylog_train_all = np.log1p(df_train['price_doc'].values)

id_test = df_test['id']



df_train.drop(['id', 'price_doc'], axis=1, inplace=True)

df_test.drop(['id'], axis=1, inplace=True)



# Build df_all = (df_train+df_test).join(df_macro)

num_train = len(df_train)

df_all = pd.concat([df_train, df_test])

df_all = pd.merge_ordered(df_all, df_macro, on='timestamp', how='left')

df_all.drop(realty_drop_cols, axis=1, inplace=True)

# print(df_all.shape)
# price_ulimit = np.log1p(1E8)

# df_all =  df_all.loc[df_all['price_doc'] < price_ulimit,:] 



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



# Remove timestamp column (may overfit the model in train)

df_all.drop(['timestamp'], axis=1, inplace=True)
for f in df_all.columns:

    if df_all[f].dtype=='object':

        lbl = preprocessing.LabelEncoder()

        lbl.fit(list(df_all[f].values.astype('str')) )

        df_all[f] = lbl.transform(list(df_all[f].values.astype('str')))
# train_df.fillna(-99, inplace=True)

# test_df.fillna(-99, inplace=True)





df_all= df_all.fillna(df_all.median())
# Convert to numpy values

X_all = df_all.values

X_all = preprocessing.normalize(X_all, norm='l1', axis=0, copy=True, return_norm=False)





# print(X_all.shape)



# Create a validation set, with last 20% of data

num_val = int(num_train * 0.2)



X_train_all = X_all[:num_train]

X_train = X_all[:num_train-num_val]

X_val = X_all[num_train-num_val:num_train]

ylog_train = ylog_train_all[:-num_val]

ylog_val = ylog_train_all[-num_val:]



X_test = X_all[num_train:]



df_columns = df_all.columns
clf = SVR(C=1.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.2, gamma='auto',

    kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)

clf.fit(X_train, ylog_train)



training_loss = np.sqrt(mean_squared_error(ylog_train, clf.predict(X_train)))

validation_loss = np.sqrt(mean_squared_error(ylog_val, clf.predict(X_val)))



print('Training loss is {}'.format(training_loss))

print('Validation loss is {}'.format(validation_loss))



test_y_SVR = np.exp(clf.predict(X_test)) - 1

df_sub = pd.DataFrame({'id': id_test, 'price_doc': test_y_SVR})

df_sub.to_csv('Predict_SVR.csv', index=False)
regNN = neural_network.MLPRegressor(hidden_layer_sizes = (100, 100, 100, 100, 100))

regNN.fit (X_train, ylog_train)

training_loss = np.sqrt(mean_squared_error(ylog_train, regNN.predict(X_train)))

validation_loss = np.sqrt(mean_squared_error(ylog_val, regNN.predict(X_val)))



print('Training loss is {}'.format(training_loss))

print('Validation loss is {}'.format(validation_loss))



test_y_regNN = np.exp(regNN.predict(X_test)) - 1

df_sub = pd.DataFrame({'id': id_test, 'price_doc': test_y_regNN})

df_sub.to_csv('Predict_train_SVR.csv', index=False)

rfr = RandomForestRegressor(n_estimators = 200, max_depth = 40, min_samples_split = 20)

rfr.fit (X_train, ylog_train)

training_loss = np.sqrt(mean_squared_error(ylog_train, rfr.predict(X_train)))

validation_loss = np.sqrt(mean_squared_error(ylog_val, rfr.predict(X_val)))



print('Training loss is {}'.format(training_loss))

print('Validation loss is {}'.format(validation_loss))





test_y_rfr = np.exp(rfr.predict(X_test)) - 1

df_sub = pd.DataFrame({'id': id_test, 'price_doc': test_y_rfr})

df_sub.to_csv('Predict_train_rfr.csv', index=False)



