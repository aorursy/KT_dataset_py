# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import os, gc, pickle, copy, datetime, warnings

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import lightgbm as lgb

import xgboost as xgb

from xgboost import plot_importance, plot_tree

import pandas_profiling

pd.set_option('display.max_columns', 100)

warnings.filterwarnings('ignore')
df_train = pd.read_csv("../input/covid19-global-forecasting-week-3/train.csv")

print(df_train.shape)

print(df_train.Date.min(), df_train.Date.max())

df_train.head()
train_min_date, train_max_date = df_train.Date.min(), df_train.Date.max()

train_min_dayofyear, train_max_dayofyear = (pd.to_datetime(train_min_date)).dayofyear, (pd.to_datetime(train_max_date)).dayofyear

print(train_min_dayofyear, train_max_dayofyear)
train_valid_cutoff_dayofyear = train_min_dayofyear + ( train_max_dayofyear - train_min_dayofyear ) // 3 * 2

train_valid_cutoff_dayofyear
df_test = pd.read_csv("../input/covid19-global-forecasting-week-3/test.csv")

print(df_test.shape)

test_min_date, test_max_date = df_test.Date.min(), df_test.Date.max()

print(test_min_date, test_max_date)

df_test.head()
# concat train and test

df_traintest = pd.concat([df_train, df_test])

print(df_train.shape, df_test.shape, df_traintest.shape)

df_traintest.head()
# concat Country/Region and Province/State

def func(x):

    try:

        x_new = x['Country_Region'] + "/" + x['Province_State']

    except:

        x_new = x['Country_Region']

    return x_new

        

df_traintest['place_id'] = df_traintest.apply(lambda x: func(x), axis=1)

tmp = np.sort(df_traintest['place_id'].unique())

print("num unique places: {}".format(len(tmp)))

print(tmp[:10])
# process date

# df_traintest['Date'] = pd.to_datetime(df_traintest['Date'])

# df_traintest['day'] = df_traintest['Date'].apply(lambda x: x.dayofyear).astype(np.int16)

# df_traintest['dayofmonth'] = df_traintest['Date'].apply(lambda x: x.day).astype(np.int16)

# df_traintest['dayofweek'] = df_traintest['Date'].apply(lambda x: x.dayofweek).astype(np.int16)

# df_traintest.head()



#     # time features

df_traintest['Date'] = pd.to_datetime(df_traintest['Date'])

time_cols = [

#     "year", "quarter", 

    "month", 

    "week", 

    "day", 

    "dayofyear", 

    "dayofweek", 

#     "is_year_end", "is_year_start", "is_quarter_end", "is_quarter_start", 

#     "is_month_end","is_month_start",

]



for attr in time_cols:

    dtype = np.int if attr == "year" else np.int8

#     df_traintest[attr] = getattr(df_traintest['Date'].dt, attr).astype(dtype)

    df_traintest[attr] = getattr(df_traintest['Date'].dt, attr)

# df_traintest["is_weekend"] = df_traintest["dayofweek"].isin([5, 6]).astype(np.int8)

# time_cols += ["is_weekend"]

print(time_cols)

df_traintest.head(10)
# calc cases and fatalities per day

df_traintest['cases/day'] = 0

df_traintest['fatal/day'] = 0

places = np.sort(df_traintest['place_id'].unique())

for place in places:

    tmp = df_traintest['ConfirmedCases'][df_traintest['place_id']==place].values

    tmp[1:] -= tmp[:-1]

    df_traintest['cases/day'][df_traintest['place_id']==place] = tmp

    tmp = df_traintest['Fatalities'][df_traintest['place_id']==place].values

    tmp[1:] -= tmp[:-1]

    df_traintest['fatal/day'][df_traintest['place_id']==place] = tmp

    

df_traintest[df_traintest['place_id']=='China/Hubei']
# aggregate cases and fatalities

def do_aggregation(df, col, mean_range, method='mean', val_cols=[]):

    df_new = copy.deepcopy(df)

    col_new = '{}_{}_({}-{})'.format(col, method, mean_range[0], mean_range[1])

    val_cols.append(col_new)

    df_new[col_new] = 0

    if method=='mean':

        tmp = df_new[col].rolling(mean_range[1]-mean_range[0]+1).mean()

    elif method=='std':

        tmp = df_new[col].rolling(mean_range[1]-mean_range[0]+1).std()

    df_new[col_new][mean_range[0]:] = tmp[:-(mean_range[0])]

    df_new[col_new][pd.isna(df_new[col_new])] = 0

    return df_new[[col_new]].reset_index(drop=True)



# def do_aggregations(df):

#     for method in ['mean']:

#         df = pd.concat([df, do_aggregation(df, 'cases/day', [1,1], method).reset_index(drop=True)], axis=1)

#         df = pd.concat([df, do_aggregation(df, 'cases/day', [1,7], method).reset_index(drop=True)], axis=1)

#         df = pd.concat([df, do_aggregation(df, 'cases/day', [8,14], method).reset_index(drop=True)], axis=1)

#         df = pd.concat([df, do_aggregation(df, 'fatal/day', [1,1], method).reset_index(drop=True)], axis=1)

#         df = pd.concat([df, do_aggregation(df, 'fatal/day', [1,7], method).reset_index(drop=True)], axis=1)

#         df = pd.concat([df, do_aggregation(df, 'fatal/day', [8,14], method).reset_index(drop=True)], axis=1)

#     return df



def do_aggregations(df, roll_ranges=[[1,1], [1,7], [8,14]], val_cols=[]):

    for method in ['mean']:

        for roll_range in roll_ranges:

            df = pd.concat([df, do_aggregation(df, 'cases/day', roll_range, method, val_cols).reset_index(drop=True)], axis=1)

            df = pd.concat([df, do_aggregation(df, 'fatal/day', roll_range, method, val_cols).reset_index(drop=True)], axis=1)

    return df
df_traintest[df_traintest['dayofyear']<0]
df_traintest2 = []

val_cols = []

roll_ranges = [[i,i] for i in range(1,8)]

roll_ranges += [[1,7], [8,14]]



for place in places[:]:

    df_tmp = df_traintest[df_traintest['place_id']==place].reset_index(drop=True)

    df_tmp = do_aggregations(df_tmp, roll_ranges=roll_ranges, val_cols=val_cols)

    df_traintest2.append(df_tmp)

df_traintest2 = pd.concat(df_traintest2).reset_index(drop=True)



val_cols = list(set(val_cols))

print(val_cols)

df_traintest2[df_traintest2['place_id']=='China/Hubei'].head(20)
roll_ranges
# add Smoking rate per country

# data of smoking rate is obtained from https://ourworldindata.org/smoking

df_smoking = pd.read_csv("../input/shareofadultswhosmoke/adults-smoking-2000-2016.csv")

print(np.sort(df_smoking['Entity'].unique())[:10])

df_smoking.head()
# extract newest data

df_smoking_recent = df_smoking.sort_values('Year', ascending=False).reset_index(drop=True)

df_smoking_recent = df_smoking_recent[df_smoking_recent['Entity'].duplicated()==False]

df_smoking_recent['Country/Region'] = df_smoking_recent['Entity']

df_smoking_recent['SmokingRate'] = df_smoking_recent['Share of adults who smoke (%)']

df_smoking_recent.head()
# merge

df_traintest3 = pd.merge(df_traintest2, df_smoking_recent[['Country/Region', 'SmokingRate']], left_on='Country_Region', right_on='Country/Region', how='left')

df_traintest3.head()
## fill na with world smoking rate

SmokingRate = df_smoking_recent['SmokingRate'][df_smoking_recent['Entity']=='World'].values[0]

print("Smoking rate of the world: {:.6f}".format(SmokingRate))

df_traintest3['SmokingRate'][pd.isna(df_traintest3['SmokingRate'])] = SmokingRate

df_traintest3.head()
world_happiness_index = pd.read_csv("../input/world-bank-datasets/World_Happiness_Index.csv")

world_happiness_grouped = world_happiness_index.groupby('Country name').nth(-1)

world_happiness_grouped.head()

world_happiness_grouped.drop("Year", axis=1, inplace=True)



df_traintest3 = pd.merge(left=df_traintest3, right=world_happiness_grouped, how='left', left_on='Country_Region', right_on='Country name')
wh_cols = world_happiness_grouped.columns.to_list()

print(wh_cols)
malaria_world_health = pd.read_csv("../input/world-bank-datasets/Malaria_World_Health_Organization.csv")



df_traintest3 = pd.merge(left=df_traintest3, right=malaria_world_health, how='left', left_on='Country_Region', right_on='Country')

df_traintest3.drop("Country", axis=1, inplace=True)



mwh_cols = [ col for col in malaria_world_health.columns.to_list() if col != "Country" ]

print(mwh_cols)
human_development_index = pd.read_csv("../input/world-bank-datasets/Human_Development_Index.csv")

human_development_index.drop(["Gross national income (GNI) per capita 2018"], axis=1, inplace=True)



df_traintest3 = pd.merge(left=df_traintest3, right=human_development_index, how='left', left_on='Country_Region', right_on='Country')

df_traintest3.drop("Country", axis=1, inplace=True)



hdi_cols = [ col for col in human_development_index.columns.to_list() if col != "Country" ]

print(hdi_cols)
# df_lat_long = pd.concat( [ pd.read_csv("../input/covid19-global-forecasting-week-1/train.csv"), pd.read_csv("../input/covid19-global-forecasting-week-3/test.csv") ] )

# df_lat_long = df_lat_long[['Country/Region', 'Province/State', 'Lat', 'Long']].drop_duplicates()

# df_lat_long = df_lat_long.rename(columns={'Country/Region': 'Country_Region', 'Province/State': 'Province_State'})

# df_lat_long['place_id'] = df_lat_long.apply(lambda x: func(x), axis=1)

# df_lat_long.drop(["Country_Region", 'Province_State'], axis=1, inplace=True)



# df_traintest3 = pd.merge(left=df_traintest3, right=df_lat_long, how='left', on='place_id')
# df_lat_long = pd.concat( [ pd.read_csv("../input/covid19-global-forecasting-week-1/train.csv"), pd.read_csv("../input/covid19-global-forecasting-week-3/test.csv") ] )

# df_lat_long = df_lat_long[['Country/Region', 'Province/State', 'Lat', 'Long']].drop_duplicates()

# df_lat_long = df_lat_long.rename(columns={'Country/Region': 'Country_Region', 'Province/State': 'Province_State'})

# df_lat_long.to_csv("lat_long.csv", index=None)
df_lat_long = pd.read_csv("../input/lat-long/lat_long.csv")

df_lat_long['place_id'] = df_lat_long.apply(lambda x: func(x), axis=1)

df_lat_long.drop(["Country_Region", 'Province_State'], axis=1, inplace=True)



df_traintest3 = pd.merge(left=df_traintest3, right=df_lat_long, how='left', on='place_id')
df_lat_long.head()
tmp = df_lat_long['place_id'].unique()

print("num unique places: {}".format(len(tmp)))
# df_tmp = pd.get_dummies(df_traintest3['Province_State'], prefix='ps')

# ps_cols = df_tmp.columns.to_list()

# print(ps_cols)

# df_traintest3 = pd.concat([df_traintest3,df_tmp],axis=1)
# df_tmp = pd.get_dummies(df_traintest3['Country_Region'], prefix='cr')

# cr_cols = df_tmp.columns.to_list()

# print(cr_cols)

# df_traintest3 = pd.concat([df_traintest3,df_tmp],axis=1)
df_traintest3[df_traintest3['place_id']=='China/Hubei']
# params

SEED = 42

params = {'num_leaves': 8,

          'min_data_in_leaf': 5,  # 42,

          'objective': 'regression',

          'max_depth': 8,

          'learning_rate': 0.02,

          'boosting': 'gbdt',

          'bagging_freq': 5,  # 5

          'bagging_fraction': 0.8,  # 0.5,

          'feature_fraction': 0.8201,

          'bagging_seed': SEED,

          'reg_alpha': 1,  # 1.728910519108444,

          'reg_lambda': 4.9847051755586085,

          'random_state': SEED,

          'metric': 'mse',

          'verbosity': 100,

          'min_gain_to_split': 0.02,  # 0.01077313523861969,

          'min_child_weight': 5,  # 19.428902804238373,

          'num_threads': 6,

          }
df_traintest3[df_traintest3.dayofyear == 72]
df_traintest3.info()
# train model to predict fatalities/day

col_target = 'fatal/day'

col_var = [

    'Lat', 'Long',

#    'cases/day_mean_(1-1)', 'cases/day_mean_(1-7)', 'cases/day_mean_(8-14)', 

#      'fatal/day_mean_(1-1)', 'fatal/day_mean_(1-7)', 'fatal/day_mean_(8-14)',

#    'cases/day_std_(1-1)', 'cases/day_std_(1-7)', 'cases/day_std_(8-14)', 

#      'fatal/day_std_(1-1)', 'fatal/day_std_(1-7)', 'fatal/day_std_(8-14)',

    'SmokingRate',

#     'dayofyear',

#     'day',

#     'dayofweek',

]

col_var += val_cols

col_var += time_cols

# extra_cols = wh_cols + mwh_cols + hdi_cols + ps_cols + cr_cols

extra_cols = wh_cols + mwh_cols + hdi_cols

col_var += extra_cols



df_train = df_traintest3[(pd.isna(df_traintest3['ForecastId'])) & (df_traintest3['dayofyear']<train_valid_cutoff_dayofyear)]

df_valid = df_traintest3[(pd.isna(df_traintest3['ForecastId'])) & (df_traintest3['dayofyear']>=train_valid_cutoff_dayofyear)]

df_test = df_traintest3[pd.isna(df_traintest3['ForecastId'])==False]

X_train = df_train[col_var].values

X_valid = df_valid[col_var].values

y_train = df_train[col_target].values

y_valid = df_valid[col_target].values

train_data = lgb.Dataset(X_train, label=y_train)

valid_data = lgb.Dataset(X_valid, label=y_valid)

num_round = 15000

model = lgb.train(params, train_data, num_round, valid_sets=[train_data, valid_data],

                  verbose_eval=100,

                  early_stopping_rounds=150,)
# display feature importance

tmp = pd.DataFrame()

tmp["feature"] = col_var

tmp["importance"] = model.feature_importance()

tmp = tmp.sort_values('importance', ascending=False)



important_features = list(tmp[0:30]['feature'])

col_var = important_features



tmp
# df_train_profile = df_train[col_var].profile_report(title='Pandas Profile Report:Train Data')
# df_train_profile
# rejected_var = df_train_profile.get_rejected_variables()

# rejected_var
important_features
X_train = df_train[col_var].values

X_valid = df_valid[col_var].values

y_train = df_train[col_target].values

y_valid = df_valid[col_target].values

train_data = lgb.Dataset(X_train, label=y_train)

valid_data = lgb.Dataset(X_valid, label=y_valid)

num_round = 15000

model = lgb.train(params, train_data, num_round, valid_sets=[train_data, valid_data],

                  verbose_eval=100,

                  early_stopping_rounds=150,)
# model = xgb.XGBRegressor(n_estimators=1000)

# eval_set = [(df_valid[col_var], df_valid[col_target])]

# model.fit(df_train[col_var], df_train[col_target], eval_metric="rmse", eval_set=eval_set, verbose=True)
# 19.30146**2
# plot = plot_importance(model, height=0.9, max_num_features=20)
# train model to predict cases/day

col_target2 = 'cases/day'

col_var2 = [

    'Lat', 'Long',

#    'cases/day_mean_(1-1)', 'cases/day_mean_(1-7)', 'cases/day_mean_(8-14)', 

#      'fatal/day_mean_(1-1)', 'fatal/day_mean_(1-7)', 'fatal/day_mean_(8-14)',

#    'cases/day_std_(1-1)', 'cases/day_std_(1-7)', 'cases/day_std_(8-14)', 

#      'fatal/day_std_(1-1)', 'fatal/day_std_(1-7)', 'fatal/day_std_(8-14)',

    'SmokingRate',

#     'day',

#     'dayofmonth',

#     'dayofweek'

]

col_var2 += val_cols

col_var2 += time_cols

# col_var2 += ps_cols

# col_var2 += cr_cols

col_var2 += extra_cols



X_train = df_train[col_var2].values

X_valid = df_valid[col_var2].values

y_train = df_train[col_target2].values

y_valid = df_valid[col_target2].values

train_data = lgb.Dataset(X_train, label=y_train)

valid_data = lgb.Dataset(X_valid, label=y_valid)

num_round = 15000

model2 = lgb.train(params, train_data, num_round, valid_sets=[train_data, valid_data],

                  verbose_eval=100,

                  early_stopping_rounds=150,)
# display feature importance

tmp = pd.DataFrame()

tmp["feature"] = col_var2

tmp["importance"] = model2.feature_importance()

tmp = tmp.sort_values('importance', ascending=False)



important_features = list(tmp[0:30]['feature'])

col_var2 = important_features



tmp
important_features
X_train = df_train[col_var2].values

X_valid = df_valid[col_var2].values

y_train = df_train[col_target2].values

y_valid = df_valid[col_target2].values

train_data = lgb.Dataset(X_train, label=y_train)

valid_data = lgb.Dataset(X_valid, label=y_valid)

num_round = 15000

model2 = lgb.train(params, train_data, num_round, valid_sets=[train_data, valid_data],

                  verbose_eval=100,

                  early_stopping_rounds=150,)
# model2 = xgb.XGBRegressor(n_estimators=1000)

# eval_set = [(df_valid[col_var2], df_valid[col_target2])]

# model.fit(df_train[col_var2], df_train[col_target2], eval_metric="rmse", eval_set=eval_set, verbose=True)
# 202.84695**2
# remove overlaps between train and test

df_traintest4 = copy.deepcopy(df_traintest3)

df_traintest4['unique'] = df_traintest4.apply(lambda x: x['place_id'] + str(x['dayofyear']), axis=1)

print(len(df_traintest4))

df_traintest4 = df_traintest4[df_traintest4['unique'].duplicated()==False]

print(len(df_traintest4))

df_traintest4[(df_traintest4['place_id']=='China/Hubei') & (df_traintest4['dayofyear']>75)].head() #2020-03-15
# count the fatalities per place until Feb.

df_tmp = df_traintest[pd.isna(df_traintest['Fatalities'])==False]

df_tmp = df_tmp[df_tmp['dayofyear']<61]

df_agg = df_tmp.groupby('place_id')['Fatalities'].agg('max').reset_index()

df_agg = df_agg.sort_values('Fatalities', ascending=False)

df_agg.head()
print(len(col_var), len(col_var2))

col_var, col_var2
# Check the predictions of some hot areas.

place = 'China/Hubei'

# place = 'Iran'

df_interest_base = df_traintest4[df_traintest4['place_id']==place].reset_index(drop=True)

df_interest = copy.deepcopy(df_interest_base)

df_interest['cases/day'] = df_interest['cases/day'].astype(np.float)

df_interest['fatal/day'] = df_interest['fatal/day'].astype(np.float)

df_interest['cases/day'][df_interest['dayofyear']>=train_valid_cutoff_dayofyear] = -1

df_interest['fatal/day'][df_interest['dayofyear']>=train_valid_cutoff_dayofyear] = -1

len_known = (df_interest['cases/day']!=-1).sum()

len_unknown = (df_interest['cases/day']==-1).sum()

print("len train: {}, len prediction: {}".format(len_known, len_unknown))

for i in range(len_unknown): # use predicted cases and fatal for next days' prediction

#     print(i)

    X_valid = df_interest[col_var].iloc[i+len_known]

    X_valid2 = df_interest[col_var2].iloc[i+len_known]

#     print(X_valid.shape)

    pred_f = model.predict(X_valid)

    pred_c = model2.predict(X_valid2)

    df_interest['fatal/day'][i+len_known] = pred_f

    df_interest['cases/day'][i+len_known] = pred_c

    df_interest = df_interest[['cases/day', 'fatal/day', 'Long', 'Lat', 'SmokingRate']+time_cols+extra_cols]

    df_interest = do_aggregations(df_interest, roll_ranges=roll_ranges)



# visualize

tmp = df_interest_base['fatal/day'].values

tmp = np.cumsum(tmp)

sns.lineplot(x=df_interest_base['dayofyear'][pd.isna(df_interest_base['Fatalities'])==False],

             y=tmp[pd.isna(df_interest_base['Fatalities'])==False], label='true')

tmp = df_interest['fatal/day'].values

tmp = np.cumsum(tmp)

sns.lineplot(x=df_interest_base['dayofyear'], y=tmp, label='pred')

plt.show()
place = 'Iran'

df_interest_base = df_traintest4[df_traintest4['place_id']==place].reset_index(drop=True)

df_interest = copy.deepcopy(df_interest_base)

df_interest['cases/day'] = df_interest['cases/day'].astype(np.float)

df_interest['fatal/day'] = df_interest['fatal/day'].astype(np.float)

df_interest['cases/day'][df_interest['dayofyear']>=train_valid_cutoff_dayofyear] = -1

df_interest['fatal/day'][df_interest['dayofyear']>=train_valid_cutoff_dayofyear] = -1

len_known = (df_interest['cases/day']!=-1).sum()

len_unknown = (df_interest['cases/day']==-1).sum()

print("len train: {}, len prediction: {}".format(len_known, len_unknown))

for i in range(len_unknown): # use predicted cases and fatal for next days' prediction

    X_valid = df_interest[col_var].iloc[i+len_known]

    X_valid2 = df_interest[col_var2].iloc[i+len_known]

#     print(X_valid.shape)

    pred_f = model.predict(X_valid)

    pred_c = model2.predict(X_valid2)

    df_interest['fatal/day'][i+len_known] = pred_f

    df_interest['cases/day'][i+len_known] = pred_c

    df_interest = df_interest[['cases/day', 'fatal/day', 'Long', 'Lat', 'SmokingRate']+time_cols+extra_cols]

    df_interest = do_aggregations(df_interest, roll_ranges=roll_ranges)



# visualize

tmp = df_interest_base['fatal/day'].values

tmp = np.cumsum(tmp)

sns.lineplot(x=df_interest_base['dayofyear'][pd.isna(df_interest_base['Fatalities'])==False],

             y=tmp[pd.isna(df_interest_base['Fatalities'])==False], label='true')

tmp = df_interest['fatal/day'].values

tmp = np.cumsum(tmp)

sns.lineplot(x=df_interest_base['dayofyear'], y=tmp, label='pred')

plt.show()
df_traintest3[df_traintest3['dayofyear']<0]
place = 'Italy'

df_interest_base = df_traintest4[df_traintest4['place_id']==place].reset_index(drop=True)

df_interest = copy.deepcopy(df_interest_base)

df_interest['cases/day'] = df_interest['cases/day'].astype(np.float)

df_interest['fatal/day'] = df_interest['fatal/day'].astype(np.float)

df_interest['cases/day'][df_interest['dayofyear']>=train_valid_cutoff_dayofyear] = -1

df_interest['fatal/day'][df_interest['dayofyear']>=train_valid_cutoff_dayofyear] = -1

len_known = (df_interest['cases/day']!=-1).sum()

len_unknown = (df_interest['cases/day']==-1).sum()

print("len train: {}, len prediction: {}".format(len_known, len_unknown))

for i in range(len_unknown): # use predicted cases and fatal for next days' prediction

    X_valid = df_interest[col_var].iloc[i+len_known]

    X_valid2 = df_interest[col_var2].iloc[i+len_known]

#     print(X_valid.shape)

    pred_f = model.predict(X_valid)

    pred_c = model2.predict(X_valid2)

    df_interest['fatal/day'][i+len_known] = pred_f

    df_interest['cases/day'][i+len_known] = pred_c

    df_interest = df_interest[['cases/day', 'fatal/day', 'Long', 'Lat', 'SmokingRate']+time_cols+extra_cols]

    df_interest = do_aggregations(df_interest, roll_ranges=roll_ranges)



# visualize

tmp = df_interest_base['fatal/day'].values

tmp = np.cumsum(tmp)

sns.lineplot(x=df_interest_base['dayofyear'][pd.isna(df_interest_base['Fatalities'])==False],

             y=tmp[pd.isna(df_interest_base['Fatalities'])==False], label='true')

tmp = df_interest['fatal/day'].values

tmp = np.cumsum(tmp)

sns.lineplot(x=df_interest_base['dayofyear'], y=tmp, label='pred')

plt.show()
# train model to predict fatalities/day

# col_target = 'fatal/day'

# col_var = [

#     'Lat', 'Long',

# #     'cases/day_(1-1)', 'cases/day_(1-7)', 'cases/day_(8-14)', 

#     'fatal/day_(1-1)', 'fatal/day_(1-7)', 'fatal/day_(8-14)',

#     'SmokingRate',

# #     'day'

# ]

df_train = df_traintest3[(pd.isna(df_traintest3['ForecastId']))]

X_train = df_train[col_var].values

X_valid = df_train[col_var].values

y_train = df_train[col_target].values

y_valid = df_train[col_target].values

train_data = lgb.Dataset(X_train, label=y_train)

valid_data = lgb.Dataset(X_valid, label=y_valid)

num_round = 575

model = lgb.train(params, train_data, num_round, valid_sets=[train_data, valid_data], verbose_eval=100)
# train model to predict cases/day

df_train = df_traintest3[(pd.isna(df_traintest3['ForecastId']))]

X_train = df_train[col_var2].values

X_valid = df_train[col_var2].values

y_train = df_train[col_target2].values

y_valid = df_train[col_target2].values

train_data = lgb.Dataset(X_train, label=y_train)

valid_data = lgb.Dataset(X_valid, label=y_valid)

num_round = 225

model2 = lgb.train(params, train_data, num_round, valid_sets=[train_data, valid_data], verbose_eval=100,)
df_traintest4[(df_traintest4['place_id']=='China/Hubei') & (df_traintest4['dayofyear']>=72)]
# predict test data

df_preds = []

for i, place in enumerate(places[:]):

    df_interest = copy.deepcopy(df_traintest4[df_traintest4['place_id']==place].reset_index(drop=True))

    df_interest['cases/day'] = df_interest['cases/day'].astype(np.float)

    df_interest['fatal/day'] = df_interest['fatal/day'].astype(np.float)

    df_interest['cases/day'][(pd.isna(df_interest['ForecastId']))==False] = -1

    df_interest['fatal/day'][(pd.isna(df_interest['ForecastId']))==False] = -1

    len_known = (df_interest['cases/day']!=-1).sum()

    len_unknown = (df_interest['cases/day']==-1).sum()

    if (i+1)%10==0:

        print("{:3d}/{}  {}, len known: {}, len unknown: {}".format(i+1, len(places), place, len_known, len_unknown), df_interest.shape)

    for j in range(len_unknown): # use predicted cases and fatal for next days' prediction

        X_valid = df_interest[col_var].iloc[j+len_known]

        X_valid2 = df_interest[col_var2].iloc[j+len_known]

#         print(X_valid.shape)

        pred_f = model.predict(X_valid)

        pred_c = model2.predict(X_valid2)

#         print(pred_f, pred_c)

        df_interest['fatal/day'][j+len_known] = pred_f

        df_interest['cases/day'][j+len_known] = pred_c

        df_interest = df_interest[['cases/day', 'fatal/day', 'Long', 'Lat', 'SmokingRate', 'ForecastId', 'place_id']+time_cols+extra_cols]

        df_interest = do_aggregations(df_interest, roll_ranges=roll_ranges)

    df_interest['fatal_pred'] = np.cumsum(df_interest['fatal/day'].values)

    df_interest['cases_pred'] = np.cumsum(df_interest['cases/day'].values)

    df_preds.append(df_interest)
# concat prediction

df_preds= pd.concat(df_preds)

df_preds = df_preds.sort_values('dayofyear')

col_tmp = ['place_id', 'ForecastId', 'dayofyear', 'cases/day', 'cases_pred', 'fatal/day', 'fatal_pred',]

df_preds[col_tmp][(df_preds['place_id']=='Afghanistan') & (df_preds['dayofyear']>75)].head(10)
# load sample submission

df_sub = pd.read_csv("../input/covid19-global-forecasting-week-3/submission.csv")

print(len(df_sub))

df_sub.head()
# merge prediction with sub

df_sub = pd.merge(df_sub, df_traintest3[['ForecastId', 'place_id', 'dayofyear']])

df_sub = pd.merge(df_sub, df_preds[['place_id', 'dayofyear', 'cases_pred', 'fatal_pred']], on=['place_id', 'dayofyear',], how='left')

df_sub.head(10)
# save

df_sub['ConfirmedCases'] = df_sub['cases_pred']

df_sub['Fatalities'] = df_sub['fatal_pred']

df_sub = df_sub[['ForecastId', 'ConfirmedCases', 'Fatalities']]

df_sub.to_csv("submission.csv", index=None)

df_sub.head(10)