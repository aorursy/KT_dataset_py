import os, gc, pickle, copy, datetime, warnings

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import lightgbm as lgb

import xgboost as xgb

from xgboost import plot_importance, plot_tree

import pandas_profiling

from sklearn.preprocessing import MinMaxScaler, RobustScaler

from sklearn import metrics

pd.set_option('display.max_columns', 100)

pd.set_option('display.max_rows', 500)

warnings.filterwarnings('ignore')
df_train = pd.read_csv("../input/covid19-global-forecasting-week-4/train.csv")

print(df_train.shape)

print(df_train.Date.min(), df_train.Date.max())

df_train.head()
train_min_date, train_max_date = df_train.Date.min(), df_train.Date.max()

train_min_dayofyear, train_max_dayofyear = (pd.to_datetime(train_min_date)).dayofyear, (pd.to_datetime(train_max_date)).dayofyear

print(train_min_dayofyear, train_max_dayofyear)
train_valid_cutoff_dayofyear = train_min_dayofyear + ( train_max_dayofyear - train_min_dayofyear ) // 3 * 2

train_valid_cutoff_dayofyear
df_test = pd.read_csv("../input/covid19-global-forecasting-week-4/test.csv")

print(df_test.shape)

test_min_date, test_max_date = df_test.Date.min(), df_test.Date.max()

print(test_min_date, test_max_date)

df_test.head()
# concat train and test

df_traintest = pd.concat([df_train, df_test])

print(df_train.shape, df_test.shape, df_traintest.shape)

df_traintest.head()
# concat Country/Region and Province/State

def concat_country_province(x):

    try:

        x_new = x['Country_Region'] + "/" + x['Province_State']

    except:

        x_new = x['Country_Region']

    return x_new

        

df_traintest['place_id'] = df_traintest.apply(lambda x: concat_country_province(x), axis=1)

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
day_before_valid = train_valid_cutoff_dayofyear

day_before_public = 92 #2020-04-01

day_before_private = df_traintest['dayofyear'][pd.isna(df_traintest['ForecastId'])].max() # last day of train

print(df_traintest['Date'][df_traintest['dayofyear']==day_before_valid].values[0])

print(df_traintest['Date'][df_traintest['dayofyear']==day_before_public].values[0])

print(df_traintest['Date'][df_traintest['dayofyear']==day_before_private].values[0])
( df_traintest[pd.isna(df_traintest['ForecastId'])].groupby('place_id')['Date'].max() ).min()
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

            

    for threshold in [1, 10, 100]:

        days_under_threshold = (df['ConfirmedCases']<threshold).sum()

        tmp = df['dayofyear'].values - 22 - days_under_threshold

        tmp[tmp<=0] = 0

        df['days_since_{}cases'.format(threshold)] = tmp

        val_cols.append('days_since_{}cases'.format(threshold))

            

    for threshold in [1, 10, 100]:

        days_under_threshold = (df['Fatalities']<threshold).sum()

        tmp = df['dayofyear'].values - 22 - days_under_threshold

        tmp[tmp<=0] = 0

        df['days_since_{}fatal'.format(threshold)] = tmp

        val_cols.append('days_since_{}fatal'.format(threshold))

    

    # process China/Hubei

    if df['place_id'][0]=='China/Hubei':

        df['days_since_1cases'] += 35 # 2019/12/8

        df['days_since_10cases'] += 35-13 # 2019/12/8-2020/1/2 assume 2019/12/8+13

        df['days_since_100cases'] += 4 # 2020/1/18

        df['days_since_1fatal'] += 13 # 2020/1/9

    return df
df_traintest[df_traintest['dayofyear']<0]
df_traintest2 = []

val_cols = []

roll_ranges = [[i,i] for i in range(1,15)]

roll_ranges += [[1,7], [8,14], [15,21]]



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



df_smoking_recent["Country/Region"] = df_smoking_recent["Country/Region"].str.replace("South Korea", "Korea, South")

df_smoking_recent["Country/Region"] = df_smoking_recent["Country/Region"].str.replace("United States", "US")



df_smoking_recent.head()
# merge

df_traintest3 = pd.merge(df_traintest2, df_smoking_recent[['Country/Region', 'SmokingRate']], left_on='Country_Region', right_on='Country/Region', how='left')

df_traintest3.drop('Country/Region', axis=1, inplace=True)

df_traintest3.head()
## fill na with world smoking rate

SmokingRate = df_smoking_recent['SmokingRate'][df_smoking_recent['Entity']=='World'].values[0]

print("Smoking rate of the world: {:.6f}".format(SmokingRate))

df_traintest3['SmokingRate'][pd.isna(df_traintest3['SmokingRate'])] = SmokingRate

df_traintest3.head()
world_happiness_index = pd.read_csv("../input/world-bank-datasets/World_Happiness_Index.csv")

world_happiness_grouped = world_happiness_index.groupby('Country name').nth(-1)

world_happiness_grouped.drop("Year", axis=1, inplace=True)



world_happiness_grouped.dropna(axis=1, how='all', inplace=True)



print(world_happiness_grouped.shape)



world_happiness_grouped.index = world_happiness_grouped.index.str.replace("Taiwan Province of China", "Taiwan*")

world_happiness_grouped.index = world_happiness_grouped.index.str.replace("United States", "US")

world_happiness_grouped.index = world_happiness_grouped.index.str.replace("South Korea", "Korea, South")

world_happiness_grouped.index = world_happiness_grouped.index.str.replace("Ivory Coast", "Cote d'Ivoire")



df_traintest3 = pd.merge(left=df_traintest3, right=world_happiness_grouped, how='left', left_on='Country_Region', right_on='Country name')
wh_cols = world_happiness_grouped.columns.to_list()

print(wh_cols)
for wh_col in wh_cols:

    df_traintest3[wh_col][pd.isna(df_traintest3[wh_col])] = world_happiness_grouped[wh_col].mean()
malaria_world_health = pd.read_csv("../input/world-bank-datasets/Malaria_World_Health_Organization.csv")



df_traintest3 = pd.merge(left=df_traintest3, right=malaria_world_health, how='left', left_on='Country_Region', right_on='Country')

df_traintest3.drop("Country", axis=1, inplace=True)



mwh_cols = [ col for col in malaria_world_health.columns.to_list() if col != "Country" ]

print(mwh_cols)
df_traintest3[['Country_Region','Estimated number of malaria cases']][pd.isna(df_traintest3['Estimated number of malaria cases'])]
df_traintest3[['Estimated number of malaria cases']].isnull().sum()
human_development_index = pd.read_csv("../input/world-bank-datasets/Human_Development_Index.csv")

human_development_index.drop(["Gross national income (GNI) per capita 2018"], axis=1, inplace=True)



human_development_index['Country'] = human_development_index['Country'].str.replace("South Korea", "Korea, South")

human_development_index['Country'] = human_development_index['Country'].str.replace("United States", "US")



df_traintest3 = pd.merge(left=df_traintest3, right=human_development_index, how='left', left_on='Country_Region', right_on='Country')

df_traintest3.drop("Country", axis=1, inplace=True)



hdi_cols = [ col for col in human_development_index.columns.to_list() if col != "Country" ]

print(hdi_cols)



for hdi_col in hdi_cols:

    df_traintest3[hdi_col][pd.isna(df_traintest3[hdi_col])] = human_development_index[hdi_col].mean()
# add additional info from countryinfo dataset

df_country = pd.read_csv("../input/countryinfo/covid19countryinfo.csv", thousands=',')

df_country = df_country[df_country['country'].duplicated()==False]

print(df_country.shape)

df_country.head()
country_info_cols = ['density', 'pop', 'fertility']

print(country_info_cols)
df_country[country_info_cols].isnull().sum()
df_traintest3 = pd.merge(left=df_traintest3, 

                         right=df_country[['country']+country_info_cols], 

                         left_on=['Country_Region'], right_on=['country'], how='left')

df_traintest3.drop('country', axis=1, inplace=True)



df_traintest3['density'][df_traintest3['place_id']=="South Sudan"] = 18

df_traintest3['density'][df_traintest3['place_id']=="Angola"] = 14.8

df_traintest3['density'][df_traintest3['place_id']=="Botswana"] = 3

df_traintest3['density'][df_traintest3['place_id']=="Burma"] = 83

df_traintest3['density'][df_traintest3['place_id']=="Burundi"] = 463

df_traintest3['density'][df_traintest3['place_id']=="Malawi"] = 129

df_traintest3['density'][df_traintest3['place_id']=="Papua New Guinea"] = 17.8

df_traintest3['density'][df_traintest3['place_id']=="Sao Tome and Principe"] = 228

df_traintest3['density'][df_traintest3['place_id']=="Sierra Leone"] = 105

df_traintest3['density'][df_traintest3['place_id']=="West Bank and Gaza"] = 758.98

df_traintest3['density'][df_traintest3['place_id']=="Western Sahara"] = 2

df_traintest3['density'][df_traintest3['place_id']=="MS Zaandam"] = 1432.0+615.0



df_traintest3['pop'][df_traintest3['place_id']=="South Sudan"] = 10.98e6

df_traintest3['pop'][df_traintest3['place_id']=="Angola"] = 30.81e6

df_traintest3['pop'][df_traintest3['place_id']=="Botswana"] = 2.254e6

df_traintest3['pop'][df_traintest3['place_id']=="Burma"] = 53.71e6

df_traintest3['pop'][df_traintest3['place_id']=="Burundi"] = 11.18e6

df_traintest3['pop'][df_traintest3['place_id']=="Malawi"] = 18.14e6

df_traintest3['pop'][df_traintest3['place_id']=="Papua New Guinea"] = 8.606e6

df_traintest3['pop'][df_traintest3['place_id']=="Sao Tome and Principe"] = 211028.0

df_traintest3['pop'][df_traintest3['place_id']=="Sierra Leone"] = 7.65e6

df_traintest3['pop'][df_traintest3['place_id']=="West Bank and Gaza"] = 4.569e6

df_traintest3['pop'][df_traintest3['place_id']=="Western Sahara"] = 567402.0

df_traintest3['pop'][df_traintest3['place_id']=="MS Zaandam"] = 1432.0+615.0



df_traintest3['fertility'][df_traintest3['place_id']=="South Sudan"] = 4.78

df_traintest3['fertility'][df_traintest3['place_id']=="Angola"] = 5.60

df_traintest3['fertility'][df_traintest3['place_id']=="Botswana"] = 2.91

df_traintest3['fertility'][df_traintest3['place_id']=="Burma"] = 2.17

df_traintest3['fertility'][df_traintest3['place_id']=="Burundi"] = 5.50

df_traintest3['fertility'][df_traintest3['place_id']=="Malawi"] = 4.30

df_traintest3['fertility'][df_traintest3['place_id']=="Papua New Guinea"] = 3.61

df_traintest3['fertility'][df_traintest3['place_id']=="Sao Tome and Principe"] = 4.37

df_traintest3['fertility'][df_traintest3['place_id']=="Sierra Leone"] = 4.36

df_traintest3['fertility'][df_traintest3['place_id']=="West Bank and Gaza"] = 3.74

df_traintest3['fertility'][df_traintest3['place_id']=="Western Sahara"] = 3.79

# df_traintest3['fertility'][df_traintest3['place_id']=="MS Zaandam"] = 0.0
df_traintest3['place_id'][pd.isna(df_traintest3['density'])].unique()
df_traintest3['place_id'][pd.isna(df_traintest3['pop'])].unique()
df_traintest3['place_id'][pd.isna(df_traintest3['fertility'])].unique()
# df_lat_long = pd.concat( [ pd.read_csv("../input/covid19-global-forecasting-week-1/train.csv"), pd.read_csv("../input/covid19-global-forecasting-week-4/test.csv") ] )

# df_lat_long = df_lat_long[['Country/Region', 'Province/State', 'Lat', 'Long']].drop_duplicates()

# df_lat_long = df_lat_long.rename(columns={'Country/Region': 'Country_Region', 'Province/State': 'Province_State'})

# df_lat_long['place_id'] = df_lat_long.apply(lambda x: concat_country_province(x), axis=1)

# df_lat_long.drop(["Country_Region", 'Province_State'], axis=1, inplace=True)



# df_traintest3 = pd.merge(left=df_traintest3, right=df_lat_long, how='left', on='place_id')
# df_lat_long = pd.concat( [ pd.read_csv("../input/covid19-global-forecasting-week-1/train.csv"), pd.read_csv("../input/covid19-global-forecasting-week-4/test.csv") ] )

# df_lat_long = df_lat_long[['Country/Region', 'Province/State', 'Lat', 'Long']].drop_duplicates()

# df_lat_long = df_lat_long.rename(columns={'Country/Region': 'Country_Region', 'Province/State': 'Province_State'})

# df_lat_long.to_csv("lat_long.csv", index=None)
# df_lat_long = pd.read_csv("../input/lat-long/lat_long.csv")

# df_lat_long['place_id'] = df_lat_long.apply(lambda x: concat_country_province(x), axis=1)

# df_lat_long.drop(["Country_Region", 'Province_State'], axis=1, inplace=True)



# df_traintest3 = pd.merge(left=df_traintest3, right=df_lat_long, how='left', on='place_id')
# df_lat_long.head()
# tmp = df_lat_long['place_id'].unique()

# print("num unique places: {}".format(len(tmp)))
df_lat_long2 = pd.read_csv("../input/coronavirus-2019ncov/covid-19-all.csv")

df_lat_long2 = df_lat_long2[["Country/Region", "Province/State", "Latitude", "Longitude"]].drop_duplicates()

df_lat_long2 = df_lat_long2.rename(columns = {"Country/Region":"Country_Region", 

                                              "Province/State":"Province_State", 

                                              "Latitude":"Lat", 

                                              "Longitude":"Long"})



df_lat_long2['place_id'] = df_lat_long2.apply(lambda x: concat_country_province(x), axis=1)

df_lat_long2.drop(["Country_Region", 'Province_State'], axis=1, inplace=True)



df_lat_long2["place_id"] = df_lat_long2["place_id"].str.replace("Ivory Coast", "Cote d'Ivoire")

df_lat_long2["place_id"] = df_lat_long2["place_id"].str.replace("South Korea", "Korea, South")

df_lat_long2["place_id"] = df_lat_long2["place_id"].str.replace("Taiwan", "Taiwan*")

df_lat_long2["place_id"] = df_lat_long2["place_id"].str.replace("Vatican City", "Holy See")



df_lat_long2 = pd.concat([df_lat_long2,

          pd.DataFrame({"place_id": ['Czechia', 'Dominica', 'Niger'],

             "Lat": [49.8175, 15.4150, 17.6078],

             "Long": [15.4730, 61.3710, 8.0817]})])



df_lat_long2 = df_lat_long2[(pd.isna(df_lat_long2["Lat"])==False) & (pd.isna(df_lat_long2["Long"])==False)]

# df_lat_long2 = df_lat_long2.groupby(["Country_Region", "Province_State"]).mean().reset_index()

df_lat_long2 = df_lat_long2.groupby(["place_id"]).mean().reset_index()

print(df_lat_long2.shape)

df_lat_long2.head()
df_traintest3 = pd.merge(left=df_traintest3, right=df_lat_long2, how='left', on=["place_id"])
assert(len(df_traintest3[(pd.isna(df_traintest3["Long"])) | (pd.isna(df_traintest3["Lat"]))])==0)
def encode_label(df, col, freq_limit=0):

    df[col][pd.isna(df[col])] = 'nan'

    tmp = df[col].value_counts()

    cols = tmp.index.values

    freq = tmp.values

    num_cols = (freq>=freq_limit).sum()

    print("col: {}, num_cat: {}, num_reduced: {}".format(col, len(cols), num_cols))



    col_new = '{}_le'.format(col)

    df_new = pd.DataFrame(np.ones(len(df), np.int16)*(num_cols-1), columns=[col_new])

    for i, item in enumerate(cols[:num_cols]):

        df_new[col_new][df[col]==item] = i



    return df_new



def get_df_le(df, col_index, col_cat):

    df_new = df[[col_index]]

    for col in col_cat:

        df_tmp = encode_label(df, col)

        df_new = pd.concat([df_new, df_tmp], axis=1)

    return df_new



df_traintest3['id_le'] = np.arange(len(df_traintest3))

df_le = get_df_le(df_traintest3, 'id_le', ['Country_Region', 'Province_State'])

df_traintest3 = pd.merge(df_traintest3, df_le, on='id_le', how='left')
le_cols = ["Country_Region_le", "Province_State_le"]

print(le_cols)
# df_tmp = pd.get_dummies(df_traintest3['Province_State'], prefix='ps')

# ps_cols = df_tmp.columns.to_list()

# print(ps_cols)

# df_traintest3 = pd.concat([df_traintest3,df_tmp],axis=1)
# df_tmp = pd.get_dummies(df_traintest3['Country_Region'], prefix='cr')

# cr_cols = df_tmp.columns.to_list()

# print(cr_cols)

# df_traintest3 = pd.concat([df_traintest3,df_tmp],axis=1)
df_wk3 = pd.read_csv("../input/covid19-country-data-wk3-release/Data Join - RELEASE.csv")

df_wk3['place_id'] = df_wk3.apply(lambda x: concat_country_province(x), axis=1)

df_wk3['Personality_uai'][df_wk3['Personality_uai']=='#NULL!'] = np.nan

df_wk3['Personality_uai'] = df_wk3['Personality_uai'].astype(np.float64)

print(df_wk3.shape)

df_wk3.head()
df_wk3[['Personality_uai']].info()
df_wk3['Personality_uai'].unique()
wk3_cols = ['Personality_uai']

print(wk3_cols)
df_wk3[wk3_cols].isnull().sum()
df_traintest3 = pd.merge(left=df_traintest3, right=df_wk3[['place_id']+wk3_cols], how='left', on=["place_id"])
df_traintest3["place_id"][pd.isna(df_traintest3['Personality_uai'])].unique()
df_traintest3[['Personality_uai']].info()
df_traintest3[df_traintest3['place_id']=='China/Hubei']
def calc_score(y_true, y_pred):

    y_true[y_true<0] = 0

    score = metrics.mean_squared_error(np.log(y_true.clip(0, 1e10)+1), np.log(y_pred[:]+1))**0.5

    return score
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

col_var += country_info_cols

col_var += le_cols

col_var += wk3_cols



# df_train = df_traintest3[(pd.isna(df_traintest3['ForecastId'])) & (df_traintest3['dayofyear']<train_valid_cutoff_dayofyear)]

# df_valid = df_traintest3[(pd.isna(df_traintest3['ForecastId'])) & (df_traintest3['dayofyear']>=train_valid_cutoff_dayofyear)]

df_train = df_traintest3[(pd.isna(df_traintest3['ForecastId'])) & (df_traintest3['dayofyear']<=day_before_valid)]

df_valid = df_traintest3[(pd.isna(df_traintest3['ForecastId'])) & (day_before_valid<df_traintest3['dayofyear']) & (df_traintest3['dayofyear']<=day_before_public)]



df_test = df_traintest3[pd.isna(df_traintest3['ForecastId'])==False]

X_train = df_train[col_var].values

X_valid = df_valid[col_var].values

print(len(X_train), len(X_valid))



# scaler = MinMaxScaler()

# scaler = RobustScaler()

# X_train = scaler.fit_transform(X_train)

# X_valid = scaler.transform(X_valid)



# y_train = df_train[col_target].values

# y_valid = df_valid[col_target].values

y_train = np.log(df_train[col_target].values.clip(0, 1e10)+1)

y_valid = np.log(df_valid[col_target].values.clip(0, 1e10)+1)



train_data = lgb.Dataset(X_train, label=y_train)

valid_data = lgb.Dataset(X_valid, label=y_valid)

num_round = 15000

model = lgb.train(params, train_data, num_round, valid_sets=[train_data, valid_data],

                  verbose_eval=100,

                  early_stopping_rounds=150,)
y_true = df_valid['fatal/day'].values

y_pred = np.exp(model.predict(X_valid))-1

score = calc_score(y_true, y_pred)

print("{:.6f}".format(score))
df_features = pd.merge( left=(pd.DataFrame(model.feature_importance(), index=col_var, columns=["importance"])).sort_values('importance', ascending=False),

                      right=df_train[col_var].isnull().sum().to_frame(name='count_null'),

                      how='left', left_index=True, right_index=True)

df_features
important_features = df_features.index[df_features['importance']>=10].to_list()

print(len(important_features))

important_features
# df_train_profile = df_train[col_var].profile_report(title='Pandas Profile Report:Train Data')
# df_train_profile
# rejected_var = df_train_profile.get_rejected_variables()

# rejected_var
col_var = important_features



X_train = df_train[col_var].values

X_valid = df_valid[col_var].values



# scaler = MinMaxScaler()

# scaler = RobustScaler()

# X_train = scaler.fit_transform(X_train)

# X_valid = scaler.transform(X_valid)



# y_train = df_train[col_target].values

# y_valid = df_valid[col_target].values

y_train = np.log(df_train[col_target].values.clip(0, 1e10)+1)

y_valid = np.log(df_valid[col_target].values.clip(0, 1e10)+1)



train_data = lgb.Dataset(X_train, label=y_train)

valid_data = lgb.Dataset(X_valid, label=y_valid)

num_round = 15000

model = lgb.train(params, train_data, num_round, valid_sets=[train_data, valid_data],

                  verbose_eval=100,

                  early_stopping_rounds=150,)

best_itr = model.best_iteration
y_true = df_valid['fatal/day'].values

y_pred = np.exp(model.predict(X_valid))-1

score = calc_score(y_true, y_pred)

print("{:.6f}".format(score))
# train with all data before public

df_train = df_traintest3[(pd.isna(df_traintest3['ForecastId'])) & (df_traintest3['dayofyear']<=day_before_public)]

df_valid = df_traintest3[(pd.isna(df_traintest3['ForecastId'])) & (df_traintest3['dayofyear']<=day_before_public)]

df_test = df_traintest3[pd.isna(df_traintest3['ForecastId'])==False]

X_train = df_train[col_var].values

X_valid = df_valid[col_var].values

y_train = np.log(df_train[col_target].values.clip(0, 1e10)+1)

y_valid = np.log(df_valid[col_target].values.clip(0, 1e10)+1)

train_data = lgb.Dataset(X_train, label=y_train)

valid_data = lgb.Dataset(X_valid, label=y_valid)

model_pub = lgb.train(params, train_data, best_itr, valid_sets=[train_data, valid_data],

                  verbose_eval=100,

                  early_stopping_rounds=150,)
# from lightgbm import LGBMRegressor

# lgb_reg = LGBMRegressor(random_state=17)
# %%time

# lgb_reg.fit(X_train, y_train)
# from sklearn.metrics import mean_squared_error



# mean_squared_error(y_valid, lgb_reg.predict(X_valid))
# param_grid = {'num_leaves': [7, 15, 31, 63], 

#               'max_depth': [3, 4, 5, 6, -1]}
# from sklearn.model_selection import train_test_split, GridSearchCV



# grid_searcher = GridSearchCV(estimator=lgb_reg, param_grid=param_grid, 

#                              cv=5, verbose=1, n_jobs=4)
# grid_searcher.fit(X_train, y_train)
# grid_searcher.best_params_, grid_searcher.best_score_
# mean_squared_error(y_valid, grid_searcher.predict(X_valid))
# num_iterations = 500

# lgb_reg2 = LGBMRegressor(random_state=17, max_depth=3, 

#                           num_leaves=7, n_estimators=num_iterations,

#                           n_jobs=1)



# param_grid2 = {'learning_rate': np.logspace(-3, 0, 10)}

# grid_searcher2 = GridSearchCV(estimator=lgb_reg2, param_grid=param_grid2,

#                                cv=5, verbose=1, n_jobs=4)

# grid_searcher2.fit(X_train, y_train)

# print(grid_searcher2.best_params_, grid_searcher2.best_score_)

# print(mean_squared_error(y_valid, grid_searcher2.predict(X_valid)))
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

col_var2 += country_info_cols

col_var2 += le_cols

col_var2 += wk3_cols



df_train = df_traintest3[(pd.isna(df_traintest3['ForecastId'])) & (df_traintest3['dayofyear']<=day_before_valid)]

df_valid = df_traintest3[(pd.isna(df_traintest3['ForecastId'])) & (day_before_valid<df_traintest3['dayofyear']) & (df_traintest3['dayofyear']<=day_before_public)]



X_train = df_train[col_var2].values

X_valid = df_valid[col_var2].values

print(len(X_train), len(X_valid))



# scaler = MinMaxScaler()

# scaler = RobustScaler()

# X_train = scaler.fit_transform(X_train)

# X_valid = scaler.transform(X_valid)



# y_train = df_train[col_target2].values

# y_valid = df_valid[col_target2].values

y_train = np.log(df_train[col_target2].values.clip(0, 1e10)+1)

y_valid = np.log(df_valid[col_target2].values.clip(0, 1e10)+1)



train_data = lgb.Dataset(X_train, label=y_train)

valid_data = lgb.Dataset(X_valid, label=y_valid)

num_round = 15000

model2 = lgb.train(params, train_data, num_round, valid_sets=[train_data, valid_data],

                  verbose_eval=100,

                  early_stopping_rounds=150,)
y_true = df_valid['cases/day'].values

y_pred = np.exp(model2.predict(X_valid))-1

score = calc_score(y_true, y_pred)

print("{:.6f}".format(score))
df_features2 = pd.merge( left=(pd.DataFrame(model2.feature_importance(), index=col_var2, columns=["importance"])).sort_values('importance', ascending=False),

                      right=df_train[col_var2].isnull().sum().to_frame(name='count_null'),

                      how='left', left_index=True, right_index=True)

df_features2
important_features2 = df_features2.index[df_features2['importance']>=18].to_list()

print(len(important_features2))

important_features2
col_var2 = important_features2



X_train = df_train[col_var2].values

X_valid = df_valid[col_var2].values



# scaler = MinMaxScaler()

# scaler = RobustScaler()

# X_train = scaler.fit_transform(X_train)

# X_valid = scaler.transform(X_valid)



# y_train = df_train[col_target2].values

# y_valid = df_valid[col_target2].values

y_train = np.log(df_train[col_target2].values.clip(0, 1e10)+1)

y_valid = np.log(df_valid[col_target2].values.clip(0, 1e10)+1)



train_data = lgb.Dataset(X_train, label=y_train)

valid_data = lgb.Dataset(X_valid, label=y_valid)

num_round = 15000

model2 = lgb.train(params, train_data, num_round, valid_sets=[train_data, valid_data],

                  verbose_eval=100,

                  early_stopping_rounds=150,)

best_itr2 = model2.best_iteration
y_true = df_valid['cases/day'].values

y_pred = np.exp(model2.predict(X_valid))-1

score = calc_score(y_true, y_pred)

print("{:.6f}".format(score))
df_train = df_traintest3[(pd.isna(df_traintest3['ForecastId'])) & (df_traintest3['dayofyear']<=day_before_public)]

df_valid = df_traintest3[(pd.isna(df_traintest3['ForecastId'])) & (df_traintest3['dayofyear']<=day_before_public)]

X_train = df_train[col_var2].values

X_valid = df_valid[col_var2].values

y_train = np.log(df_train[col_target2].values.clip(0, 1e10)+1)

y_valid = np.log(df_valid[col_target2].values.clip(0, 1e10)+1)

train_data = lgb.Dataset(X_train, label=y_train)

valid_data = lgb.Dataset(X_valid, label=y_valid)

model2_pub = lgb.train(params, train_data, best_itr2, valid_sets=[train_data, valid_data],

                  verbose_eval=100,

                  early_stopping_rounds=150,)
# train model to predict fatalities/day

df_train = df_traintest3[(pd.isna(df_traintest3['ForecastId'])) & (df_traintest3['dayofyear']<=day_before_public)]

df_valid = df_traintest3[(pd.isna(df_traintest3['ForecastId'])) & (day_before_public<df_traintest3['dayofyear'])]

df_test = df_traintest3[pd.isna(df_traintest3['ForecastId'])==False]

X_train = df_train[col_var].values

X_valid = df_valid[col_var].values

y_train = np.log(df_train[col_target].values.clip(0, 1e10)+1)

y_valid = np.log(df_valid[col_target].values.clip(0, 1e10)+1)

train_data = lgb.Dataset(X_train, label=y_train)

valid_data = lgb.Dataset(X_valid, label=y_valid)

num_round = 15000

model = lgb.train(params, train_data, num_round, valid_sets=[train_data, valid_data],

                  verbose_eval=100,

                  early_stopping_rounds=150,)



best_itr = model.best_iteration
# train with all data

df_train = df_traintest3[(pd.isna(df_traintest3['ForecastId']))]

df_valid = df_traintest3[(pd.isna(df_traintest3['ForecastId']))]

X_train = df_train[col_var].values

X_valid = df_valid[col_var].values

y_train = np.log(df_train[col_target].values.clip(0, 1e10)+1)

y_valid = np.log(df_valid[col_target].values.clip(0, 1e10)+1)

train_data = lgb.Dataset(X_train, label=y_train)

valid_data = lgb.Dataset(X_valid, label=y_valid)

model_pri = lgb.train(params, train_data, best_itr, valid_sets=[train_data, valid_data],

                  verbose_eval=100,

                  early_stopping_rounds=150,)
# train model to predict cases/day

df_train = df_traintest3[(pd.isna(df_traintest3['ForecastId'])) & (df_traintest3['dayofyear']<=day_before_public)]

df_valid = df_traintest3[(pd.isna(df_traintest3['ForecastId'])) & (day_before_public<df_traintest3['dayofyear'])]

X_train = df_train[col_var2].values

X_valid = df_valid[col_var2].values

y_train = np.log(df_train[col_target2].values.clip(0, 1e10)+1)

y_valid = np.log(df_valid[col_target2].values.clip(0, 1e10)+1)

train_data = lgb.Dataset(X_train, label=y_train)

valid_data = lgb.Dataset(X_valid, label=y_valid)

model2 = lgb.train(params, train_data, num_round, valid_sets=[train_data, valid_data],

                  verbose_eval=100,

                  early_stopping_rounds=150,)

best_itr2 = model2.best_iteration
# train with all data

df_train = df_traintest3[(pd.isna(df_traintest3['ForecastId']))]

df_valid = df_traintest3[(pd.isna(df_traintest3['ForecastId']))]

X_train = df_train[col_var2].values

X_valid = df_valid[col_var2].values

y_train = np.log(df_train[col_target2].values.clip(0, 1e10)+1)

y_valid = np.log(df_valid[col_target2].values.clip(0, 1e10)+1)

train_data = lgb.Dataset(X_train, label=y_train)

valid_data = lgb.Dataset(X_valid, label=y_valid)

model2_pri = lgb.train(params, train_data, best_itr2, valid_sets=[train_data, valid_data],

                  verbose_eval=100,

                  early_stopping_rounds=150,)
# model2 = xgb.XGBRegressor(n_estimators=1000)

# eval_set = [(df_valid[col_var2], df_valid[col_target2])]

# model.fit(df_train[col_var2], df_train[col_target2], eval_metric="rmse", eval_set=eval_set, verbose=True)
print(df_traintest['Date'][df_traintest['dayofyear']==day_before_valid].values[0])

print(df_traintest['Date'][df_traintest['dayofyear']==day_before_public].values[0])

print(df_traintest['Date'][df_traintest['dayofyear']==day_before_private].values[0])
# remove overlap for public LB prediction

df_tmp = df_traintest3[

    ((df_traintest3['dayofyear']<=day_before_public)  & (pd.isna(df_traintest3['ForecastId'])))

    | ((day_before_public<df_traintest3['dayofyear']) & (pd.isna(df_traintest3['ForecastId'])==False))].reset_index(drop=True)

# df_tmp = df_tmp.drop([

#     'cases/day_(1-1)', 'cases/day_(1-7)', 'cases/day_(8-14)', 'cases/day_(15-21)', 

#     'fatal/day_(1-1)', 'fatal/day_(1-7)', 'fatal/day_(8-14)', 'fatal/day_(15-21)',

#     'days_since_1cases', 'days_since_10cases', 'days_since_100cases',

#     'days_since_1fatal', 'days_since_10fatal', 'days_since_100fatal',

#                                ],  axis=1)

df_tmp = df_tmp.drop(val_cols, axis=1)

df_traintest9 = []

for i, place in enumerate(places[:]):

    df_tmp2 = df_tmp[df_tmp['place_id']==place].reset_index(drop=True)

    df_tmp2 = do_aggregations(df_tmp2, roll_ranges=roll_ranges)

    df_traintest9.append(df_tmp2)

df_traintest9 = pd.concat(df_traintest9).reset_index(drop=True)

df_traintest9[df_traintest9['dayofyear']>day_before_public-2].head()
# remove overlap for private LB prediction

df_tmp = df_traintest3[

    ((df_traintest3['dayofyear']<=day_before_private)  & (pd.isna(df_traintest3['ForecastId'])))

    | ((day_before_private<df_traintest3['dayofyear']) & (pd.isna(df_traintest3['ForecastId'])==False))].reset_index(drop=True)

# df_tmp = df_tmp.drop([

#     'cases/day_(1-1)', 'cases/day_(1-7)', 'cases/day_(8-14)', 'cases/day_(15-21)', 

#     'fatal/day_(1-1)', 'fatal/day_(1-7)', 'fatal/day_(8-14)', 'fatal/day_(15-21)',

#     'days_since_1cases', 'days_since_10cases', 'days_since_100cases',

#     'days_since_1fatal', 'days_since_10fatal', 'days_since_100fatal',

#                                ],  axis=1)

df_tmp = df_tmp.drop(val_cols, axis=1)

df_traintest10 = []

for i, place in enumerate(places[:]):

    df_tmp2 = df_tmp[df_tmp['place_id']==place].reset_index(drop=True)

    df_tmp2 = do_aggregations(df_tmp2, roll_ranges=roll_ranges)

    df_traintest10.append(df_tmp2)

df_traintest10 = pd.concat(df_traintest10).reset_index(drop=True)

df_traintest10[df_traintest10['dayofyear']>day_before_private-2].head()
# predict test data in public

# predict the cases and fatatilites one day at a time and use the predicts as next day's feature recursively.

df_preds = []

for i, place in enumerate(places[:]):

    df_interest = copy.deepcopy(df_traintest9[df_traintest9['place_id']==place].reset_index(drop=True))

    df_interest['cases/day'][(pd.isna(df_interest['ForecastId']))==False] = -1

    df_interest['fatal/day'][(pd.isna(df_interest['ForecastId']))==False] = -1

    len_known = (df_interest['dayofyear']<=day_before_public).sum()

    len_unknown = (day_before_public<df_interest['dayofyear']).sum()

    for j in range(len_unknown): # use predicted cases and fatal for next days' prediction

        X_valid = df_interest[col_var].iloc[j+len_known]

        X_valid2 = df_interest[col_var2].iloc[j+len_known]

        pred_f = model_pub.predict(X_valid)

        pred_c = model2_pub.predict(X_valid2)

        pred_c = (np.exp(pred_c)-1).clip(0, 1e10)

        pred_f = (np.exp(pred_f)-1).clip(0, 1e10)

        df_interest['fatal/day'][j+len_known] = pred_f

        df_interest['cases/day'][j+len_known] = pred_c

        df_interest['Fatalities'][j+len_known] = df_interest['Fatalities'][j+len_known-1] + pred_f

        df_interest['ConfirmedCases'][j+len_known] = df_interest['ConfirmedCases'][j+len_known-1] + pred_c

#         print(df_interest['ConfirmedCases'][j+len_known-1], df_interest['ConfirmedCases'][j+len_known], pred_c)

#         df_interest = df_interest.drop([

#             'cases/day_(1-1)', 'cases/day_(1-7)', 'cases/day_(8-14)', 'cases/day_(15-21)', 

#             'fatal/day_(1-1)', 'fatal/day_(1-7)', 'fatal/day_(8-14)', 'fatal/day_(15-21)',

#             'days_since_1cases', 'days_since_10cases', 'days_since_100cases',

#             'days_since_1fatal', 'days_since_10fatal', 'days_since_100fatal',



#                                        ],  axis=1)

        df_interest = df_interest.drop(val_cols,  axis=1)

        df_interest = do_aggregations(df_interest, roll_ranges=roll_ranges)

    if (i+1)%10==0:

        print("{:3d}/{}  {}, len known: {}, len unknown: {}".format(i+1, len(places), place, len_known, len_unknown), df_interest.shape)

    df_interest['fatal_pred'] = np.cumsum(df_interest['fatal/day'].values)

    df_interest['cases_pred'] = np.cumsum(df_interest['cases/day'].values)

    df_preds.append(df_interest)

df_preds = pd.concat(df_preds)

df_preds.to_csv("df_preds.csv", index=None)
# predict test data in private

df_preds_pri = []

for i, place in enumerate(places[:]):

    df_interest = copy.deepcopy(df_traintest10[df_traintest10['place_id']==place].reset_index(drop=True))

    df_interest['cases/day'][(pd.isna(df_interest['ForecastId']))==False] = -1

    df_interest['fatal/day'][(pd.isna(df_interest['ForecastId']))==False] = -1

    len_known = (df_interest['dayofyear']<=day_before_private).sum()

    len_unknown = (day_before_private<df_interest['dayofyear']).sum()

    for j in range(len_unknown): # use predicted cases and fatal for next days' prediction

        X_valid = df_interest[col_var].iloc[j+len_known]

        X_valid2 = df_interest[col_var2].iloc[j+len_known]

        pred_f = model_pri.predict(X_valid)

        pred_c = model2_pri.predict(X_valid2)

        pred_c = (np.exp(pred_c)-1).clip(0, 1e10)

        pred_f = (np.exp(pred_f)-1).clip(0, 1e10)

        df_interest['fatal/day'][j+len_known] = pred_f

        df_interest['cases/day'][j+len_known] = pred_c

        df_interest['Fatalities'][j+len_known] = df_interest['Fatalities'][j+len_known-1] + pred_f

        df_interest['ConfirmedCases'][j+len_known] = df_interest['ConfirmedCases'][j+len_known-1] + pred_c

#         print(df_interest['ConfirmedCases'][j+len_known-1], df_interest['ConfirmedCases'][j+len_known], pred_c)

#         df_interest = df_interest.drop([

#             'cases/day_(1-1)', 'cases/day_(1-7)', 'cases/day_(8-14)', 'cases/day_(15-21)', 

#             'fatal/day_(1-1)', 'fatal/day_(1-7)', 'fatal/day_(8-14)', 'fatal/day_(15-21)',

#             'days_since_1cases', 'days_since_10cases', 'days_since_100cases',

#             'days_since_1fatal', 'days_since_10fatal', 'days_since_100fatal',



#                                        ],  axis=1)

        df_interest = df_interest.drop(val_cols,  axis=1)

        df_interest = do_aggregations(df_interest, roll_ranges=roll_ranges)

    if (i+1)%10==0:

        print("{:3d}/{}  {}, len known: {}, len unknown: {}".format(i+1, len(places), place, len_known, len_unknown), df_interest.shape)

    df_interest['fatal_pred'] = np.cumsum(df_interest['fatal/day'].values)

    df_interest['cases_pred'] = np.cumsum(df_interest['cases/day'].values)

    df_preds_pri.append(df_interest)

df_preds_pri = pd.concat(df_preds_pri)

df_preds_pri.to_csv("df_preds_pri.csv", index=None)
places_sort = df_traintest10[['place_id', 'ConfirmedCases']][df_traintest10['dayofyear']==day_before_private]

places_sort = places_sort.sort_values('ConfirmedCases', ascending=False).reset_index(drop=True)['place_id'].values

print(len(places_sort))

places_sort[:5]
print("Fatalities / Public")

plt.figure(figsize=(30,30))

for i in range(30):

    plt.subplot(5,6,i+1)

    idx = i * 10

    df_interest = df_preds[df_preds['place_id']==places_sort[idx]].reset_index(drop=True)

    tmp = df_interest['fatal/day'].values

    tmp = np.cumsum(tmp)

#     print(len(tmp), places_sort[idx])

    sns.lineplot(x=df_interest['dayofyear'], y=tmp, label='pred')

    df_interest2 = df_traintest10[(df_traintest10['place_id']==places_sort[idx]) & (df_traintest10['dayofyear']<=day_before_private)].reset_index(drop=True)

    sns.lineplot(x=df_interest2['dayofyear'].values, y=df_interest2['Fatalities'].values, label='true')

    plt.title(places_sort[idx])

plt.show()
print("Confirmed Cases / Public")

plt.figure(figsize=(30,30))

for i in range(30):

    plt.subplot(5,6,i+1)

    idx = i * 10

    df_interest = df_preds[df_preds['place_id']==places_sort[idx]].reset_index(drop=True)

    tmp = df_interest['cases/day'].values

    tmp = np.cumsum(tmp)

    sns.lineplot(x=df_interest['dayofyear'], y=tmp, label='pred')

    df_interest2 = df_traintest10[(df_traintest10['place_id']==places_sort[idx]) & (df_traintest10['dayofyear']<=day_before_private)].reset_index(drop=True)

    sns.lineplot(x=df_interest2['dayofyear'].values, y=df_interest2['ConfirmedCases'].values, label='true')

    plt.title(places_sort[idx])

plt.show()
print("Fatalities / Private")

plt.figure(figsize=(30,30))

for i in range(30):

    plt.subplot(5,6,i+1)

    idx = i * 10

    df_interest = df_preds_pri[df_preds_pri['place_id']==places_sort[idx]].reset_index(drop=True)

    tmp = df_interest['fatal/day'].values

    tmp = np.cumsum(tmp)

    sns.lineplot(x=df_interest['dayofyear'], y=tmp, label='pred')

    df_interest2 = df_traintest10[(df_traintest10['place_id']==places_sort[idx]) & (df_traintest10['dayofyear']<=day_before_private)].reset_index(drop=True)

    sns.lineplot(x=df_interest2['dayofyear'].values, y=df_interest2['Fatalities'].values, label='true')

    plt.title(places_sort[idx])

plt.show()
print("ConfirmedCases / Private")

plt.figure(figsize=(30,30))

for i in range(30):

    plt.subplot(5,6,i+1)

    idx = i * 10

    df_interest = df_preds_pri[df_preds_pri['place_id']==places_sort[idx]].reset_index(drop=True)

    tmp = df_interest['cases/day'].values

    tmp = np.cumsum(tmp)

    sns.lineplot(x=df_interest['dayofyear'], y=tmp, label='pred')

    df_interest2 = df_traintest10[(df_traintest10['place_id']==places_sort[idx]) & (df_traintest10['dayofyear']<=day_before_private)].reset_index(drop=True)

    sns.lineplot(x=df_interest2['dayofyear'].values, y=df_interest2['ConfirmedCases'].values, label='true')

    plt.title(places_sort[idx])

plt.show()
# # remove overlaps between train and test

# df_traintest4 = copy.deepcopy(df_traintest3)

# df_traintest4['unique'] = df_traintest4.apply(lambda x: x['place_id'] + str(x['dayofyear']), axis=1)

# print(len(df_traintest4))

# df_traintest4 = df_traintest4[df_traintest4['unique'].duplicated()==False]

# print(len(df_traintest4))

# df_traintest4[(df_traintest4['place_id']=='China/Hubei') & (df_traintest4['dayofyear']>75)].head() #2020-03-15
# # count the fatalities per place until Feb.

# df_tmp = df_traintest[pd.isna(df_traintest['Fatalities'])==False]

# df_tmp = df_tmp[df_tmp['dayofyear']<61]

# df_agg = df_tmp.groupby('place_id')['Fatalities'].agg('max').reset_index()

# df_agg = df_agg.sort_values('Fatalities', ascending=False)

# df_agg.head()
print(len(col_var), len(col_var2))

col_var, col_var2
# # Check the predictions of some hot areas.

# place = 'China/Hubei'

# # place = 'Iran'

# df_interest_base = df_traintest4[df_traintest4['place_id']==place].reset_index(drop=True)

# df_interest = copy.deepcopy(df_interest_base)

# df_interest['cases/day'] = df_interest['cases/day'].astype(np.float)

# df_interest['fatal/day'] = df_interest['fatal/day'].astype(np.float)

# df_interest['cases/day'][df_interest['dayofyear']>=train_valid_cutoff_dayofyear] = -1

# df_interest['fatal/day'][df_interest['dayofyear']>=train_valid_cutoff_dayofyear] = -1

# len_known = (df_interest['cases/day']!=-1).sum()

# len_unknown = (df_interest['cases/day']==-1).sum()

# print("len train: {}, len prediction: {}".format(len_known, len_unknown))

# for i in range(len_unknown): # use predicted cases and fatal for next days' prediction

# #     print(i)

#     X_valid = df_interest[col_var].iloc[i+len_known]

#     X_valid2 = df_interest[col_var2].iloc[i+len_known]

# #     print(X_valid.shape)

#     pred_f = model.predict(X_valid)

#     pred_c = model2.predict(X_valid2)

#     df_interest['fatal/day'][i+len_known] = pred_f

#     df_interest['cases/day'][i+len_known] = pred_c

#     df_interest = df_interest[['cases/day', 'fatal/day', 'Long', 'Lat', 'SmokingRate']+time_cols+extra_cols+country_info_cols+le_cols+wk3_cols]

#     df_interest = do_aggregations(df_interest, roll_ranges=roll_ranges)



# # visualize

# tmp = df_interest_base['fatal/day'].values

# tmp = np.cumsum(tmp)

# sns.lineplot(x=df_interest_base['dayofyear'][pd.isna(df_interest_base['Fatalities'])==False],

#              y=tmp[pd.isna(df_interest_base['Fatalities'])==False], label='true')

# tmp = df_interest['fatal/day'].values

# tmp = np.cumsum(tmp)

# sns.lineplot(x=df_interest_base['dayofyear'], y=tmp, label='pred')

# plt.show()
# place = 'Iran'

# df_interest_base = df_traintest4[df_traintest4['place_id']==place].reset_index(drop=True)

# df_interest = copy.deepcopy(df_interest_base)

# df_interest['cases/day'] = df_interest['cases/day'].astype(np.float)

# df_interest['fatal/day'] = df_interest['fatal/day'].astype(np.float)

# df_interest['cases/day'][df_interest['dayofyear']>=train_valid_cutoff_dayofyear] = -1

# df_interest['fatal/day'][df_interest['dayofyear']>=train_valid_cutoff_dayofyear] = -1

# len_known = (df_interest['cases/day']!=-1).sum()

# len_unknown = (df_interest['cases/day']==-1).sum()

# print("len train: {}, len prediction: {}".format(len_known, len_unknown))

# for i in range(len_unknown): # use predicted cases and fatal for next days' prediction

#     X_valid = df_interest[col_var].iloc[i+len_known]

#     X_valid2 = df_interest[col_var2].iloc[i+len_known]

# #     print(X_valid.shape)

#     pred_f = model.predict(X_valid)

#     pred_c = model2.predict(X_valid2)

#     df_interest['fatal/day'][i+len_known] = pred_f

#     df_interest['cases/day'][i+len_known] = pred_c

#     df_interest = df_interest[['cases/day', 'fatal/day', 'Long', 'Lat', 'SmokingRate']+time_cols+extra_cols+country_info_cols+le_cols+wk3_cols]

#     df_interest = do_aggregations(df_interest, roll_ranges=roll_ranges)



# # visualize

# tmp = df_interest_base['fatal/day'].values

# tmp = np.cumsum(tmp)

# sns.lineplot(x=df_interest_base['dayofyear'][pd.isna(df_interest_base['Fatalities'])==False],

#              y=tmp[pd.isna(df_interest_base['Fatalities'])==False], label='true')

# tmp = df_interest['fatal/day'].values

# tmp = np.cumsum(tmp)

# sns.lineplot(x=df_interest_base['dayofyear'], y=tmp, label='pred')

# plt.show()
# place = 'Italy'

# df_interest_base = df_traintest4[df_traintest4['place_id']==place].reset_index(drop=True)

# df_interest = copy.deepcopy(df_interest_base)

# df_interest['cases/day'] = df_interest['cases/day'].astype(np.float)

# df_interest['fatal/day'] = df_interest['fatal/day'].astype(np.float)

# df_interest['cases/day'][df_interest['dayofyear']>=train_valid_cutoff_dayofyear] = -1

# df_interest['fatal/day'][df_interest['dayofyear']>=train_valid_cutoff_dayofyear] = -1

# len_known = (df_interest['cases/day']!=-1).sum()

# len_unknown = (df_interest['cases/day']==-1).sum()

# print("len train: {}, len prediction: {}".format(len_known, len_unknown))

# for i in range(len_unknown): # use predicted cases and fatal for next days' prediction

#     X_valid = df_interest[col_var].iloc[i+len_known]

#     X_valid2 = df_interest[col_var2].iloc[i+len_known]

# #     print(X_valid.shape)

#     pred_f = model.predict(X_valid)

#     pred_c = model2.predict(X_valid2)

#     df_interest['fatal/day'][i+len_known] = pred_f

#     df_interest['cases/day'][i+len_known] = pred_c

#     df_interest = df_interest[['cases/day', 'fatal/day', 'Long', 'Lat', 'SmokingRate']+time_cols+extra_cols+country_info_cols+le_cols+wk3_cols]

#     df_interest = do_aggregations(df_interest, roll_ranges=roll_ranges)



# # visualize

# tmp = df_interest_base['fatal/day'].values

# tmp = np.cumsum(tmp)

# sns.lineplot(x=df_interest_base['dayofyear'][pd.isna(df_interest_base['Fatalities'])==False],

#              y=tmp[pd.isna(df_interest_base['Fatalities'])==False], label='true')

# tmp = df_interest['fatal/day'].values

# tmp = np.cumsum(tmp)

# sns.lineplot(x=df_interest_base['dayofyear'], y=tmp, label='pred')

# plt.show()
# # train model to predict fatalities/day

# # col_target = 'fatal/day'

# # col_var = [

# #     'Lat', 'Long',

# # #     'cases/day_(1-1)', 'cases/day_(1-7)', 'cases/day_(8-14)', 

# #     'fatal/day_(1-1)', 'fatal/day_(1-7)', 'fatal/day_(8-14)',

# #     'SmokingRate',

# # #     'day'

# # ]

# df_train = df_traintest3[(pd.isna(df_traintest3['ForecastId']))]

# X_train = df_train[col_var].values

# X_valid = df_train[col_var].values

# y_train = df_train[col_target].values

# y_valid = df_train[col_target].values

# train_data = lgb.Dataset(X_train, label=y_train)

# valid_data = lgb.Dataset(X_valid, label=y_valid)

# # num_round = 575

# num_round = 15000

# model = lgb.train(params, train_data, num_round, valid_sets=[train_data, valid_data], verbose_eval=100)
# # train model to predict cases/day

# df_train = df_traintest3[(pd.isna(df_traintest3['ForecastId']))]

# X_train = df_train[col_var2].values

# X_valid = df_train[col_var2].values

# y_train = df_train[col_target2].values

# y_valid = df_train[col_target2].values

# train_data = lgb.Dataset(X_train, label=y_train)

# valid_data = lgb.Dataset(X_valid, label=y_valid)

# # num_round = 225

# num_round = 15000

# model2 = lgb.train(params, train_data, num_round, valid_sets=[train_data, valid_data], verbose_eval=100,)
# # predict test data

# df_preds = []

# for i, place in enumerate(places[:]):

#     df_interest = copy.deepcopy(df_traintest4[df_traintest4['place_id']==place].reset_index(drop=True))

#     df_interest['cases/day'] = df_interest['cases/day'].astype(np.float)

#     df_interest['fatal/day'] = df_interest['fatal/day'].astype(np.float)

#     df_interest['cases/day'][(pd.isna(df_interest['ForecastId']))==False] = -1

#     df_interest['fatal/day'][(pd.isna(df_interest['ForecastId']))==False] = -1

#     len_known = (df_interest['cases/day']!=-1).sum()

#     len_unknown = (df_interest['cases/day']==-1).sum()

#     if (i+1)%10==0:

#         print("{:3d}/{}  {}, len known: {}, len unknown: {}".format(i+1, len(places), place, len_known, len_unknown), df_interest.shape)

#     for j in range(len_unknown): # use predicted cases and fatal for next days' prediction

#         X_valid = df_interest[col_var].iloc[j+len_known]

#         X_valid2 = df_interest[col_var2].iloc[j+len_known]

# #         print(X_valid.shape)

#         pred_f = model.predict(X_valid)

#         pred_c = model2.predict(X_valid2)

# #         print(pred_f, pred_c)

#         df_interest['fatal/day'][j+len_known] = pred_f

#         df_interest['cases/day'][j+len_known] = pred_c

#         df_interest = df_interest[['cases/day', 'fatal/day', 'Long', 'Lat', 'SmokingRate', 'ForecastId', 'place_id']+time_cols+extra_cols+country_info_cols+le_cols+wk3_cols]

#         df_interest = do_aggregations(df_interest, roll_ranges=roll_ranges)

#     df_interest['fatal_pred'] = np.cumsum(df_interest['fatal/day'].values)

#     df_interest['cases_pred'] = np.cumsum(df_interest['cases/day'].values)

#     df_preds.append(df_interest)
len(df_preds), len(df_preds_pri)
# merge 2 preds

df_preds[df_preds['dayofyear']>day_before_private] = df_preds_pri[df_preds['dayofyear']>day_before_private]
df_preds.to_csv("df_preds2.csv", index=None)
# # concat prediction

# df_preds= pd.concat(df_preds)

# df_preds = df_preds.sort_values('dayofyear')

# col_tmp = ['place_id', 'ForecastId', 'dayofyear', 'cases/day', 'cases_pred', 'fatal/day', 'fatal_pred',]

# df_preds[col_tmp][(df_preds['place_id']=='Afghanistan') & (df_preds['dayofyear']>75)].head(10)
# load sample submission

df_sub = pd.read_csv("../input/covid19-global-forecasting-week-4/submission.csv")

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