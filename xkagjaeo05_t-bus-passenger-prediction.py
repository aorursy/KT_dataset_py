import pandas as pd

import numpy as np



import matplotlib.pyplot as plt

import seaborn as sns



import missingno as msno



from glob import glob

import os, random, time, gc, warnings



from tqdm import tqdm_notebook



import lightgbm as lgbm

from sklearn.linear_model import Ridge, Lasso

from sklearn.ensemble import RandomForestRegressor

from sklearn.tree import DecisionTreeRegressor



from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import KFold, StratifiedKFold

from sklearn.metrics import mean_squared_error



from catboost import CatBoostRegressor

from sklearn.feature_selection import RFECV





from sklearn.cluster import KMeans



from datetime import datetime



from math import sqrt



import folium

from folium import Marker, Icon, CircleMarker



from pdpbox import pdp, info_plots



warnings.filterwarnings('ignore')



pd.set_option('max_columns', 500)

pd.set_option('max_rows', 500)



%matplotlib inline
# 상위 directory인 input에 무엇이 들어있나 확인합니다.

os.listdir('../input/')
# glob 함수를 이용하여 주최 측이 제공한 데이터셋을 확인합니다.

glob('../input/dacon-bus-dataset/*.*')
# 첫 10개의 row를 출력해보도록 합시다.

!head ../input/dacon-bus-dataset/train.csv
# 데이터를 load합니다.

# train/test --> string형식으로 저장되어 있는 `date` column은 datetime형식으로 수집합니다.

# bus_bts    --> string형식으로 저장되어 있는 `geton_date`, `getoff_date` columns은 datetime형식으로 수집합니다.

def load_dataset(path):

    train = pd.read_csv(path + 'train.csv', parse_dates=['date'])

    test  = pd.read_csv(path + 'test.csv', parse_dates=['date'])

    df_bus = pd.read_csv(path + 'bus_bts.csv', parse_dates = ['geton_date', 'getoff_date'] )

    sample_submission = pd.read_csv(path + 'submission_sample.csv')

    return train, test, df_bus, sample_submission



path = '../input/dacon-bus-dataset/'

%time train, test, df_bus, sample_submission = load_dataset(path)
# Train/Test-set을 각각 체크해봅니다.

display(train.head(3))



display(test.head(3))
print("-- Size -- ")

print(f"Train-set : {train.shape}")

print(f"Test-set  : {test.shape}")
# Train-set에만 있는 칼럼은?

train.columns.difference( test.columns )
# Train/Test-set은 어떻게 분리되었을까?

display(train.head(3))

display(test.head(3))
# Train-set의 id는?

print("Min/Max of id in Train-set")

display( train['id'].agg(['min','max']) )



print('='* 80)

print(f'Size : {len(train)}')
# test-set의 id는?

print("Min/Max of id in Test-set")

display( test['id'].agg(['min','max']) )



print('='* 80)

print(f'Size : {len(test)}')
# Train-set의 date는?

print("Min/Max of date in Train-set")

display( train['date'].agg(['min','max']) )



print('='* 80)

print(f'Size : {len(train)}')
# Train-set의 date는?

print("Min/Max of date in Test-set")

display( test['date'].agg(['min','max']) )



print('='* 80)

print(f'Size : {len(test)}')
# Train/Test의 date 분포는?



# Figure을 정의

plt.figure(figsize = (12,8))



# Train/Test-set 각각에서 특정 date가 몇 번 등장했는지 시각화 시킴

train['date'].value_counts().sort_index().plot(color='b', lw=2, label='train')

test['date'].value_counts().sort_index().plot(color='r',  lw=2, label='test')



plt.legend()

plt.xlabel("date")

plt.ylabel("# of rows")

plt.title("Distribution of date in Train/Test-set");
# Train/Test-set의 고유한 bus_route_id를 구함.

train_bus_route_id_set = set(train['bus_route_id'])

test_bus_route_id_set  = set(test['bus_route_id'])





# Train/Test-set 고유한 bus_route의 개수를 구함.

print(f"Train-set에 있는 고유한 bus_route의 개수 : { len(train_bus_route_id_set) }")

print(f"Test-set에 있는 고유한 bus_route의 개수 : { len(test_bus_route_id_set) }")



# Train/Test-set 모두에 포함되어있는 bus_route를 구함.

print('='* 80)

common_bus_route_id = train_bus_route_id_set.intersection(test_bus_route_id_set)

print(f"Train/Test-set에 공통으로 포함되어 있는 bus_route 개수 : {len(common_bus_route_id)}")



# Train-set에만 있는 bus_route를 구함.

print('='* 80)

only_train_bus_route = train_bus_route_id_set.difference(test_bus_route_id_set)

print(f"Train-set에만 있는 bus_route는 총 { len(only_train_bus_route) }개 입니다.")

print(f"Train-set에만 있는 bus_route는 : { sorted(only_train_bus_route ) }")



# Test-set에만 있는 bus_route를 구함.

print('='* 80)

only_test_bus_route = test_bus_route_id_set.difference(train_bus_route_id_set)

print(f"Test-set에만 있는 bus_route는 총 { len(only_test_bus_route) }개 입니다.")

print(f"Test-set에만 있는 bus_route는 : { sorted( only_test_bus_route ) }")
# Test-set에만 있는 bus_route_id와 Train/Test-set모두에 등장하는 bus_route_id의 탑승/하차 칼럼들의 합을 비교해보자



print("오직 Test-set에만 존재하는 bus_route_id")

display(test[test['bus_route_id'].isin(only_test_bus_route)].head() )



print("="*80)

print("Train/Test-set 모두에 존재하는 bus_route_id")

display(test[test['bus_route_id'].isin(common_bus_route_id)].head() )
# 탑승 관련 columns & 하차 관련 columns

ride_columns = [col for col in test.columns if '_ride' in col] + ['bus_route_id','date']

take_off_columns = [col for col in test.columns if '_takeoff' in col] + ['bus_route_id','date']
# 두 경우의 탑승 관련 columns 비교

plt.figure(figsize=(12,5))



test[test['bus_route_id'].isin(only_test_bus_route)][ride_columns].groupby(['date','bus_route_id'])['8~9_ride'].sum().groupby('date').mean().plot(color='b', lw=2, label='only in Test-set')

test[test['bus_route_id'].isin(common_bus_route_id)][ride_columns].groupby(['date','bus_route_id'])['8~9_ride'].sum().groupby('date').mean().plot(color='r', lw=2, label='Both in Train/Test-set')

plt.legend()

plt.title("Average number of passengers\nbus_route_id only in Test-set VS bus_route_id both in Train/Test-set ");
# Missing Values

msno.matrix(train)
# Missing Value 확인

print("Train-set")

display( train.isnull().sum() )



print('=' * 80)



print("Test-set")

display( test.isnull().sum() )

# Target Variable의 분포를 살펴보자

target_col = '18~20_ride'



train[target_col].value_counts().sort_index()
# Dist-plot을 그려보도록 한다.

# --> (1) 0이 굉장히 많다. 

# --> (2) right-skewed된 형태이며, 값이 매우 큰 outlier들이 존재한다.

sns.distplot( train[target_col] )
# log1p transformation을 적용해봐도 정규분포에 근사한 모양을 보이지 않는다.

sns.distplot( np.log1p( train[target_col] ) )
# 탑승 관련 columns & 하차 관련 columns

ride_columns = [col for col in test.columns if '_ride' in col]

take_off_columns = [col for col in test.columns if '_takeoff' in col] 
# Train-set의 승차관련 칼럼들의 rowsum

display( train[train[target_col]==0][ride_columns].sum(axis=1).agg(['min','max']) )



# Train-set의 하차관련 칼럼들의 rowsum

display( train[train[target_col]==0][take_off_columns].sum(axis=1).agg(['min','max']) )



# Train-set의 승하차관련 칼럼들의 rowsum

display( train[train[target_col]==0][ride_columns + take_off_columns].sum(axis=1).agg(['min','max']) )
# (1)의 경우에는 어떤 것들이 있나 예시를 통해 살펴보도록 하자

# 하나의 station_name에 여러 개의 station_code가 기록되어 있는 경우는 어떤 상황인가?

multiple_station_name = train.groupby('station_name')['station_code'].nunique()

multiple_station_name = multiple_station_name[multiple_station_name>=7]

print(multiple_station_name)



df_sample = train[train['station_name'].isin(multiple_station_name.index)][['station_code','station_name','latitude','longitude']]

df_sample = df_sample.drop_duplicates().reset_index(drop=True)

df_sample
def generateMap(default_location=[33.35098, 126.79807], default_zoom_start=10):

    base_map = folium.Map(location=default_location, 

                          control_scale=True, 

                          zoom_start=default_zoom_start)

    

    # 여러 개의 정거장에 대해서 Icon 생성하기

    for row in df_sample.itertuples():

        station_code, station_name, latitude, longitude = row[1:]

        

        # Create Icon

        if station_name == '금악리':

            icon = Icon(color='red',icon='station')

        else:

            icon = Icon(color='blue',icon='station')

                

        # Add Marker

        Marker(location=[ latitude , longitude], 

               popup=f'station_code : {station_code} station_name : {station_name}',

               icon = icon).add_to(base_map)

        

    

    base_map.save('하나의 station_name에 여러개의 station_code.html')

    return base_map



generateMap()
display( train.groupby('station_name')['station_code'].nunique().value_counts() )
# station_name를 기준으로 삼는다면?

# --> 하나의 station_name가 여러 개의 latitude, longitude를 갖는 것으로 보임

display( train.groupby('station_name')['latitude'].nunique().value_counts() )

display( train.groupby('station_name')['longitude'].nunique().value_counts() )
# station_code를 기준으로 삼는다면?

# --> station_code에는 1개의  station_name이 매핑되어있음.

display( train.groupby('station_code')['station_name'].nunique().value_counts() )
# station_code를 기준으로 삼는다면?

# --> station_code는 latitude, longitude와 1대1 관계를 만족함

display( train.groupby('station_code')['latitude'].nunique().value_counts() )

display( train.groupby('station_code')['longitude'].nunique().value_counts() )
# station_code를 기준으로 삼는다면?

# --> station_code는 in_out와 1대1 관계를 만족함

display( train.groupby('station_code')['in_out'].nunique().value_counts() )
# date, bus_route_id, station_code이 특정 날짜에 몇번 등장했는지 재확인하기 

display( train.groupby(['date','bus_route_id','station_code']).size().value_counts() )

print('='* 80)

print(f'Train-set size : {len(train)}')
# local_train/local_test를 만든다.

local_train = train[train['date']<='2019-09-24'].reset_index(drop=True)

local_test  = train[train['date']>'2019-09-24'].reset_index(drop=True)
# categorical variable인 'bus_route_id','in_out','station_code','station_name' 에 대해선 label_encoding을 적용해주고,

# numeric variable들에 대해선 있는 그대로 학습을 시켜보도록 한다.

lbl = LabelEncoder()



# Implement Label Encoding 

cat_cols = ['bus_route_id','in_out','station_code','station_name']

for col in tqdm_notebook( cat_cols ):

    # local_train과 local_test를 concat하여 temp_df에 저장

    temp_df = pd.concat([ local_train[[col]], local_test[[col]] ] , axis=0)

    

    # Label-Encoding을 fitting함

    lbl.fit( temp_df[col] )

    

    # local_train/local_test에 label_encoding한 값을 대입함

    local_train[col] = lbl.transform(local_train[col])

    local_test[col] = lbl.transform(local_test[col])
local_train.head()
# 모델에 쓰일 parameter 정의하기

n_splits= 5

NUM_BOOST_ROUND = 100000

SEED = 1993

lgbm_param = {'objective':'rmse',

              'boosting_type': 'gbdt',

              'random_state':1993,

              'learning_rate':0.3,

              'subsample':0.7,

              'tree_learner': 'serial',

              'colsample_bytree':0.78,

              'early_stopping_rounds':50,

              'subsample_freq': 1,

              'reg_lambda':7,

              'reg_alpha': 5,

              'num_leaves': 96,

              'seed' : SEED

            }
# 제거해야하는 columns들 정의

drop_cols = ['id','date', target_col]



# local_train/local_test에 대한 label 정의

local_train_label = local_train[target_col]

local_test_label  = local_test[target_col]
# local_train/local_test의 예측값을 저장하기 위한 OOF 만들기 & CV를 저장할 list 정의

oof_train = np.zeros((local_train.shape[0], ))

oof_test = np.zeros((local_test.shape[0], ))



cv_list = []





# Kfold 정의

kfolds = KFold(n_splits=n_splits, random_state=1993, shuffle=True)



# Fold별로 학습진행

for ind, (trn_ind, val_ind) in tqdm_notebook( enumerate( kfolds.split( X = local_train, y = local_train_label ) ) ):

    

    # Train/Valid-set을 정의하기

    X_train , y_train = local_train.iloc[trn_ind].drop(drop_cols, 1), local_train_label[trn_ind]

    X_valid , y_valid = local_train.iloc[val_ind].drop(drop_cols, 1), local_train_label[val_ind]

    

    # dtrain/dvalid 정의

    dtrain = lgbm.Dataset(X_train, y_train)

    dvalid = lgbm.Dataset(X_valid, y_valid)

    

    # model 정의&학습

    model = lgbm.train(lgbm_param , dtrain, NUM_BOOST_ROUND, 

                       valid_sets=(dtrain, dvalid), 

                       valid_names=('train','valid'), 

                       verbose_eval= 100)

    

    # local_valid/local_test에 대한 예측

    valid_pred = model.predict(X_valid)

    test_pred  = model.predict( local_test.drop(drop_cols,1) )

    

    # CV를 저장

    cv_list.append( sqrt( mean_squared_error( y_valid, valid_pred )  ) )

    

    # OOF에 예측값을 저장

    oof_train[val_ind] = valid_pred

    oof_test += test_pred/n_splits

    print('='*80)

    

print(f"<LOCAL_TRAIN> OVERALL RMSE : {sqrt( mean_squared_error( local_train_label, oof_train ) )}")

print(f"<LOCAL_TEST>  OVERALL RMSE : {sqrt( mean_squared_error( local_test_label, oof_test ) )}")
# 실제값과 예측값의 분포 비교

fig, axes = plt.subplots( 1, 2, figsize=(20, 8), sharex=True, sharey=True)



# y=x를 그리기 위하여

x_range = np.linspace(0, 300, 1000)



# <SUBPLOT 1> : local_train에 대한 예측/실제값 비교

axes[0].scatter( local_train_label, oof_train )

axes[0].set_xlabel("Prediction")

axes[0].set_ylabel("Real")



# y=x 그리기

axes[0].plot(x_range, x_range, color='r')



# <SUBPLOT 2> : local_test에 대한 예측/실제값 비교

axes[1].scatter( local_test_label, oof_test )

axes[1].set_xlabel("Prediction")

axes[1].set_ylabel("Real")



# y=x 그리기

axes[1].plot(x_range, x_range, color='r');



# Super Title 

plt.suptitle('Comparison between Prediction VS Real');
# 실제값 vs 예측값 비교

plt.figure(figsize=(12,6))



sns.distplot( oof_train, color='r' , label='Prediction for Local-Train')

sns.distplot( local_train_label, color='b', label='Real' )

plt.legend()

plt.title("Comparing the real vs prediction in Local-Train");

# 실제값 vs 예측값 비교

plt.figure(figsize=(12,6))



sns.distplot( oof_test, color='r' , label='Prediction for Local-Test')

sns.distplot( local_test_label, color='b', label='Real' )

plt.legend()

plt.title("Comparing the real vs prediction in Local-Test");
del local_train, local_test, local_train_label, local_test_label; gc.collect();
# train_label 정의

train_label = train[target_col]
# categorical variable에 대해서는 Label-Encoding을 수행 

# --> One-Hot Encoding가 바람직하다고 생각되나 메모리 문제로 실행할 수 없음.

lbl = LabelEncoder()



# Implement Label Encoding 

cat_cols = ['bus_route_id','in_out','station_code','station_name']

for col in tqdm_notebook( cat_cols ):

    

    # Label-Encoding을 fitting함

    lbl.fit( train[col] )

    

    # local_train/local_test에 label_encoding한 값을 대입함

    train[col] = lbl.transform(train[col])
# 각 모델에 대한 oof 정의

ridge_oof_train = np.zeros((train.shape[0]))

lasso_oof_train = np.zeros((train.shape[0]))

dt_oof_train = np.zeros((train.shape[0]))

rf_oof_train = np.zeros((train.shape[0]))

lgbm_oof_train = np.zeros((train.shape[0]))



# Kfold 정의

kfolds = KFold(n_splits=n_splits, random_state=1993, shuffle=True)



# Fold별로 학습진행

for ind, (trn_ind, val_ind) in tqdm_notebook( enumerate( kfolds.split( X = train, y = train_label ) ) ):

    

    # Train/Valid-set을 정의하기

    X_train , y_train = train.iloc[trn_ind].drop(drop_cols, 1), train_label[trn_ind]

    X_valid , y_valid = train.iloc[val_ind].drop(drop_cols, 1), train_label[val_ind]

    

    # (1) Ridge

    print("---TRAINING RIDGE---")

    ridge = Ridge(random_state = 1993)

    

    ridge.fit(X_train, y_train)

    

    ridge_valid_pred = ridge.predict(X_valid)

    ridge_oof_train[val_ind] = ridge_valid_pred

    

    # (2) Lasso

    print("---TRAINING LASSO---")

    lasso = Lasso(random_state = 1993)

    

    lasso.fit(X_train, y_train)

    

    lasso_valid_pred = lasso.predict(X_valid)

    lasso_oof_train[val_ind] = lasso_valid_pred

    

    # (3) Decision Tree

    print("---TRAINING DECISION TREE---")

    dt = DecisionTreeRegressor(random_state=231)

    

    dt.fit(X_train, y_train)

    

    dt_valid_pred = dt.predict(X_valid)

    dt_oof_train[val_ind] = dt_valid_pred

    

    

    # (4) Random Forest

    print("---TRAINING RANDOM FOREST---")

    rf = RandomForestRegressor(random_state=231, n_estimators=20 )

    

    rf.fit(X_train, y_train)

    

    rf_valid_pred = rf.predict(X_valid)

    rf_oof_train[val_ind] = rf_valid_pred

    

    # (5) Light GBM

    print("---TRAINING LIGHT GBM---")

    # dtrain/dvalid 정의

    dtrain = lgbm.Dataset(X_train, y_train)

    dvalid = lgbm.Dataset(X_valid, y_valid)

    

    # model 정의&학습

    model = lgbm.train(lgbm_param , dtrain, NUM_BOOST_ROUND, 

                       valid_sets=(dtrain, dvalid), 

                       valid_names=('train','valid'), 

                       verbose_eval= 0)

    

    # local_valid/local_test에 대한 예측

    lgbm_valid_pred = model.predict(X_valid)

        

    lgbm_oof_train[val_ind] = lgbm_valid_pred

    print('='*80)

    

print(f"<Ridge> OVERALL RMSE         : {sqrt( mean_squared_error( train_label, ridge_oof_train ) )}")

print(f"<Lasso> OVERALL RMSE         : {sqrt( mean_squared_error( train_label, lasso_oof_train ) )}")

print(f"<Decision-Tree> OVERALL RMSE : {sqrt( mean_squared_error( train_label, dt_oof_train ) )}")

print(f"<Random-Forest> OVERALL RMSE : {sqrt( mean_squared_error( train_label, rf_oof_train ) )}")

print(f"<Light-GBM> OVERALL RMSE     : {sqrt( mean_squared_error( train_label, lgbm_oof_train ) )}")
# Figure을 정의한다.

plt.figure(figsize=(24,5))



# Ridge의 Coef를 barplot으로 그린다.

plt.bar( train.drop(drop_cols,1).columns,  ridge.coef_ )



# y=0인 horizental한 선을 그린다.

plt.axhline(y=0, color='r', linestyle='-')



plt.xticks(rotation=45)

plt.title("Coef of Ridge Model");
# Figure을 정의한다.

plt.figure(figsize=(24,5))



# lasso의 Coef를 barplot으로 그린다.

plt.bar( train.drop(drop_cols,1).columns,  lasso.coef_ )



# y=0인 horizental한 선을 그린다.

plt.axhline(y=0, color='r', linestyle='-')



plt.xticks(rotation=45)

plt.title("Coef of lasso Model");
# 상관관계를 살펴보도록 하자.

train.corr()[target_col].sort_values()
# Figure을 정의한다.

plt.figure(figsize=(24,5))



# Ridge의 Coef를 barplot으로 그린다.

plt.bar( train.drop(drop_cols,1).columns,  rf.feature_importances_ )



# y=0인 horizental한 선을 그린다.

plt.axhline(y=0, color='r', linestyle='-')



plt.xticks(rotation=45)

plt.title("Coef of Random Forest Model");
# Figure을 정의한다.

plt.figure(figsize=(24,5))



# Ridge의 Coef를 barplot으로 그린다.

plt.bar( train.drop(drop_cols,1).columns,  model.feature_importance() )



# y=0인 horizental한 선을 그린다.

plt.axhline(y=0, color='r', linestyle='-')



plt.xticks(rotation=45)

plt.title("Coef of Random Forest Model");
# 전체 데이터를 사용할 시 너무 많은 시간이 소요되어 일부 샘플만 사용하도록 하겠습니다.

sample = train.drop(drop_cols,1).sample(1000)
# PDP Plot For 11~12_ride

pdp_ = pdp.pdp_isolate(

    model= model, dataset=sample, model_features=list(sample), feature='11~12_ride'

)

fig, axes = pdp.pdp_plot(pdp_, '11~12_ride')
# PDP Plot For 8~9_takeoff

pdp_ = pdp.pdp_isolate(

    model= model, dataset=sample, model_features=list(sample), feature='8~9_takeoff'

)

fig, axes = pdp.pdp_plot(pdp_, '8~9_takeoff')

# PDP Plot For latitude

pdp_ = pdp.pdp_isolate(

    model= model, dataset=sample, model_features=list(sample), feature='latitude'

)

fig, axes = pdp.pdp_plot(pdp_, 'latitude')

# PDP Plot For longitude

pdp_ = pdp.pdp_isolate(

    model= model, dataset=sample, model_features=list(sample), feature='longitude'

)

fig, axes = pdp.pdp_plot(pdp_, 'longitude')

# Interactive PDP Plot For latitude,longitude

pdp_ = pdp.pdp_interact(

    model= model, dataset=sample, model_features=list(sample), features=['latitude','longitude']

)



fig, axes = pdp.pdp_interact_plot(pdp_interact_out=pdp_,

                                  feature_names=['latitude','longitude'],

                                  plot_type='grid',

                                  x_quantile=True,

                                  plot_pdp=False)
# 모델에 쓰일 parameter 정의하기

n_splits= 5

NUM_BOOST_ROUND = 100000

SEED = 1993

lgbm_param = {'objective':'rmse',

              'boosting_type': 'gbdt',

              'random_state':1993,

              'learning_rate':0.1,

              'subsample':0.7,

              'tree_learner': 'serial',

              'colsample_bytree':0.78,

              'early_stopping_rounds':50,

              'subsample_freq': 1,

              'reg_lambda':7,

              'reg_alpha': 5,

              'num_leaves': 96,

              'seed' : SEED

            }
# 데이터를 load합니다.

# train/test --> string형식으로 저장되어 있는 `date` column은 datetime형식으로 수집합니다.

# bus_bts    --> string형식으로 저장되어 있는 `geton_date`, `getoff_date` columns은 datetime형식으로 수집합니다.

def load_dataset(path):

    train = pd.read_csv(path + 'train.csv', parse_dates=['date'])

    test  = pd.read_csv(path + 'test.csv', parse_dates=['date'])

    df_bus = pd.read_csv(path + 'bus_bts.csv', parse_dates = ['geton_date', 'getoff_date'] )

    sample_submission = pd.read_csv(path + 'submission_sample.csv')

    return train, test, df_bus, sample_submission



path = '../input/dacon-bus-dataset/'

%time train, test, df_bus, sample_submission = load_dataset(path)
# categorical variable에 대해서는 Label-Encoding을 수행 

# --> One-Hot Encoding가 바람직하다고 생각되나 메모리 문제로 실행할 수 없음.

lbl = LabelEncoder()



# Implement Label Encoding 

cat_cols = ['bus_route_id','in_out','station_code','station_name']

for col in tqdm_notebook( cat_cols ):

    

    # Label-Encoding을 fitting함

    lbl.fit( train[[col]].append(test[[col]]) )

    

    # train/test label_encoding한 값을 대입함

    train[col] = lbl.transform(train[col])

    test[col] = lbl.transform(test[col])
# 각 모델에 대한 oof 정의

lgbm_oof_train = np.zeros((train.shape[0]))



# Kfold 정의

kfolds = KFold(n_splits=n_splits, random_state=1993, shuffle=True)



# Fold별로 학습진행

for ind, (trn_ind, val_ind) in tqdm_notebook( enumerate( kfolds.split( X = train, y = train_label ) ) ):

    

    # Train/Valid-set을 정의하기

    X_train , y_train = train.iloc[trn_ind].drop(drop_cols, 1), train_label[trn_ind]

    X_valid , y_valid = train.iloc[val_ind].drop(drop_cols, 1), train_label[val_ind]

    

    

    

    # (5) Light GBM

    print("---TRAINING LIGHT GBM---")

    # dtrain/dvalid 정의

    dtrain = lgbm.Dataset(X_train, y_train)

    dvalid = lgbm.Dataset(X_valid, y_valid)

    

    # model 정의&학습

    model = lgbm.train(lgbm_param , dtrain, NUM_BOOST_ROUND, 

                       valid_sets=(dtrain, dvalid), 

                       valid_names=('train','valid'), 

                       verbose_eval= 100)

    

    # local_valid/local_test에 대한 예측

    lgbm_valid_pred = model.predict(X_valid)

        

    lgbm_oof_train[val_ind] = lgbm_valid_pred

    print('='*80)

    

print(f"<Light-GBM> OVERALL RMSE     : {sqrt( mean_squared_error( train_label, lgbm_oof_train ) )}")
# 피쳐 중요도 확인

df_imp = pd.DataFrame(data = {'col': model.feature_name(),

                              'imp': model.feature_importance()})

df_imp = df_imp.sort_values(by='imp', ascending=False).reset_index(drop=True)

df_imp 
# 승하차 간격을 2시간 간격으로 설정할 수는 없는가? (3시간으로 설정해도 ok -> 결국 실험의 영역)

dawn_ride_cols, dawn_takoff_cols = ['6~7_ride','7~8_ride'], ['6~7_takeoff','7~8_takeoff']

morning_ride_cols, morning_takeoff_cols = ['8~9_ride','9~10_ride'], ['8~9_takeoff','9~10_takeoff']

noon_ride_cols, noon_takeoff_cols = ['10~11_ride','11~12_ride'], ['10~11_takeoff','11~12_takeoff']



# df 가공

def modify_terms(df):

    # ride columns

    df['dawn_ride'] = df[dawn_ride_cols].sum(axis=1)

    df['morning_ride'] = df[morning_ride_cols].sum(axis=1)

    df['noon_ride'] = df[noon_ride_cols].sum(axis=1)

    

    # takeoff columns

    df['dawn_takeoff'] = df[dawn_takoff_cols].sum(axis=1)

    df['morning_takeoff'] = df[morning_takeoff_cols].sum(axis=1)

    df['noon_takeoff'] = df[noon_takeoff_cols].sum(axis=1)

    

    # drop columns

    drop_cols = dawn_ride_cols + morning_ride_cols + noon_ride_cols + dawn_takoff_cols + morning_takeoff_cols + noon_takeoff_cols

    df = df.drop(drop_cols, 1)

    

    return df

    



train = modify_terms(train)

test = modify_terms(test)
# 각 모델에 대한 oof 정의

lgbm_oof_train = np.zeros((train.shape[0]))



# Kfold 정의

kfolds = KFold(n_splits=n_splits, random_state=1993, shuffle=True)



# Fold별로 학습진행

for ind, (trn_ind, val_ind) in tqdm_notebook( enumerate( kfolds.split( X = train, y = train_label ) ) ):

    

    # Train/Valid-set을 정의하기

    X_train , y_train = train.iloc[trn_ind].drop(drop_cols, 1), train_label[trn_ind]

    X_valid , y_valid = train.iloc[val_ind].drop(drop_cols, 1), train_label[val_ind]

    

    

    

    # (5) Light GBM

    print("---TRAINING LIGHT GBM---")

    # dtrain/dvalid 정의

    dtrain = lgbm.Dataset(X_train, y_train)

    dvalid = lgbm.Dataset(X_valid, y_valid)

    

    # model 정의&학습

    model = lgbm.train(lgbm_param , dtrain, NUM_BOOST_ROUND, 

                       valid_sets=(dtrain, dvalid), 

                       valid_names=('train','valid'), 

                       verbose_eval= 100)

    

    # local_valid/local_test에 대한 예측

    lgbm_valid_pred = model.predict(X_valid)

        

    lgbm_oof_train[val_ind] = lgbm_valid_pred

    print('='*80)

    

print(f"<Light-GBM> OVERALL RMSE     : {sqrt( mean_squared_error( train_label, lgbm_oof_train ) )}")
# 피쳐 중요도 확인

df_imp = pd.DataFrame(data = {'col': model.feature_name(),

                              'imp': model.feature_importance()})

df_imp = df_imp.sort_values(by='imp', ascending=False).reset_index(drop=True)

df_imp 
# 요일 정보 추가

train['weekday'] = train['date'].dt.weekday

test['weekday']  = test['date'].dt.weekday



# 공휴일 정보 추가

# -> EDA필요

holidays = [datetime(2019, 9 ,12), datetime(2019, 9, 13), datetime(2019, 9 ,14), datetime(2019, 10,3), datetime(2019, 10,9) ]

train['is_holiday'] = train['date'].apply(lambda x: x in holidays).astype(np.int8)

test['is_holiday']  = test['date'].apply(lambda x: x in holidays).astype(np.int8)
# Mean Encoding

# (1) 일자별로 dawn, morning, noon에 각각 몇몇의 승객이 탑승하였는가

# (2) 일자별로 dawn, morning, noon에 각각 몇몇의 승객이 하차하였는가

# - 기준 :

# - (1) bus_route_id

# - (2) bus_route_id , station_code

# - (3) station_code



# (1) bus_route_id 기준



# 탑승

train['avg_dawn_ride_bus_route_id'] = train.groupby(['date','bus_route_id'])['dawn_ride'].transform('mean') 

train['avg_morning_ride_bus_route_id'] = train.groupby(['date','bus_route_id'])['morning_ride'].transform('mean') 

train['avg_noon_ride_bus_route_id'] = train.groupby(['date','bus_route_id'])['noon_ride'].transform('mean') 



test['avg_dawn_ride_bus_route_id'] = test.groupby(['date','bus_route_id'])['dawn_ride'].transform('mean') 

test['avg_morning_ride_bus_route_id'] = test.groupby(['date','bus_route_id'])['morning_ride'].transform('mean') 

test['avg_noon_ride_bus_route_id'] = test.groupby(['date','bus_route_id'])['noon_ride'].transform('mean') 



# 하차

train['avg_dawn_takeoff_bus_route_id'] = train.groupby(['date','bus_route_id'])['dawn_takeoff'].transform('mean') 

train['avg_morning_takeoff_bus_route_id'] = train.groupby(['date','bus_route_id'])['morning_takeoff'].transform('mean') 

train['avg_noon_takeoff_bus_route_id'] = train.groupby(['date','bus_route_id'])['noon_takeoff'].transform('mean') 



test['avg_dawn_takeoff_bus_route_id'] = test.groupby(['date','bus_route_id'])['dawn_takeoff'].transform('mean') 

test['avg_morning_takeoff_bus_route_id'] = test.groupby(['date','bus_route_id'])['morning_takeoff'].transform('mean') 

test['avg_noon_takeoff_bus_route_id'] = test.groupby(['date','bus_route_id'])['noon_takeoff'].transform('mean') 



# (2) bus_route_id, station_code 기준

# train['avg_dawn_ride_bus_route_id_station_code'] = train.groupby(['date','bus_route_id','station_code'])['dawn_ride'].transform('mean') 

# train['avg_morning_ride_bus_route_id_station_code'] = train.groupby(['date','bus_route_id','station_code'])['morning_ride'].transform('mean') 

# train['avg_noon_ride_bus_route_id_station_code'] = train.groupby(['date','bus_route_id','station_code'])['noon_ride'].transform('mean') 



# test['avg_dawn_ride_bus_route_id_station_code'] = test.groupby(['date','bus_route_id','station_code'])['dawn_ride'].transform('mean') 

# test['avg_morning_ride_bus_route_id_station_code'] = test.groupby(['date','bus_route_id','station_code'])['morning_ride'].transform('mean') 

# test['avg_noon_ride_bus_route_id_station_code'] = test.groupby(['date','bus_route_id','station_code'])['noon_ride'].transform('mean') 



# (3) station_code 기준

# train['avg_dawn_ride_station_code'] = train.groupby(['date','station_code'])['dawn_ride'].transform('mean') 

# train['avg_morning_ride_bus_station_code'] = train.groupby(['date','station_code'])['morning_ride'].transform('mean') 

# train['avg_noon_ride_station_code'] = train.groupby(['date','station_code'])['noon_ride'].transform('mean') 



# test['avg_dawn_ride_station_code'] = test.groupby(['date','station_code'])['dawn_ride'].transform('mean') 

# test['avg_morning_ride_bus_station_code'] = test.groupby(['date','station_code'])['morning_ride'].transform('mean') 

# test['avg_noon_ride_station_code'] = test.groupby(['date','station_code'])['noon_ride'].transform('mean') 



# 각 모델에 대한 oof 정의

lgbm_oof_train = np.zeros((train.shape[0]))



# Kfold 정의

kfolds = KFold(n_splits=n_splits, random_state=1993, shuffle=True)



# Fold별로 학습진행

for ind, (trn_ind, val_ind) in tqdm_notebook( enumerate( kfolds.split( X = train, y = train_label ) ) ):

    

    # Train/Valid-set을 정의하기

    X_train , y_train = train.iloc[trn_ind].drop(drop_cols, 1), train_label[trn_ind]

    X_valid , y_valid = train.iloc[val_ind].drop(drop_cols, 1), train_label[val_ind]

    

    # Target- Mean Encoding

    X_train['label'] = y_train

    d = X_train.groupby(['station_code'])['label'].mean().to_dict()

    X_train['station_code_te'] = X_train['station_code'].apply(lambda x: d.get(x))

    X_valid['station_code_te'] = X_valid['station_code'].apply(lambda x: d.get(x))

    

    X_train= X_train.drop('label',1)

    

    

    # (5) Light GBM

    print("---TRAINING LIGHT GBM---")

    # dtrain/dvalid 정의

    dtrain = lgbm.Dataset(X_train, y_train)

    dvalid = lgbm.Dataset(X_valid, y_valid)

    

    # model 정의&학습

    model = lgbm.train(lgbm_param , dtrain, NUM_BOOST_ROUND, 

                       valid_sets=(dtrain, dvalid), 

                       valid_names=('train','valid'), 

                       verbose_eval= 100)

    

    # local_valid/local_test에 대한 예측

    lgbm_valid_pred = model.predict(X_valid)

        

    lgbm_oof_train[val_ind] = lgbm_valid_pred

    print('='*80)

    

print(f"<Light-GBM> OVERALL RMSE     : {sqrt( mean_squared_error( train_label, lgbm_oof_train ) )}")
# 날씨 정보

df_weather = pd.read_csv('../input/dacon-bus-dataset/jeju_weather_dataset', encoding='cp949')

df_weather = df_weather[['일시','강수량(mm)']]

df_weather.columns = ['date','precipitation']



# date의 type을 string에서 datetime으로 변환

df_weather['date'] = pd.to_datetime( df_weather['date'] )



# 대회 기간에 해당하는 데이터만 사용하도록 함

df_weather = df_weather[(df_weather['date']>='2019-08-31 00:00:00')&(df_weather['date']<='2019-10-16 23:00:00')].reset_index(drop=True)



# 대회 규정상 해당 날짜의 15시까지 정보만 사용할 수 있음

df_weather['hour'] = df_weather['date'].dt.hour

df_weather['date'] = df_weather['date'].dt.date



# 전날의 강수량을 정보를 대입할 때 사용

df_prevday_weather = df_weather.groupby('date')['precipitation'].sum().reset_index()

df_prevday_weather.columns = ['prev_date', 'prevday_precipitation']



# 해당 날짜의 강수량을 구함

df_weather = df_weather[df_weather['hour']<=15].reset_index(drop=True)



# 00~15시까지의 강수량을 피쳐로 사용

df_weather = df_weather.groupby('date')['precipitation'].sum().reset_index()



# Train/Test-set과 join하기 위하여 column의 타입을 datetime으로 변환한다.

df_prevday_weather['prev_date'] = pd.to_datetime( df_prevday_weather['prev_date']  )

df_weather['date'] = pd.to_datetime( df_weather['date']  )
# 전날짜에 대하여 Train/Test-set과 강수량 정보를 join



# Train/Test-set에 대하여 전날을 구함

train['prev_date'] = train['date'] - pd.Timedelta('1 day')

test['prev_date'] = test['date'] - pd.Timedelta('1 day')



train = pd.merge(train, df_prevday_weather , on ='prev_date',  how ='left')

test = pd.merge(test, df_prevday_weather , on ='prev_date',how ='left')



# prev_date 칼럼은 삭제해줌

train = train.drop('prev_date',1)

test = test.drop('prev_date',1)





# 해당날짜에 대하여 Train/Test-set과 강수량 정보를 join

train = pd.merge( train, df_weather , on ='date', how='left')

test = pd.merge( test, df_weather , on ='date', how='left')
# 각 모델에 대한 oof 정의

lgbm_oof_train = np.zeros((train.shape[0]))



# Kfold 정의

kfolds = KFold(n_splits=n_splits, random_state=1993, shuffle=True)



# Fold별로 학습진행

for ind, (trn_ind, val_ind) in tqdm_notebook( enumerate( kfolds.split( X = train, y = train_label ) ) ):

    

    # Train/Valid-set을 정의하기

    X_train , y_train = train.iloc[trn_ind].drop(drop_cols, 1), train_label[trn_ind]

    X_valid , y_valid = train.iloc[val_ind].drop(drop_cols, 1), train_label[val_ind]

    

    # Target- Mean Encoding

    X_train['label'] = y_train

    d = X_train.groupby(['station_code'])['label'].mean().to_dict()

    X_train['station_code_te'] = X_train['station_code'].apply(lambda x: d.get(x))

    X_valid['station_code_te'] = X_valid['station_code'].apply(lambda x: d.get(x))

    

    X_train= X_train.drop('label',1)

    

    

    # (5) Light GBM

    print("---TRAINING LIGHT GBM---")

    # dtrain/dvalid 정의

    dtrain = lgbm.Dataset(X_train, y_train)

    dvalid = lgbm.Dataset(X_valid, y_valid)

    

    # model 정의&학습

    model = lgbm.train(lgbm_param , dtrain, NUM_BOOST_ROUND, 

                       valid_sets=(dtrain, dvalid), 

                       valid_names=('train','valid'), 

                       verbose_eval= 100)

    

    # local_valid/local_test에 대한 예측

    lgbm_valid_pred = model.predict(X_valid)

        

    lgbm_oof_train[val_ind] = lgbm_valid_pred

    print('='*80)

    

print(f"<Light-GBM> OVERALL RMSE     : {sqrt( mean_squared_error( train_label, lgbm_oof_train ) )}")
print(f"<Light-GBM> OVERALL RMSE     : {sqrt( mean_squared_error( train_label, lgbm_oof_train ) )}")
train.head()
# 해당 딕셔너리에 bus_route_id 별 정차 순서를 구하도록 함

bus_route_sequence = {}



# 모든 bus_route_id 수집

combined = train.append(test, ignore_index=True)

all_bus_route_ids = set(combined['bus_route_id'])



for bus_route_id in tqdm_notebook( all_bus_route_ids ) :

    # bus_route_id별 station_code를 오름차순으로 순서매김함

    df_bus_route = combined[combined['bus_route_id']==bus_route_id]

    sorted_station_codes = np.unique(df_bus_route['station_code'])

    

    # dictionary에 해당 정류장이 몇번째 정차 정류장인지 기입

    bus_route_sequence[bus_route_id] = {station_code: ind for ind, station_code in enumerate( list(sorted_station_codes) )}
# 몇 번째 정류장인지를 피쳐로 생성

train['nth_station']= train[['bus_route_id','station_code']].apply(lambda x: bus_route_sequence.get(x[0]).get(x[1]), axis=1)

test['nth_station'] = test[['bus_route_id','station_code']].apply(lambda x: bus_route_sequence.get(x[0]).get(x[1]), axis=1)
# 해당 bus_route_id에는 몇 개의 정류장이 있는지

bus_route_id_total_station_count_dict = combined.groupby('bus_route_id')['station_code'].nunique().to_dict()



train['bus_route_id_total_staion_count'] = train['bus_route_id'].apply(lambda x: bus_route_id_total_station_count_dict.get(x) )

test['bus_route_id_total_staion_count']  = test['bus_route_id'].apply(lambda x: bus_route_id_total_station_count_dict.get(x) )
# 뒤에서부터 몇 번째 정류정인지

train['nth_station_backward']= train['nth_station'] - train['bus_route_id_total_staion_count']

test['nth_station_backward'] = test['nth_station'] - test['bus_route_id_total_staion_count']
# 각 모델에 대한 oof 정의

lgbm_oof_train = np.zeros((train.shape[0]))



# Kfold 정의

kfolds = KFold(n_splits=n_splits, random_state=1993, shuffle=True)



# Fold별로 학습진행

for ind, (trn_ind, val_ind) in tqdm_notebook( enumerate( kfolds.split( X = train, y = train_label ) ) ):

    

    # Train/Valid-set을 정의하기

    X_train , y_train = train.iloc[trn_ind].drop(drop_cols, 1), train_label[trn_ind]

    X_valid , y_valid = train.iloc[val_ind].drop(drop_cols, 1), train_label[val_ind]

    

    # Target- Mean Encoding

    X_train['label'] = y_train

    d = X_train.groupby(['station_code'])['label'].mean().to_dict()

    X_train['station_code_te'] = X_train['station_code'].apply(lambda x: d.get(x))

    X_valid['station_code_te'] = X_valid['station_code'].apply(lambda x: d.get(x))

    

    X_train= X_train.drop('label',1)

    

    

    # (5) Light GBM

    print("---TRAINING LIGHT GBM---")

    # dtrain/dvalid 정의

    dtrain = lgbm.Dataset(X_train, y_train)

    dvalid = lgbm.Dataset(X_valid, y_valid)

    

    # model 정의&학습

    model = lgbm.train(lgbm_param , dtrain, NUM_BOOST_ROUND, 

                       valid_sets=(dtrain, dvalid), 

                       valid_names=('train','valid'), 

                       categorical_feature= ['bus_route_id','station_code'],

                       verbose_eval= 100)

    

    # local_valid/local_test에 대한 예측

    lgbm_valid_pred = model.predict(X_valid)

        

    lgbm_oof_train[val_ind] = lgbm_valid_pred

    print('='*80)

    

print(f"<Light-GBM> OVERALL RMSE     : {sqrt( mean_squared_error( train_label, lgbm_oof_train ) )}")
# 중복되지 않는 위경도 값들을 수집함

combined = train[['latitude','longitude']].append(test[['latitude','longitude']])

combined = combined.drop_duplicates()



# kmeans를 통하여 군집화

kmeans = KMeans(n_clusters= int(sqrt(len(combined)) ), random_state=1993)

kmeans.fit( combined )



train['station_code_kmeans'] = kmeans.predict(train[['latitude','longitude']])

test['station_code_kmeans']  = kmeans.predict(test[['latitude','longitude']])
# 각 모델에 대한 oof 정의

lgbm_oof_train = np.zeros((train.shape[0]))



# Kfold 정의

kfolds = KFold(n_splits=n_splits, random_state=1993, shuffle=True)



# Fold별로 학습진행

for ind, (trn_ind, val_ind) in tqdm_notebook( enumerate( kfolds.split( X = train, y = train_label ) ) ):

    

    # Train/Valid-set을 정의하기

    X_train , y_train = train.iloc[trn_ind].drop(drop_cols, 1), train_label[trn_ind]

    X_valid , y_valid = train.iloc[val_ind].drop(drop_cols, 1), train_label[val_ind]

    

    # Target- Mean Encoding

    X_train['label'] = y_train

    d = X_train.groupby(['station_code'])['label'].mean().to_dict()

    X_train['station_code_te'] = X_train['station_code'].apply(lambda x: d.get(x))

    X_valid['station_code_te'] = X_valid['station_code'].apply(lambda x: d.get(x))

    

    X_train= X_train.drop('label',1)

    

    

    # (5) Light GBM

    print("---TRAINING LIGHT GBM---")

    # dtrain/dvalid 정의

    dtrain = lgbm.Dataset(X_train, y_train)

    dvalid = lgbm.Dataset(X_valid, y_valid)

    

    # model 정의&학습

    model = lgbm.train(lgbm_param , dtrain, NUM_BOOST_ROUND, 

                       valid_sets=(dtrain, dvalid), 

                       valid_names=('train','valid'), 

                       categorical_feature= ['bus_route_id','station_code', 'station_code_kmeans'],

                       verbose_eval= 100)

    

    # local_valid/local_test에 대한 예측

    lgbm_valid_pred = model.predict(X_valid)

        

    lgbm_oof_train[val_ind] = lgbm_valid_pred

    print('='*80)

    

print(f"<Light-GBM> OVERALL RMSE     : {sqrt( mean_squared_error( train_label, lgbm_oof_train ) )}")
lgbm_param = {'objective': 'rmse',

             'boosting_type': 'gbdt',

             'random_state': 1993,

             'learning_rate': 0.1,

             'subsample': 0.7,

             'tree_learner': 'serial',

             'colsample_bytree': 0.78,

#              'early_stopping_rounds': 50,

             'subsample_freq': 1,

             'reg_lambda': 7,

             'reg_alpha': 5,

             'num_leaves': 96,

             'seed': 1993}
reg_model = lgbm.LGBMRegressor(**lgbm_param)

rfe = RFECV(estimator=reg_model, step=1, cv=KFold(n_splits=5, shuffle=False, random_state=231), scoring='neg_mean_squared_error', verbose=2)

rfe.fit(train.drop(drop_cols,1), train_label)
df_rank = pd.DataFrame(data = {'col': list(train.drop(drop_cols,1)) , 'imp': rfe.ranking_})

use_cols = list(df_rank[df_rank['imp']==1]['col'])
lgbm_param = {'objective': 'rmse',

             'boosting_type': 'gbdt',

             'random_state': 1993,

             'learning_rate': 0.1,

             'subsample': 0.7,

             'tree_learner': 'serial',

             'colsample_bytree': 0.78,

             'early_stopping_rounds': 50,

             'subsample_freq': 1,

             'reg_lambda': 7,

             'reg_alpha': 5,

             'num_leaves': 96,

             'seed': 1993}
# 각 모델에 대한 oof 정의

lgbm_oof_train = np.zeros((train.shape[0]))



# Kfold 정의

kfolds = KFold(n_splits=n_splits, random_state=1993, shuffle=True)



# Fold별로 학습진행

for ind, (trn_ind, val_ind) in tqdm_notebook( enumerate( kfolds.split( X = train, y = train_label ) ) ):

    

    # Train/Valid-set을 정의하기

    X_train , y_train = train.iloc[trn_ind].drop(drop_cols, 1), train_label[trn_ind]

    X_valid , y_valid = train.iloc[val_ind].drop(drop_cols, 1), train_label[val_ind]

    

    # Target- Mean Encoding

    X_train['label'] = y_train

    d = X_train.groupby(['station_code'])['label'].mean().to_dict()

    X_train['station_code_te'] = X_train['station_code'].apply(lambda x: d.get(x))

    X_valid['station_code_te'] = X_valid['station_code'].apply(lambda x: d.get(x))

    

    X_train= X_train.drop('label',1)

    

    

    # (5) Light GBM

    print("---TRAINING LIGHT GBM---")

    # dtrain/dvalid 정의

    dtrain = lgbm.Dataset(X_train[use_cols], y_train)

    dvalid = lgbm.Dataset(X_valid[use_cols], y_valid)

    

    # model 정의&학습

    model = lgbm.train(lgbm_param , dtrain, NUM_BOOST_ROUND, 

                       valid_sets=(dtrain, dvalid), 

                       valid_names=('train','valid'), 

                       categorical_feature= ['bus_route_id','station_code', 'station_code_kmeans'],

                       verbose_eval= 100)

    

    # local_valid/local_test에 대한 예측

    lgbm_valid_pred = model.predict(X_valid[use_cols])

        

    lgbm_oof_train[val_ind] = lgbm_valid_pred

    print('='*80)

    

print(f"<Light-GBM> OVERALL RMSE     : {sqrt( mean_squared_error( train_label, lgbm_oof_train ) )}")
# 모델에 쓰일 parameter 정의하기

n_splits= 5

NUM_BOOST_ROUND = 100000

SEED = 1993

lgbm_param = {'objective':'rmse',

              'boosting_type': 'gbdt',

              'random_state':1993,

              'learning_rate':0.01,

              'subsample':0.7,

              'tree_learner': 'serial',

              'colsample_bytree':0.68,

              'early_stopping_rounds':50,

              'subsample_freq': 1,

              'reg_lambda':7,

              'reg_alpha': 5,

              'num_leaves': 96,

              'seed' : SEED

            }



n_rounds = 100000

cat_params = {

        'n_estimators': n_rounds,

        'learning_rate': 0.08,

        'eval_metric': 'RMSE',

        'loss_function': 'RMSE',

        'random_seed': 42,

        'metric_period': 500,

        'od_wait': 500,

        'task_type': 'GPU',

       'l2_leaf_reg' : 3,

        'depth': 8,

    }
target_col = '18~20_ride'

drop_cols = ['date','id',target_col]

train_label = train[target_col]
# 형식을 맞춰주기 위해서 Test-set에 '18~20_ride' columns을 만들어줌

test[target_col] = np.NaN
# 각 모델에 대한 oof 정의

lgbm_oof_train = np.zeros((train.shape[0]))

lgbm_oof_test = np.zeros((test.shape[0]))



# Kfold 정의

kfolds = KFold(n_splits=n_splits, random_state=1993, shuffle=True)





# Fold별로 학습진행

for ind, (trn_ind, val_ind) in tqdm_notebook( enumerate( kfolds.split( X = train, y = train_label ) ) ):

    

    # Train/Valid-set을 정의하기

    X_train , y_train = train.iloc[trn_ind].drop(drop_cols, 1), train_label[trn_ind]

    X_valid , y_valid = train.iloc[val_ind].drop(drop_cols, 1), train_label[val_ind]

    

    # Target- Mean Encoding

    X_train['label'] = y_train

    d = X_train.groupby(['station_code'])['label'].mean().to_dict()

    X_train['station_code_te'] = X_train['station_code'].apply(lambda x: d.get(x))

    X_valid['station_code_te'] = X_valid['station_code'].apply(lambda x: d.get(x))

    test['station_code_te'] = test['station_code'].apply(lambda x: d.get(x))

    

    X_train= X_train.drop('label',1)

    

    

    # (5) Light GBM

    print("---TRAINING LIGHT GBM---")

    # dtrain/dvalid 정의

    dtrain = lgbm.Dataset(X_train, y_train)

    dvalid = lgbm.Dataset(X_valid, y_valid)

    

    # model 정의&학습

    model = lgbm.train(lgbm_param , dtrain, NUM_BOOST_ROUND, 

                       valid_sets=(dtrain, dvalid), 

                       valid_names=('train','valid'), 

                       categorical_feature= ['bus_route_id','station_code', 'station_code_kmeans'],

                       verbose_eval= 100)

    

    # local_valid/local_test에 대한 예측

    lgbm_valid_pred = model.predict(X_valid)

    lgbm_test_pred = model.predict(test.drop(drop_cols, 1))

        

    lgbm_oof_train[val_ind] = lgbm_valid_pred

    lgbm_oof_test += lgbm_test_pred/ n_splits

    print('='*80)

    

print(f"<Light-GBM> OVERALL RMSE     : {sqrt( mean_squared_error( train_label, lgbm_oof_train ) )}")
# 각 모델에 대한 oof 정의

cat_oof_train = np.zeros((train.shape[0]))

cat_oof_test = np.zeros((test.shape[0]))



# Kfold 정의

kfolds = KFold(n_splits=n_splits, random_state=1993, shuffle=True)





# Fold별로 학습진행

for ind, (trn_ind, val_ind) in tqdm_notebook( enumerate( kfolds.split( X = train, y = train_label ) ) ):

    

    # Train/Valid-set을 정의하기

    X_train , y_train = train.iloc[trn_ind].drop(drop_cols, 1), train_label[trn_ind]

    X_valid , y_valid = train.iloc[val_ind].drop(drop_cols, 1), train_label[val_ind]

    

    # Target- Mean Encoding

    X_train['label'] = y_train

    d = X_train.groupby(['station_code'])['label'].mean().to_dict()

    X_train['station_code_te'] = X_train['station_code'].apply(lambda x: d.get(x))

    X_valid['station_code_te'] = X_valid['station_code'].apply(lambda x: d.get(x))

    test['station_code_te'] = test['station_code'].apply(lambda x: d.get(x))

    

    X_train= X_train.drop('label',1)

    

    

    # (5) CATBOOST

    print("---TRAINING CATBOOST---")

    

    # model 정의&학습

    model = CatBoostRegressor(**cat_params)

    

    model.fit( X_train, y_train, eval_set = (X_valid, y_valid), 

              cat_features  = ['bus_route_id','station_code', 'station_code_kmeans'],

              use_best_model=True,

              verbose=True)

    

    # local_valid/local_test에 대한 예측

    cat_valid_pred = model.predict(X_valid)

    cat_test_pred = model.predict(test.drop(drop_cols, 1))

        

    cat_oof_train[val_ind] = cat_valid_pred

    cat_oof_test += cat_test_pred/ n_splits

    print('='*80)

    

print(f"<CATBOOST> OVERALL RMSE     : {sqrt( mean_squared_error( train_label, cat_oof_train ) )}")
# 제출 파일 만들기

ensemble_pred = 0.5 * ( lgbm_oof_test+ cat_oof_test )

sample_submission[target_col] = np.clip( ensemble_pred, 0 , max(ensemble_pred) )
# Train-set의 실제값과 예측값 비교

plt.figure(figsize=(12,6))



sns.distplot( train_label, color='r' , label='real')

sns.distplot( 0.5*(lgbm_oof_train + cat_oof_train), color='b', label='prediction' )

plt.legend()

plt.title("Real Vs Prediction");

# Train-set/Test-set의  예측값 비교

plt.figure(figsize=(12,6))



sns.distplot( 0.5*(lgbm_oof_train + cat_oof_train), color='r' , label='Train')

sns.distplot( ensemble_pred, color='b', label='Test' )

plt.legend()

plt.title("Prediction for Train/Test-set");

from IPython.display import FileLink



sample_submission.to_csv('lgbm_catboost_ensemble.csv', index=False)
FileLink('lgbm_catboost_ensemble.csv')