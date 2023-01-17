import pandas as pd

import numpy as np

from sklearn.preprocessing import LabelEncoder

from catboost import CatBoostRegressor

from sklearn.model_selection import train_test_split



import geopy as geo

import geopy.distance as dist



from tqdm import tqdm_notebook 

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline
air_reserve = pd.read_csv('../input/air_reserve.csv')

air_store_info = pd.read_csv('../input/air_store_info.csv')

air_visit_data = pd.read_csv('../input/air_visit_data.csv')

data_info = pd.read_csv('../input/date_info.csv')

hpg_reserve = pd.read_csv('../input/hpg_reserve.csv')

hpg_store_info = pd.read_csv('../input/hpg_store_info.csv')

store_id_relation = pd.read_csv('../input/store_id_relation.csv')
air_reserve.head()
air_store_info.head()
air_visit_data.head()
#Приведем данные теста к нормальному виду

test = pd.read_csv('../input/sample_submission.csv')

test['air_store_id'] = test.id.str.split('_').str.get(0) + '_' + test.id.str.split('_').str.get(1)

test['visit_date'] = test.id.str.split('_').str.get(2)

test['vis'] = test['visitors']

test.drop(['id', 'visitors'],axis=1, inplace=True)

test['visitors'] = test['vis']

test.drop(['vis'],axis=1, inplace=True)

test.head()
#Добавим данные по магазинам 

train = pd.merge(air_visit_data,air_store_info, on='air_store_id')

test = pd.merge(test, air_store_info, on='air_store_id')
train['id'] = train['air_store_id'] +'_'+train['visit_date']

test['id'] = test['air_store_id'] +'_'+test['visit_date']

air_reserve['visit_date'] =  air_reserve.visit_datetime.str.split(' ').str.get(0)

air_reserve['id'] = air_reserve['air_store_id'] + '_' + air_reserve['visit_date']

reserv = air_reserve.pivot_table(['reserve_visitors'],['id'], aggfunc='sum', fill_value = 0)

reserv['id'] = reserv.index

reserv = reserv[['id','reserve_visitors']]

air_reserve['visit_datetime'] = pd.to_datetime(air_reserve['visit_datetime'])

air_reserve['reserve_datetime'] = pd.to_datetime(air_reserve['reserve_datetime'])

air_reserve['period'] = air_reserve['visit_datetime']-air_reserve['reserve_datetime']

air_reserve['period'] = air_reserve['period'] / np.timedelta64(1, 'h')

reserv['period'] = air_reserve.pivot_table(['period'],['id'], aggfunc='mean', fill_value = 0)['period']

reserv.shape
train = train.merge(reserv, 'left', on='id')

test = test.merge(reserv, 'left', on='id')

train.shape, test.shape
train['day'] = train['visit_date'].str.split('-').str.get(2)

train['month'] = train['visit_date'].str.split('-').str.get(1) 



test['day'] = test['visit_date'].str.split('-').str.get(2)

test['month'] = test['visit_date'].str.split('-').str.get(1) 

train.head()
test.head()
hpg_byair_reserve = hpg_reserve.merge(store_id_relation, 'left', on='hpg_store_id')

hpg_air_reserve = hpg_byair_reserve.dropna()

hpg_air_reserve.head()
hpg_air_reserve['visit_date'] = hpg_air_reserve['visit_datetime'].str.split(' ').str.get(0)

hpg_air_reserve['id'] = hpg_air_reserve['air_store_id'] + '_' + hpg_air_reserve['visit_date']

hpg_reserv = hpg_air_reserve.pivot_table(['reserve_visitors'],['id'], aggfunc='sum', fill_value = 0)

hpg_reserv['id'] = hpg_reserv.index

hpg_reserv = hpg_reserv[['id','reserve_visitors']]
hpg_air_reserve['visit_datetime'] = pd.to_datetime(hpg_air_reserve['visit_datetime'])

hpg_air_reserve['reserve_datetime'] = pd.to_datetime(hpg_air_reserve['reserve_datetime'])

hpg_air_reserve['period'] = hpg_air_reserve['visit_datetime']-hpg_air_reserve['reserve_datetime']

hpg_air_reserve['period'] = hpg_air_reserve['period'] / np.timedelta64(1, 'h')

hpg_reserv['period'] = hpg_air_reserve.pivot_table(['period'],['id'], aggfunc='mean', fill_value = 0)['period']

hpg_reserv.head()
hpg_reserv['hpg_reserve_visitors'] = hpg_reserv['reserve_visitors']

hpg_reserv['hpg_period'] = hpg_reserv['period']



hpg_reserv.drop(['reserve_visitors', 'period'], axis=1, inplace=True)
train = train.merge(hpg_reserv, 'left', on='id')

test = test.merge(hpg_reserv, 'left', on='id')
data_info['visit_date'] = data_info['calendar_date']

data_info_norm = data_info.drop('calendar_date', axis=1)
train = train.merge(data_info_norm, 'left', on='visit_date')

test = test.merge(data_info_norm, 'left', on='visit_date')
train.shape, test.shape
train['day'] = pd.to_numeric(train['day'])

train['month'] = pd.to_numeric(train['month'])

test['day'] = pd.to_numeric(test['day'])

test['month'] = pd.to_numeric(test['month'])









train['day_of_year'] = train['day'] + train['month']*100

test['day_of_year'] = test['day'] + test['month']*100

train.dtypes





train.drop(['visit_date', 'id'], axis=1, inplace=True)

test.drop(['visit_date', 'id'], axis=1, inplace=True)
tokio = [35.6895, 139.69171] 

dist_tokio = []



for index, row in tqdm_notebook(train.iterrows()):

    dist_tokio = np.append(dist_tokio, 

                              dist.great_circle(

                                  (row['latitude'], row['longitude']),

                                  (tokio[0], tokio[1])).km)



train['dist_tokio']=dist_tokio



dist_tokio = []



for index, row in tqdm_notebook(test.iterrows()):

    dist_tokio = np.append(dist_tokio, 

                              dist.great_circle(

                                  (row['latitude'], row['longitude']),

                                  (tokio[0], tokio[1])).km)



test['dist_tokio']=dist_tokio

train['dist_day'] = train['dist_tokio']/train['day_of_year']

test['dist_day'] = test['dist_tokio']/test['day_of_year']







train['day_dist'] = train['day_of_year']/train['dist_tokio']

test['day_dist'] = test['day_of_year']/test['dist_tokio']






le = LabelEncoder()

train['air_store_id'] = le.fit_transform(train.air_store_id)

test['air_store_id'] = le.transform(test.air_store_id)



train['day_of_week'] = le.fit_transform(train.day_of_week)

test['day_of_week'] = le.transform(test.day_of_week)



train['air_genre_name'] = le.fit_transform(train.air_genre_name)

test['air_genre_name'] = le.transform(test.air_genre_name)



train['air_area_name'] = le.fit_transform(train.air_area_name)

test['air_area_name'] = le.transform(test.air_area_name)



y = np.log(train['visitors']+1)

X = train.drop(['visitors'], axis=1)

X.fillna(0, inplace=True)

X.head()
categorical_features_indices = np.where(X.dtypes != np.float)[0]
X_tr, X_val, y_tr, y_val = train_test_split(X, y, train_size=0.80, random_state=11568)
model = CatBoostRegressor(loss_function='RMSE', depth=3, learning_rate=0.4, iterations=1000, 

    random_seed=18, 

    od_type='Iter',

    od_wait=20,

)



model.fit(

    X_tr, y_tr, use_best_model=True,

    cat_features=categorical_features_indices,

    eval_set=(X_val, y_val),

    verbose=False,  

    plot=True,

)
X_test = test.drop(['visitors'], axis=1)

X_test.fillna(0, inplace=True)

X_test.head()

pred = model.predict(X_test)

pred = np.exp(pred)-1

submission = pd.read_csv('../input/sample_submission.csv')

submission.visitors = pred

submission['visitors'] = submission['visitors'].apply(lambda x: 0 if x < 0 else x) 



submission.to_csv('cat_pred.csv', index=False)
pd.DataFrame(model.feature_importances_, index=X.columns, )