import numpy as np

import pandas as pd

import lightgbm as lgb



from sklearn import  model_selection, linear_model

from sklearn.metrics import mean_squared_log_error, recall_score

from sklearn.model_selection import cross_validate

from sklearn.dummy import DummyRegressor

from sklearn.preprocessing import OneHotEncoder

from sklearn.neighbors import KNeighborsRegressor

from matplotlib import pyplot



import os

import matplotlib.pyplot as plt
data_dir = '../input/'
def rmsle(est, x, y_test):

    """

    Метрика rmsle



    :est: обученный экземпляр модели

    :x: объекты на которых нужно предсказать значение

    :y_test: фактическое значение объектов

    :returns: значение метрики rmsle на выборке x

    """



    predict = est.predict(x)

    predict = [x if x > 0 else 0 for x in predict]

    return np.sqrt(mean_squared_log_error(y_test, predict ))





def regr_score(x_train, y_train, regr, scoring):

    """

    Расчет кроссвалидации и вывод на экран



    :x_train: обучающая выборка

    :y_train: целевое значение

    :regr: экземпляр модели

    :scoring: метрика

    """

    scores = cross_validate(regr, 

                            x_train, 

                            y_train, 

                            scoring=scoring,

                            cv=5, 

                            return_train_score=False)

    

    scores_list = scores[list(scores.keys())[-1]]

    print(scores_list)

    print(f'mean score -- {np.mean(scores_list)}')

    

    

def get_data():

    df_x = pd.read_csv(f'{data_dir}/train.csv')

    df_x = df_x.fillna(-1)

        

    y = df_x['label']

    df_x = df_x.drop(['label', 'status', 'short', 'activity_title', 'title_activity_type',

                      'activity_description', 'title_direction', 'comment_direction'], 

                     axis=1)

    return df_x, y

df_x, y = get_data()

print(df_x.shape)

print(len(y))

print(df_x.columns)
x_train = np.array(df_x)

y_train = np.array(y)
# Посмотрим, какие есть файлы и положим их в словарь

dfs = {}

for i, x in enumerate(os.listdir(data_dir)):

    file_name = x.split('.')[0]

    print(f'{i} -- {file_name}')

    dfs[file_name] = pd.read_csv(f'{data_dir}/{x}')
# Наивная модель, где предсказанием является среднее значение, полученное на обучающей выборке

regr = DummyRegressor(strategy='mean')

regr_score(x_train, y_train, regr, rmsle)

print()



# Наивная модель, где предсказанием является медиана, полученная на обучающей выборке

# Эта статистика менее подвержена выбросам, поэтому, возможно, даст лучшее качество

regr = DummyRegressor(strategy='median')

regr_score(x_train, y_train, regr, rmsle)

print()



# Градиентный бустинг от Микрософта

regr = lgb.LGBMRegressor()

regr_score(x_train, y_train, regr, rmsle)

print()



# К средних соседей

regr = KNeighborsRegressor()

regr_score(x_train, y_train, regr, rmsle)

regr = lgb.LGBMRegressor(n_estimators=68)

regr_score(x_train, y_train, regr, rmsle)

print()



regr = KNeighborsRegressor(n_neighbors=4, weights='distance',  p=1)

regr_score(x_train, y_train, regr, rmsle)

pd.Series(sorted(y_train)).plot()
pd.Series(sorted([x for x in y_train if x < np.quantile(y_train, 0.95)])).plot()
# Формирование новой выборки без выбросов

x_train = np.array(df_x)

y_train = np.array(y)



x_train = x_train[[True if x < np.quantile(y_train, 0.95) else False for x in y_train]]

y_train = [x for x in y_train if x < np.quantile(y_train, 0.95)]

# Наивная модель где предсказанием является среднее значение полученное на обучающей выборке

regr = DummyRegressor(strategy='mean')

regr_score(x_train, y_train, regr, rmsle)

print()



# Наивная модель где предсказанием является медиана полученная на обучающей выборке

# Эта статистика менее подвержена выбросам, поэтому возможно даст лучшее качество

regr = DummyRegressor(strategy='median')

regr_score(x_train, y_train, regr, rmsle)

print()



# Градиентный бустинг от Микрософта

regr = lgb.LGBMRegressor()

regr_score(x_train, y_train, regr, rmsle)

print()



# К средних соседей

regr = KNeighborsRegressor()

regr_score(x_train, y_train, regr, rmsle)

regr = lgb.LGBMRegressor(n_estimators=66, num_leaves=38)

regr_score(x_train, y_train, regr, rmsle)

print()



regr = KNeighborsRegressor(n_neighbors=4, weights='distance',  p=1)

regr_score(x_train, y_train, regr, rmsle)

x_train = np.array(df_x)

y_train = np.array(y)



print(x_train.shape)



enc = OneHotEncoder(handle_unknown='ignore')

enc.fit(x_train)

x_train = enc.transform(x_train).toarray()



print(x_train.shape)
regr = lgb.LGBMRegressor()

regr_score(x_train, y_train, regr, rmsle)
x_train = np.array(df_x)

y_train = np.array(y)



# тут происходит мерж исходных данных и дополнительных, которые мы считали в словарь dfs



df_tmp = pd.merge(df_x, dfs['activity_author'].groupby('activity_id').count().reset_index(),  

                  how='left', 

                  left_on='activity_id', right_on='activity_id', 

                  suffixes=('_x', 'activity_author'))



df_tmp = pd.merge(df_tmp, dfs['event'].groupby('run_id').count().reset_index(),  

                  how='left', 

                  left_on='run_id', right_on='run_id', 

                  suffixes=('_x', '_event'))



df_tmp = pd.merge(df_tmp, dfs['user_role'].drop_duplicates('user_id'),

                  how='left', 

                  left_on='user_id', right_on='user_id', 

                  suffixes=('_x', '_event'))



regr = lgb.LGBMRegressor()

regr_score(df_tmp, y_train, regr, rmsle)

print()





regr = KNeighborsRegressor(n_neighbors=4, weights='distance',  p=1)

regr_score(x_train, y_train, regr, rmsle)



print(df_tmp.shape)

print(len(y))



df_tmp = df_tmp.fillna(-1)

x_train = np.array(df_tmp)

y_train = np.array(y)



x_train = x_train[[True if x < np.quantile(y_train, 0.95) else False for x in y_train]]

y_train = [x for x in y_train if x < np.quantile(y_train, 0.95)]



print(x_train.shape)

print(len(y_train))
regr = lgb.LGBMRegressor(n_estimators=60, num_leaves=39)

regr_score(x_train, y_train, regr, rmsle)

print()





regr = KNeighborsRegressor(n_neighbors=4, weights='distance',  p=1)

regr_score(x_train, y_train, regr, rmsle)



regr = lgb.LGBMRegressor()

regr.fit(x_train, y_train)

pyplot.figure(figsize=(20,10))

pyplot.bar(df_tmp.columns, regr.feature_importances_)

pyplot.show()
enc = OneHotEncoder(handle_unknown='ignore')

enc.fit(x_train)

x_train = enc.transform(x_train).toarray()
x_train.shape
regr = lgb.LGBMRegressor()

regr_score(x_train, y_train, regr, rmsle)

regr = lgb.LGBMRegressor()

regr.fit(x_train, y_train)

pyplot.figure(figsize=(20,10))

pyplot.bar(range(len(regr.feature_importances_)), regr.feature_importances_)

pyplot.show()
df_tmp = pd.merge(df_x, dfs['activity_author'].groupby('activity_id').count().reset_index(),  

                  how='left', 

                  left_on='activity_id', right_on='activity_id', 

                  suffixes=('_x', 'activity_author'))



df_tmp = pd.merge(df_tmp, dfs['event'].groupby('run_id').count().reset_index(),  

                  how='left', 

                  left_on='run_id', right_on='run_id', 

                  suffixes=('_x', '_event'))



df_tmp = pd.merge(df_tmp, dfs['user_role'].drop_duplicates('user_id'),

                  how='left', 

                  left_on='user_id', right_on='user_id', 

                  suffixes=('_x', '_event'))
df_x_test = pd.read_csv(f'{data_dir}/test.csv')

df_x_test = df_x_test.fillna(-1)

df_x_test = df_x_test.drop(['short', 'activity_title', 'title_activity_type',

            'activity_description', 'title_direction', 'comment_direction'], axis=1)

df_test = pd.merge(df_x_test, dfs['activity_author'].groupby('activity_id').count().reset_index(),  

                  how='left', 

                  left_on='activity_id', right_on='activity_id', 

                  suffixes=('_x', 'activity_author'))



df_test = pd.merge(df_test, dfs['event'].groupby('run_id').count().reset_index(),  

                  how='left', 

                  left_on='run_id', right_on='run_id', 

                  suffixes=('_x', '_event'))



df_test = pd.merge(df_test, dfs['user_role'].drop_duplicates('user_id'),

                  how='left', 

                  left_on='user_id', right_on='user_id', 

                  suffixes=('_x', '_event'))



print(df_tmp.columns.values)

print(df_test.columns.values)

df_tmp = df_tmp.fillna(-1)

df_test = df_test.fillna(-1)





x_train = np.array(df_tmp)

x_test = np.array(df_test)

y_train = np.array(y)

df_tmp = df_tmp.fillna(-1)

x_train = np.array(df_tmp)



y_train = np.array(y)



x_train = x_train[[True if x < np.quantile(y_train, 0.95) else False for x in y_train]]

y_train = [x for x in y_train if x < np.quantile(y_train, 0.95)]



df_test = df_test.fillna(-1)

x_test = np.array(df_test)

regr = lgb.LGBMRegressor(n_estimators=68)

regr.fit(x_train, y_train)
%%time

test_pred = regr.predict(x_test)
submit = pd.concat([df_test['id_bet'], pd.Series(test_pred)], axis=1)

submit.columns=['id_bet', 'label']

submit.to_csv('submit_baseline.csv', index=False)