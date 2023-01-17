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
from lightgbm import LGBMClassifier, Dataset

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score

from matplotlib import pyplot as plt



#функция для загрузки покупок

def load_purchases(path, cols=None, nrows=None):

    """

        path: путь до файла с покупками,

        cols: list колонок для загрузки,

        nrows: количество строк датафрейма

    """

    

    dtypes = {

        'regular_points_received': np.float16,

        'express_points_received': np.float16,

        'regular_points_spent': np.float16,

        'express_points_spent': np.float16,

        'purchase_sum': np.float32,

        'product_quantity': np.float16,

        'trn_sum_from_iss': np.float32,

        'trn_sum_from_red': np.float16,        

    }

    if cols:

        purchases = pd.read_csv(path, dtype=dtypes, parse_dates=['transaction_datetime'], nrows=nrows, usecols=cols)

    else:

        purchases = pd.read_csv(path, dtype=dtypes, parse_dates=['transaction_datetime'], nrows=nrows)

    purchases['purchase_sum'] = np.round(purchases['purchase_sum'], 2)

    

    return purchases

# загружаем данные

train_df = pd.read_csv('../input/x5-uplift-valid/data/train.csv')

test_df = pd.read_csv('../input/x5-uplift-valid/data/test.csv')



clients = pd.read_csv('../input/x5-uplift-valid/data/clients2.csv')



puchases_train = load_purchases('../input/x5-uplift-valid/train_purch/train_purch.csv')

puchases_test = load_purchases('../input/x5-uplift-valid/test_purch/test_purch.csv')



product_df = pd.read_csv('../input/x5-uplift-valid/data/products.csv')



train_df['new_target'] = 0

train_df.loc[(train_df['treatment_flg'] == 1) & (train_df['target'] == 1), 'new_target'] = 1

train_df.loc[(train_df['treatment_flg'] == 0) & (train_df['target'] == 0), 'new_target'] = 1
# медианный чек

median_chech_sum_train = puchases_train.drop_duplicates('transaction_id').groupby('client_id')['purchase_sum'].median()

median_chech_sum_test = puchases_test.drop_duplicates('transaction_id').groupby('client_id')['purchase_sum'].median()



train_df['median_sum'] = train_df['client_id'].map(median_chech_sum_train)

test_df['median_sum'] = test_df['client_id'].map(median_chech_sum_test)
train_df
# пенсионер или нет

train_df['age'] = train_df['client_id'].map(clients.set_index('client_id')['age'])

test_df['age'] = test_df['client_id'].map(clients.set_index('client_id')['age'])

train_df['gender'] = train_df['client_id'].map(clients.set_index('client_id')['gender'])

test_df['gender'] = test_df['client_id'].map(clients.set_index('client_id')['gender'])

train_df['is_retired'] = (np.array([x == 'F' for x in train_df['gender']]) * np.array([x >= 60 for x in train_df['age']]) + np.array([x == 'M' for x in train_df['gender']]) * np.array([x >= 65 for x in train_df['age']])).astype('int')

test_df['is_retired'] = (np.array([x == 'F' for x in test_df['gender']]) * np.array([x >= 60 for x in test_df['age']]) + np.array([x == 'M' for x in test_df['gender']]) * np.array([x >= 65 for x in test_df['age']])).astype('int')
train_df
# level_1

lvl1Dict = {}

for lvl in product_df['level_1']:

    if lvl not in lvl1Dict:

        lvl1Dict.update({lvl : 2 ** len(lvl1Dict)})

        

mapper = product_df.set_index('product_id')['level_1']

puchases_train['level_1'] = [lvl1Dict[x] for x in puchases_train['product_id'].map(mapper)]

puchases_test['level_1'] = [lvl1Dict[x] for x in puchases_test['product_id'].map(mapper)]



lvl1_mapper_train = puchases_train.groupby('client_id')['level_1'].agg(lambda x: np.bitwise_or.reduce(x.values))

train_df['level_1_bitmap'] = train_df['client_id'].map(lvl1_mapper_train)



lvl1_mapper_test = puchases_test.groupby('client_id')['level_1'].agg(lambda x: np.bitwise_or.reduce(x.values))

test_df['level_1_bitmap'] = test_df['client_id'].map(lvl1_mapper_test)
train_df
# level_2

lvl2Dict = {}

for lvl in product_df['level_2']:

    if lvl not in lvl2Dict:

        lvl2Dict.update({lvl : 2 ** len(lvl2Dict)})

        

mapper = product_df.set_index('product_id')['level_2']

puchases_train['level_2'] = [lvl2Dict[x] for x in puchases_train['product_id'].map(mapper)]

puchases_test['level_2'] = [lvl2Dict[x] for x in puchases_test['product_id'].map(mapper)]



lvl2_mapper_train = puchases_train.groupby('client_id')['level_2'].agg(lambda x: np.bitwise_or.reduce(x.values))

train_df['level_2_bitmap'] = train_df['client_id'].map(lvl2_mapper_train)



lvl2_mapper_test = puchases_test.groupby('client_id')['level_2'].agg(lambda x: np.bitwise_or.reduce(x.values))

test_df['level_2_bitmap'] = test_df['client_id'].map(lvl2_mapper_test)

train_df
# алкоголь собственной марки

alco_mapper = product_df.set_index('product_id')['is_alcohol']

trademark_mapper = product_df.set_index('product_id')['is_own_trademark']

puchases_train['is_alcohol'] = puchases_train['product_id'].map(alco_mapper)

puchases_test['is_alcohol'] = puchases_test['product_id'].map(alco_mapper)

puchases_train['is_own_trademark'] = puchases_train['product_id'].map(trademark_mapper)

puchases_test['is_own_trademark'] = puchases_test['product_id'].map(trademark_mapper)

puchases_train['is_own_alcohol'] = np.array(puchases_train['is_own_trademark']) * np.array(puchases_train['is_alcohol'])

puchases_test['is_own_alcohol'] = np.array(puchases_test['is_own_trademark']) * np.array(puchases_test['is_alcohol'])



alco_mapper_train = puchases_train.groupby('client_id')['is_own_alcohol'].sum()

train_df['own_alco_produts_count'] = train_df['client_id'].map(alco_mapper_train)



alco_mapper_test = puchases_test.groupby('client_id')['is_own_alcohol'].sum()

test_df['own_alco_produts_count'] = test_df['client_id'].map(alco_mapper_test)



train_df['is_own_alcohol'] = (train_df['own_alco_produts_count'] > 0).astype('int')

test_df['is_own_alcohol'] = (test_df['own_alco_produts_count'] > 0).astype('int')
train_df
# создаем трейн и валидацию

cols = ['is_retired', 'median_sum', 'is_own_alcohol', 'level_1_bitmap', 'level_2_bitmap']

x_train, x_valid, y_train, y_valid = train_test_split(train_df[cols], train_df['new_target'], test_size=.2)



# обучение модели

params = {

#     'boosting_type': 'gbdt',

    'n_estimators': 1000, # максимальное количество деревьев

    'max_depth': 2, # глубина одного дерева

    'learning_rate' : 0.1, # скорость обучения

    'num_leaves': 3, # количество листьев в дереве

    'min_data_in_leaf': 50, # минимальное количество наблюдений в листе

    'lambda_l1': 0, # параметр регуляризации

    'lambda_l2': 0, # параметр регуляризации

    

    'early_stopping_rounds': 20, # количество итераций без улучшения целевой метрики

}



lgbm = LGBMClassifier(**params)



lgbm = lgbm.fit(x_train, y_train, verbose=50, eval_set=[(x_valid, y_valid)], eval_metric='AUC')

predicts = lgbm.predict_proba(x_valid)[:, 1]



#gini

2*roc_auc_score(y_valid, predicts) - 1