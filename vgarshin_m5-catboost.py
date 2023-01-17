import os

import gc

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from datetime import datetime, timedelta

from catboost import Pool, CatBoostRegressor

pd.set_option('display.max_columns', None)

from catboost.utils import get_gpu_device_count

from tqdm.notebook import tqdm

print('available GPU devices catboost:', get_gpu_device_count())
DATA_DIR = '/kaggle/input/m5-forecasting-accuracy'

MODEL_VER = 'v0'

BACKWARD_LAGS = 60

END_D = 1913

CUT_D = END_D - int(365 * 1.2)

END_DATE = '2016-04-24'

print(datetime.strptime(END_DATE, '%Y-%m-%d'))
CALENDAR_DTYPES = {

    'date':             'str',

    'wm_yr_wk':         'int16', 

    'weekday':          'object',

    'wday':             'int16', 

    'month':            'int16', 

    'year':             'int16', 

    'd':                'object',

    'event_name_1':     'object',

    'event_type_1':     'object',

    'event_name_2':     'object',

    'event_type_2':     'object',

    'snap_CA':          'int16', 

    'snap_TX':          'int16', 

    'snap_WI':          'int16'

}

PARSE_DATES = ['date']

SPRICES_DTYPES = {

    'store_id':    'object', 

    'item_id':     'object', 

    'wm_yr_wk':    'int16',  

    'sell_price':  'float32'

}
def get_df(is_train=True, backward_lags=None):

    strain = pd.read_csv('{}/sales_train_validation.csv'.format(DATA_DIR))

    print('read train:', strain.shape)

    cat_cols = ['id', 'item_id', 'dept_id','store_id', 'cat_id', 'state_id']

    last_day = int(strain.columns[-1].replace('d_', ''))

    print('first day is:', CUT_D)

    print('last day is:', last_day)

    if not is_train:

        for day in range(last_day + 1, last_day + 28 + 28 + 1):

            strain['d_{}'.format(day)] = np.nan

        value_vars = [col for col in strain.columns 

                      if (col.startswith('d_') and (int(col.replace('d_', '')) >= END_D - backward_lags))]

    else:

        value_vars = [col for col in strain.columns 

                      if (col.startswith('d_') and (int(col.replace('d_', '')) >= CUT_D))]

    strain = pd.melt(

        strain,

        id_vars = cat_cols,

        value_vars = value_vars,

        var_name = 'd',

        value_name = 'sales'

    )

    print('melted train:', strain.shape)

    calendar = pd.read_csv('{}/calendar.csv'.format(DATA_DIR), dtype=CALENDAR_DTYPES, parse_dates=PARSE_DATES)

    print('read calendar:', calendar.shape)

    strain = strain.merge(calendar, on='d', copy=False)

    del calendar

    gc.collect()

    print('calendar merge done')

    sprices = pd.read_csv('{}/sell_prices.csv'.format(DATA_DIR), dtype=SPRICES_DTYPES)

    print('read prices:', sprices.shape)

    strain = strain.merge(

        sprices, 

        on=['store_id', 'item_id', 'wm_yr_wk'], 

        copy=False

    )

    del sprices

    gc.collect()

    print('prices merge done')

    print('begin train date:', strain['date'].min())

    print('end train date:', strain['date'].max())

    if not is_train:

        strain = strain.loc[

            strain['date'] >= (datetime.strptime(END_DATE, '%Y-%m-%d') - timedelta(days=backward_lags))

        ]

    print('date cut train:', strain.shape)

    print('cut train date:', strain['date'].min())

    print('end train date:', strain['date'].max())

    return strain
def make_features(strain):

    print('in dataframe:', strain.shape)

    lags = [7, 28]

    windows= [7, 28]

    wnd_feats = ['id', 'item_id']

    lag_cols = ['lag_{}'.format(lag) for lag in lags ]

    for lag, lag_col in zip(lags, lag_cols):

        strain[lag_col] = strain[['id', 'sales']].groupby('id')['sales'].shift(lag)

    print('lag sales done')

    for wnd_feat in wnd_feats:

        for wnd in windows:

            for lag_col in lag_cols:

                wnd_col = '{}_{}_rmean_{}'.format(lag_col, wnd_feat, wnd)

                strain[wnd_col] = strain[[wnd_feat, lag_col]].groupby(wnd_feat)[lag_col].transform(

                    lambda x: x.rolling(wnd).mean()

                )

        print('rolling mean sales for feature done:', wnd_feat)

    date_features = {

        'week_num': 'weekofyear',

        'quarter': 'quarter',

        'mday': 'day'

    }

    for date_feat_name, date_feat_func in date_features.items():

        strain[date_feat_name] = getattr(strain['date'].dt, date_feat_func).astype('int16')

    print('date features done')

    strain['d'] = strain['d'].apply(lambda x: int(x.replace('d_', '')))  

    print('out dataframe:', strain.shape)

    return strain
%%time

strain = get_df(is_train=True, backward_lags=None)

strain = make_features(strain)
drop_cols = ['id', 'sales', 'date', 'wm_yr_wk', 'weekday']

train_cols = strain.columns[~strain.columns.isin(drop_cols)]

cat_cols = [

    'item_id', 'dept_id', 'store_id', 'cat_id', 'state_id', 

    'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2'

]

strain[cat_cols] = strain[cat_cols].fillna(0)
%%time

val_size = int(strain.shape[0] * .15)

val_idxs = np.random.choice(strain.index.values, val_size, replace=False)

train_idxs = np.setdiff1d(strain.index.values, val_idxs)

train_pool = Pool(

    strain.loc[train_idxs][train_cols], 

    strain.loc[train_idxs]['sales'],

    cat_features=cat_cols

)

val_pool = Pool(

    strain.loc[val_idxs][train_cols], 

    strain.loc[val_idxs]['sales'],

    cat_features=cat_cols

)

del strain

gc.collect()
model = CatBoostRegressor(

    iterations=1000,

    task_type='GPU',

    verbose=0,

    loss_function='RMSE',

    boosting_type='Plain', #use to overcome the “Out of memory” error when training on GPU 

    depth=8,

    #gpu_cat_features_storage='CpuPinnedMemory', #use to overcome the “Out of memory” error when training on GPU 

    #max_ctr_complexity=2 #use to overcome the “Out of memory” error when training on GPU 

)

model.fit(

    train_pool,

    eval_set = val_pool,

    plot=True   

)

del train_pool, val_pool

gc.collect()
model.save_model('model_{}.cbm'.format(MODEL_VER))
feat_importances = sorted(

    [(f, v) for f, v in zip(train_cols, model.get_feature_importance())],

    key=lambda x: x[1],

    reverse=True

)

threshold = .25

labels = [x[0] for x in feat_importances if x[1] > threshold]

values = [x[1] for x in feat_importances if x[1] > threshold]

fig, ax = plt.subplots(figsize=(8, 8))

y_pos = np.arange(len(labels))

ax.barh(y_pos, values)

ax.set_yticks(y_pos)

ax.set_yticklabels(labels)

ax.invert_yaxis()

ax.set_xlabel('Performance')

ax.set_title('feature importances')

plt.show()
%%time

spred = get_df(is_train=False, backward_lags=BACKWARD_LAGS)

for pred_day in tqdm(range(1, 28 + 28 + 1)):

    pred_date = datetime.strptime(END_DATE, '%Y-%m-%d') + timedelta(days=pred_day)

    pred_date_back = pred_date - timedelta(days=BACKWARD_LAGS + 1)

    print('-' * 70)

    print('forecast day forward:', pred_day, '| forecast date:', pred_date) 

    spred_data = spred[(spred['date'] >= pred_date_back) & (spred['date'] <= pred_date)].copy()

    spred_data = make_features(spred_data)

    spred_data = spred_data.loc[spred['date'] == pred_date, train_cols]

    spred_data[cat_cols] = spred_data[cat_cols].fillna(0)

    spred.loc[spred['date'] == pred_date, 'sales'] = model.predict(spred_data)

del spred_data

gc.collect()
spred_subm = spred.loc[spred['date'] > END_DATE, ['id', 'd', 'sales']].copy()

last_d = int(spred.loc[spred['date'] == END_DATE, 'd'].unique()[0].replace('d_', ''))

print('last d num:', last_d)

spred_subm['d'] = spred_subm['d'].apply(lambda x: 'F{}'.format(int(x.replace('d_', '')) - last_d))

spred_subm.loc[spred_subm['sales'] < 0, 'sales'] = 0
f_cols = ['F{}'.format(x) for x in range(1, 28 + 28 + 1)]

spred_subm = spred_subm.set_index(['id', 'd']).unstack()['sales'][f_cols].reset_index()

spred_subm.fillna(0, inplace=True)

spred_subm.sort_values('id', inplace=True)

spred_subm.reset_index(drop=True, inplace=True)
f_cols_val = ['F{}'.format(x) for x in range(1, 28 + 1)]

f_cols_eval = ['F{}'.format(x) for x in range(28 + 1, 28 + 28 + 1)]

spred_subm_eval = spred_subm.copy()

spred_subm.drop(columns=f_cols_eval, inplace=True)

spred_subm_eval.drop(columns=f_cols_val, inplace=True)

spred_subm_eval.columns = spred_subm.columns

spred_subm_eval['id'] = spred_subm_eval['id'].str.replace('validation', 'evaluation')

spred_subm = pd.concat([spred_subm, spred_subm_eval], axis=0, sort=False)

spred_subm.reset_index(drop=True, inplace=True)

spred_subm.to_csv('submission.csv', index=False)

print('submission saved:', spred_subm.shape)