# import libraries
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import json
import gc
import os
# common function to reduce the memory usage thus allowing us to work with larger datasets
def reduce_mem_usage(df, verbose=True):
    """
    Common function to reduce the size of the entries in a pandas DataFrame.
    """
    import numpy as np
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

# simple function to read the data in the competition files
def readData(submission_only=False,PATH='/kaggle/input/'):
    import pandas as pd
    print('Reading files...')
    submission = pd.read_csv(PATH+'m5-forecasting-accuracy/sample_submission.csv')
    if submission_only:
        return submission
    else:
        calendar = pd.read_csv(PATH+'m5-forecasting-accuracy/calendar.csv')
        calendar = reduce_mem_usage(calendar)
        print('Calendar has {} rows and {} columns'.format(calendar.shape[0], calendar.shape[1]))

        sell_prices = pd.read_csv(PATH+'m5-forecasting-accuracy/sell_prices.csv')
        sell_prices = reduce_mem_usage(sell_prices)
        print('Sell prices has {} rows and {} columns'.format(sell_prices.shape[0], sell_prices.shape[1]))

        sales_train_validation = pd.read_csv(PATH+'m5-forecasting-accuracy/sales_train_validation.csv')
        print('Sales train validation has {} rows and {} columns'.format(sales_train_validation.shape[0], sales_train_validation.shape[1]))

    return calendar, sell_prices, sales_train_validation, submission


# process the data to get it into a tabular format; pd.melt is especially useful to 'unpack' the target variable (demand)
def melt_and_merge(nrows=5.5e7):
    calendar, sell_prices, sales_train_validation, submission = readData()
    # drop some calendar features
    calendar.drop(['weekday', 'wday', 'month', 'year'], inplace = True, axis = 1)
    
    # melt sales data, get it ready for training
    sales_train_validation = pd.melt(sales_train_validation, id_vars = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'], 
                                     var_name = 'day', value_name = 'demand')
    
    #print('Melted sales train validation has {} rows and {} columns'.format(sales_train_validation.shape[0], sales_train_validation.shape[1]))
    sales_train_validation = reduce_mem_usage(sales_train_validation)
    
    # seperate test dataframes
    test1_rows = [row for row in submission['id'] if 'validation' in row]
    #test2_rows = [row for row in submission['id'] if 'evaluation' in row]
    test1 = submission[submission['id'].isin(test1_rows)]
    #test2 = submission[submission['id'].isin(test2_rows)]
    
    # change column names
    test1.columns = ['id'] + ['d_{}'.format(i) for i in range(1914,1942)]
    #test2.columns = ['id'] + ['d_{}'.format(i) for i in range(1942,1970)]


    # get product table
    product = sales_train_validation[['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']].drop_duplicates()
    
    # merge with product table
    #test2['id'] = test2['id'].str.replace('_evaluation','_validation')
    test1 = test1.merge(product, how = 'left', on = 'id')
    #test2 = test2.merge(product, how = 'left', on = 'id')
    #test2['id'] = test2['id'].str.replace('_validation','_evaluation')
    
    # 
    test1 = pd.melt(test1, id_vars = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'], var_name = 'day',
                    value_name = 'demand')
    #test2 = pd.melt(test2, id_vars = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'], var_name = 'day',
    #                value_name = 'demand')
    
    sales_train_validation = pd.concat([sales_train_validation, test1], axis = 0) # include test2 later
    
    del test1#, test2
    gc.collect()
    
    # delete first entries otherwise memory errors
    sales_train_validation = sales_train_validation.loc[nrows:]
    
    # delete test2 for now
    #data = data[data['part'] != 'test2']
    
    sales_train_validation = pd.merge(sales_train_validation, calendar, how = 'left', left_on = ['day'], right_on = ['d'])
    sales_train_validation.drop(['d', 'day'], inplace = True, axis = 1)
    
    # get the sell price data (this feature should be very important)
    sales_train_validation = sales_train_validation.merge(sell_prices, on = ['store_id', 'item_id', 'wm_yr_wk'], how = 'left')
    print('Our final dataset to train has {} rows and {} columns'.format(sales_train_validation.shape[0], sales_train_validation.shape[1]))
    
    del calendar, sell_prices; gc.collect();
    
    return sales_train_validation
# this function fills up the Nan values and encodes the categorical variables
def transform(data):
    from sklearn.preprocessing import LabelEncoder
    # convert to datetime object
    data['date'] = pd.to_datetime(data.date)
    
    # fill NaN features with unknown
    nan_features = ['event_name_1', 'event_type_1', 'event_name_2', 'event_type_2']
    for feature in nan_features:
        data[feature].fillna('unknown', inplace = True)
    
    # Encode categorical features
    cat = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2']
    for feature in cat:
        encoder = LabelEncoder()
        data[feature] = encoder.fit_transform(data[feature])
    
    return data
# this function computes useful laging features from the target variable and the price
# to convert what is initially a sequence to sequence mapping into a regression task (sequence to one),
# the author used lagged values starting from the minimum lag of 28 days, which is the forecasting horizon
def simple_fe(data):
    
    # rolling demand features
    data['lag_t28'] = data.groupby(['id'])['demand'].transform(lambda x: x.shift(28))
    data['lag_t29'] = data.groupby(['id'])['demand'].transform(lambda x: x.shift(29))
    data['lag_t30'] = data.groupby(['id'])['demand'].transform(lambda x: x.shift(30))
    data['rolling_mean_t7'] = data.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(7).mean())
    data['rolling_std_t7'] = data.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(7).std())
    data['rolling_mean_t30'] = data.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(30).mean())
    data['rolling_mean_t90'] = data.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(90).mean())
    data['rolling_mean_t180'] = data.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(180).mean())
    data['rolling_std_t30'] = data.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(30).std())
    data['rolling_skew_t30'] = data.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(30).skew())
    data['rolling_kurt_t30'] = data.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(30).kurt())
    
    
    # price features
    data['lag_price_t1'] = data.groupby(['id'])['sell_price'].transform(lambda x: x.shift(1))
    data['price_change_t1'] = (data['lag_price_t1'] - data['sell_price']) / (data['lag_price_t1'])
    data['rolling_price_max_t365'] = data.groupby(['id'])['sell_price'].transform(lambda x: x.shift(1).rolling(365).max())
    data['price_change_t365'] = (data['rolling_price_max_t365'] - data['sell_price']) / (data['rolling_price_max_t365'])
    data['rolling_price_std_t7'] = data.groupby(['id'])['sell_price'].transform(lambda x: x.rolling(7).std())
    data['rolling_price_std_t30'] = data.groupby(['id'])['sell_price'].transform(lambda x: x.rolling(30).std())
    data.drop(['rolling_price_max_t365', 'lag_price_t1'], inplace = True, axis = 1)
    
    # time features
    data['date'] = pd.to_datetime(data['date'])
    data['year'] = data['date'].dt.year
    data['month'] = data['date'].dt.month
    data['week'] = data['date'].dt.week
    data['day'] = data['date'].dt.day
    data['dayofweek'] = data['date'].dt.dayofweek
    
    
    return data
new_fe = True
if new_fe:
    data = melt_and_merge(nrows=2.75e7)
    submission = readData(submission_only=True)
    data = transform(data)
    print('There are {:e} / {:e} NaN entries in the sell_price column'.format(data.sell_price.isna().sum(),data.shape[0]))
    data = simple_fe(data)
    data = reduce_mem_usage(data)
    #data.to_pickle('engineered_data.pkl')
else:
    data = pd.read_pickle('engineered_data.pkl')
    submission = readData(submission_only=True)
from hyperopt import hp, tpe
from hyperopt.fmin import fmin
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import make_scorer
import lightgbm as lgb
from sklearn.metrics import mean_squared_error

# define list of features
features = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'year', 'month', 'week', 'day', 'dayofweek', 'event_name_1',
            'event_type_1', 'event_name_2', 'event_type_2', 'snap_CA', 'snap_TX', 'snap_WI', 'sell_price', 'lag_t28', 'lag_t29',
            'lag_t30', 'rolling_mean_t7', 'rolling_std_t7', 'rolling_mean_t30', 'rolling_mean_t90', 'rolling_mean_t180', 
            'rolling_std_t30', 'price_change_t1', 'price_change_t365', 'rolling_price_std_t7', 'rolling_price_std_t30', 
            'rolling_skew_t30', 'rolling_kurt_t30']

def optimize_parameters(x_train,max_evals=20):
    # define fixed hyperparameters
    params = {
        'tree_learner':'voting',
        'boosting_type': 'gbdt',
        'objective': 'tweedie',
        'tweedie_variance_power': 1.1,
        'metric': 'rmse',
        'subsample': 0.5,
        'subsample_freq': 1,
        'sub_feature' : 0.8,
        'sub_row' : 0.75,
        'bagging_freq' : 1,
        'lambda_l2' : 0.1,
        'verbosity': 1,
        'boost_from_average': True,
        'n_jobs': -1,
        'learning_rate':0.1,
        'seed': 3008,
        'verbose': -1}
    
    # define floating hyperparameters
    space = {
        'n_estimators': hp.quniform('n_estimators', 25, 600, 25),
        'max_depth': hp.quniform('max_depth', 1, 6, 1),
        'num_leaves': hp.quniform('num_leaves', 10, 120, 1)
    }
    
    # define the objective function to optimize the hyperparameters
    def objective(floating_params):
        params['n_estimators'] = int(floating_params['n_estimators'])
        params['max_depth'] = int(floating_params['max_depth'])
        params['num_leaves'] = int(floating_params['num_leaves'])
        print(params)
        regressor = lgb.LGBMRegressor(**params)
        
        x_sample = x_train.sample(int(x_train.shape[0]/50))
        x_train_sample, y_train_sample = x_sample[features], x_sample['demand']
        
        score = cross_val_score(regressor, x_train_sample, y_train_sample, cv=StratifiedKFold(),
                                scoring=make_scorer(mean_squared_error, greater_is_better=False)
                                ).mean()
        print("rmse {:.3f} params {}".format(score, params))
        return score

    best = fmin(fn=objective,
                space=space,
                algo=tpe.suggest,
                max_evals=max_evals)
    
    best['num_iterations'] =  1500
    best['n_estimators'] = int(best['n_estimators']); best['max_depth'] = int(best['max_depth']); 
    best['num_leaves'] = int(best['num_leaves'])
    
    with open('params.txt', 'w') as file:
        file.write(json.dumps(params))
    
    return params
    
def optimized_lgb(data,update_hyperparameters=False):
    
    # going to evaluate with the last 28 days
    x_train = data[data['date'] <= '2016-03-27']
    y_train = x_train['demand']
    x_val = data[(data['date'] > '2016-03-27') & (data['date'] <= '2016-04-24')]
    y_val = x_val['demand']
    test = data[(data['date'] > '2016-04-24')]
    del data
    gc.collect()
    

    
    if (update_hyperparameters and os.path.exist('/kaggle/working/params.txt')):
        with open('params.txt') as params_file:    
            params = json.load(params_file)
    else:
        params = optimize_parameters(x_train)
        
    train_set = lgb.Dataset(x_train[features], y_train)
    val_set = lgb.Dataset(x_val[features], y_val)
    del x_train, y_train
    gc.collect()
    
    model = lgb.train(params, train_set, num_boost_round = 2500, early_stopping_rounds = 50, 
                      valid_sets = [train_set, val_set], verbose_eval = 100)
    
    val_pred = model.predict(x_val[features])
    val_score = np.sqrt(mean_squared_error(val_pred, y_val))
    print(f'Our val rmse score is {val_score}')
    y_pred = model.predict(test[features])
    test['demand'] = y_pred
    
    return test

def append_predictions(test, submission):
    predictions = test[['id', 'date', 'demand']]
    predictions = pd.pivot(predictions, index = 'id', columns = 'date', values = 'demand').reset_index()
    predictions.columns = ['id'] + ['F' + str(i + 1) for i in range(28)]

    evaluation_rows = [row for row in submission['id'] if 'evaluation' in row] 
    evaluation = submission[submission['id'].isin(evaluation_rows)]

    validation = submission[['id']].merge(predictions, on = 'id')
    final = pd.concat([validation, evaluation])
    final.to_csv('submission.csv', index = False)
# eliminate the first 90 datapoints
data = data[~data.rolling_mean_t180.isna()]
data = reduce_mem_usage(data)
test = optimized_lgb(data,update_hyperparameters=False)
append_predictions(test, submission)