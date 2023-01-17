class MonthTimeValidation(object):

    def __init__(self, month_to_test_set=2, time_col='timestamp'):

        self.month_to_test_set = month_to_test_set

        self.time_col = time_col

        

    def split(self, df):

        split_col = df[self.time_col].dt.month

        split_col = split_col.reset_index(drop=True)

        

        for max_month in range(1,13-self.month_to_test_set):

            train_idx = split_col[split_col <= max_month].index.tolist()

            test_idx = split_col[(split_col > max_month) & (split_col <= max_month+self.month_to_test_set)].index.tolist()

            yield train_idx, test_idx

            

import numpy as np





def reduce_mem_usage(df, verbose=True):

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





from sklearn.metrics import mean_squared_error



def LRMSE(y_true, y_pred):

    return (mean_squared_error(y_true,y_pred))**(1/2)
class NaiveMeanModel(object):

    def __init__(self, values_to_count_mean, target_variable_name, value_to_fillna=0):

        self.values_to_count_mean = values_to_count_mean

        self.target_variable_name = target_variable_name

        self.value_to_fillna = value_to_fillna

        

        self.counted_stats = None 

        

    def fit(self, X, y=None):

        if len(set(self.values_to_count_mean) & set(X.columns)) < len(self.values_to_count_mean):

            raise ValueError('Columns to count stats not in df')

            

        self.counted_stats = X.groupby(self.values_to_count_mean)[self.target_variable_name].mean().reset_index()

        

    def predict(self, X):

        if self.target_variable_name in X.columns:

            prediction =  X.merge(self.counted_stats, on=self.values_to_count_mean, how='left')[self.target_variable_name+'_y']

        else:

            prediction =  X.merge(self.counted_stats, on=self.values_to_count_mean, how='left')[self.target_variable_name]

            

        print(str(prediction.isna().sum()) + ' Nan detected')

        return prediction.fillna(self.value_to_fillna)
import pandas as pd

import numpy as np

import gc



from os import path
data_path = '/kaggle/input/ashrae-energy-prediction/'
train = reduce_mem_usage(pd.read_csv(path.join(data_path,'train.csv')))

test = reduce_mem_usage(pd.read_csv(path.join(data_path,'test.csv')))



building_metadata = reduce_mem_usage(pd.read_csv(path.join(data_path,'building_metadata.csv')))
def prepare_data(df, build_metadata, is_test=False):

    df['timestamp'] = pd.to_datetime(df['timestamp'])

    df = df.sort_values('timestamp')

    

    df = df.merge(build_metadata, on='building_id', how='left')

    

    df['day_of_week'] = df['timestamp'].dt.weekday

    df['hour'] = df['timestamp'].dt.hour

    

    if not is_test:

        df['meter_reading'] = np.log1p(df['meter_reading'])

    

    return df
train = prepare_data(train, building_metadata)

test = prepare_data(test, building_metadata, is_test=True)
del building_metadata

gc.collect()
def time_val(data, model, metric=LRMSE, target_var_name='meter_reading'):

    time_validation_split = MonthTimeValidation()

    

    results = []

    for train_idx, test_idx in time_validation_split.split(data):

        model.fit(data.iloc[train_idx])

        pred = model.predict(data.iloc[test_idx])

        itter_metric = metric(data.iloc[test_idx][target_var_name], pred)

        

        print('Itter metric: '+str(itter_metric))

        results.append(itter_metric)

        

        gc.collect()

        

    return results
naive_mean_model = NaiveMeanModel(['building_id','meter','day_of_week','hour'],'meter_reading')
naive_mean_model_results = time_val(train, naive_mean_model)

print('Result: {} +/- {}'.format(round(np.mean(naive_mean_model_results),5), round(np.std(naive_mean_model_results),5)))
naive_mean_model.fit(train)

test['meter_reading'] = naive_mean_model.predict(test)
test['meter_reading'] = np.expm1(test['meter_reading'])
test.head()
test[['row_id','meter_reading']].to_csv('naive_mean_predictor.csv', index=False)