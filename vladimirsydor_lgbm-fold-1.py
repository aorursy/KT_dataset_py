MAX_BOOST_ROUNDS = 7000

EARLY_STOPPING = 200

BATCH_SIZE = 50000

FOLD_NUMBER = 0



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



from sklearn.model_selection import StratifiedKFold



def sort_X_by_Y(x_list, y_list):

    return [x for _, x in sorted(zip(y_list,x_list), key=lambda pair: pair[0])]



class NaiveMeanModel(object):

    def __init__(self, values_to_count_mean, target_variable_name, value_to_fillna=0, out_of_fold_col_stratify='building_id'):

        self.values_to_count_mean = values_to_count_mean

        self.target_variable_name = target_variable_name

        self.value_to_fillna = value_to_fillna

        self.out_of_fold_col_stratify = out_of_fold_col_stratify

        

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

        return prediction.fillna(self.value_to_fillna).reset_index(drop=True)

    

    def out_of_fold_predict(self, X):

        kf_nmm = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        

        results_nmm = []

        indexes_nmm = []

        for train_idx_nmm, test_idx_nmm in kf_nmm.split(X, X['building_id']):

            self.fit(X.iloc[train_idx_nmm])

            results_nmm += list(self.predict(X.iloc[test_idx_nmm]))

            indexes_nmm += list(test_idx_nmm)

            

        return sort_X_by_Y(results_nmm, indexes_nmm)

import lightgbm as lgb

from tqdm import tqdm



import matplotlib.pyplot as plt

import seaborn as sns



def plotImp(model, col_names , num = 20):

    feature_imp = pd.DataFrame({'Value':model.feature_importance(),'Feature':col_names})

    plt.figure(figsize=(40, 20))

    sns.set(font_scale = 5)

    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[0:num])

    

    plt.title('LightGBM Features (avg over folds)')

    plt.show()



class MyRegressor(object):

    def __init__(self, ml_params, categoricals, cols_to_drop=[], tgt_variable='meter_reading'):

        self.ml = None

        self.ml_params = ml_params

        

        self.tgt_variable = tgt_variable

        self.categoricals = categoricals

        self.cols_to_drop = cols_to_drop

        self.predictors = None

        

    def fit(self, X, X_val=None, plot_feature_imp=True):        

        y = X[self.tgt_variable]

        X = X.drop(columns=[self.tgt_variable] + self.cols_to_drop)

        col_names = X.columns

        self.predictors = list(col_names)

        

        X = X[self.predictors].values.astype(np.float32)

        

        X = lgb.Dataset(X, label=y,feature_name=self.predictors, categorical_feature=self.categoricals)        

        if X_val is not None:

            y_val = X_val[self.tgt_variable]

            X_val = X_val.drop(columns=[self.tgt_variable] + self.cols_to_drop)

            

            X_val = X_val[self.predictors].values.astype(np.float32)

            X_val = lgb.Dataset(X_val, label=y_val, 

                                feature_name = self.predictors, categorical_feature=self.categoricals)

            

            self.ml = lgb.train(self.ml_params,

                                X,

                                num_boost_round=MAX_BOOST_ROUNDS,

                                valid_sets=(X, X_val),

                                early_stopping_rounds=EARLY_STOPPING,

                                verbose_eval = 50)

        else:

            self.ml = lgb.train(self.ml_params,

                                X,

                                valid_sets=(X),

                                num_boost_round=MAX_BOOST_ROUNDS,

                                verbose_eval = 50)

        if plot_feature_imp:

            plotImp(self.ml, col_names)

            

        return self

    

    def predict(self, X):

        cols_to_drop = list(set(['row_id', self.tgt_variable] + self.cols_to_drop) & set(X.columns))

        

        batches = int(np.ceil(X.shape[0]/BATCH_SIZE))

        

        res=[]

        for i in tqdm(range(batches)):

            res.append(self.ml.predict( X.iloc[i*BATCH_SIZE:(i+1)*BATCH_SIZE].drop(columns=cols_to_drop)[self.predictors].values.astype(np.float32) ))

            

        return np.concatenate(res)
import pandas as pd

import numpy as np

import gc



from os import path



%matplotlib inline
cat_columns = [

    "building_id", "meter", "site_id", "primary_use", "had_air_temperature", "had_cloud_coverage",

    "had_dew_temperature", "had_precip_depth_1_hr", "had_sea_level_pressure", "had_wind_direction",

    "had_wind_speed", "tm_day_of_week", "tm_hour_of_day"

]
X_train = reduce_mem_usage(pd.read_parquet('/kaggle/input/baseline-preprocessing/X_train.parquet.gzip'))

X_test = reduce_mem_usage(pd.read_parquet('/kaggle/input/baseline-preprocessing/X_test.parquet.gzip'))
print(X_train.columns)
def one_fold_predict(data, model, metric=LRMSE, target_var_name='meter_reading', test_to_predict=None):    

    print('Starting Validation')

    print('Fold {}'.format(FOLD_NUMBER))

    

    model.fit(data[data['k_folds'] != FOLD_NUMBER].reset_index(drop=True), data[data['k_folds'] == FOLD_NUMBER].reset_index(drop=True))

    pred = model.predict(data[data['k_folds'] == FOLD_NUMBER].reset_index(drop=True))

        

    if test_to_predict is not None:

        test_prediction = model.predict(test_to_predict)

            

    itter_metric = metric(data.loc[data['k_folds'] == FOLD_NUMBER, target_var_name], pred)

    print('Fold metric: '+str(itter_metric))

    

    gc.collect()

     

    if test_to_predict is not None:

        return itter_metric, test_prediction

    else:

        return itter_metric
boost_model = MyRegressor(ml_params={

            "objective": "regression",

            "boosting": "gbdt",

            "num_leaves": 82,

            "learning_rate": 0.05,

            "feature_fraction": 0.85,

            "reg_lambda": 1,

            "metric": "rmse",

            'seed':42

            }, categoricals=cat_columns, cols_to_drop=['k_folds'])
rf_res, X_test['meter_reading'] = one_fold_predict(X_train, boost_model, test_to_predict=X_test)
gc.collect()
X_test.head()
X_test[['row_id','meter_reading']].to_csv('submission.csv', index=False)