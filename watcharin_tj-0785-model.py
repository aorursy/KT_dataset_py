import pprint



import datetime as dt

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import seaborn as sns



from scipy import stats

from scipy.fftpack import fft



class AnalysisUtils:

    @classmethod

    def coverage(cls, data):

        return data.apply(lambda v: pd.Series([v.dtype, 1.0 - float(v.isnull().sum()) / len(v), v.isnull().sum(), v.nunique()], 

                                              ['Type', 'Coverage', 'Total NA\'s', '#Unique']), axis=0)

    

    @classmethod

    def summary(cls, data):

        def inner(v):

            return pd.Series([v.min(), np.nanpercentile(v, 25), v.mean(), v.median(), np.nanpercentile(v, 75), v.max()], 

                             ['min', 'Q1', 'mean', 'median', 'Q3', 'max'])

        

        def inner_datetime(v):

            return pd.Series([v.min(), v.max()], ['min', 'max'])

        

        def summarize_categorical_series(series, output_na = True):

            cnt = series.value_counts()

            num_na = series.isnull().sum()

            head = cnt.head()

            others_cnt = cnt.iloc[len(head):].sum()

            

            print(head)

            

            if others_cnt > 0:

                print('Others: {}'.format(others_cnt))

                

            if num_na > 0:

                print('NA: {}'.format(num_na))

        

        NUMERIC_TYPES = ['uint8', 'int8', 'uint16', 'uint32', 'int32', 'int64', 'float64']

        

        # Take care of input of Type "Series".

        if isinstance(data, pd.core.series.Series):

            num_na = data.isnull().sum()

            

            if data.dtype in NUMERIC_TYPES:

                print(inner(data))

            elif data.dtype in ['<M8[ns]']:

                print(inner_datetime(data))

            else:

                summarize_categorical_series(data, False)

        

            print("NA: {} (from {}; {})".format(num_na, len(data), float(num_na) / len(data)))

            

            return

        

        print('Shape: {}'.format(data.shape))

        print('Coverage:')

        print(AnalysisUtils.coverage(data))

    

        # Summarize numeric columns

        numeric_idx = [x in NUMERIC_TYPES for x in data.dtypes]

        if sum(numeric_idx) > 0:

            numeric_cols = list(data.dtypes[numeric_idx].index)

            rv = data[numeric_cols].apply(inner)

            print('\n{}'.format(rv))

        

        # Summarize datetime columns

        datetime_idx = [x in ['<M8[ns]'] for x in data.dtypes]

        if sum(datetime_idx) > 0:

            datetime_cols = list(data.dtypes[datetime_idx].index)

            rv = data[datetime_cols].apply(inner_datetime)

            print('\n{}'.format(rv))

        

        for c in data.columns:

            print('\n' + c)

            summarize_categorical_series(data[c])

            

class DateService:

    

    @staticmethod

    def month_number(s, cutoff_date):

        stamp = pd.Timestamp(cutoff_date)

        

        return 12 * (stamp.year - s.dt.year) + (stamp.month - s.dt.month)

    

    @staticmethod

    def week_number(s, cutoff_date):

        stamp = pd.Timestamp(cutoff_date)

        

        return np.ceil((stamp - s).dt.days / 7).astype(int)

    

class TsOp:

    

    @staticmethod

    def argmin(row):

        return len(row) - 1 - np.argmin(row)

        

    @staticmethod

    def argmax(row):

        return len(row) - 1 - np.argmax(row)

    

    @staticmethod

    def autocorrelation(row, lag=1):

        n = len(row)



        if n < lag + 2:

            return 0



        return stats.pearsonr(row[lag:], row[:n - lag])[0]



    @staticmethod

    def delta(row, lag=1, relative=0):

        if lag < len(row):

            diff = row[-1] - row[-1 - lag]

            

            if relative == 0:

                return diff

            else:

                base = abs(row[-1] + row[-1 - lag])

                

                if base > 0.0:

                    return diff / base

                else:

                    return 2e9

                           

        return 0

        

    @staticmethod

    def fourier(row, f=0):

        n = len(row)



        if n < 3:

            return 0



        return np.real(fft(row))[f]



    @staticmethod

    def ts_rank(df):

        return df.rank(axis=1).iloc[:, -1]

    

    @staticmethod

    def ts_zscore(df):

        return (df.iloc[:, -1] - df.mean(axis=1)) / df.std(axis=1)

    

    @staticmethod

    def up_down_ratio(row):

        if len(row) == 0:

            return 0



        up = (row[1:] > row[:-1]).sum()

        down = (row[1:] < row[:-1]).sum()

        

        return up / (down + 1)

    

class TJUtils:

    @classmethod

    def cont_effect(cls, data, predictor, label, target_value = 1, bins = 10, quantile_cut=True):

        if data[predictor].nunique() < bins:

            TJUtils.discrete_effect(data, predictor, label, target_value)

            return

        

        if quantile_cut:

            g = data.groupby(pd.qcut(data[predictor], q=bins, duplicates='drop'))

        else:

            g = data.groupby(pd.cut(data[predictor], bins=bins))

            

        avg = g[label].mean()

        cnt = g[label].size()

        

        sns.pointplot(x=list(range(len(avg))), y=avg, ci=None)

        

        x = plt.gca().axes.get_xlim()

        benchmark = table(data[label])[target_value]

        plt.plot(x, len(x) * [benchmark], color='orange', linestyle='--')

        

        print(pd.DataFrame({'bin': range(len(avg)), 

                            'signal': avg, 

                            'diff': avg - benchmark, 

                            'count': cnt, 

                            'frac': cnt / cnt.sum()}, 

                           columns=['bin', 'signal', 'diff', 'count', 'frac']))

        

    @classmethod

    def discrete_effect(cls, data, predictor, label, target_value = 1, **kwargs):

        g = data.groupby(predictor)

        avg = g[label].mean()

        cnt = g[label].size()

        

        sns.barplot(x=predictor, y=label, data=data, ci=None, order=avg.index, **kwargs)

        

        benchmark = table(data[label])[target_value]

        

        # Add a benchmark line

        x = plt.gca().axes.get_xlim()

        plt.plot(x, len(x) * [benchmark], color='orange', linestyle='--')

        

        print(pd.DataFrame({'signal': avg, 'diff': avg - benchmark, 'count': cnt, 'frac': cnt / cnt.sum()}, 

                           columns=['signal', 'diff', 'count', 'frac']))



    @classmethod

    def drop_const_columns(cls, data):

        mask = data.nunique() == 1

        all_cols = data.columns.values

        to_drop = all_cols[mask]

        

        data.drop(to_drop, axis=1, inplace=True)

        

    @classmethod

    def hist(cls, series, figsize=(15, 8)):

        fig, ax = plt.subplots(figsize=figsize)

        sns.distplot(series, kde=False, ax=ax)



        

def percentile(series):

    return pd.Series([np.nanpercentile(series, x) for x in range(101)], list(range(101)))





def reduce_mem_usage(props, exclude_columns=[]):

    start_mem_usg = props.memory_usage().sum() / 1024**2 

    print("Memory usage of properties dataframe is :",start_mem_usg," MB")

    NAlist = [] # Keeps track of columns that have missing values filled in. 

    for col in props.columns:

        if props[col].dtype != object and col not in exclude_columns:  # Exclude strings

            

            # Print current column type

            print("Column: ",col)

            print("dtype before: ",props[col].dtype)

            

            # make variables for Int, max and min

            IsInt = False

            mx = props[col].max()

            mn = props[col].min()

            

            # Integer does not support NA, therefore, NA needs to be filled

            if not np.isfinite(props[col]).all(): 

                NAlist.append(col)

                props[col].fillna(mn-1,inplace=True)  

                   

            # test if column can be converted to an integer

            asint = props[col].fillna(0).astype(np.int64)

            result = (props[col] - asint)

            result = result.sum()

            if result > -0.01 and result < 0.01:

                IsInt = True



            

            # Make Integer/unsigned Integer datatypes

            if IsInt:

                if mn >= 0:

                    if mx < 255:

                        props[col] = props[col].astype(np.uint8)

                    elif mx < 65535:

                        props[col] = props[col].astype(np.uint16)

                    elif mx < 4294967295:

                        props[col] = props[col].astype(np.uint32)

                    else:

                        props[col] = props[col].astype(np.uint64)

                else:

                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:

                        props[col] = props[col].astype(np.int8)

                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:

                        props[col] = props[col].astype(np.int16)

                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:

                        props[col] = props[col].astype(np.int32)

                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:

                        props[col] = props[col].astype(np.int64)    

            

            # Make float datatypes 32 bit

            else:

                props[col] = props[col].astype(np.float32)

            

            # Print new column type

            print("dtype after: ",props[col].dtype)

            print("******************************")

    

    # Print final result

    print("___MEMORY USAGE AFTER COMPLETION:___")

    mem_usg = props.memory_usage().sum() / 1024**2 

    print("Memory usage is: ",mem_usg," MB")

    print("This is ",100*mem_usg/start_mem_usg,"% of the initial size")

    return props, NAlist





def autocorrelation_columns(df, columns, lookback=1):

    n = len(columns)

    rows = len(df)



    if n < lookback + 2:

        return np.zeros(rows)



    reshaped = [[df[c].iloc[i] for c in columns] for i in range(rows)]

    rv = np.zeros(rows)

    

    for i in range(rows):

        v = reshaped[i]



        rv[i] = stats.pearsonr(v[lookback:], v[:n - lookback])[0]

        

    return rv





def fourier_columns(df, columns, f=0):

    n = len(columns)

    rows = len(df)



    if n < 3:

        return np.zeros(rows)



    reshaped = [[df[c].iloc[i] for c in columns] for i in range(rows)]

    rv = np.zeros(rows)

    

    for i in range(rows):

        v = reshaped[i]



        rv[i] = np.real(fft(v))[f]

        

    return rv





def summary(data):

    AnalysisUtils.summary(data)

    

    

def table(series, normalize=True):

    if isinstance(series, np.ndarray):

        series = pd.Series(series)

        

    return series.value_counts(normalize=normalize)

KEY = 'id'

LABEL = 'label'
import os

import json

import math

import pickle

import pprint

import sys

import time



import datetime as dt

import lightgbm as lgb

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import seaborn as sns



from abc import ABC, abstractmethod



from functools import partial

from pandas.io.json import json_normalize

from scipy import stats

from scipy.fftpack import fft



from sklearn.ensemble import RandomForestClassifier



from sklearn.metrics import confusion_matrix

from sklearn.metrics import mean_squared_error, roc_auc_score, roc_curve



from sklearn.model_selection import cross_val_score

from sklearn.model_selection import KFold

from sklearn.model_selection import train_test_split



from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import OneHotEncoder



np.set_printoptions(suppress=True)



pd.set_option('display.max_rows', 150)

pd.set_option('display.max_columns', 300)

pd.set_option('display.width', 120)

pd.set_option('float_format', '{:7.2f}'.format)

        

### Abstract classes ###

class DataCleaner:

    def __init__(self, data_input):

        self.data_input = data_input



    @abstractmethod

    def correct_types(self):

        pass

    

    def clean(self):

        self.clean_data = [x.copy() for x in self.data_input.get_dataframes()]

        

        self.correct_types()

        self.handle_na()

        

        return self.clean_data

        

    def convert_to_str_columns(self, idx, columns):

        for c in columns:

            self.clean_data[idx][c] = self.clean_data[idx][c].astype(str)

    

    @abstractmethod

    def handle_na(self):

        pass

    

class DataInput:

    @abstractmethod

    def get_dataframes(self):

        pass



class FeatureEngineering:

    def __init__(self, clean_data):

        self.clean_data = clean_data



    # This method should return a single dataframe containing every feature.

    @abstractmethod

    def transform(self, *args, **kwargs):

        pass

    

class PredictionModel:

    def __init__(self, data):

        self.data = data



    @abstractmethod

    def encode(self, data, preserve_order = True):

        pass

    

    @abstractmethod

    def evaluate(self):

        pass

    

    def fit(self):

        # self.encoded_data = self.encode(self.data, preserve_order=False)

        pass

    

    @abstractmethod

    def is_score(self):

        pass

    

    @abstractmethod

    def produce_solution(self):

        pass



    def solve(self, test_data):

        self.encoded_test_data = self.encode(test_data)

        

    @abstractmethod

    def split_X_y(self, df):

        pass

### Abstract classes ###
class ChurnDataInput(DataInput):

    def __init__(self, file_demo, file_txn, file_train, file_test):

        self.df_demo = pd.read_csv(file_demo, dtype={KEY: 'str'})

        self.df_txn = pd.read_csv(file_txn, dtype={KEY: 'str'})

        self.df_train = pd.read_csv(file_train, dtype={KEY: 'str'})

        self.df_test = pd.read_csv(file_test, dtype={KEY: 'str'})

        

        reduce_mem_usage(self.df_demo, ['n1', 'n2'])

        reduce_mem_usage(self.df_txn)

        reduce_mem_usage(self.df_train)

        reduce_mem_usage(self.df_test)

    

    def get_dataframes(self):

        return (self.df_demo, self.df_txn, self.df_train, self.df_test)

    

class ChurnDataCleaner(DataCleaner):

    def __init__(self, data_input):

        super(ChurnDataCleaner, self).__init__(data_input)

    

    def correct_types(self):

        start = time.clock()

        self.convert_to_str_columns(0, ['c0', 'c1', 'c2', 'c3', 'c4'])

        self.convert_to_str_columns(1, ['old_cc_no', 'c5', 'c6', 'c7'])

        print('correct_types: Elapsed {}s'.format(time.clock() - start))

        

    def handle_na(self):

        start = time.clock()

        self.clean_data[0]['n1'].fillna(self.clean_data[0]['n1'].median(), inplace=True)

        self.clean_data[0]['n2'].fillna(self.clean_data[0]['n2'].median(), inplace=True)

        print('handle_na: Elapsed {}s'.format(time.clock() - start))
data_in = ChurnDataInput('demo.csv', 'txn.csv', 'train.csv', 'test.csv')
dfs = data_in.get_dataframes()
dfs[0].head()
summary(dfs[0])
summary(dfs[1])
summary(dfs[2])
summary(dfs[3])
summary(dfs[4])
%%time

data_cleaner = ChurnDataCleaner(data_in)

clean_data = data_cleaner.clean()
x = clean_data[1].copy()

x['month_number'] = np.clip((x['n3'] / 30).apply(np.floor), a_min=1, a_max=12)

x['month_number'] = x['month_number'].astype(int)



grouped = x.groupby('month_number')
x.head()
res = grouped['n4'].mean().rename('amount')

res
pd.DataFrame(res)
res.to_csv('mean_spending_by_month.csv', header=True)
summary(clean_data[0])
summary(clean_data[1])
summary(clean_data[2])
summary(clean_data[3])
sentout = pd.read_csv('catSentout2.csv')

sentout['Cat'] = sentout['Cat'].astype(int)

sentout.head()
summary(sentout)
class ChurnFeatureEngineering(FeatureEngineering):

    def __init__(self, clean_data, cache):

        super(ChurnFeatureEngineering, self).__init__(clean_data)

        self.cache = cache

        

        self.transformed_train = None

        self.transformed_test = None



    def transform(self, *args, **kwargs):

        if self.transformed_train is not None and self.transformed_test is not None:

            if kwargs['train']:

                return self.transformed_train.copy()

            else:

                return self.transformed_test.copy()

            

        demo_df = self.clean_data[0].copy()

        txn_df = self.clean_data[1].copy()

        train_df = self.clean_data[2].copy()

        test_df = self.clean_data[3].copy()

        

        txn_df['spending_cat'] = sentout['Cat']

        

        rv = pd.merge(demo_df, train_df, how='left', on=KEY)

        

        # Separate features and labels

        labels = rv[LABEL]

        rv = rv.drop([LABEL], axis=1)

        

        numeric = ['n{}'.format(x) for x in range(3, 8)]

        

        ### Simple aggregation ###

        start = time.clock()

        sub_columns = numeric.copy()

        sub_columns.insert(0, KEY)



        sub_df = txn_df[sub_columns]

        grouped = sub_df.groupby(KEY)

        agg_names = ['sum', 'mean', 'count', 'nunique', 'std', 'max', 'min']

        

        for agg in agg_names:

            cache_key = 'col_{}'.format(agg)

            

            if cache_key not in self.cache:

                if agg == 'sum':

                    self.cache[cache_key] = grouped.sum()

                elif agg == 'mean':

                    self.cache[cache_key] = grouped.mean()

                elif agg == 'count':

                    self.cache[cache_key] = grouped.count()

                elif agg == 'nunique':

                    self.cache[cache_key] = grouped.nunique()

                elif agg == 'std':

                    self.cache[cache_key] = grouped.std()

                elif agg == 'max':

                    self.cache[cache_key] = grouped.max()

                elif agg == 'min':

                    self.cache[cache_key] = grouped.min()

                

            s = self.cache[cache_key].rename(columns={k: '{}_{}'.format(k, agg) for k in numeric}).fillna(0)

            rv = rv.join(s, on=KEY)

            

        rv.drop(['n4_count', 'n5_count', 'n6_count', 'n7_count'], axis=1, inplace=True)

        print('Simple Aggregation: Elapsed {}s'.format(time.clock() - start))

        ### Simple aggregation ###

        

        ### Filter-and-Aggregate Template ###

        filter_names = ['old_cc_label', 'spending_cat']

        numeric_fields = ['n{}'.format(x) for x in range(3, 8)]

        agg_names = ['mean', 'sum', 'count', 'nunique', 'max', 'min']

        possible_cat_values = {

            'old_cc_label': range(13),

            # 'spending_cat': [x for x in range(10)] + [11, 12, 13]

            'spending_cat': [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13]

        }



        sub_columns = numeric.copy()

        sub_columns.insert(0, KEY)



        # For each filter

        for ft in filter_names:

            # For each possible value in that filter

            for val in possible_cat_values[ft]:

                start = time.clock()

                sub_df = txn_df[txn_df[ft] == val][sub_columns]

                grouped = sub_df.groupby(KEY)



                # For each aggregation function

                for agg in agg_names:

                    cache_key = 'col_{}_{}_{}'.format(ft, val, agg)



                    if cache_key not in self.cache:

                        print('Building Key: {}'.format(cache_key))

                        

                        if agg == 'sum':

                            self.cache[cache_key] = grouped.sum()

                        elif agg == 'mean':

                            self.cache[cache_key] = grouped.mean()

                        elif agg == 'count':

                            self.cache[cache_key] = grouped.count()

                        elif agg == 'nunique':

                            self.cache[cache_key] = grouped.nunique()

                        elif agg == 'max':

                            self.cache[cache_key] = grouped.max()

                        elif agg == 'min':

                            self.cache[cache_key] = grouped.min()

                            

                    s = self.cache[cache_key].rename(columns={k: '{}_{}_{}_{}'.format(k, ft, val, agg) 

                                                                  for k in sub_df.columns}).fillna(0)

                        

                    rv = rv.join(s, on=KEY).fillna(0)

                    

                print('Filter-and-Aggregate {}-{}: Elapsed {}s'.format(ft, val, time.clock() - start))

        

        rv.drop(['n{}_old_cc_label_{}_count'.format(x, y) for x in range(4, 8) for y in range(13)], axis=1, inplace=True)

        rv.drop(['n{}_spending_cat_{}_count'.format(x, y) for x in range(4, 8) for y in possible_cat_values['spending_cat']], 

                axis=1, inplace=True)

        ### Filter-and-Aggregate Template ###

        

        # Reconstruct the dataframe with labels

        rv[LABEL] = labels

        

        self.transformed_train = rv[rv[LABEL].notnull()].copy()

        self.transformed_test = rv[rv[LABEL].isna()].copy()

        

        if kwargs['train']:

            return self.transformed_train.copy()

        else:

            return self.transformed_test.copy()
if os.path.exists('cache.pkl'):

    f = open("cache.pkl", "rb")

    cache = pickle.load(f)

    f.close()

    

    print('Cache loaded')

else:

    cache = {}
f = open("cache.pkl", "wb")

pickle.dump(cache, f)

f.close()
list(cache.keys())
cnt = 0



for k in list(cache.keys()):

    if k.startswith('col_spending_cat'):

        devnull = cache.pop(k)

        cnt += 1

        

print('Pop {} key(s)'.format(cnt))
%%time

fe = ChurnFeatureEngineering(clean_data, cache)

data_fe = fe.transform(train=True)

test_fe = fe.transform(train=False)
cache['data_fe'] = data_fe

cache['test_fe'] = test_fe
data_fe.shape
summary(data_fe)
data_fe.head()
data_fe.columns.values
summary(test_fe)
class ChurnPredictionModel(PredictionModel):

    def __init__(self, data, fe, num_regressors=1):

        super(ChurnPredictionModel, self).__init__(data)



        self.fe = fe

        self.num_regressors = num_regressors

        

    def create_default_model(self):        

#         gb = GradientBoostingClassifier(random_state=86225)

#         lg = LogisticRegression(class_weight='balanced', random_state=86225)

#         ada = AdaBoostClassifier(random_state=86225)

#         et = ExtraTreesClassifier(n_estimators=200, n_jobs=7, criterion='gini', class_weight='balanced', max_depth=3, min_samples_split=0.05, min_samples_leaf=0.03, 

#                                   random_state=86225)

        

        # cf = XGBClassifier(n_estimators=150, n_jobs=3, random_state=86225, scale_pos_weight=scale_pos_weight)

        reg = XGBRegressor(n_estimators=1500, n_jobs=3, random_state=86225)

    

#         knn = KNeighborsClassifier(n_jobs=7)



        # parameters = {'n_estimators':[30], 'max_depth':[3], 'min_samples_split': [15], 'criterion': ['gini'], 'max_features': [0.05]}

#         parameters = {'n_estimators':[150], 'max_depth':[3], 'min_samples_split': [0.01], 'max_features': [0.05], 

#                       'learning_rate': [0.1]}

        

        # grid_clf = GridSearchCV(gb, parameters, cv=5, n_jobs=3, scoring='accuracy')

        

        return reg

        # return VotingClassifier(estimators=[('rf', rf), ('lg', lg), ('gb', gb), ('ada', ada), ('et', et), ('xgb', xgb), ('knn', knn)], voting='soft', weights=[1] * 7)

        

        # return GradientBoostingClassifier(random_state=86225)

    

    def encode(self, data, use_existing_onehot = False):

        if not use_existing_onehot:

            self.enc = OneHotEncoder(categorical_features=[1, 2, 3, 4, 5], sparse=False)

            self.enc.fit(data)

        

        rv = self.enc.transform(data)

        

        return rv

    

    def evaluate(self):

        start = time.clock()

        total_scores = list()

        

        for i in range(15):

            ### Fit training part ###

            clf, reg = self.create_default_model()

            df = self.data.sample(frac=1, random_state=77545 + i)

            

            X, y = self.split_X_y(df)

            X_encoded = self.encode(X, use_existing_onehot=True)

            

            X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.70, random_state=77545 + i)

            

            y_binary = y_train.copy()

            y_binary[y_binary > 0] = 1

            clf.fit(X_train, y_binary)

            

            idx = y_train > 0.0

            X_train_reg = X_train[idx].copy()

            y_train_reg = y_train[idx].copy()



            reg.fit(X_train_reg, y_train_reg)

            ### Fit training part ###

            

            pred_class = clf.predict(X_test)

            pred_reg = reg.predict(X_test)



            pred = pred_class * pred_reg

            score = self.scoring(pred, y_test)

            

            total_scores.append(score)

    

        print('Model Evaluation: Elapsed {}s'.format(time.clock() - start))

        

        return np.array(total_scores)

    

    def fit(self, **kwargs):

        super(ChurnPredictionModel, self).fit()

        

        start = time.clock()

        self.regressor = self.create_default_model()



        self.X, self.y = self.split_X_y(self.data, LABEL)

        self.X_encoded = self.encode(self.X)



        self.regressor.fit(self.X_encoded, self.y)

        print('Regression Fitting: Elapsed {}s'.format(time.clock() - start))

        

    def importance(self):

        rv = self.column_map.copy()

        rv['importance'] = self.regressor.feature_importances_



        return rv

    

    def is_score(self):        

        self.pred = self.regressor.predict(self.X_encoded)

        

        score = np.square(self.y - self.pred) / np.square(np.minimum(2 * np.abs(self.y), np.abs(self.pred)) + np.abs(self.y))

        score = 100 - 100 * score.sum() / len(self.pred)

        

        return score

    

    def produce_solution(self):

        np.savetxt('O-TJ2018-REGCEN-10582.csv', self.solutions, fmt='%f')

        

    def scoring(self, pred, actual):

        sol = np.zeros(len(pred))

        mask = actual + pred > 0.0

        sol[mask] = np.absolute(actual[mask] - pred[mask]) / ((actual[mask] + pred[mask]) / 2)

        smape = sol.mean() * 100

                

        return 1.0 - smape / 200

    

    def solve(self, test_data):

        X, y = self.split_X_y(test_data, LABEL)

        X_encoded = self.encode(X, use_existing_onehot=True)



        self.solutions = self.regressor.predict(X_encoded)

        

        return self.solutions



    def split_X_y(self, df, lb):

        labels = df[lb].copy().values

        

        X = df.drop([KEY], axis=1).drop(LABEL, axis=1)

        

        return X, labels
class LightModel(ChurnPredictionModel):



    def __init__(self, data, fe):

        super(LightModel, self).__init__(data, fe)

        

        self.weight = {}

        

    def fit(self, **kwargs):        

        self.data[LABEL] = self.data[LABEL].astype(int)

        

        self.X, self.y = self.split_X_y(self.data, LABEL)

        self.X_encoded = self.encode(self.X)

        

        params = {

            "objective" : "multiclass",

            'metric': "None",

            "num_class": 13,

            # "num_leaves" : 120,

            'max_depth': -1,

            "learning_rate" : 0.1,

            "bagging_fraction" : 0.8,

            "feature_fraction" : 0.015,

            "bagging_freq" : 5,

            "bagging_seed" : 86225,

            "verbosity" : -1

        }



        train_kwargs = {

            'num_boost_round': kwargs['num_boost_round'] if 'num_boost_round' in kwargs else 1000, 

            'early_stopping_rounds': 100,

            'verbose_eval': kwargs['verbose_eval'] if 'verbose_eval' in kwargs else 10

        }

        

        def custom_eval(y_pred, train_data):

            y_true = train_data.get_label()

            

            weight = self.get_weight(len(y_true))



            targets = np.array([y_true]).reshape(-1)

            indicator = np.eye(13)[targets]



            y_pred = y_pred.reshape((len(y_true), 13), order='F')

            y_pred = np.clip(y_pred, a_min=0.0001, a_max=None)

            score = -(weight * indicator * np.log(y_pred)).sum() / len(y_true)

            

            return ('Weighted Log Loss', score, False)

        

        # X_train, X_test, y_train, y_test = train_test_split(self.X_encoded, self.y, test_size=0.3, random_state=77545)

        

        kf = KFold(n_splits=kwargs['n_splits'] if 'n_splits' in kwargs else 5, 

                   random_state=75341, shuffle=True)

        

        self.regressor = []

        

        for train_index, test_index in kf.split(self.X_encoded):

            start = time.clock()

            X_train, X_test = self.X_encoded[train_index], self.X_encoded[test_index]

            y_train, y_test = self.y[train_index], self.y[test_index]

            

            lgtrain = lgb.Dataset(X_train, label=y_train)

            lgval = lgb.Dataset(X_test, label=y_test)



            self.regressor.append(lgb.train(params, lgtrain, valid_sets=[lgval], feval=custom_eval, **train_kwargs))

        

            print('Regression Fitting: Elapsed {}s'.format(time.clock() - start))

        

    def is_score(self):

        self.pred = None

        

        for i in range(len(self.regressor)):

            if self.pred is None:

                self.pred = self.regressor[i].predict(self.X_encoded, num_iteration=self.regressor[i].best_iteration)

            else:

                self.pred += self.regressor[i].predict(self.X_encoded, num_iteration=self.regressor[i].best_iteration)

        self.pred /= self.pred.sum(axis=1, keepdims=True)

        

        weight = self.get_weight(len(self.y))

        

        targets = np.array([self.y]).reshape(-1)

        indicator = np.eye(13)[targets]



        score = -(weight * indicator * np.log(self.pred)).sum() / len(self.y)

        

        return score

    

    def solve(self, test_data):

        X, y = self.split_X_y(test_data, LABEL)

        X_encoded = self.encode(X, use_existing_onehot=True)



        self.solutions = None

        

        for i in range(len(self.regressor)):

            if self.solutions is None:

                self.solutions = self.regressor[i].predict(X_encoded, num_iteration=self.regressor[i].best_iteration)

            else:

                self.solutions += self.regressor[i].predict(X_encoded, num_iteration=self.regressor[i].best_iteration)

        self.solutions /= self.solutions.sum(axis=1, keepdims=True)

        

        return self.solutions

    

    def get_weight(self, sz):

        if sz not in self.weight:

            weight = self.data[LABEL].value_counts()

            weight_series = weight.sort_index() / 100000

            self.weight[sz] = np.array([np.array(weight_series)] * sz)

            

        return self.weight[sz]
# model = ChurnPredictionModel(data_fe, fe, num_regressors=1)

model = LightModel(data_fe, fe)
%%time

model.fit(num_boost_round=40000, verbose_eval=100, n_splits=5)
model.is_score()
model.pred[:5]
summary(test_fe)
solutions = model.solve(test_fe)
solutions
len(solutions)
first = pd.DataFrame({KEY: test_fe[KEY]}).reset_index(drop=True)

second = pd.DataFrame(solutions)

prod = pd.concat([first, second], axis=1).rename(columns={x: 'class{}'.format(x) for x in range(13)})

prod.to_csv('pred.csv', index=False, header=True)
prod[['class{}'.format(x) for x in range(13)]].sum(axis=1)
prod.shape
x = pd.DataFrame({'y_true': model.y, 'y_pred': model.pred})

x['upper'] = np.square(x['y_true'] - x['y_pred'])

x['lower'] = np.square(np.minimum(2 * np.abs(x['y_true']), np.abs(x['y_pred'])) + np.abs(x['y_true']))

x['root_lower'] = np.minimum(2 * np.abs(x['y_true']), np.abs(x['y_pred'])) + np.abs(x['y_true'])

x['min_term'] = np.minimum(2 * np.abs(x['y_true']), np.abs(x['y_pred']))

x['loss'] = np.square(x['y_true'] - x['y_pred']) / np.square(np.minimum(2 * np.abs(x['y_true']), np.abs(x['y_pred'])) + np.abs(x['y_true']))

x['diff'] = x['y_pred'] - x['y_true']
x.head()
summary(x)
percentile(x['diff'])
x[x['loss'] > 0.1]