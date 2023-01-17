print('loading libs...')

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import lightgbm as lgb

import warnings

warnings.filterwarnings("ignore")

import os

import gc

import datetime

import time

from tqdm import tqdm

from scipy import stats

from sklearn.model_selection import KFold

from sklearn.metrics import roc_auc_score

from pandas.tseries.holiday import USFederalHolidayCalendar as calendar

import seaborn as sns

import matplotlib.pyplot as plt

print('done')
# loading the funcs
# func for loading data

def DataLoading(path, df_name):

    files = os.listdir(f'{path}')

    for i in range(len(files)):

        s0 = files[i]

        s1 = files[i][:-4]

        s2 = files[i][-4:]

        if s2 =='.csv':

            print('loading:'+ s1 + '...')

            globals()[s1] = pd.read_csv(f'{path}'+ s0)

            df_name.append(s1)

        elif s2 == '.pkl':

            print('loading:'+ s1 + '...')

            globals()[s1] = pd.read_pickle(f'{path}'+ s0)

            df_name.append(s1)

        else:

            pass

    print('successfully loading: ')

    print(df_name)

    print('done')

    return df_name



    

# func for data analysis(based on https://www.kaggle.com/kabure/extensive-eda-and-modeling-xgb-hyperopt)

def DataStatistics(df):   

    summary = pd.DataFrame(df.dtypes,columns=['dtypes'])

    summary = summary.reset_index()

    summary['Name'] = summary['index']

    summary = summary[['Name','dtypes']]

    summary['Missing'] = df.isnull().sum().values 

    summary['Missing_percentage'] = round((summary['Missing']/df.shape[0])*100, 1)

    summary['Uniques'] = df.nunique().values

    summary['First Value'] = df.loc[0].values

    summary['Second Value'] = df.loc[1].values

    summary['Third Value'] = df.loc[2].values

       

    for name in summary['Name'].value_counts().index:

        summary.loc[summary['Name'] == name, 'Entropy'] = round(stats.entropy(df[name].value_counts(normalize=True), base=2),2) 

    summary.set_index('Name',inplace=True)

    summary = summary.T

    return summary





# func for showing data

def DataShowing(df_name, start=0, end=20, seeall = False):

    if seeall:

        pd.set_option('display.max_rows', None)

        pd.set_option('display.max_columns', None)   

    df_name.sort(reverse=True)

    for i in range(len(df_name)):

        s = df_name[i]

        df = globals()[s]

        print('data shape of ' + s + ':' + f'{df.shape}')

        df = df.iloc[:,start:end]

        if df.empty:

            pass

        else:

            print('looking over the statistics of all features of ' + s + '...')

            display(DataStatistics(df))

            print('looking over the statistics of the num_type_features of ' + s + '...')

            display(df.describe())

   

                    

                    

# func for merging data

def DataMerging(df1, df2, on='TransactionID', how='left'):

    print('merging data...')

    df = pd.merge(df1, df2, on=on, how=how)

    print('done')

    return df

                    

                    

# func for processing data pipeline

def DataProcessing(df1, df2):

    print('translate DT...')

    ProcessDT(df1)

    ProcessDT(df2)

    df3 = df1['isFraud'].copy()

    df1 = df1.drop('isFraud', axis=1)

    colsnum =  df1.dtypes.reset_index()

    colsnum = colsnum[colsnum[0]!='object'].index

    print('translate number features in train...')

    colsnum = tqdm(colsnum) 

    for idx in colsnum:

        ProcessNumt2Obj(df1,idx)

    print('translate number features in test...')

    colsnum = tqdm(colsnum) 

    for idx in colsnum:

        ProcessNumt2Obj(df2,idx)

    print('reduce memory...')

    df1 = ReduceMemUsage(df1)

    df2 = ReduceMemUsage(df2)

    print('translate all features...')

    df1['isFraud'] = df3

    colsnum = range(len(df1.columns)-1)

    colsnum = tqdm(colsnum)

    for idx in colsnum:

        ProcessObj(df1,df2,idx)

    print('fillna...')

    df1 = df1.fillna(0)

    df2 = df2.fillna(0)

    print('reducing memory...')

    df1 = ReduceMemUsage(df1)

    df2 = ReduceMemUsage(df2)

    print('dropping target...')

    df4 = df1.drop('isFraud', axis=1)

    df5 = df2.copy()

    train_cols = list(df1.columns)

    print('Done')

    return df3, df4, df5      



# convert number type to object

def ProcessNumt2Obj(df,idx,sp=20):

    temp = df.iloc[:,idx]

    df.iloc[:,idx] = pd.qcut(temp,sp,labels=False,duplicates='drop')

    return df

                                     

                    

# func for processing object(using the percentage of isFraud to repalce the object)

def ProcessObj(df1,df2,idx):

    temp = df1.iloc[:,idx]

    temp_t = df2.iloc[:,(idx)]

    L = temp.unique().tolist()

    L1 = temp_t.unique().tolist()

    L.extend(L1)

    L  = list(set(L))

    if temp.isnull().any(): 

        temp_0 = df1[temp.isna()]

        temp_t_0 = df2[temp_t.isna()]

        temp_1 = temp_0[temp_0.isFraud == 1]

        rep = temp_1.shape[0]/(temp_0.shape[0]+0.0001)

        rep = round(rep, 4)

        #print(rep)

        df1.iloc[temp_0.index,idx]=rep

        df2.iloc[temp_t_0.index,idx]=rep

        for i in range(len(L)):

            #print(L[i])

            temp_0 = df1[temp == L[i]]

            temp_t_0 = df2[temp_t == L[i]]

            temp_1 = temp_0[temp_0.isFraud == 1]

            rep = temp_1.shape[0]/(temp_0.shape[0]+0.0001)

            rep = round(rep, 4)

            #print(rep)

            df1.iloc[temp_0.index,idx]=rep

            df2.iloc[temp_t_0.index,idx]=rep

    else:

        for i in range(len(L)):

            #print(L[i])

            temp_0 = df1[temp == L[i]]

            temp_t_0 = df2[temp_t == L[i]]

            temp_1 = temp_0[temp_0.isFraud == 1]

            rep = temp_1.shape[0]/(temp_0.shape[0]+0.0001)

            rep = round(rep, 4)

            #print(rep)

            df1.iloc[temp_0.index,idx]=rep

            df2.iloc[temp_t_0.index,idx]=rep

    df1.iloc[:,idx]=df1.iloc[:,idx].astype('float16')

    df2.iloc[:,(idx-1)]=df2.iloc[:,idx].astype('float16')

    

    

    

# func for processing datetime data(based on https://www.kaggle.com/kyakovlev/ieee-fe-with-some-eda)

def ProcessDT(df):

    START_DATE = datetime.datetime.strptime('2017-11-30', '%Y-%m-%d')

    dates_range = pd.date_range(start='2017-10-01', end='2019-01-01')

    us_holidays = calendar().holidays(start=dates_range.min(), end=dates_range.max())

    



    df['DT'] = df['TransactionDT'].apply(lambda x: (START_DATE + datetime.timedelta(seconds = x)))

    df['DT_M'] = ((df['DT'].dt.year-2017)*12 + df['DT'].dt.month).astype(np.int8)

    df['DT_W'] = ((df['DT'].dt.year-2017)*52 + df['DT'].dt.weekofyear).astype(np.int8)

    df['DT_D'] = ((df['DT'].dt.year-2017)*365 + df['DT'].dt.dayofyear).astype(np.int16)

    

    df['DT_hour'] = (df['DT'].dt.hour).astype('object')

    df['DT_day_week'] = (df['DT'].dt.dayofweek).astype('object')

    df['DT_day_month'] = (df['DT'].dt.day).astype('object')

        

    # Possible solo feature

    df['is_december'] = df['DT'].dt.month

    df['is_december'] = (df['is_december']==12).astype('object')



    # Holidays

    df['is_holiday'] = (df['DT'].dt.date.astype('datetime64').isin(us_holidays)).astype('object')

    df.drop(['DT','DeviceInfo'],axis=1,inplace=True)

    

    

# reduce memory usage

def ReduceMemUsage(df, verbose=True):

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

                                       
# setting the params

DF_NAME= []

PATH = '../input/ieee-fraud-detection/'
# loading data

df_name = DataLoading(PATH, DF_NAME)
# looking at the original data

# DataShowing(df_name)
# merging data

train = DataMerging(train_transaction, train_identity)

test = DataMerging(test_transaction, test_identity)
del train_transaction, train_identity, test_transaction, test_identity

gc.collect()
# looking at the merged data

df_name = ['X_train','X_test']

DataShowing(df_name)
# processing data

y_train, X_train, X_test = DataProcessing(train, test)
# saving processed data

X_train.to_pickle('X_train.pkl')

X_test.to_pickle('X_test.pkl')

y_train.to_pickle('y_train.pkl')

sample_submission.to_pickle('sample_submission.pkl')