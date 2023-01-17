# To print multiple output in a cell

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = 'all'
## Import all the required libraries



import pandas as pd

import numpy as np

import os



import matplotlib.pyplot as plt

# % matplotlib inline
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

train=pd.read_csv('../input/janatahack-demand-forecasting-analytics-vidhya/train.csv')



test=pd.read_csv('../input/janatahack-demand-forecasting-analytics-vidhya/test.csv')



sample=pd.read_csv('../input/janatahack-demand-forecasting-analytics-vidhya/sample_submission.csv')
train.head(10)

print('Shape of training data is {}'.format(train.shape))



print('-------------'*5)



test.head()

print('Shape of test data is {}'.format(test.shape))



print('--------------'*5)



sample.head()
train.describe()
train.isnull().sum()
train['week'].unique()
# Number of units sold in accordance with the week



train.groupby('week').sum()['units_sold'].plot(figsize=(12,8))

font = {'family': 'serif',

        'color':  'darkred',

        'weight': 'normal',

        'size': 26,

        }

plt.xlabel('Week',fontdict=font)

plt.ylabel('units_sold',fontdict=font)

# amount earned through sales in each week



train.groupby('week').sum()['total_price'].plot(figsize=(12,8))

font = {'family': 'serif',

        'color':  'darkred',

        'weight': 'normal',

        'size': 26,

        }

plt.xlabel('Week',fontdict=font)

plt.ylabel('total_price',fontdict=font)
train['store_id'].unique()
## product sold by each of the store





train.groupby('store_id').sum()['units_sold'].plot(figsize=(15,8),kind='bar')

font = {'family': 'serif',

        'color':  'darkred',

        'weight': 'normal',

        'size': 26,

        }

plt.xlabel('store_id',fontdict=font)

plt.ylabel('units_sold',fontdict=font)
## Product was on display at a prominent place at the store



# Impact on sales on the basis of display

train.groupby(['is_display_sku','store_id']).sum()['units_sold']
# join test and train data



train['train_or_test']='train'

test['train_or_test']='test'

df=pd.concat([train,test])
df.head()

df.shape
# function to utilize date time column i.e '''week'''



def create_week_date_featues(dataframe):



    df['Month'] = pd.to_datetime(df['week']).dt.month



    df['Day'] = pd.to_datetime(df['week']).dt.day



    df['Dayofweek'] = pd.to_datetime(df['week']).dt.dayofweek



    df['DayOfyear'] = pd.to_datetime(df['week']).dt.dayofyear



    df['Week'] = pd.to_datetime(df['week']).dt.week



    df['Quarter'] = pd.to_datetime(df['week']).dt.quarter 



    df['Is_month_start'] = pd.to_datetime(df['week']).dt.is_month_start



    df['Is_month_end'] = pd.to_datetime(df['week']).dt.is_month_end



    df['Is_quarter_start'] = pd.to_datetime(df['week']).dt.is_quarter_start



    df['Is_quarter_end'] = pd.to_datetime(df['week']).dt.is_quarter_end



    df['Is_year_start'] = pd.to_datetime(df['week']).dt.is_year_start



    df['Is_year_end'] = pd.to_datetime(df['week']).dt.is_year_end



    df['Semester'] = np.where(df['week'].isin([1,2]),1,2)



    df['Is_weekend'] = np.where(df['week'].isin([5,6]),1,0)



    df['Is_weekday'] = np.where(df['week'].isin([0,1,2,3,4]),1,0)



    df['Days_in_month'] = pd.to_datetime(df['week']).dt.days_in_month

    

    return df
df=create_week_date_featues(df)
df.head(5)

df.shape
from sklearn.preprocessing import LabelEncoder
col=['store_id','sku_id','Is_month_start','Is_month_end','Is_quarter_start','Is_quarter_end','Is_year_start','Is_year_end']
for i in col:

    df = pd.get_dummies(df, columns=[i])
df.head()
# col2=['Is_month_start','Is_month_end','Is_quarter_start','Is_quarter_end','Is_year_start','Is_year_end']



# for i in col2:

#     df = pd.get_dummies(df, columns=[i])
# drop the columns

df.drop(['record_ID','week'],inplace=True,axis=1)
df.head()

df.shape
# Total price columns



df['total_price'].plot(kind='hist')
df['total_price']=np.log1p(df['total_price'])

df['total_price'].plot(kind='hist')
df['base_price'].plot(kind='hist')
df['base_price']=np.log1p(df['base_price'])

df['base_price'].plot(kind='hist')
df.head()
train_1=df.loc[df.train_or_test.isin(['train'])]

test_1=df.loc[df.train_or_test.isin(['test'])]

train_1.drop(columns={'train_or_test'},axis=1,inplace=True)

test_1.drop(columns={'train_or_test'},axis=1,inplace=True)
train_1.head()

train_1.shape

test_1.shape

test_1.head()
test_1.drop(['units_sold'],axis=1,inplace=True)
train_1.shape

test_1.shape
x=train_1.drop(['units_sold'],axis=1)

y=train_1['units_sold']
x=x.values

test_data=test_1.values



y=y.values
x.shape

test_data.shape

from sklearn.model_selection import train_test_split
# x_train,x_valid,y_train,y_valid=train_test_split(x,y,test_size=0.35)
# x_train.shape
from xgboost import XGBRegressor

from xgboost import plot_importance

import xgboost as xgb



# function to plot all features based out of its importance.

def plot_features(booster, figsize):    

    fig, ax = plt.subplots(1,1,figsize=figsize)

    return plot_importance(booster=booster, ax=ax)



import time
def rmsle(y_true, y_pred):

    return np.sqrt(np.mean(np.power(np.log1p(y_true)-np.log1p(y_pred), 2)))
# Perform cross-validation

from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
model = XGBRegressor(

    max_depth=12,

    booster = "gbtree",

    n_estimators=200,

    eval_metric = 'rmse',

    min_child_weight=300, 

    colsample_bytree=0.8, 

    subsample=0.8, 

    eta=0.3,

    seed=42,

    objective='reg:linear')
kfold, scores = KFold(n_splits=5, shuffle=True, random_state=0), list()



for train, test in kfold.split(x):

    x_train, x_test =x[train], x[test]

    y_train, y_test = y[train], y[test]

    model.fit(x_train, y_train,verbose=True,

              eval_set=[(x_train, y_train), (x_test, y_test)],

              early_stopping_rounds = 50)

#     preds = model.predict(x_test)

#     score = rmsle(y_test, preds)

#     scores.append(score)

#     print(score)

    

    

# print("Average: ", sum(scores)/len(scores))
# # defint the model parameters



# ts = time.time()



# model = XGBRegressor(

#     max_depth=12,

#     booster = "gbtree",

#     n_estimators=500,

#     min_child_weight=350, 

#     colsample_bytree=0.8, 

#     subsample=0.8, 

#     eta=0.3,

#     seed=42,

#     objective='reg:linear')



# model.fit(

#     x_train, 

#     y_train, 

#     eval_metric="rmse", 

#     eval_set=[(x_train, y_train), (x_valid, y_valid)], 

#     verbose=True, 

#     early_stopping_rounds = 100)



# time.time() - ts
test_data.shape
# create prediction on test data



pred=model.predict(test_data)
len(pred)
# sample['units_sold']=pred.round()



sample['units_sold']=pred

sample.head()
sample['units_sold'].unique()
sample['units_sold']=abs(sample['units_sold']).astype('int')
sample.to_csv('submission_xgb.csv',index=False,encoding='utf-8')
sub_path = "../input/testing-minmaxbestbasestacking"

all_files = os.listdir(sub_path)

print(all_files)
for f in all_files:

    print(f)
d=pd.read_csv('../input/testing-minmaxbestbasestacking/866_126527_us_submission_lgbm_22.csv')

d.head()

d.shape
# # Read and concatenate submissions



outs = [pd.read_csv(os.path.join(sub_path, f), index_col=0) for f in all_files ]

concat_sub = pd.concat(outs, axis=1)

concat_sub.head()
cols = list(map(lambda x: "target" + str(x), range(len(concat_sub.columns))))



cols = list(map(lambda x: "target" + str(x), range(len(concat_sub.columns))))

concat_sub.columns = cols
concat_sub.head()
concat_sub.reset_index(inplace=True)
concat_sub.head()
ncol = concat_sub.shape[1]

ncol
# get the data fields ready for stacking

concat_sub['target_mean'] = concat_sub.iloc[:, 1:ncol].mean(axis=1)

concat_sub['target_median'] = concat_sub.iloc[:, 1:ncol].median(axis=1)

concat_sub.head()
concat_sub.describe()
# Create 1st submission for the mean with round

# concat_sub['target_mean'].round()

col2=['record_ID','units_sold']
# concat_sub['target'] = concat_sub['target_mean']



concat_sub['target'] = concat_sub['target_mean'].round()

data=concat_sub[['record_ID', 'target']]

data.columns=col2
data.head()
# data.to_csv('submission_mean.csv', index=False, float_format='%.6f')



data.to_csv('submission_mean_round.csv', index=False, float_format='%.6f')
# Create 1st submission for the median
data_med=data
data_med['units_sold']=concat_sub['target_median']

data_med
data_med.to_csv('submission_median.csv', index=False, float_format='%.6f')
data['units_sold']= 1.85/3 * data['units_sold'] + 1.15/3 * data_med['units_sold']
data.to_csv('submission_mean_median_blend_2.csv', index=False, float_format='%.6f')