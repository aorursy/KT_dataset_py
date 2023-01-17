#!pip install category_encoders
#!pip install bayes_opt
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from category_encoders.ordinal import OrdinalEncoder
from category_encoders.woe import WOEEncoder
from category_encoders.target_encoder import TargetEncoder
from category_encoders.sum_coding import SumEncoder
from category_encoders.m_estimate import MEstimateEncoder
from category_encoders.leave_one_out import LeaveOneOutEncoder
from category_encoders.helmert import HelmertEncoder
from category_encoders.cat_boost import CatBoostEncoder
from category_encoders.james_stein import JamesSteinEncoder
from category_encoders.one_hot import OneHotEncoder
from scipy.special import boxcox1p
#from bayes_opt import BayesianOptimization
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train = pd.read_csv('/kaggle/input/demand-forecasting/train_0irEZ2H.csv')
test = pd.read_csv('/kaggle/input/demand-forecasting/test_nfaJ3J5.csv')
print(train.shape)
train.head()
train.info()
train.describe()
train['units_sold'].describe()
train.isnull().sum()
sns.distplot(train["units_sold"])
print("skewness = ", train['units_sold'].skew())
cor = train.corr()
plt.figure(figsize= (15,12))
sns.heatmap(cor, annot = True,)
plt.subplot(1,2)
sns.distplot()
plt.hist()
plt.boxplot()
plt.scatter()
import statsmodels.api as sm 
sm.tsa.seasonal_decompose(train.units_sold).plot() 
result = sm.tsa.stattools.adfuller(train.units_sold) 
plt.show()
#Imputing missing value with the relevant total price
train.total_price=train.total_price.fillna(469.5375)
#New Feature Creation functions

def gen_count_id(train,test,col,name):
    temp=train.groupby(col)['record_ID'].count().reset_index().rename(columns={'record_ID':name})
    train=pd.merge(train,temp,how='left',on=col)
    test=pd.merge(test,temp,how='left',on=col)
    train[name]=train[name].astype(float)
    test[name]=test[name].astype(float)
    train[name].fillna(np.median(temp[name]),inplace=True)
    test[name].fillna(np.median(temp[name]),inplace=True)
    return train,test

def gen_average_units(train,test,col,name):
    temp=train.groupby(col)['units_sold'].mean().reset_index().rename(columns={'units_sold':name})
    train=pd.merge(train,temp,how='left',on=col)
    test=pd.merge(test,temp,how='left',on=col)
    train[name].fillna(np.median(temp[name]),inplace=True)
    test[name].fillna(np.median(temp[name]),inplace=True)
    return train,test

def gen_average_price(train,test,col,price='base_price',name='name'):
    temp=train.groupby(col)[price].mean().reset_index().rename(columns={price:name})
    train=pd.merge(train,temp,how='left',on=col)
    test=pd.merge(test,temp,how='left',on=col)
    train[name].fillna(np.median(temp[name]),inplace=True)
    test[name].fillna(np.median(temp[name]),inplace=True)
    return train,test
train,test = gen_count_id(train,test,col=['sku_id','store_id'],name='count_id_sku_store') #Genearting count of records per 'sku-id & store-id' 
#train,test = gen_count_id(train,test,col=['sku_id','store_id'],name='count_id_sku_store') #Genearting count of records per 'sku-id & store-id' 
train,test = gen_count_id(train,test,col=['sku_id'],name='count_id_sku') #Genearting count of records per 'sku-id'
train,test = gen_count_id(train,test,col=['store_id'],name='count_id_store') #Genearting count of records per 'store-id'
train,test = gen_average_units(train,test,col=['sku_id','store_id'],name='count_sku_store_id') #Genearting average units sold per 'sku-id & store-id'
train,test = gen_average_units(train,test,col=['store_id'],name='count_store_id') #Genearting average units sold per 'store-id'
train,test = gen_average_units(train,test,col=['sku_id'],name='count_sku_id') #Genearting average units sold per 'sku-id'

train,test = gen_average_price(train,test,col=['sku_id','store_id'],price='base_price',name='price_sku_store') #Genearting average base price per 'sku-id & store-id'
train,test = gen_average_price(train,test,col=['sku_id','store_id'],price='total_price',name='price_to_sku_store') #Genearting average total price per 'sku-id & store-id'
train,test = gen_average_price(train,test,col=['store_id'],price='base_price',name='price_store_id') #Genearting average base price per 'store-id'
train,test = gen_average_price(train,test,col=['sku_id'],price='base_price',name='price_sku_id') #Genearting average base price per 'sku-id'
train,test = gen_average_price(train,test,col=['store_id'],price='total_price',name='price_to_store_id') #Genearting average total price per 'store-id'
train,test = gen_average_price(train,test,col=['sku_id'],price='total_price',name='price_to_sku_id') #Genearting average total price per 'sku-id'
train
#Converting week feature
le = OrdinalEncoder()
train['week_1']=le.fit_transform(train['week'])
le = OrdinalEncoder()
test['week_1']=le.fit_transform(test['week'])+130
train
