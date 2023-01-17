# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import featuretools as ft
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import pickle
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
new = pd.read_csv('../input/new_merchant_transactions.csv',parse_dates =["purchase_date"])
his = pd.read_csv('../input/historical_transactions.csv',parse_dates =["purchase_date"])
train = pd.read_csv( '../input/train.csv',parse_dates =["first_active_month"])
test = pd.read_csv( '../input/test.csv',parse_dates =["first_active_month"])
merchants = pd.read_csv( '../input/merchants.csv')
train.shape
train_sub = train.loc[np.random.choice(train.index, 10000, replace=False)]
merchants = merchants.drop_duplicates(subset="merchant_id",keep="first")
print(train.shape[0]==train["card_id"].nunique())
print(merchants.shape[0]==merchants["merchant_id"].nunique())
#print(merchants[merchants["merchant_id"].duplicated()])
new_drop=new.drop(['purchase_date',"merchant_category_id",
                                                   "subsector_id","city_id",
                                                   "state_id"], axis=1)
his_drop=his.drop(['purchase_date',"merchant_category_id",
                                                   "subsector_id","city_id",
                                                   "state_id"], axis=1)
mer_drop = merchants.drop(['merchant_group_id',"merchant_category_id",
                                                   "subsector_id","most_recent_sales_range",
                                                   "most_recent_purchases_range",
                           'city_id','state_id'], axis=1)
train_his=train_sub[["card_id"]].merge(his_drop, how='left', on="card_id")
print(train_his.shape)
train_his.head()
#train_his_transactions=train_his_transactions.dropna()
train_his_sub=train_his
#train_his_sub=train_his.loc[np.random.choice(train_his.index, 1000000, replace=False)]
print(train_his_sub.shape)
train_his_sub.head()
authorized_flag = pd.get_dummies(train_his_sub['authorized_flag'])
authorized_flag.columns = ['authorized_flag_N', 'authorized_flag_Y']
train_his_sub=train_his_sub.drop(['authorized_flag'], axis=1)
train_his_sub=pd.concat([train_his_sub, authorized_flag], axis=1)
train_his_sub.head()
category_1 = pd.get_dummies(train_his_sub['category_1'])
category_1.head()
category_1.columns = ['category_1_N', 'category_1_Y']
train_his_sub=train_his_sub.drop(['category_1'], axis=1)
train_his_sub=pd.concat([train_his_sub, category_1], axis=1)
train_his_sub.head()
category_2 = pd.get_dummies(train_his_sub['category_2'])
#category_2.head()
category_2.columns = ['category_2_1', 'category_2_2',"category_2_3","category_2_4","category_2_5"]
train_his_sub=train_his_sub.drop(['category_2'], axis=1)
train_his_sub=pd.concat([train_his_sub, category_2], axis=1)
train_his_sub.head()
category_3 = pd.get_dummies(train_his_sub['category_3'])
#category_3.head()
category_3.columns = ['category_3_A', 'category_3_B',"category_3_C"]
train_his_sub=train_his_sub.drop(['category_3'], axis=1)
train_his_sub=pd.concat([train_his_sub, category_3], axis=1)
train_his_sub.head()
train_his_sub.columns
train_his_sub.columns.values[1:]=["his_trans_" + str(col) for col in list(train_his_sub)[1:]]

train_his_sub.columns.values[2] = "merchant_id"
train_his_sub.columns.values
mer_drop = merchants.drop(['merchant_group_id',"merchant_category_id",
                                                   "subsector_id","most_recent_sales_range",
                                                   "most_recent_purchases_range",
                           'city_id','state_id'], axis=1)
mer_drop.columns.values[1:]=["his_mer_" + str(col) for col in mer_drop.columns.values[1:]] 
mer_drop.columns
train_his_sub_mer = train_his_sub.merge(mer_drop,how="left",on='merchant_id')
train_his_sub_mer.shape
train_his_sub_mer.head()
category_1 = pd.get_dummies(train_his_sub_mer['his_mer_category_1'])
category_1.head()
category_1.columns = ['his_mer_category_1_N', 'his_mer_category_1_Y']
train_his_sub_mer=train_his_sub_mer.drop(['his_mer_category_1'], axis=1)
train_his_sub_mer=pd.concat([train_his_sub_mer, category_1], axis=1)
train_his_sub_mer.head()
category_2 = pd.get_dummies(train_his_sub_mer['his_mer_category_2'])
#category_2.head()
category_2.columns = ['his_mer_category_2_1', 'his_mer_category_2_2',"his_mer_category_2_3","his_mer_category_2_4","his_mer_category_2_5"]
train_his_sub_mer=train_his_sub_mer.drop(['his_mer_category_2'], axis=1)
train_his_sub_mer=pd.concat([train_his_sub_mer, category_2], axis=1)
train_his_sub_mer.head()
category_4 = pd.get_dummies(train_his_sub_mer['his_mer_category_4'])
#category_1.head()
category_4.columns = ['his_mer_category_4_N', 'his_mer_category_4_Y']
train_his_sub_mer=train_his_sub_mer.drop(['his_mer_category_4'], axis=1)
train_his_sub_mer=pd.concat([train_his_sub_mer, category_4], axis=1)
train_his_sub_mer.head()
with open('train_his_sub_mer.pickle', 'wb') as f:
    pickle.dump(train_his_sub_mer, f)
with open('train_his_sub_mer.pickle', 'rb') as f:
    train_his_sub_mer = pickle.load(f)
## extract history transactions record for training data

train_new=train_sub[["card_id"]].merge(new_drop, how='left', on="card_id")

print(train_new.shape)
train_new.head()

## **one hot encode** for train_his_sub : authorized_flag; category_1; category_2; category_3
train_new['authorized_flag']=train_new['authorized_flag'].fillna("N")
authorized_flag = pd.get_dummies(train_new['authorized_flag'])
#print(train_new['authorized_flag'].value_counts())
#print(train_new['authorized_flag'].isna().sum())
authorized_flag.columns = ['authorized_flag_N', 'authorized_flag_Y']
train_new=train_new.drop(['authorized_flag'], axis=1)
train_new=pd.concat([train_new, authorized_flag], axis=1)

train_new.head()

category_1 = pd.get_dummies(train_new['category_1'])
category_1.head()
category_1.columns = ['category_1_N', 'category_1_Y']
train_new=train_new.drop(['category_1'], axis=1)
train_new=pd.concat([train_new, category_1], axis=1)
train_new.head()

category_2 = pd.get_dummies(train_new['category_2'])
#category_2.head()
category_2.columns = ['category_2_1', 'category_2_2',"category_2_3","category_2_4","category_2_5"]
train_new=train_new.drop(['category_2'], axis=1)
train_new=pd.concat([train_new, category_2], axis=1)
train_new.head()

category_3 = pd.get_dummies(train_new['category_3'])
#category_3.head()
category_3.columns = ['category_3_A', 'category_3_B',"category_3_C"]
train_new=train_new.drop(['category_3'], axis=1)
train_new=pd.concat([train_new, category_3], axis=1)
train_new.head()

## rename column for merging simplicity

print(train_new.columns)

train_new.columns.values[1:]=["new_trans_" + str(col) for col in list(train_new)[1:]]


train_new.columns.values[2] = "merchant_id"

print(train_new.columns.values)


mer_drop = merchants.drop(['merchant_group_id',"merchant_category_id",
                                                   "subsector_id","most_recent_sales_range",
                                                   "most_recent_purchases_range",
                           'city_id','state_id'], axis=1)
mer_drop.columns.values[1:]=["new_mer_" + str(col) for col in mer_drop.columns.values[1:]] 
print(mer_drop.columns)

## merge with merchants (X : his transaction, Y : merchant)

train_new_mer = train_new.merge(mer_drop,how="left",on='merchant_id')
train_new_mer.shape

train_new_mer.head()

## **one hot encode** for train_his_sub_mer(merchants part) : category_1; category_2; category_3

category_1 = pd.get_dummies(train_new_mer['new_mer_category_1'])
#category_1.head()
category_1.columns = ['new_mer_category_1_N', 'new_mer_category_1_Y']
train_new_mer=train_new_mer.drop(['new_mer_category_1'], axis=1)
train_new_mer=pd.concat([train_new_mer, category_1], axis=1)
train_new_mer.head()

category_2 = pd.get_dummies(train_new_mer['new_mer_category_2'])
#category_2.head()
category_2.columns = ['new_mer_category_2_1', 'new_mer_category_2_2',"new_mer_category_2_3","new_mer_category_2_4","new_mer_category_2_5"]
train_new_mer=train_new_mer.drop(['new_mer_category_2'], axis=1)
train_new_mer=pd.concat([train_new_mer, category_2], axis=1)
train_new_mer.head()

category_4 = pd.get_dummies(train_new_mer['new_mer_category_4'])
#category_1.head()
category_4.columns = ['new_mer_category_4_N', 'new_mer_category_4_Y']
train_new_mer=train_new_mer.drop(['new_mer_category_4'], axis=1)
train_new_mer=pd.concat([train_new_mer, category_4], axis=1)
train_new_mer.head()

with open('train_new_mer.pickle', 'wb') as f:
    pickle.dump(train_new_mer, f)

with open('train_new_mer.pickle', 'rb') as f:
    train_new_mer = pickle.load(f)

train_new_mer.head()
train_new_mer.columns.values
train_new_mer.shape
def aggregate_new_trans(data):  
    agg_func = {
        'card_id': ['size'], #num_trans
        'new_trans_installments': ['sum', 'mean','median', 'max', 'min', 'std', 'nunique'],
        'merchant_id': ['nunique'],
        'new_trans_month_lag': ['mean', 'max', 'min', 'std', 'nunique'],
        'new_trans_purchase_amount': ['sum', 'mean', 'max', 'min', 'std', 'nunique'],
        'new_trans_authorized_flag_Y': ['mean'],
        'new_trans_category_1_Y': ['mean'],
        'new_trans_category_2_1': ['mean'],
        'new_trans_category_2_2': ['mean'],
        'new_trans_category_2_3': ['mean'],
        'new_trans_category_2_4': ['mean'],
        'new_trans_category_2_5': ['mean'],
        'new_trans_category_3_A': ['mean'],
        'new_trans_category_3_B': ['mean'],
        'new_trans_category_3_C': ['mean'],
        'new_mer_numerical_1':['mean'],
        'new_mer_numerical_2':['mean'],
        'new_mer_avg_sales_lag3':['mean'],
        'new_mer_avg_purchases_lag3':['mean'],
        'new_mer_active_months_lag3':['mean'], 
        'new_mer_avg_sales_lag6':['mean'],
        'new_mer_avg_purchases_lag6':['mean'],
        'new_mer_active_months_lag6':['mean'],
        'new_mer_avg_sales_lag12':['mean'],
        'new_mer_avg_purchases_lag12':['mean'],
        'new_mer_active_months_lag12':['mean'],
        'new_mer_category_1_Y':['mean'], 
        'new_mer_category_2_1':['mean'],
        'new_mer_category_2_2':['mean'], 
        'new_mer_category_2_3':['mean'],
        'new_mer_category_2_4':['mean'], 
        'new_mer_category_2_5':['mean'],
        'new_mer_category_4_Y':['mean']
    }    
    agg_trans = data.groupby(['card_id']).agg(agg_func)
    agg_trans.columns = ['_'.join(col).strip() for col in agg_trans.columns.values]
    agg_trans.reset_index(inplace=True)
    
    return agg_trans

#hist_sum = aggregate_trans(histdata, 'hist_')
new_sum = aggregate_new_trans(train_new_mer)
new_sum.head()
new_sum.shape
def aggregate_his_trans(data):  
    agg_func = {
        'card_id': ['size'], #num_trans
        'his_trans_installments': ['sum', 'mean','median', 'max', 'min', 'std', 'nunique'],
        'merchant_id': ['nunique'],
        'his_trans_month_lag': ['mean', 'max', 'min', 'std', 'nunique'],
        'his_trans_purchase_amount': ['sum', 'mean', 'max', 'min', 'std', 'nunique'],
        'his_trans_authorized_flag_Y': ['mean'],
        'his_trans_category_1_Y': ['mean'],
        'his_trans_category_2_1': ['mean'],
        'his_trans_category_2_2': ['mean'],
        'his_trans_category_2_3': ['mean'],
        'his_trans_category_2_4': ['mean'],
        'his_trans_category_2_5': ['mean'],
        'his_trans_category_3_A': ['mean'],
        'his_trans_category_3_B': ['mean'],
        'his_trans_category_3_C': ['mean'],
        'his_mer_numerical_1':['mean'],
        'his_mer_numerical_2':['mean'],
        'his_mer_avg_sales_lag3':['mean'],
        'his_mer_avg_purchases_lag3':['mean'],
        'his_mer_active_months_lag3':['mean'], 
        'his_mer_avg_sales_lag6':['mean'],
        'his_mer_avg_purchases_lag6':['mean'],
        'his_mer_active_months_lag6':['mean'],
        'his_mer_avg_sales_lag12':['mean'],
        'his_mer_avg_purchases_lag12':['mean'],
        'his_mer_active_months_lag12':['mean'],
        'his_mer_category_1_Y':['mean'], 
        'his_mer_category_2_1':['mean'],
        'his_mer_category_2_2':['mean'], 
        'his_mer_category_2_3':['mean'],
        'his_mer_category_2_4':['mean'], 
        'his_mer_category_2_5':['mean'],
        'his_mer_category_4_Y':['mean']
    }    
    agg_trans = data.groupby(['card_id']).agg(agg_func)
    agg_trans.columns = ['_'.join(col).strip() for col in agg_trans.columns.values]
    agg_trans.reset_index(inplace=True)
    
    return agg_trans

#hist_sum = aggregate_trans(histdata, 'hist_')
his_sum = aggregate_his_trans(train_his_sub_mer)
with open('new_sum.pickle', 'wb') as f:
    pickle.dump(new_sum, f)
with open('his_sum.pickle', 'wb') as f:
    pickle.dump(his_sum, f)
his_sum.shape
