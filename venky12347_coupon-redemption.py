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
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

from datetime import datetime

from catboost import CatBoostClassifier

import catboost as cb
train_data = pd.read_csv("/kaggle/input/amexpert-2019/train.csv")

cust_demo_data = pd.read_csv("/kaggle/input/amexpert-2019/customer_demographics.csv")

campaign_data = pd.read_csv("/kaggle/input/amexpert-2019/campaign_data.csv")

coupon_item_mapping = pd.read_csv("/kaggle/input/amexpert-2019/coupon_item_mapping.csv")

customer_transaction_data = pd.read_csv("/kaggle/input/amexpert-2019/customer_transaction_data.csv")

item_data = pd.read_csv("/kaggle/input/amexpert-2019/item_data.csv")
cust_demo_data["no_of_children"].fillna(0, inplace = True) 

cust_demo_data["marital_status"] = np.where(cust_demo_data["marital_status"].isna(),

                                            np.where(cust_demo_data["no_of_children"]==0,"Single","Married"),

         cust_demo_data["marital_status"])
train_data.info()
train_data_merge = pd.merge(train_data,campaign_data,how="inner",on="campaign_id")
train_data_merge = pd.merge(train_data_merge,coupon_item_mapping,how="inner",on="coupon_id")
train_data_merge = pd.merge(train_data_merge,item_data,how="inner",on="item_id")
train_data_merge = pd.merge(train_data_merge,cust_demo_data,how="left",on="customer_id")
pd.unique(train_data_merge[['id','campaign_id','coupon_id',

                            'customer_id','redemption_status','item_id']].values.ravel())
train_data_merge.info()
train_data_merge.drop_duplicates(keep="first", inplace=True)
train_data_merge.info()
customer_transaction_data.info()
train_data_merge = pd.merge(train_data_merge,customer_transaction_data,how="left",on=['customer_id','item_id'])
train_data_merge.info()
train_data_merge.drop_duplicates(keep="first", inplace=True)
train_data_merge.info()
train_data_merge['quantity'].fillna(0,inplace=True)
train_data_merge['other_discount'].fillna(0,inplace=True)
train_data_merge['coupon_discount'].fillna(0,inplace=True)
train_data_merge.info()
train_data_merge.columns
columns = ['campaign_type','brand_type', 'category', 

           'quantity','other_discount', 'coupon_discount','marital_status',

           'rented','income_bracket','age_range','brand']

x_data = train_data_merge[columns]

y_data = train_data_merge['redemption_status']
x_data.isnull().sum()
x_data['age_range'].fillna('100+',inplace=True)

x_data['marital_status'].fillna('XX',inplace=True)

x_data['rented'].fillna(2,inplace=True)

x_data['income_bracket'].fillna(99,inplace=True)
from catboost import CatBoostClassifier

categorical_var = np.where(x_data.dtypes != np.float)[0]

print('\nCategorical Variables indices : ',categorical_var)
cat_model = CatBoostClassifier(iterations=50,learning_rate=0.7)

cat_model.fit(x_data,y_data,cat_features = categorical_var,plot=False)
y_pred = cat_model.predict(x_data)

from sklearn.metrics import confusion_matrix,accuracy_score

confusion_matrix = confusion_matrix(y_data, y_pred)

print(confusion_matrix)

print("Accuracy:",accuracy_score(y_data, y_pred))
test_data = pd.read_csv("/kaggle/input/amexpert-2019/test_QyjYwdj.csv")
test_data_merge = pd.merge(test_data,campaign_data,how="inner",on="campaign_id")

test_data_merge = pd.merge(test_data_merge,coupon_item_mapping,how="inner",on="coupon_id")

test_data_merge = pd.merge(test_data_merge,item_data,how="inner",on="item_id")

test_data_merge = pd.merge(test_data_merge,cust_demo_data,how="left",on="customer_id")
test_data_merge = pd.merge(test_data_merge,customer_transaction_data,how="left",on=['customer_id','item_id'])
test_data_merge.drop_duplicates(keep="first", inplace=True)
test_data_merge['quantity'].fillna(0,inplace=True)

test_data_merge['other_discount'].fillna(0,inplace=True)

test_data_merge['coupon_discount'].fillna(0,inplace=True)

test_data_merge['age_range'].fillna('100+',inplace=True)

test_data_merge['marital_status'].fillna('XX',inplace=True)

test_data_merge['rented'].fillna(2,inplace=True)

test_data_merge['income_bracket'].fillna(99,inplace=True)
x_test_data = test_data_merge[columns]

#y_data = train_data_merge['redemption_status']
y_pred = cat_model.predict(x_test_data)
test_data_merge['redemption_status'] = y_pred
columns = ['id','redemption_status']

submission_data = test_data_merge[columns]
final_submit_data = submission_data.groupby("id").max()["redemption_status"]
final_submit_data.to_csv('submission.csv', index=False)