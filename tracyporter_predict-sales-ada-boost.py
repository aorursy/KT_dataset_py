#load modules

import os

import pandas as pd

import numpy as np

from pandas import read_csv

from pandas import datetime



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



#Load datasets

train=pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/sales_train.csv")

test=pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/test.csv")

sample=pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/sample_submission.csv")

items=pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/items.csv")

item_cat=pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/item_categories.csv")

shops=pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/shops.csv")
#convert date to datetime format

train['date'] = pd.to_datetime(train['date'],format = '%d.%m.%Y')

train
#create pivot table

dataset = train.pivot_table(index = ['shop_id','item_id'],values = ['item_cnt_day'],columns = ['date_block_num'],fill_value = 0,aggfunc='sum')

dataset.reset_index(inplace = True)

dataset
ID = test.ID

ID
test = test.drop(['ID'], axis=1)

test
#merge pivot table with test set

dataset = pd.merge(test,dataset,on = ['item_id','shop_id'],how = 'left')

dataset
#check for any null values

dataset.isnull().sum().sum()
#fill all NaN values with 0

dataset.fillna(0,inplace = True)

dataset.isnull().sum().sum()
#drop shop_id and item_id

dataset.drop(['shop_id','item_id'],inplace = True, axis = 1)

dataset
#split the dataset in two

# the last column is our label

y_train = dataset.iloc[:,-1:]

#drop last column of data

X_train = dataset.iloc[:, :-1]

#drop first colum of data

X_test = dataset.iloc[:,1:]

# lets have a look on the shape 

print(X_train.shape,y_train.shape,X_test.shape)
from sklearn.ensemble import AdaBoostRegressor

from sklearn.ensemble import RandomForestRegressor



model = AdaBoostRegressor(base_estimator = RandomForestRegressor(max_depth=10), random_state=0, n_estimators=3000).fit(X_train, y_train)

y_pred = model.predict(X_train)

y_pred
print(model.score(X_train, y_train))
pred = model.predict(X_test)

pred
# creating submission file 

submission_file = pred

# we will keep every value between 0 and 20

submission_file = submission_file.clip(0,20)

# creating dataframe with required columns 

submission_trp = pd.DataFrame({'ID':ID,'item_cnt_month':submission_file.ravel()})

# creating csv file from dataframe

submission_trp.to_csv('submission.csv',index = False)

submission_trp.head(20)