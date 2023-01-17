# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow import keras



from datetime import datetime, date

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



train = pd.read_csv('../input/competitive-data-science-predict-future-sales/sales_train.csv')

test = pd.read_csv('../input/competitive-data-science-predict-future-sales/test.csv')

sample_sub = pd.read_csv('../input/competitive-data-science-predict-future-sales/sample_submission.csv')

items = pd.read_csv('../input/competitive-data-science-predict-future-sales/items.csv')

item_cats = pd.read_csv('../input/competitive-data-science-predict-future-sales/item_categories.csv')

shops = pd.read_csv('../input/competitive-data-science-predict-future-sales/shops.csv')



print("----------Top-10- Record----------")

print(train.head(10))

print("-----------Information-----------")

print(train.info())

print("-----------Data Types-----------")

print(train.dtypes)

print("----------Missing value-----------")

print(train.isnull().sum())

print("----------Null value-----------")

print(train.isna().sum())

print("----------Shape of Data----------")

print(train.shape)



train['date'] = pd.to_datetime(train['date'], format = '%d.%m.%Y')



sum(train['item_cnt_day'] == 0)



## Assume the item_cnt_day is zero if it is not recorded



train.head()



dataset = train.pivot_table(index = ['shop_id','item_id'],values = ['item_cnt_day'],columns = ['date_block_num'],fill_value = 0, aggfunc=np.sum)



dataset.reset_index(inplace = True)

dataset.head()



# Now we will merge our pivot table with the test_data because we want to keep the data of items we have

# predict

dataset_left = pd.merge(test,dataset,on = ['item_id','shop_id'],how = 'left')



dataset_left.fillna(0, inplace= True)

dataset_left.head()



dataset_left.drop(['shop_id','item_id','ID'],inplace = True, axis = 1)

dataset_left.head()



X_train = np.expand_dims(dataset_left.values[:,:-1],axis = 2)

y_train = dataset_left.values[:,-1:]

X_test = np.expand_dims(dataset_left.values[:,1:],axis = 2)



print(X_train.shape,y_train.shape,X_test.shape)



model = keras.models.Sequential([

    keras.layers.LSTM(64, return_sequences=True, input_shape=[33, 1]),

    keras.layers.LSTM(64),

    keras.layers.Dropout(0.4),

    keras.layers.Dense(1)

])



model.compile(loss="mse", optimizer="adam", metrics=['mean_squared_error'])

history = model.fit(X_train,y_train, batch_size=256, epochs=10)



# creating submission file 

submission_pfs = model.predict(X_test)

# we will keep every value between 0 and 20

submission_pfs = submission_pfs.clip(0,20)

# creating dataframe with required columns 

submission = pd.DataFrame({'ID':test['ID'],'item_cnt_month':submission_pfs.ravel()})

# creating csv file from dataframe

submission.to_csv('sub_pfs.csv',index = False)