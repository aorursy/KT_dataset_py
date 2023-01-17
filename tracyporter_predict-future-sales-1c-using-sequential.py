#load modules

import os

import pandas as pd

import numpy as np

from pandas import read_csv

from pandas import datetime

from keras.models import Sequential

from keras.layers import LSTM,Dense,Dropout
#loading data 

os.listdir('../input')

item_cat = pd.read_csv('../input/item_categories.csv')

items = pd.read_csv('../input/items.csv')

shops = pd.read_csv('../input/shops.csv')

sample_submission = pd.read_csv('../input/sample_submission.csv')
df_train = pd.read_csv('../input/sales_train.csv')

df_train
df_train.info()
df_train.isnull().sum().sum()
df_train.info()
df_train.describe()
#convert date to datetime format

df_train['date'] = pd.to_datetime(df_train['date'],format = '%d.%m.%Y')
#create pivot table

dataset = df_train.pivot_table(index = ['shop_id','item_id'],values = ['item_cnt_day'],columns = ['date_block_num'],fill_value = 0,aggfunc='sum')

dataset.reset_index(inplace = True)

dataset
#Load test set

df_test = pd.read_csv('../input/test.csv')

df_test
#merge pivot table with test set

dataset = pd.merge(df_test,dataset,on = ['item_id','shop_id'],how = 'left')
#check for any null values

dataset.isnull().sum().sum()
#fill all NaN values with 0

dataset.fillna(0,inplace = True)

dataset.isnull().sum().sum()
dataset.info()
#drop shop_id and item_id

dataset.drop(['shop_id','item_id','ID'],inplace = True, axis = 1)

dataset
#split the dataset in two

#keep all columns execpt the last one 

X_train = np.expand_dims(dataset.values[:,:-1],axis = 2)

# the last column is our label

y_train = dataset.values[:,-1:]

# for test we keep all the columns execpt the first one

X_test = np.expand_dims(dataset.values[:,1:],axis = 2)

# lets have a look on the shape 

print(X_train.shape,y_train.shape,X_test.shape)
# create sequential model

my_model = Sequential()

my_model.add(LSTM(units = 64,input_shape = (33,1)))

my_model.add(Dropout(0.4))

my_model.add(Dense(1))

my_model.compile(loss = 'mse',optimizer = 'adam', metrics = ['mean_squared_error'])

my_model.summary()
#fit the model

my_model.fit(X_train,y_train,batch_size = 4096,epochs = 10)
# creating submission file 

submission_file = my_model.predict(X_test)

# we will keep every value between 0 and 20

submission_file = submission_file.clip(0,20)

# creating dataframe with required columns 

submission_trp = pd.DataFrame({'ID':df_test['ID'],'item_cnt_month':submission_file.ravel()})

# creating csv file from dataframe

submission_trp.to_csv('submission.csv',index = False)

submission_trp