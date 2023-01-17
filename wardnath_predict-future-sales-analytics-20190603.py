# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

from keras.models import Sequential

from keras.layers import LSTM,Dense,Dropout

from keras import optimizers





import pandas as pd

import numpy as np

import gc

import matplotlib.pyplot as plt

%matplotlib inline 



pd.set_option('display.max_rows', 600)

pd.set_option('display.max_columns', 50)



import lightgbm as lgb

from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score

from tqdm import tqdm_notebook



from itertools import product



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
input_dir = '../input/'

pd.set_option('display.max_colwidth', 20)

pd.set_option('display.max_columns', 500)

shops = pd.read_csv(input_dir + 'shops.csv')

print("Shops:\n", shops.head(), '\n', shops.shape, '\n')

item_cat = pd.read_csv(input_dir + 'item_categories.csv')

print("item_categories: \n", item_cat.head(), '\n', item_cat.shape, '\n')

sales_data = pd.read_csv(input_dir + 'sales_train.csv')

print("sales_train: \n", sales_data.head(), '\n', sales_data.shape, '\n')

items = pd.read_csv(input_dir + 'items.csv')

print("items: \n", items.head(), '\n', items.shape, '\n')

sample_submission = pd.read_csv(input_dir + 'sample_submission.csv')

print("sample_submission: \n", sample_submission.head(), '\n', sample_submission.shape, '\n')

test_data = pd.read_csv(input_dir + 'test.csv')

print("test: \n", test_data.head(), '\n', test_data.shape, '\n')
def simple_eda(df):

    print("----------TOP 5 RECORDS--------")

    print(df.head(5))

    print("----------INFO-----------------")

    print(df.info())

    print("----------Describe-------------")

    print(df.describe())

    print("----------Columns--------------")

    print(df.columns)

    print("----------Data Types-----------")

    print(df.dtypes)

    print("-------Missing Values----------")

    print(df.isnull().sum())

    print("-------NULL values-------------")

    print(df.isna().sum())

    print("-----Shape Of Data-------------")

    print(df.shape)
print("Sales Data -------------------------> \n")

simple_eda(sales_data)

print("Test data -------------------------> \n")

simple_eda(test_data)

print("Item Categories -------------------------> \n")

simple_eda(item_cat)

print("Items -------------------------> \n")

simple_eda(items)

print("Shops -------------------------> \n")

simple_eda(shops)

print("Sample Submission -------------------------> \n")

simple_eda(sample_submission)
sales_data['date'] = pd.to_datetime(sales_data['date'],format = '%d.%m.%Y')

dataset = sales_data.pivot_table(index = ['shop_id','item_id'],values = ['item_cnt_day'],columns = ['date_block_num'],fill_value = 0)
dataset.reset_index(inplace = True)

dataset = pd.merge(test_data,dataset,on = ['item_id','shop_id'],how = 'left')

dataset.fillna(0,inplace = True)

dataset.drop(['shop_id','item_id','ID'],inplace = True, axis = 1)

print(dataset.head())
# X we will keep all columns execpt the last one 

X_train = np.expand_dims(dataset.values[:,:-1],axis = 2)

# the last column is our label

y_train = dataset.values[:,-1:]



# for test we keep all the columns execpt the first one

X_test = np.expand_dims(dataset.values[:,1:],axis = 2)



# lets have a look on the shape 

print(X_train.shape,y_train.shape,X_test.shape)
# our defining our model 

my_model = Sequential()

my_model.add(LSTM(units = 256,input_shape = (33,1)))

my_model.add(Dropout(0.5))

my_model.add(Dense(1))



sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

my_model.compile(loss = 'mse',optimizer = sgd, metrics = ['mean_squared_error'])

my_model.summary()
my_model.fit(X_train,y_train,batch_size = 1000,epochs = 5)

# creating submission file 

submission_pfs = my_model.predict(X_test)

# we will keep every value between 0 and 20

submission_pfs = submission_pfs.clip(0,20)

# creating dataframe with required columns 

submission = pd.DataFrame({'ID':test_data['ID'],'item_cnt_month':submission_pfs.ravel()})

# creating csv file from dataframe

submission.to_csv('sub_pfs.csv',index = False)