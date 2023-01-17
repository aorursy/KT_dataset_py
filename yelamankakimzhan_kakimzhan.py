# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, GRU
import seaborn as sns
import matplotlib.pyplot as plt
submission=pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/sample_submission.csv")
train=pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/sales_train.csv")
test=pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/test.csv")
items=pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/items.csv")
item_cat=pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/item_categories.csv")
shop=pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/shops.csv")
print("Shape of train:", train.shape)
print("Shape of test:",test.shape)

train.head()
train.info()
train["date"] = pd.to_datetime(train["date"],format='%d.%M.%Y')
item_cat.head()
items.head()
shop.head()

train[train['item_price'] < 0].count()
train[train['item_cnt_day'] < 0].count()
train = train[(train['item_price'] > 0) & (train['item_cnt_day'] > 0)]
df = train.pivot_table(index=['item_id','shop_id'], columns=['date_block_num'], values=['item_cnt_day'], fill_value=0) 
df
df1 = pd.merge(test, df, on=['item_id', 'shop_id'], how='left')
df1.fillna(0, inplace=True)
df1.head()
df1.drop(['ID', 'shop_id', 'item_id'], axis=1, inplace=True)
X_train = np.expand_dims(df1.values[:, :-1], axis=2) 
y_train = df1.values[:, -1:] 
X_test = np.expand_dims(df1.values[:, 1:], axis=2)
model = Sequential()
model.add(GRU(units=128, return_sequences=True,input_shape=(33,1)))
model.add(Dropout(0.3))
model.add(GRU(units=32))
model.add(Dense(1))

model.compile(loss='mse',
              optimizer='adam',
              metrics=['mean_squared_error'])
model.summary()
reg = model.fit(X_train, y_train, batch_size=512, epochs=10)
LSTM_prediction = model.predict(X_test)
submission = pd.DataFrame()
submission['ID'] = test['ID']
submission['Predict'] = LSTM_prediction.ravel()
submission.to_csv('submission.csv',index=False)
len(LSTM_prediction)
submission.head()


