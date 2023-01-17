import numpy as np

import pandas as pd



default_path = '../input/'
train_df = pd.read_csv(default_path+'sales_train.csv')

items_df = pd.read_csv(default_path+'items.csv')

test_df = pd.read_csv(default_path+'test.csv')



print(train_df.shape, test_df.shape)
train_df['date'] = pd.to_datetime(train_df['date'], format='%d.%m.%Y')
train_df.head()
dataset = train_df.pivot_table(index=['item_id', 'shop_id'],values=['item_cnt_day'], columns='date_block_num', fill_value=0)
dataset = dataset.reset_index()

dataset.head()
dataset = pd.merge(test_df, dataset, on=['item_id', 'shop_id'], how='left')

dataset = dataset.fillna(0)

dataset.head()
dataset = dataset.drop(['shop_id', 'item_id', 'ID'], axis=1)

dataset.head()
X_train = np.expand_dims(dataset.values[:, :-1], axis=2)

y_train = dataset.values[:, -1:]



X_test = np.expand_dims(dataset.values[:, 1:], axis=2)

print(X_train.shape, y_train.shape, X_test.shape)
from keras.models import Sequential

from keras.layers import LSTM, Dense, Dropout
model = Sequential()

model.add(LSTM(64, input_shape=(33, 1)))

model.add(Dropout(0.3))

model.add(Dense(1))



model.compile(loss='mse',

             optimizer='adam',

             metrics=['mean_squared_error'])



model.summary()
model_fitting = model.fit(X_train, y_train, batch_size=4096, epochs=10)
model_prediction = model.predict(X_test)

model_prediction = model_prediction.clip(0, 20)
submission = pd.DataFrame({'ID': test_df['ID'], 'item_cnt_month': model_prediction.ravel()})

submission.to_csv('submission.csv',index=False)