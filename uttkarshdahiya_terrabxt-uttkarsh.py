import pandas as pd

import numpy as np

import matplotlib.pylab as plt

import seaborn as sns

import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler

from tensorflow import keras

Random_seed = 42

np.random.seed(Random_seed)

tf.random.set_seed(Random_seed)
df = pd.read_csv('../input/into-the-future/train.csv',index_col = [1],parse_dates = [1])

df_test = pd.read_csv('../input/into-the-future/test.csv',index_col = [1],parse_dates = [1])
df = df.drop(['id'],axis = 1)

df.dtypes
test = df_test

test.dtypes
df.isnull().values.any()
df.head()

df['hour'] = df.index.hour

df['minute'] = df.index.minute

df['second'] = df.index.second

df.describe()

test['hour'] = test.index.hour

test['minute'] = test.index.minute

test['second'] = test.index.second

test.describe()
sns.lineplot(x=df.index,y='feature_2',data =df)

sns.set(rc={'figure.figsize':(12,8)})
sns.lineplot(x=df.index,y='feature_1',data =df)

sns.set(rc={'figure.figsize':(12,8)})
#df = df.iloc[1:]
df.head()
train = df

train_size = len(df)

test_size = len(test)



print(len(train), len(test))
scaler = MinMaxScaler()
scaler.fit(train[['feature_1']].to_numpy())
scaled_train_f1 = scaler.transform(train[['feature_1']].to_numpy())

scaled_test_f1 = scaler.transform(test[['feature_1']].to_numpy())



second_square = np.square(train[['second']])

test_second_square = np.square(test[['second']])



train_feature = np.c_[scaled_train_f1,train[['hour']],train[['minute']],train[['second']]]

test_feature = np.c_[scaled_test_f1,test[['hour']],test[['minute']],test[['second']]]

scaler.fit(train[['feature_2']].to_numpy())

scaled_train_f2 = scaler.transform(train[['feature_2']].to_numpy())
model = keras.Sequential([

    keras.layers.Flatten(input_shape=(4,)),

    keras.layers.Dense(180, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),

    keras.layers.Dropout(0.2),

    keras.layers.Dense(180, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),

    keras.layers.Dropout(0.2),

    

    keras.layers.Dense(1, activation='sigmoid', kernel_regularizer=keras.regularizers.l2(0.01)),

])
model.compile(optimizer='adam',

              loss='mse'

              )
hist = model.fit(train_feature, scaled_train_f2, shuffle=False,

          batch_size=32, 

                 epochs=100,validation_split=0.1)
plt.plot(hist.history['loss'])

plt.plot(hist.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Val'], loc='upper right')

plt.show()
prediction = model.predict(test_feature)
prediction.shape
feature_2 = scaler.inverse_transform(prediction)
ids = test[['id']].to_numpy()

ids = ids.flatten()



feature_2 = feature_2.flatten()

d = {'id': ids, 'feature_2': feature_2}

submission = pd.DataFrame(data = d)

submission.to_csv('submission.csv',index=False)
submission