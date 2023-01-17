# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from tensorflow import keras

from tensorflow.keras import Sequential

from tensorflow.keras.layers import Dense

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/mockdatadec/MOCK_DATA_DEC.csv')

test = pd.read_csv('../input/mockedtest/MOCK_TEST.csv')
train[train['x1'] == 0.01] = None

train[train['x2'] == 0.01] = None

test[test['x1'] == 0.01] = None

test[test['x2'] == 0.01] = None

train[train['x1'] == 0.00] = None

train[train['x2'] == 0.00] = None

test[test['x1'] == 0.00] = None

test[test['x2'] == 0.00] = None

train[train['x1'] == -0.01] = None

train[train['x2'] == -0.01] = None

test[test['x1'] == -0.01] = None

test[test['x2'] == -0.01] = None
#train['id'] = train.id.astype('int')
train.loc[(train.x1 > 0) & (train.x2 < 0), 'target'] = 1

train.loc[(train.x1 < 0) & (train.x2 > 0), 'target'] = 1

train.loc[train.target != 1, 'target'] = 0
train.head(10)
train.describe()
x = train.x1

y = train.x2
target = train.target
x1 = train.loc[train.target > 0].x1

y1 = train.loc[train.target > 0].x2

x2 = train.loc[train.target == 0].x1

y2 = train.loc[train.target == 0].x2

plt.plot(x1,y1,'o',color='orange')

plt.plot(x2,y2,'bo')

plt.show()
test.head(10)
x_test = test.x1

y_test = test.x2
plt.plot(x_test,y_test, 'ro')

plt.show()
train.isna().sum()
test.isna().sum()
train.dropna(0, inplace=True)

test.dropna(0, inplace=True)

print('done')
test.isna().sum()
test['id'] = test.id.astype('int')
target = train.target
del train['target'], train['id']
x_train, x_validation, y_train, y_validation = train_test_split(train.values, target.values, test_size=0.2, random_state=42)
optimAdam = keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
model = Sequential()

model.add(Dense(4,activation='relu'))

model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer=optimAdam, metrics=['accuracy'])
bst_model_path = 'model.h5'

model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True)
history = model.fit(x_train, y_train, validation_data=([x_validation], y_validation), epochs=50, batch_size=50, callbacks=[model_checkpoint], verbose=2)

plt.plot(history.history['val_loss'])

plt.plot(history.history['loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
model.load_weights('./model.h5')
test_ids = test.id
del test['id']
preds = model.predict_classes(test, batch_size=10, verbose=2)
test['target'] = preds
print(test.head(10))
x1_test = test.loc[test.target > 0].x1

y1_test = test.loc[test.target > 0].x2

x2_test = test.loc[test.target == 0].x1

y2_test = test.loc[test.target == 0].x2
old_x1 = x1_test

old_y1 = y1_test

old_x2 = x2_test

old_y2 = y2_test
plt.figure(figsize=(9, 4))

plt.subplot(121)

plt.title('Predicted test')

plt.plot(x1_test,y1_test,'o',color='orange')

plt.plot(x2_test,y2_test,'bo')

plt.subplot(122)

plt.title('Train dataset')

plt.plot(x1,y1,'o',color='orange')

plt.plot(x2,y2,'bo')

plt.show()
train['x1x2'] = train['x1'] * train['x2']

test['x1x2'] = test['x1'] * test['x2']
train.head(10)
x_train, x_validation, y_train, y_validation = train_test_split(train.values, target.values, test_size=0.2, random_state=4)
model = Sequential()

model.add(Dense(4,activation='relu'))

model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer=optimAdam, metrics=['accuracy'])

bst_model_path = 'model.h5'

model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True)
history = model.fit(x_train, y_train, validation_data=([x_validation], y_validation), epochs=50, batch_size=50, callbacks=[model_checkpoint], verbose=2)
plt.plot(history.history['val_loss'])

plt.plot(history.history['loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
del test['target']
model.load_weights('./model.h5')

preds = model.predict_classes(test, batch_size=10, verbose=2)

test['target'] = preds
x1_test = test.loc[test.target > 0].x1

y1_test = test.loc[test.target > 0].x2

x2_test = test.loc[test.target == 0].x1

y2_test = test.loc[test.target == 0].x2
plt.figure(figsize=(9, 4))

plt.subplot(121)

plt.title('New Results')

plt.plot(x1_test,y1_test,'o',color='orange')

plt.plot(x2_test,y2_test,'bo')

plt.subplot(122)

plt.title('Old results')

plt.plot(old_x1,old_y1,'o',color='orange')

plt.plot(old_x2,old_y2,'bo')

plt.show()