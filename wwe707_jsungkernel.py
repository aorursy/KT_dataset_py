# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
THRESHOLD = 73 

RND_SEED = 56
####### 1. Import training data and extract ids #######

train_data = pd.read_csv('../input/train.csv', nrows=3e6)

raw_ids_all = train_data["Id"]

raw_ids = raw_ids_all.unique()
train_data.shape
for column in train_data.columns:

    train_data[column] = train_data[column].astype(np.float32)

train_data['Id'] = train_data['Id'].astype(np.int32)
####### 2. Remove ids with only NaNs in the "Ref" column #######

train_raw_tmp = train_data[~np.isnan(train_data.Ref)]

raw_ids_tmp = train_raw_tmp["Id"].unique()

train_new = train_data[np.in1d(raw_ids_all, raw_ids_tmp)]
train_new = train_new.fillna(0.0)

# train_new = train_new.reset_index(drop=True)
####### 4. Define and exclude outliers from training set #######

train_new_group = train_new.groupby('Id')

df = pd.DataFrame(train_new_group['Expected'].mean()) # mean, or any value

meaningful_ids = np.array(df[df['Expected'] < THRESHOLD].index)
train_final = train_new[np.in1d(train_new.Id, meaningful_ids)]
def pad_series(X, target_len=19):

    seq_len = X.shape[0]

    pad_size = target_len-seq_len

    if (pad_size > 0):

        X = np.pad(X, ((0,pad_size), (0,0)), 'constant', constant_values=0.)

    return X, seq_len
data_pd_gp = train_final.groupby("Id")

data_size = len(data_pd_gp)
INPUT_WIDTH = 19



X_train = np.empty((data_size, INPUT_WIDTH, 22))

seq_lengths = np.zeros(data_size)

y_train = np.zeros(data_size)



i = 0

for _, group in data_pd_gp:

    group_array = np.array(group)

    X, seq_length = pad_series(group_array[:,1:23], target_len=INPUT_WIDTH) 

    y = group_array[0,23]

    X_train[i,:,:] = X[:,:]

    seq_lengths[i] = seq_length

    y_train[i]= y

    i += 1

    

X_train.shape, y_train.shape
from keras.layers import RNN, Input, Dense, CuDNNLSTM, AveragePooling1D, TimeDistributed, Bidirectional, Flatten

from keras.models import Model
def get_model_lite(shape=(19,22)):

    inp = Input(shape)

    x = Bidirectional(CuDNNLSTM(32, return_sequences=True))(inp)

    x = TimeDistributed(Dense(32))(x)

    x = Bidirectional(CuDNNLSTM(64, return_sequences=True))(inp)

    x = TimeDistributed(Dense(64))(x)

    x = Bidirectional(CuDNNLSTM(128, return_sequences=True))(x)

    x = TimeDistributed(Dense(1))(x)

    x = AveragePooling1D(pool_size=2)(x)

    x = Flatten()(x)

    x = Dense(1)(x)



    model = Model(inp, x)

    return model
m16 = get_model_lite((19,22))
from keras.optimizers import SGD

sgd = SGD(lr=0.01, momentum=0.9, nesterov=True)

m16.compile(sgd, loss='mae')
m16.fit(X_train, y_train,batch_size=32, epochs=1, verbose=1)
del train_data

del raw_ids_all

del raw_ids

del train_raw_tmp

del raw_ids_tmp

del train_new

del train_new_group

del df

del meaningful_ids

del train_final

del data_pd_gp

del data_size

del X_train

del y_train

del seq_lengths
####### 6. Preprocess the test data #######

test_raw = pd.read_csv('../input/test.csv')

test_raw_ids_all = test_raw["Id"]

test_raw_ids = np.array(test_raw_ids_all.unique())



# Convert all NaNs to zero

test_new = test_raw.fillna(0.0)

# test_new = test_new.reset_index(drop=True)









INPUT_WIDTH = 19



data_pd_gp = test_new.groupby("Id")

data_size = len(data_pd_gp)



X_test = np.empty((data_size, INPUT_WIDTH, 22))

seq_lengths = np.zeros(data_size)



i = 0

for _, group in data_pd_gp:

    group_array = np.array(group)

    X, seq_length = pad_series(group_array[:,1:23], target_len=INPUT_WIDTH) 

    X_test[i,:,:] = X[:,:]

    seq_lengths[i] = seq_length

    i += 1

    

X_test.shape
output = m16.predict(X_test, batch_size=32,verbose=1)
my_submission = pd.DataFrame({'Id': np.arange(1,output.shape[0]+1), 'Expected': output[:,0]})

# you could use any filename. We choose submission here

my_submission.to_csv('submission.csv', index=False)