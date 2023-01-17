import os

import numpy as np

import pandas as pd



THRESHOLD = 73 

N_FOLDS = 21

RND_SEED = 56
def pad_series(X, target_len=19):

    seq_len = X.shape[0]

    pad_size = target_len - seq_len

    if (pad_size > 0):

        X = np.pad(X, ((0, pad_size), (0, 0)), 'constant', constant_values=0.)

    return X, seq_len
####### 1. Import training data and extract ids #######

train_raw = pd.read_csv("../input/train.csv", nrows=3500000)

raw_ids_all = train_raw["Id"]

raw_ids = raw_ids_all.unique()



####### 2. Remove ids with only NaNs in the "Ref" column #######

train_raw_tmp = train_raw[~np.isnan(train_raw.Ref)]

raw_ids_tmp = train_raw_tmp["Id"].unique()

train_new = train_raw[np.in1d(raw_ids_all, raw_ids_tmp)]



####### 3. Convert all NaN to zero #######

train_new = train_new.fillna(0.0)



####### 4. Define and exclude outliers from training set #######

train_new_group = train_new.groupby('Id')

df = pd.DataFrame(train_new_group['Expected'].mean()) # mean, or any value

meaningful_ids = np.array(df[df['Expected'] < THRESHOLD].index)



train_final = train_new[np.in1d(train_new.Id, meaningful_ids)]

train_final.head(10)
data_pd_gp = train_final.groupby("Id")

data_size = len(data_pd_gp)
INPUT_WIDTH = 19



X_train = np.empty((data_size, INPUT_WIDTH, 22))

seq_lengths = np.zeros(data_size)

y_train = np.zeros(data_size)



i = 0

for _, group in data_pd_gp:

    group_array = np.array(group)

    X, seq_length = pad_series(group_array[:, 1:23], target_len=INPUT_WIDTH)

    y = group_array[0, 23]

    X_train[i, :, :] = X[:, :]

    seq_lengths[i] = seq_length

    y_train[i] = y

    i += 1

    

X_train.shape, y_train.shape
from keras.layers import Input, Bidirectional, TimeDistributed, AveragePooling1D, Flatten, Dense, CuDNNLSTM

from keras.models import Model

def get_model(shape=(19,22)):

    inp = Input(shape)

    x = Bidirectional(CuDNNLSTM(64, return_sequences=True))(inp)

    x = TimeDistributed(Dense(64))(x)

    x = Bidirectional(CuDNNLSTM(128, return_sequences=True))(x)

    x = TimeDistributed(Dense(1))(x)

    x = AveragePooling1D(pool_size=2)(x)

    x = Flatten()(x)

    x = Dense(1)(x)

    

    model = Model(inp, x)

    return model



model = get_model((19, 22))
from keras.optimizers import SGD



sgd = SGD(lr=0.01, momentum=0.9, nesterov=True)

model.compile(sgd, loss='mae')



model.fit(X_train, y_train, batch_size=32, epochs=1, verbose=1)
####### 6. Preprocess the test data #######

test_raw = pd.read_csv("../input/test.csv")

test_raw_ids_all = test_raw["Id"]

test_raw_ids = np.array(test_raw_ids_all.unique())



# Convert all NaNs to zero

test_new = test_raw.fillna(0.0)
INPUT_WIDTH = 19



data_pd_gp = test_new.groupby("Id")

data_size = len(data_pd_gp)



X_test = np.empty((data_size, INPUT_WIDTH, 22))

seq_lengths = np.zeros(data_size)



i = 0

for _, group in data_pd_gp:

    group_array = np.array(group)

    X, seq_length = pad_series(group_array[:, 1:23], target_len=INPUT_WIDTH)

    X_test[i, :, :] = X[:, :]

    seq_lengths[i] = seq_length

    i += 1

    

X_test.shape
output = model.predict(X_test, batch_size=32, verbose=1)
submission = pd.DataFrame(output)

submission.index += 1

submission.index.names = ['Id']

submission = submission.rename(columns={ submission.columns[0]: "Expected" })
submission.head()
submission.to_csv("submission.csv")