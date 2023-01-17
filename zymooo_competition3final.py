# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import pandas as pd
import numpy as np
import tensorflow
import tensorflow.keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


batch_size = 256
hidden_size = 400

def load_dataset(filename):
    df = pd.read_csv(filename)
    df.sort_values(['session_id', 'timestamp'], inplace=True, ascending=True)

    dtcol = pd.DatetimeIndex(df['timestamp'])
    df['day'] = dtcol.dayofweek
    df['month'] = dtcol.month
    df['hour'] = dtcol.hour
    df.drop('timestamp', axis=1, inplace=True)

    return df

train_data_df = load_dataset('/kaggle/input/etipgdla2020c3/train_data.csv')
train_data_df.head()
feature_columns = list(train_data_df.columns.values)
feature_columns.remove('session_id')
train_data_df_groups = train_data_df.groupby(['session_id'])

train_labels_df = pd.read_csv('/kaggle/input/etipgdla2020c3/train_labels.csv')
train_labels_df['buy'] = train_labels_df['buy'].astype(int)

num_train_of_sessions = len(train_labels_df['session_id'])
train_labels_dict = train_labels_df.set_index('session_id').to_dict(orient='dict')['buy']

train_session_ids, validation_sessions_ids = train_test_split(list(train_labels_df['session_id']), test_size=0.02,
                                                              shuffle=False)
train_data_df.head()
class DataGenerator(tensorflow.keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, groups, labels_dict, session_ids, batch_size=32, shuffle=True):
        'Initialization'
        self.groups = groups
        self.labels_dict = labels_dict
        self.session_ids = session_ids
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.session_ids) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        session_ids = [self.session_ids[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(session_ids)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.session_ids))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, session_ids):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = list()
        y = list()
        # Generate data
        for i, id in enumerate(session_ids):
            group = np.array(self.groups.get_group(id).drop('session_id', axis=1)).tolist()
            X.append(group)
            y.append(self.labels_dict[id])

        X = tensorflow.keras.preprocessing.sequence.pad_sequences(X, dtype='float')
        return X, np.array(y).astype(float)

train_data_gen = DataGenerator(train_data_df_groups, train_labels_dict, train_session_ids, batch_size=batch_size)
validation_data_gen = DataGenerator(train_data_df_groups, train_labels_dict, validation_sessions_ids,
                                    batch_size=batch_size)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM,BatchNormalization,Activation
from tensorflow.keras.optimizers import SGD, Adadelta, Adam, RMSprop
# import keras_metrics as km
import tensorflow.keras.backend as K


def f1(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2 * p * r / (p + r + K.epsilon())
    f1 = tensorflow.where(tensorflow.compat.v1.is_nan(f1), tensorflow.zeros_like(f1), f1)
    return K.mean(f1)


def p(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    return K.mean(p)


def r(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)

    r = tp / (tp + fn + K.epsilon())
    return K.mean(r)

def weighted_binary_crossentropy(y_true, y_pred):
    logloss = -(y_true * K.log(y_pred) * 1.0 + (1 - y_true) * K.log(1 - y_pred) * 1.0)
    return K.mean(logloss, axis=-1)

model = Sequential()
model.add(LSTM(hidden_size, input_shape=(None, len(train_data_df.columns) - 1), return_sequences=True))
model.add(LSTM(100, input_shape=(None, len(train_data_df.columns) - 1),return_sequences=False))
model.add(Dense(1, activation='sigmoid'))
model.summary()

model.compile(loss=weighted_binary_crossentropy, optimizer='nadam', metrics=['accuracy', f1, p, r])
history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=len(train_session_ids) // batch_size,
    use_multiprocessing=False,
    epochs=4,
    validation_steps=len(validation_sessions_ids) // batch_size,
    validation_data=validation_data_gen
)
test_data_df = load_dataset('/kaggle/input/etipgdla2020c3/test_data.csv')
test_data_df_groups = test_data_df.groupby(['session_id'])

test_labels_df = pd.read_csv('/kaggle/input/etipgdla2020c3/test_labels.csv')
test_labels_df['buy'] = 0

num_test_of_sessions = len(test_labels_df['session_id'])
test_labels_dict = test_labels_df.set_index('session_id').to_dict(orient='dict')['buy']
test_session_ids = list(test_labels_df['session_id'])

test_data_gen = DataGenerator(test_data_df_groups, test_labels_dict, test_session_ids, batch_size=100, shuffle=False)
predict = model.predict_generator(test_data_gen, steps=len(test_session_ids))

test_labels_df['buy']=predict>0.05
test_labels_df.to_csv("submission.csv", header=True, index=False)