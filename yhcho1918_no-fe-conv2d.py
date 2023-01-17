#
# Setting for obtaining reproducible results
#

import numpy as np
import tensorflow as tf
import random as rn

# The below is necessary in Python 3.2.3 onwards to
# have reproducible behavior for certain hash-based operations.
# See these references for further details:
# https://docs.python.org/3.4/using/cmdline.html#envvar-PYTHONHASHSEED
# https://github.com/keras-team/keras/issues/2280#issuecomment-306959926

import os
os.environ['PYTHONHASHSEED'] = '0'

# The below is necessary for starting Numpy generated random numbers
# in a well-defined initial state.

np.random.seed(1234)

# The below is necessary for starting core Python generated random numbers
# in a well-defined state.

#rn.seed(12345)

# Force TensorFlow to use single thread.
# Multiple threads are a potential source of
# non-reproducible results.

session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, 
                              inter_op_parallelism_threads=1)

from keras import backend as K

# The below tf.set_random_seed() will make random number generation
# in the TensorFlow backend have a well-defined initial state.
# For further details, 
# see: https://www.tensorflow.org/api_docs/python/tf/set_random_seed

tf.set_random_seed(1234)

sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import keras
df_train = pd.read_csv('../input/X_train.csv', encoding='cp949')
df_test = pd.read_csv('../input/X_test.csv', encoding='cp949')
y_train = pd.read_csv('../input/y_train.csv').gender
IDtest = df_test.cust_id.unique()
df_test.head()
level = 'gds_grp_nm'
items = list(set(df_train[level]) | set(df_test[level]))
ncol = len(items)

def make_image(df):
    x = pd.DataFrame({'cust_id': df.cust_id.unique()})
    y = pd.DataFrame({level: items})
    z = pd.DataFrame({'week_day': np.arange(7)})
    df_pad = (
        x.assign(key=1)
        .merge(y.assign(key=1), on="key")
        .merge(z.assign(key=1), on="key")
        .drop("key", axis=1)
        .assign(amount=0)
    )
    df['week_day'] = pd.to_datetime(df.tran_date).dt.weekday
    df_all = pd.concat([df, df_pad], sort=False, axis=0)
    x = pd.pivot_table(df_all.query('amount >= 0'), 
                       index=['cust_id','week_day'], columns=level, 
                       values='amount', aggfunc=np.size, fill_value=0)
    x = np.array(x).reshape(-1,7*ncol) - 1
    return x.reshape(-1,7,ncol,1)
#    x = np.array(x).reshape(-1,7*ncol) - 1
#    return (x / x.max(1).reshape(-1,1)).reshape(-1,7,ncol,1)

X_train = make_image(df_train)
X_test = make_image(df_test)

X_train.shape, X_test.shape
import seaborn as sns
fig, ax = plt.subplots(figsize=(20,10))
sns.heatmap(X_train.reshape(-1,7,ncol)[300], ax=ax)
plt.show()
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping

model = Sequential()
model.add(layers.Conv2D(1, kernel_size=(1, 1),strides=(1,1), activation='tanh', input_shape=(7,ncol,1)))
#model.add(layers.MaxPooling2D(pool_size=(2, 1)))
#model.add(layers.Conv2D(1, kernel_size=(1, 1),strides=(1,1), activation='tanh'))
model.add(layers.Flatten())
model.add(layers.Dropout(0.2))
#model.add(layers.Dense(16, activation='tanh'))
model.add(layers.Dense(1, activation='sigmoid'))
model.summary()

model.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['acc'])
#model.compile(optimizer=RMSprop(lr=1e-4), loss='binary_crossentropy', metrics=['acc'])
history = model.fit(X_train, y_train, 
                    epochs=50, batch_size=128, 
                    validation_split=0.2, callbacks=[EarlyStopping(patience=5)])

plt.plot(history.history["loss"], label="train loss")
plt.plot(history.history["val_loss"], label="validation loss")
plt.legend()
plt.title("Loss")
plt.show()
pred = model.predict(X_test)[:,0]
fname = 'submissions.csv'
submissions = pd.concat([pd.Series(IDtest, name="cust_id"), pd.Series(pred, name="gender")] ,axis=1)
submissions.to_csv(fname, index=False)
print("'{}' is ready to submit." .format(fname))