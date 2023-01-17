# Load libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

%matplotlib inline
sns.set()

import warnings
warnings.filterwarnings("ignore")
train_df = pd.read_csv('../input/train_V2.csv')
test_df  = pd.read_csv('../input/test_V2.csv')

submission_df = pd.DataFrame()
submission_df['Id'] = test_df["Id"].copy()
train_df.info()
def null_percentage(column):
    df_name = column.name
    nans = np.count_nonzero(column.isnull().values)
    total = column.size
    frac = nans / total
    perc = int(frac * 100)
    print('%d%% or %d missing values from [ %s ] column.' % (perc, nans, df_name))

def check_nan(df):
    columns = df.columns
    for col in columns: null_percentage(df[col])
check_nan(train_df)
# Just one missing value exists, DROP it.
train_df = train_df.dropna()
train_df.reset_index(drop=True)
train_df.describe().T
# Drop columns
train_df2 = train_df.drop(columns=['Id', 'groupId', 'matchId'])
test_df2 = test_df.drop(columns=['Id', 'groupId', 'matchId'])
from sklearn.preprocessing import LabelEncoder as LE

le = LE()
train_df2['matchType'] = le.fit_transform(train_df2['matchType'].astype(str))
test_df2['matchType'] = le.fit_transform(test_df2['matchType'].astype(str))
# Split training dataset into train/validation set (ratio = 7:3)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler as SS

X = train_df2.iloc[:, 0:-1]; y = train_df2.iloc[:, -1]

scaler_train, scaler_test = SS(), SS()
scaler_train.fit_transform(X)
scaler_test.fit_transform(test_df2)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=1040941203)
X_tr = np.array(X_train).reshape(X_train.shape[0], X_train.shape[1], 1).astype('float32')
X_vd = np.array(X_val).reshape(X_val.shape[0], X_val.shape[1], 1).astype('float32')

y_tr = np.array(y_train)
y_vd = np.array(y_val)
X_test = np.array(test_df2)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1).astype('float32')
import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv1D
from keras.initializers import random_uniform
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import EarlyStopping
#hyperparameters
SEED = 1040941203
hidden_initializer = random_uniform(seed=SEED)
dropout_rate = 0.2

# create model
model = Sequential()
model.add(Conv1D(20, 5, input_shape = X_tr.shape[1:3]))
model.add(LeakyReLU(alpha=0.1))
model.add(Conv1D(10, 1))
model.add(Flatten())
model.add(Dense(128, input_dim=25, kernel_initializer=hidden_initializer))
model.add(LeakyReLU(alpha=0.1))
model.add(Dense(32))
model.add(LeakyReLU(alpha=0.1))
model.add(Dense(1, activation='linear'))

model.summary()
model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])

early_stopping = EarlyStopping(patience = 3)
hist = model.fit(X_tr, y_tr, epochs=50, batch_size=5000, validation_data=(X_vd, y_vd), callbacks=[early_stopping])
fig, loss_ax = plt.subplots(figsize=(15,15))

mae_ax = loss_ax.twinx()

loss_ax.plot(hist.history['loss'], 'y', label='train loss', linewidth=5)
loss_ax.plot(hist.history['val_loss'], 'r', label='val loss', linewidth=9)

mae_ax.plot(hist.history['mean_absolute_error'], 'b', label='train mae')
mae_ax.plot(hist.history['val_mean_absolute_error'], 'g', label='val mae')

loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
mae_ax.set_ylabel('MAE')

loss_ax.legend(loc='upper left')
mae_ax.legend(loc='upper right')

plt.show()
y_pred = model.predict(X_test, batch_size=1000, verbose=True)
submission_df['winPlacePerc'] = y_pred
submission_df.to_csv('submission_convnet.csv', index=False)