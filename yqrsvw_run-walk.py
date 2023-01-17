from keras.callbacks import Callback

from keras.callbacks import EarlyStopping

from keras.layers import Dense

from keras.models import Sequential

from keras.utils import np_utils

from keras.utils.np_utils import to_categorical

from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.cross_validation import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import KFold

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import seaborn as sns
from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
data = pd.read_csv('../input/dataset.csv')

data.info()
data.describe()
data.shape
data.isnull().any()
data.head()
print(pd.value_counts(data['activity'].values))
SENSOR_DATA_COLUMNS = [

    'acceleration_x',

    'acceleration_y',

    'acceleration_z',

    'gyro_x',

    'gyro_y',

    'gyro_z'

]



data_left = pd.DataFrame()

data_left = data[data['wrist'] == 0]



data_right = pd.DataFrame()

data_right = data[data['wrist'] == 1]
for c in SENSOR_DATA_COLUMNS:

    plt.figure(figsize=(10,5))

    plt.title("Sensor data distribution for both wrists")

    sns.distplot(data_left[c], label='left')

    sns.distplot(data_right[c], label='right')

    plt.legend()

    plt.show()
dataset = data.values



X = dataset[:,5:11].astype(float)

Y = dataset[:,4].astype(int)
def create_model():

    global model

    model = Sequential()

    model.add(Dense(15, input_dim=6, activation='relu'))

    model.add(Dense(15, activation='relu'))

    model.add(Dense(15, activation='relu'))

    model.add(Dense(2, activation='softmax'))

    

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model
class LossHistory(Callback):

    def on_train_begin(self, logs={}):

        self.losses = []



    def on_batch_end(self, batch, logs={}):

        self.losses.append(logs.get('loss'))



loss_history = LossHistory()

early_stopping = EarlyStopping(monitor='val_acc', patience=20)
estimator = KerasClassifier(create_model, epochs=200, batch_size=100, verbose=False)



X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=5)

Y_test = to_categorical(Y_test)



results = estimator.fit(

    X_train,

    Y_train,

    callbacks=[loss_history, early_stopping], validation_data=(X_test, Y_test)

)
results
kfold = KFold(n_splits=10, shuffle=True, random_state=5)

cv_results = cross_val_score(estimator, X_test, Y_test, cv=kfold)

print("Baseline on test data: %.2f%% (%.2f%%)" % (cv_results.mean()*100, cv_results.std()*100))