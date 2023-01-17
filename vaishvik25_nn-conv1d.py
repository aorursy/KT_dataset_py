import pandas as pd

import numpy as np

from sklearn.preprocessing import MinMaxScaler,StandardScaler

from keras.models import Sequential

from keras.layers import Dense,Dropout

import keras
X_train=pd.read_pickle('../input/ieee-fe-data/X_train.pkl')

test=pd.read_pickle('../input/ieee-fe-data/test.pkl')

y_train=pd.read_pickle('../input/ieee-fe-data/y_train.pkl')
print(X_train.shape,test.shape,y_train.shape)
num_preds = 5

num_features = test.shape[1]

learning_rate_init = 0.02



def get_model():

    inp = keras.layers.Input((num_features,))

    x = keras.layers.Reshape((num_features,1))(inp)

    x = keras.layers.Conv1D(32,num_preds, activation='elu')(x)

    x = keras.layers.BatchNormalization()(x)

    x = keras.layers.Conv1D(24,1, activation='elu')(x)

    x = keras.layers.BatchNormalization()(x)

    x = keras.layers.Conv1D(16,1, activation='elu')(x)

    x = keras.layers.BatchNormalization()(x)

    x = keras.layers.Conv1D(4,1, activation='elu')(x)

    x = keras.layers.Flatten()(x)

    x = keras.layers.BatchNormalization()(x)

    out = keras.layers.Dense(1, activation='sigmoid')(x)

    return keras.Model(inputs=inp, outputs=out)

# fit the keras model on the dataset



model = get_model()

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])



model.fit(X_train, y_train, epochs=150, batch_size=400, verbose=1)
predictions = model.predict(test,batch_size=2000)
sample_submission=pd.read_csv("../input/ieee-fraud-detection/sample_submission.csv")

sample_submission['isFraud'] = predictions

sample_submission.to_csv('NN_w_conv1.csv',index=False)
(sample_submission['isFraud']).value_counts()