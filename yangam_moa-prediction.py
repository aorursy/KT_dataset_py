from keras.layers import Input,LSTM,Dense

from keras.models import Model

import pandas as pd
train_features=pd.read_csv("../input/lish-moa/train_features.csv")

test_features=pd.read_csv("../input/lish-moa/test_features.csv")

train_targets_nonscored=pd.read_csv("../input/lish-moa/train_targets_nonscored.csv")

train_targets_scored=pd.read_csv("../input/lish-moa/train_targets_scored.csv")
train_targets_nonscored.head()
len(train_targets_nonscored)
train_targets_scored.head()
len(train_targets_scored)
y_train=train_targets_scored
len(train_features)
len(test_features)
train_features.head()
train_features.dtypes
s=(train_features.dtypes=='object')

object_cols=list(s[s].index)

print(object_cols)
train_features['cp_type'].unique()
train_features['cp_dose'].unique()
test_features.head()
test_features['cp_type'].unique()
test_features['cp_dose'].unique()
train_features['cp_dose']=pd.get_dummies(train_features['cp_dose'])

test_features['cp_dose']=pd.get_dummies(test_features['cp_dose'])
train_features['cp_type']=pd.get_dummies(train_features['cp_type'])

test_features['cp_type']=pd.get_dummies(test_features['cp_type'])
train_features.head()
test_features.head()
X_train=train_features.drop('sig_id',axis=1)
X_test1=test_features.drop('sig_id',axis=1)
X_train.head()
X_test1.head()
X_train.shape[1]
import tensorflow as tf

import tensorflow_addons as tfa

import numpy as np
from keras.models import Sequential
y_train=y_train.drop('sig_id',axis=1)
model = tf.keras.models.Sequential([

    tf.keras.layers.Input(X_train.shape[1]),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dropout(0.5),

    tfa.layers.WeightNormalization(tf.keras.layers.Dense(2048, activation="relu")),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dropout(0.5),

    tfa.layers.WeightNormalization(tf.keras.layers.Dense(1048, activation="relu")),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dropout(0.5),

    tfa.layers.WeightNormalization(tf.keras.layers.Dense(y_train.shape[1], activation="relu"))

])

model.compile(loss="binary_crossentropy", optimizer="adam", metrics= ['accuracy'])
model.summary()
m= model.fit(X_train, y_train, verbose=2, epochs=50,batch_size=32)
from sklearn.preprocessing import StandardScaler
X_test1 = StandardScaler().fit_transform(X_test1)
y_pre= model.predict(X_test1)
y_pre
submission = pd.read_csv("../input/lish-moa/sample_submission.csv")



columns = list(submission.columns)

columns.remove('sig_id')



for i in range(len(columns)):

    submission[columns[i]] = y_pre[:, i]



submission.to_csv('submission.csv', index=False)