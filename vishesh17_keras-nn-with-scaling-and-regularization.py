import tensorflow as tf

import pandas as pd

import os

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from keras import Sequential

from keras import layers

from keras import backend as K

from keras.layers.core import Dense

from keras import regularizers

from keras.layers import Dropout

from keras.constraints import max_norm
# Import data

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train.shape
test.shape
#Check num of cases in label 

print(train.target.value_counts())

print(train.target.value_counts()[1]/train.target.value_counts()[0])
train_features = train.drop(['target', 'ID_code'], axis=1)

train_targets = train['target']

test_features = test.drop(['ID_code'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(train_features, train_targets, test_size = 0.25, random_state = 50)
# Feature Scaling

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)

test_features = sc.transform(test_features)
# Add RUC metric to monitor NN

def auc(y_true, y_pred):

    auc = tf.metrics.auc(y_true, y_pred)[1]

    K.get_session().run(tf.local_variables_initializer())

    return auc
input_dim = X_train.shape[1]

input_dim
# Try early stopping

#from keras.callbacks import EarlyStopping

#callback = EarlyStopping(monitor='loss', min_delta=0, patience=10, verbose=0, mode='auto', baseline=None, restore_best_weights=True)
model = Sequential()

# Input layer

model.add(Dense(units = 200, activation = "relu", input_dim = input_dim, kernel_initializer = "normal", kernel_regularizer=regularizers.l2(0.005), 

                kernel_constraint = max_norm(5.)))

# Add dropout regularization

model.add(Dropout(rate=0.2))



# First hidden layer

model.add(Dense(units = 200, activation='relu', kernel_regularizer=regularizers.l2(0.005), kernel_constraint=max_norm(5)))

# Add dropout regularization

model.add(Dropout(rate=0.1))



# Second hidden layer

model.add(Dense(100, activation='relu', kernel_regularizer=regularizers.l2(0.005), kernel_constraint=max_norm(5)))

# Add dropout regularization

model.add(Dropout(rate=0.1))



# Third hidden layer

model.add(Dense(50, activation='tanh', kernel_regularizer=regularizers.l2(0.005), kernel_constraint=max_norm(5)))

# Add dropout regularization

model.add(Dropout(rate=0.1))



# Output layer

model.add(layers.Dense(units = 1, activation='sigmoid'))



model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', auc])

model.summary()
model.fit(X_train, y_train, batch_size = 16384, epochs = 125, validation_data = (X_test, y_test))#, callbacks = [callback])
y_pred = model.predict_proba(X_test)

roc_auc_score(y_test, y_pred)
id_code_test = test['ID_code']

# Make predicitions

pred = model.predict(test_features)

pred_ = pred[:,0]
pred_
# To CSV

my_submission = pd.DataFrame({"ID_code" : id_code_test, "target" : pred_})
my_submission
my_submission.to_csv('submission.csv', index = False, header = True)