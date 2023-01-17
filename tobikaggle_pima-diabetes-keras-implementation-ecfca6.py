# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

%matplotlib inline



from keras.models import Sequential

from keras.layers import Dense

from keras.callbacks import ModelCheckpoint



seed = 42

np.random.seed(seed)
# load Pima dataset

pdata = pd.read_csv('../input/diabetes.csv')

pdata.head()
pdata.describe()
# let's remove the 0-entries for these fields



zero_fields = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']



def check_zero_entries(data, fields):

    """ List number of 0-entries in each of the given fields"""

    for field in fields:

        print('field %s: num 0-entries: %d' % (field, len(data.loc[ data[field] == 0, field ])))



check_zero_entries(pdata, zero_fields)
# First - split into Train/Test

from sklearn.model_selection import train_test_split



features = list(pdata.columns.values)

features.remove('Outcome')

print(features)

X = pdata[features]

y = pdata['Outcome']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)



print(X_train.shape)

print(X_test.shape)
# lets fix the 0-entry for a field in the dataset with its mean value

def impute_zero_field(data, field):

    nonzero_vals = data.loc[data[field] != 0, field]

    avg = np.sum(nonzero_vals) / len(nonzero_vals)

    k = len(data.loc[ data[field] == 0, field])   # num of 0-entries

    data.loc[ data[field] == 0, field ] = avg

    print('Field: %s; fixed %d entries with value: %.3f' % (field, k, avg))
# Fix it for Train dataset

for field in zero_fields:

    impute_zero_field(X_train, field)
# double check for the Train dataset

check_zero_entries(X_train, zero_fields)
# Fix for Test dataset

for field in zero_fields:

    impute_zero_field(X_test, field)
# double check for the Test dataset

check_zero_entries(X_test, zero_fields)
# Ensure that fieldnames aren't included

X_train = X_train.values

y_train = y_train.values

X_test  = X_test.values

y_test  = y_test.values
NB_EPOCHS = 1000  # num of epochs to test for

BATCH_SIZE = 5



## Create our model

model = Sequential()



# 1st layer: input_dim=8, 12 nodes, RELU

model.add(Dense(12, input_dim=8, init='uniform', activation='relu'))

# 2nd layer: 8 nodes, RELU

model.add(Dense(8, init='uniform', activation='relu'))

# output layer: dim=1, activation sigmoid

model.add(Dense(1, init='uniform', activation='sigmoid' ))



# Compile the model

model.compile(loss='binary_crossentropy',   # since we are predicting 0/1

             optimizer='Nadam',

             metrics=['accuracy'])



# checkpoint: store the best model

ckpt_model = 'pima-weights.best.hdf5'

checkpoint = ModelCheckpoint(ckpt_model, 

                            monitor='val_acc',

                            verbose=1,

                            save_best_only=True,

                            mode='max')

callbacks_list = [checkpoint]



print('Starting training...')

# train the model, store the results for plotting

history = model.fit(X_train,

                    y_train,

                    validation_data=(X_test, y_test),

                    nb_epoch=NB_EPOCHS,

                    batch_size=BATCH_SIZE,

                    callbacks=callbacks_list,

                    verbose=0)
# Model accuracy

plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('Model Accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'])

plt.show()
# Model Losss

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model Loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'])

plt.show()
# print final accuracy

scores = model.evaluate(X_test, y_test, verbose=0)

print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))