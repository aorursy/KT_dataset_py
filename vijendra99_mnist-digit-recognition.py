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
from keras.models import Sequential

from keras.utils import np_utils

from keras.layers.core import Dense, Activation, Dropout

from keras.layers import Input
# Training Set

train = pd.read_csv('../input/train.csv')

labels = train.iloc[:, 0].values.astype('int32')

X_train = train.iloc[:, 1:].values.astype('float32')



# Test Set

test = pd.read_csv('../input/test.csv')

X_test = (test.values).astype('float32')



# Convert labels integer to catedorical binary data

y_train = np_utils.to_categorical(labels)
# Scaling 

scaled_value = np.max(X_train)

X_train /= scaled_value

X_test /= scaled_value 



# Mean

mean = np.mean(X_train)

X_train -= mean

X_test -= mean



# Shapes

input_dim = X_train.shape[1]

# print(input_dim)

nb_classes = y_train.shape[1]



# multi layer perceptron implementation by using 

model = Sequential()



# model.add(Input(input_dim))



# 1st hidden layer

# model.add(Dense(128))

model.add(Dense(128, input_dim=input_dim))

model.add(Activation('relu'))

model.add(Dropout(0.15))



# 2nd hidden layer

model.add(Dense(128))

model.add(Activation('relu'))

model.add(Dropout(0.15))



# output layer

model.add(Dense(nb_classes))

model.add(Activation('softmax'))





# Taking categorical cross entropy as loss, and rmsprop as optimizer

model.compile(loss = 'categorical_crossentropy', optimizer = 'rmsprop', metrics=['accuracy'])



print('Traing the given train dataset...')

# model.fit(X_train, y_train, epochs = 10, batch_size = 16, validation_split = 0.1, show_accuracy = True, verbose = 2) )

model.fit(X_train, y_train, epochs = 10, batch_size=16, validation_split=0.1,verbose=2)



print('predictions on test classes...')

model.predict_classes(X_test, verbose = 0)



# Summarizing the model

model.summary()
# # Read the file for writing predictions on test classes

# submission_df = pd.read_csv(os.path.abspath('sample_submission.csv'), header=0)



# # Making a pandas series of the predicted labels

# pred_labels_series = pd.Series(pred_labels, dtype='int32')

# submission_df['Label'] = pred_labels_series



# # Identifying the ImageId column as the index column

# submission_df = submission_df.set_index(['ImageId'])



# # displaying the first 20 predictions

# print(submission_df.head(20))



# # Saving to file "sample_submission.csv"

# submission_df.to_csv(os.path.abspath('sample_submission.csv'))

preds = model.predict_classes(X_test, verbose = 0)



def write_preds(preds, fname):

    pd.DataFrame({"ImageId": list(range(1,len(preds)+1)), "Label": preds}).to_csv(fname, index=False, header=True)



write_preds(preds, "sample_submission.csv")