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
train = pd.read_csv("../input/train.csv")
train.head()
train.shape
train.describe()
test = pd.read_csv("../input/test.csv")
test.head()
test.describe()
test.shape
# split randomly the train1 dataset
from sklearn.cross_validation import train_test_split
num_train ,num_test = train_test_split(train,test_size=0.3) 
num_train.shape
num_test.shape
# specify X_train, X_test
X_train = num_train.drop('label',axis=1)
X_test = num_test.drop('label',axis=1)
# specify y_train, y_test
y_train = num_train.label
y_test = num_test.label
# Specify the parameters
n_train = 29400 # number of training examples
n_test = 12600 # number of test examples
height, width, depth = 28, 28, 1 # MNIST images are 28x28 and greyscale
n_classes = 10 # there are 10 classes (1 per digit)
# convert to array
X_train = X_train.values
X_test = X_test.values
# reshape the data
X_train = X_train.reshape(n_train, height*width) # flatten train data to 1D
X_test = X_test.reshape(n_test, height*width) # flatten test data to 1D
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
# normalise data to [0,1] range
X_train /= 255
X_test /= 255
# one-hot label encoding
from keras.utils import np_utils
Y_train = np_utils.to_categorical(y_train, n_classes)
Y_test = np_utils.to_categorical(y_test, n_classes)
# Build the DPP
# Importing the Keras libraries and packages
from keras.models import Model
from keras.layers import Input, Dense, Dropout

inp = Input(shape=(height*width,))
hidden_1 = Dense(512, activation='relu')(inp)
dropout_1 = Dropout(0.2)(hidden_1)
hidden_2 = Dense(512, activation='relu')(dropout_1)
dropout_2 = Dropout(0.2)(hidden_2)
hidden_3 = Dense(512, activation='relu')(dropout_2)
dropout_3 = Dropout(0.2)(hidden_3)
out = Dense(n_classes, activation='softmax')(dropout_3)

model = Model(inputs=inp, outputs=out)



# Compile the model
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Fit the model to the training set
model.fit(X_train, Y_train, epochs = 100, batch_size = 32, verbose=1, validation_split=0.1)


# Evaluate the model
model.evaluate(X_test, Y_test, verbose=1)
# convert to array
test = test.values
test = test.reshape(28000, height*width) # flatten test data to 1D
test = test.astype('float32')
# normalise data to [0,1] range
test /= 255
# predict on the test set
predictions = model.predict(test)
predictions.shape
predictions = predictions.astype('int')
predictions[0:6,:]
predictions.shape
inv_preds = predictions.argmax(1)
inv_preds.shape
inv_preds = inv_preds[np.newaxis]
inv_preds1 = inv_preds.T
inv_preds1
ImageId  = list(range(1,len(test)+1))
ImageId= np.array(ImageId)
ImageId
ImageId = ImageId[np.newaxis]
ImageId = ImageId.T
ImageId
dnn = np.column_stack((ImageId,inv_preds1))
dnn
# Prepare submission file 
dnn1 = pd.DataFrame(dnn,columns=('ImageId','Label'))

# prepare the csv file
dnn1.to_csv('dnn2.csv',index=False)