# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split



import tensorflow as tf

from tensorflow.contrib.keras import models

from tensorflow.contrib.keras import layers

from tensorflow.contrib.keras import losses,optimizers,metrics

from keras.utils.np_utils import to_categorical



%matplotlib inline

sns.set_style('whitegrid')

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
print('Shape of training data is {}.'.format(train.shape))

print('Shape of test data is {}.'.format(test.shape))
train.head()
# # select a random sample of dataset without replacement

# train = train.sample(n=10000,replace=False,random_state=101)

# test = test.sample(n=2000,replace=False,random_state=101)
# Check that we have about equal numbers in each class

train['label'].value_counts()
y_train = train['label']

X_train = train.drop('label',axis=1)

print('Shapes of training labels and data are {} and {}.'.format(y_train.shape,X_train.shape))
X_train.max().max(),X_train.min().min()
X_train /= 255

test /= 255
# check to make sure

print(X_train.max().max(),X_train.min().min())

print('\n')

print(test.max().max(),test.min().min())
y_train = to_categorical(y_train, num_classes = 10)

y_train.shape
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=101)
print('Shapes of training labels and data are {} and {}.'.format(y_train.shape,X_train.shape))

print('\n')

print('Shapes of validation labels and data are {} and {}.'.format(y_val.shape,X_val.shape))
test_image = X_train.values.reshape(-1,28,28,1)

test_image.shape
indx = 8 # change this to see different numbers

plt.imshow(test_image[indx][:,:,0],cmap='Greys')

print('What is the label? {}'.format(np.where(y_train[indx]==1)[0]))
cnn_kmodel = models.Sequential()
cnn_kmodel.add(layers.Conv2D(filters = 32, kernel_size = (6,6),padding = 'Same', 

                 activation ='relu', input_shape = (28,28,1)))

cnn_kmodel.add(layers.MaxPool2D(pool_size=(2,2)))

cnn_kmodel.add(layers.Dropout(0.25))



cnn_kmodel.add(layers.Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))

cnn_kmodel.add(layers.MaxPool2D(pool_size=(2,2), strides=(2,2)))

cnn_kmodel.add(layers.Dropout(0.25))



cnn_kmodel.add(layers.Flatten())

cnn_kmodel.add(layers.Dense(512, activation = "relu"))

cnn_kmodel.add(layers.Dropout(0.5))

cnn_kmodel.add(layers.Dense(10, activation = "softmax"))
# optimizer

opt = tf.keras.optimizers.Adam()
cnn_kmodel.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['accuracy'])
epochs = 10

batch_size = 50
cnn_kmodel.fit(X_train.values.reshape(-1,28,28,1), y_train, batch_size=batch_size, epochs=epochs,

               validation_data=(X_val.values.reshape(-1,28,28,1), y_val), shuffle=True)
plt.figure(figsize=[9,6])

plt.plot(cnn_kmodel.history.history['loss'], color='b', label="Training loss")

plt.plot(cnn_kmodel.history.history['val_loss'], color='r', label="validation loss")

plt.legend(loc='best')

plt.xlabel('Epochs')

plt.ylabel('Loss')
predict_vals = cnn_kmodel.predict_classes(X_val.values.reshape(-1,28,28,1))
yval_label = np.argmax(y_val,axis=1) 

print(yval_label)
print('Metric for Validation set')

print('Classification report:')

print(classification_report(predict_vals,yval_label))

print('\n')

print('Accuracy score is {:6.3f}.'.format(accuracy_score(predict_vals,yval_label)))
plt.figure(figsize=[9,6])

sns.heatmap(confusion_matrix(predict_vals,yval_label),cmap='gist_gray',cbar=False,

            annot=True,fmt='d',linewidths=0.5)

plt.title('Confusion Matrix for Validation Set')
sub_results = cnn_kmodel.predict(test.values.reshape(-1,28,28,1))



sub_results = np.argmax(sub_results, axis=1)



submission = pd.DataFrame({'ImageId':np.arange(1,sub_results.size +1),'Label':sub_results})

submission.to_csv("submission.csv",index=False)