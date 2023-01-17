# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
RAW_TRAIN_DATA = pd.read_csv('/kaggle/input/fashionmnist/fashion-mnist_train.csv')

RAW_TEST_DATA = pd.read_csv('/kaggle/input/fashionmnist/fashion-mnist_test.csv')

RAW_TRAIN_DATA.shape
RAW_TRAIN_DATA_X  = RAW_TRAIN_DATA[RAW_TRAIN_DATA.columns.difference(['label'])] 

RAW_TRAIN_DATA_Y  = pd.DataFrame(RAW_TRAIN_DATA['label'],columns=['label'])

RAW_TEST_DATA_X  = RAW_TEST_DATA[RAW_TEST_DATA.columns.difference(['label'])]

RAW_TEST_DATA_Y  = pd.DataFrame(RAW_TEST_DATA['label'],columns=['label'])
print("Shape of training data is :{}".format(RAW_TRAIN_DATA_X.shape))

print("Shape of test data is :{}".format(RAW_TEST_DATA_X.shape))

print("Shape of training labels is :{}".format(RAW_TRAIN_DATA_Y.shape))

print("shape of test labels is :{}".format(RAW_TEST_DATA_Y.shape))
RAW_TRAIN_DATA_Y.head()
import seaborn as sns

sns.countplot(data=RAW_TRAIN_DATA_Y,x='label')
label_to_name = {0:'T-shirt/top',1:'Trouser',2:'Pullover',3:'Dress',4:'Coat',5:'Sandal',6:'Shirt',7:'Sneaker',8:'Bag',9:'Ankle boot'}
TRAIN_X = RAW_TRAIN_DATA_X/255

TEST_X = RAW_TEST_DATA_X/255
import keras

from keras import models

from keras.layers import Dense, Dropout

from keras.utils import to_categorical

from keras.datasets import mnist

from keras.utils.vis_utils import model_to_dot

from IPython.display import SVG

from keras import optimizers





INPUT_SHAPE = 784 #Sequential layers need to know the dimension of features. Note: Do not pass number or records as its irrelevant.

NUM_CLASSES = 10 #We know there are 10 different object types in our dataset. 



#SETUP HYPER PARAMETERS FOR THE MODEL

BATCH_SIZE = 32 #Generally keep a multiple of 32 as machines process better when values are 2**power

EPOCHS = 40 #Lets start with 10 epochs and later see how our model is performing and tune this parameter

ADAM_LEARNING_RATE = 0.001 # This variable is only applicable to Adam optimizer

ADAM_DECAY_RATE = 0.0 #Decay rate for Adam optimizer



#Convert labels into one hot vector

RAW_TRAIN_DATA_Y_ = keras.utils.to_categorical(RAW_TRAIN_DATA_Y,10)

RAW_TEST_DATA_Y_ = keras.utils.to_categorical(RAW_TEST_DATA_Y,10)



#We will using a callback to control running the epochs if there is no improvement in accuract after certain epochs. This will avoid wasting resources

early_stop = keras.callbacks.EarlyStopping(monitor='loss', min_delta=0, patience=5, verbose=0, mode='auto', baseline=None, restore_best_weights=False)

#patience: number of epochs with no improvement after which training will be stopped. Here we will wait for 5 epochs because we have seen that literally with every epoch there has been a small improvement
#Build neural network

model = models.Sequential()

model.add(Dense(512, activation='relu', input_shape=(INPUT_SHAPE,)))

model.add(Dropout(0.25))

model.add(Dense(256, activation='relu'))

model.add(Dropout(0.25))

model.add(Dense(128, activation='relu'))

model.add(Dense(10, activation='softmax'))#last layer is Softmax since we will be outputting the probability distribution and not a single prediction



# Compile model

#model.compile(optimizer='rmsprop',#RMSProp is a starting point and we will try Adam optimizer which is mix of RMSProp and Momentum

#              loss='categorical_crossentropy',#since  we have a multilabel output

#              metrics=['accuracy'])



adam = optimizers.Adam(lr=ADAM_LEARNING_RATE,amsgrad=True,decay=ADAM_DECAY_RATE)

model.compile(optimizer=adam,#RMSProp is a starting point and we will try Adam optimizer which is mix of RMSProp and Momentum

              loss='categorical_crossentropy',#since  we have a multilabel output

              metrics=['accuracy'])



# Train model

history = model.fit(TRAIN_X, RAW_TRAIN_DATA_Y_,

          batch_size=BATCH_SIZE,

          epochs=EPOCHS,

          callbacks=[early_stop])        
accuracy  = model.evaluate(x=TEST_X,y=RAW_TEST_DATA_Y_,batch_size=32)
print('Test set accuracy is {} %'.format(accuracy[1]*100))
history_df = pd.DataFrame(history.history)

sns.lineplot(y='loss',data=history_df,x=history_df.index,legend='brief')

sns.lineplot(y='acc',data=history_df,x=history_df.index,legend='brief')