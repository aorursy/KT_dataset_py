from keras.models import Sequential

import numpy as np

from keras.layers.core import Dense, Dropout, Activation

from keras.layers import Conv2D, MaxPooling2D, Flatten

from keras.optimizers import SGD, Adam

from keras.utils import np_utils

#from keras.datasets import mnist

import pandas as pd
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
def load_data():

    train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

    test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv').values



    x_train = train.iloc[:,1:].values.astype('float32')



    # y_train is the label for each of x_train data

    y_train = train.iloc[:,0].values.astype('float32')



    x_test = test.astype('float32')



    # our training data are the images that need to be transfromed as numpy array (two-dimension matrix)

    # each of (28* 28) image will be saved as a vector that only has one row 

    # the number of rows of training data depends on how many data we have

    # the number of columns of training data depends on the dimension of the image (in this case are 28 * 28 = 784)

    x_train = x_train.reshape(x_train.shape[0], 28 * 28)

    x_test = x_test.reshape(x_test.shape[0], 28 * 28)



    # convert class vectors to binary class matrices 

    y_train = np_utils.to_categorical(y_train,10)

    x_train = x_train

    x_test = x_test



    # add some noises

    x_test = np.random.normal(x_test) 



    # the inputs need to be divided by 255 since they are image pixel so as to ensure the value is between 0 ~ 1

    x_train = x_train / 255

    x_test = x_test / 255 

    

    return x_train, y_train, x_test
if __name__ == '__main__':

    # load training data and testing data

    x_train, y_train, x_test = load_data()

    # input channels: x_train.shape[0]

    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)

    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)



    model = Sequential()

    model.add(Conv2D(100, (3, 3), input_shape = (28, 28, 1)))

    model.add(MaxPooling2D((2, 2)))

    model.add(Dropout(0.5))



    model.add(Conv2D(100, (3, 3)))

    model.add(MaxPooling2D((2, 2)))

    model.add(Dropout(0.5))



    model.add(Flatten())

    model.add(Dense(units = 200, activation = 'relu'))

    model.add(Dense(units = 10, activation='softmax'))

    model.summary()



    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])



    model.fit(x_train, y_train, batch_size = 100, epochs = 30)



    result_train = model.evaluate(x_train, y_train)

    pred_result = model.predict_classes(x_test,verbose=0)



    submission = pd.DataFrame({'ImageId': list(range(1,len(pred_result)+1)), 'Label': pred_result}).to_csv('Digit Recognizer.csv', index=False, header=True)

    print('Training Accuracy:',result_train[1])
