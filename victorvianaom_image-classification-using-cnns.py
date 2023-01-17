from keras.datasets import mnist

import keras



def load_and_preprocess():

    r, c = 28, 28

    num_classes = 10

    (x_train, y_train), (x_test, y_test) = mnist.load_data()



    x_train = x_train.reshape(x_train.shape[0], r, c, 1)

    x_test = x_test.reshape(x_test.shape[0], r, c, 1)



    input_shape = (r, c, 1)

    

    x_train = x_train.astype('float32')

    x_test = x_test.astype('float32')

    

    x_train /= 255

    x_test /= 255

    

    y_train = keras.utils.to_categorical(y_train, num_classes)

    y_test = keras.utils.to_categorical(y_test, num_classes)

    

    return (x_train,x_test,y_train,y_test,input_shape)
from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D



x_train, x_test, y_train, y_test, input_shape = load_and_preprocess()

num_classes=10



model = Sequential()



#Now defining 2 convolution layers

model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=input_shape))

model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(num_classes, activation='softmax'))



# Next, we minimize errors by using the AdaDelta algorithm:

model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])



model.fit(x_train, y_train, batch_size=10, epochs=10, validation_data=(x_test, y_test))



score = model.evaluate(x_test, y_test, verbose=0)



print('Test loss:', score[0])

print('Test accuracy:', score[1])