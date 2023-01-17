from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Activation

from tensorflow.keras.layers import Flatten

from tensorflow.keras.layers import Dense

import tensorflow as tf

import numpy as np

import matplotlib.pyplot as plt

plt.style.use('ggplot')



def visualizar_img (class_name):

    for i in range(16):

        plt.subplot(4,4,i+1)

        plt.xticks([])

        plt.yticks([])

        plt.grid(False)

        plt.imshow(X_train[i], cmap=plt.cm.binary)

    plt.show()
from tensorflow.keras.datasets import cifar10

(X_train, y_train), (X_test, y_test) = cifar10.load_data()



nb_epoch = 200

batch_size = 128

num_classes = 10



class_names = ['Avião', 'Automóvel', 'Pássaro', 'Gato', 'Cervo', 'Cachorro', 'Sapo', 'Cavalo', 'Barco', 'Caminhão']



visualizar_img(class_names)

X_train.shape
X_test.shape
y_train.shape
import tensorflow as tf

n_classes = 10



X_train = X_train.reshape(50000, 32 * 32 * 3)

X_test = X_test.reshape(10000, 32 * 32 * 3)



X_train = X_train.astype('float32')

X_test = X_test.astype('float32')

X_train /= 255.0

X_test /= 255.0



y_train = tf.keras.utils.to_categorical(y_train)

y_test = tf.keras.utils.to_categorical(y_test)
y_train
X_train
from keras.datasets import cifar10

from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation, Flatten

from keras.utils import np_utils



num_classes = 10



model = Sequential()

model.add(Dense(1024, input_shape=(3072, )))

model.add(Activation('relu'))

model.add(Dropout(0.2))

model.add(Dense(512))

model.add(Activation('relu'))

model.add(Dropout(0.2))

model.add(Dense(512))

model.add(Activation('relu'))

model.add(Dropout(0.2))

model.add(Dense(10))

model.add(Activation('softmax'))



model.summary()

model.compile(loss="categorical_crossentropy", optimizer='adam', metrics = ['accuracy'])
cb = [ tf.keras.callbacks.EarlyStopping(monitor='val_loss',

           patience=500,

           restore_best_weights=True)

       ]



print('treinamento...')



# training

history = model.fit(X_train, y_train,

                    batch_size=batch_size,

                    nb_epoch=nb_epoch,

                    verbose=1,

                    validation_data=(X_test, y_test))



#history = model.fit(X_train, 

#          y_train, 

#          epochs=2000, 

#          callbacks=cb,

#          verbose=2,

#          validation_data=(X_test, y_test),

#          )

print(history)

    


import matplotlib.pyplot as plt



print(history.history.keys())



plt.plot(history.history['accuracy'])



plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()





plt.plot(history.history['loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
loss, acc = model.evaluate(X_test, y_test, verbose=1)



print('Test loss:', loss)

print('Test acc:', acc)


predict = model.predict(X_test)



predict_classes = model.predict_classes(X_test)



predict, predict_classes
(X_train, y_train), (X_test, y_test) = cifar10.load_data()



from sklearn.metrics import classification_report



print(classification_report(y_test, predict_classes))






