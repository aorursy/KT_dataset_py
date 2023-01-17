import sys
import keras
import keras as ks
import matplotlib
from matplotlib import pyplot
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import Adam
(X_train,Y_train),(X_test,Y_test) = cifar10.load_data()

# Ananlyzing the shape of the data(Should be the first step of training)
print("shape of X_train and Y_train "+ str(X_train.shape)+" "+str(Y_train.shape))
print("shape of X_test and Y_test "+ str(X_test.shape)+" "+str(Y_test.shape))
for i in range(5):
    pyplot.subplot(330 + 1 + i)
    pyplot.imshow(X_train[i])
    print(Y_train[i])
pyplot.show()
def load():
    (X_train,Y_train),(X_test,Y_test) = cifar10.load_data()
    Y_train=ks.utils.to_categorical(Y_train)
    Y_test=ks.utils.to_categorical(Y_test)
    return X_train,Y_train,X_test,Y_test
def normalize(train,test):
    trainNorm=train/255.0
    testNorm=test/255.0
    return trainNorm,testNorm
def getmodel():
    # Adding sequential to add layers sequentially as per our needs.
    model = ks.Sequential()
    model.add(Conv2D(32,(3,3),activation='relu',kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
    model.add(Conv2D(32,(3,3),activation='relu',kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2,2)))
    model.add(Conv2D(64,(3,3),activation='relu',kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(64,(3,3),activation='relu',kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2,2)))
    model.add(Conv2D(128,(3,3),activation='relu',kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(128,(3,3),activation='relu',kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2,2)))
    
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(10, activation='softmax'))
    
    opt = Adam(learning_rate=0.01,name="Adam")
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model
def summarize_diagnostics(history):
    # plot loss
    pyplot.subplot(211)
    pyplot.title('Cross Entropy Loss')
    pyplot.plot(history.history['loss'], color='blue', label='train')
    pyplot.plot(history.history['val_loss'], color='orange', label='test')
    # plot accuracy
    pyplot.subplot(212)
    pyplot.title('Classification Accuracy')
    pyplot.plot(history.history['accuracy'], color='blue', label='train')
    pyplot.plot(history.history['val_accuracy'], color='orange', label='test')
    # save plot to file
    filename = sys.argv[0].split('/')[-1]
    pyplot.savefig(filename + '_plot.png')
    pyplot.close()
def trainModel():
    X_train,Y_train,X_test,Y_test=load()
    X_train,X_test=normalize(X_train,X_test)
    model=getmodel()
    history=model.fit(X_train,Y_train,epochs=100,batch_size=64, validation_data=(X_test, Y_test), verbose=0)
    summarize_diagnostics(history)
    _, acc = model.evaluate(X_test, Y_test, verbose=0)
    print('> %.3f' % (acc * 100.0))
trainModel()
