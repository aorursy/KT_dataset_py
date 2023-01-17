import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import keras

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2

from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from keras.optimizers import rmsprop
from keras.callbacks import CSVLogger
import os
import random
import gc
random.seed(1)


files = []

os.chdir('/kaggle/working/')

for dirname, _, filenames in os.walk('/kaggle/input/animals10/raw-img'):
    for filename in filenames:
        dirname_arr = dirname.split("/")
        if(random.random() < 0.2):
            files.append(os.path.join(dirname, filename) +','+dirname_arr[5]+',test')
        else:
            files.append(os.path.join(dirname, filename)+','+dirname_arr[5]+',train')
            
files = np.array(files)
print(files.size)

def get_image_data(file):
    img = image.load_img(file, target_size=(100, 100, 3))

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis = 0)
    return x
    
x_train = []
y_train = []
x_test = []
y_test = []
animals = ('gatto', 'elefante', 'cane', 'scoiattolo', 'mucca', 'farfalla', 'gallina', 'cavallo', 'ragno', 'pecora')

for i in range(1, files.size):
    file_path, animal, phase  = files[i].split(',')
    
    animal_types = keras.utils.to_categorical(animals.index(animal), 10)
    image_data = np.array(get_image_data(file_path))
    
    if phase == 'train':
        x_train.append(image_data)
        y_train.append(animal_types)
    elif phase == 'test':
        x_test.append(image_data)
        y_test.append(animal_types)
    print('Bir '+animal+' '+phase+' verisine eklendi. '+str(i)+'. veri')
        
        
x_train = np.array(x_train, 'float32')
y_train = np.array(y_train, 'float32')
x_test = np.array(x_test, 'float32')
y_test = np.array(y_test, 'float32')

x_train /= 255
x_test /= 255

x_train = x_train.reshape(x_train.shape[0], 100, 100, 3)
x_train = x_train.astype('float32')
x_test = x_test.reshape(x_test.shape[0], 100, 100, 3)
x_test = x_test.astype('float32')

print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
batch_size = 53
epochs = 69
gc.collect()
alexnet = Sequential()

# Layer 1
alexnet.add(Conv2D(96, (11, 11), input_shape=(100, 100, 3), padding='same', kernel_regularizer=l2(0)))
alexnet.add(BatchNormalization())
alexnet.add(Activation('relu'))
alexnet.add(MaxPooling2D(pool_size=(2, 2)))

# Layer 2
alexnet.add(Conv2D(256, (5, 5), padding='same'))
alexnet.add(BatchNormalization())
alexnet.add(Activation('relu'))
alexnet.add(MaxPooling2D(pool_size=(2, 2)))

# Layer 3
alexnet.add(ZeroPadding2D((1, 1)))
alexnet.add(Conv2D(512, (3, 3), padding='same'))
alexnet.add(BatchNormalization())
alexnet.add(Activation('relu'))
alexnet.add(MaxPooling2D(pool_size=(2, 2)))

# Layer 4
alexnet.add(ZeroPadding2D((1, 1)))
alexnet.add(Conv2D(1024, (3, 3), padding='same'))
alexnet.add(BatchNormalization())
alexnet.add(Activation('relu'))

# Layer 5
alexnet.add(ZeroPadding2D((1, 1)))
alexnet.add(Conv2D(1024, (3, 3), padding='same'))
alexnet.add(BatchNormalization())
alexnet.add(Activation('relu'))
alexnet.add(MaxPooling2D(pool_size=(2, 2)))

# Layer 6
alexnet.add(Flatten())
alexnet.add(Dense(3072))
alexnet.add(BatchNormalization())
alexnet.add(Activation('relu'))
alexnet.add(Dropout(0.5))

# Layer 7
alexnet.add(Dense(4096))
alexnet.add(BatchNormalization())
alexnet.add(Activation('relu'))
alexnet.add(Dropout(0.5))

# Layer 8
alexnet.add(Dense(10))
alexnet.add(BatchNormalization())
alexnet.add(Activation('softmax'))

fit = True #Train etmek için fit=True olarak değiştiriniz.

if fit == True:
    gen = ImageDataGenerator(horizontal_flip=True)
    gen.fit(x_train)
    train_generator = gen.flow(x_train, y_train, batch_size=batch_size)

    alexnet.compile(loss='categorical_crossentropy'
        , optimizer=rmsprop(lr=0.0001, decay=1e-6)
        , metrics=['accuracy']
    )
    #alexnet.fit_generator(x_train, y_train, epochs=epochs, callbacks=callbacks, verbose=1) #Tüm veri kümesi için eğit
    alexnet.fit_generator(train_generator, validation_data=(x_test, y_test), steps_per_epoch=batch_size, epochs=epochs, verbose=1, callbacks=[CSVLogger('training_animal.log', separator=',', append=False)]) #rastgele bir eğtiim yap
    alexnet.save('animal.h5')
else:
    alexnet = keras.models.load_model('/kaggle/working/animal.h5') #Öğrenilmiş ağırlıkları yükle
    alexnet.history = pd.read_csv('training_animal.log', sep=',', engine='python')
def loss_accuracy_graph(history):
    loss_history = history.history['loss']
    acc_history = history.history['accuracy']
    val_loss_history = history.history['val_loss']
    val_acc_history = history.history['val_accuracy']
    epoch = [(i + 1) for i in range(epochs)]

    plt.rcParams.update({'font.size': 12})
    plt.plot(epoch, loss_history, color='red', label='Eğitim Hatası')
    plt.plot(epoch, val_loss_history, color='orange', label='Test Hatası')
    plt.xlabel('Epoch')
    plt.ylabel('Kayıp')
    plt.title('Hata Grafiği\n')
    plt.legend()
    plt.show()
    plt.close()
    
    plt.plot(epoch, acc_history, color='green', label='Eğitim Başarısı')
    plt.plot(epoch, val_acc_history, color='blue', label='Test Başarısı')
    plt.xlabel('Epoch')
    plt.ylabel('Kazanç')
    plt.title('Kazanç Grafiği\n')
    plt.legend()
    plt.show()
    plt.close()
test_img_path = "/kaggle/input/validation/fil1.jpg"

img_orj = image.load_img(test_img_path)
img = image.load_img(test_img_path, target_size=(100, 100, 3))

x = image.img_to_array(img)
x = np.expand_dims(x, axis = 0)

x /= 255

custom = alexnet.predict(x)

custom[0]=(custom[0]/1)*100

if fit == True:
    loss_accuracy_graph(alexnet.history)
else:
    loss_accuracy_graph(alexnet)
#1
objects = ('kedi', 'fil', 'köpek', 'sincap', 'inek', 'kelebek', 'tavuk', 'at', 'örümcek', 'koyun')

y_pos = np.arange(len(objects))

plt.rcParams.update({'font.size': 12})
plt.bar(y_pos, custom[0], align='center', alpha=0.5, color='g')
plt.xticks(y_pos, objects, rotation='vertical')
plt.ylabel('yüzde')
plt.title('tür')
plt.show()

#2
x = np.array(x, 'float32')
x = x.reshape([100, 100, 3]);
plt.axis('off')
plt.gray()
plt.imshow(img_orj)
plt.show()