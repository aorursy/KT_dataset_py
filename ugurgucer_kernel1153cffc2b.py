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
types = []

os.chdir('/kaggle/working/')

for dirname, _, filenames in os.walk('/kaggle/input/100-bird-species/'):
    for filename in filenames:
        dirname_arr = dirname.split("/")
        if(dirname_arr[4] == 'train' or dirname_arr[4] == 'test'):
            files.append(os.path.join(dirname, filename) +','+dirname_arr[5]+','+dirname_arr[4])
            types.append(dirname_arr[5])
            
files = np.array(files)
types = np.array(types)
types = np.unique(types)
types = np.sort(types)

print(files.size)
def get_image_data(file):
    with image.load_img(file, target_size=(112, 112, 3)) as img:
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis = 0)
        img.close()
    return x    
x_train =  []
y_train = []
x_valid =  []
y_valid = []

for i in range(1, files.size):
    file_path, bird, phase  = files[i].split(',')
    
    bird_types = keras.utils.to_categorical(np.where(types == bird)[0][0], 200)
    image_data = np.array(get_image_data(file_path))
    
    if phase == 'train':
        x_train.append(image_data)
        y_train.append(bird_types)
    elif phase == 'test':
        x_valid.append(image_data)
        y_valid.append(bird_types)

    del bird_types
    del image_data
    del file_path
    del bird
    del phase

x_train = np.array(x_train, 'float32')
y_train = np.array(y_train, 'float32')
x_valid = np.array(x_valid, 'float32')
y_valid = np.array(y_valid, 'float32')

x_train /= 255
x_valid /= 255

x_train = x_train.reshape(x_train.shape[0], 112, 112, 3)
x_train = x_train.astype('float32')
x_valid = x_valid.reshape(x_valid.shape[0], 112, 112, 3)
x_valid = x_valid.astype('float32')

print(x_train.shape[0], 'train samples')
print(x_valid.shape[0], 'test samples')
batch_size = 32
epochs = 128
gc.collect()
alexnet = Sequential()

# Layer 1
alexnet.add(Conv2D(96, (11, 11), input_shape=(112, 112, 3), padding='same', kernel_regularizer=l2(0)))
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
alexnet.add(Dense(200))
alexnet.add(BatchNormalization())
alexnet.add(Activation('softmax'))

fit = False #Train etmek için fit=True olarak değiştiriniz.

if fit == True:
    gen = ImageDataGenerator(horizontal_flip=True)
    gen.fit(x_train)
    train_generator = gen.flow(x_train, y_train, batch_size=batch_size)

    alexnet.compile(loss='categorical_crossentropy'
        , optimizer=rmsprop(lr=0.0001, decay=1e-6)
        , metrics=['accuracy']
    )
    #alexnet.fit_generator(x_train, y_train, epochs=epochs, callbacks=callbacks, verbose=1) #Tüm veri kümesi için eğit
    alexnet.fit_generator(train_generator, validation_data=(x_valid, y_valid), steps_per_epoch=batch_size, epochs=epochs, verbose=1, callbacks=[CSVLogger('training_bird.log', separator=',', append=False)]) #rastgele bir eğtiim yap
    alexnet.save('birds.h5')
else:
    alexnet = keras.models.load_model('/kaggle/working/birds.h5') #Öğrenilmiş ağırlıkları yükle
    alexnet.history = pd.read_csv('training_bird.log', sep=',', engine='python')
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
    
def get_n_best_fit(data, labels, n):
    values = np.sort(data)
    indexes = np.argsort(data)
    best_n_data = values[-1*n:]
    best_n_data_indxs = indexes[-1*n:]
    
    return best_n_data[::-1], labels[best_n_data_indxs][::-1]

test_img_path = "/kaggle/input/100-bird-species/valid/VERMILION FLYCATHER/1.jpg"

img_orj = image.load_img(test_img_path)
img = image.load_img(test_img_path, target_size=(112, 112, 3))

x = image.img_to_array(img)
x = np.expand_dims(x, axis = 0)

x /= 255

custom = alexnet.predict(x)
best_n_fit, best_label = get_n_best_fit(custom[0], types, 10)

best_n_fit = (best_n_fit / 1) * 100
if fit == True:
    loss_accuracy_graph(alexnet.history)
else:
    loss_accuracy_graph(alexnet)

y_pos = np.arange(len(best_label))

plt.rcParams.update({'font.size': 12})
plt.bar(y_pos, best_n_fit, align='center', alpha=0.5, color='g')
plt.xticks(y_pos, best_label, rotation='vertical')
plt.ylabel('Benzerlik (%)')
plt.title('Kuş Türleri')
#plt.savefig('books_read.png', dpi= 2400)
plt.show()

#2
x = np.array(x, 'float32')
x = x.reshape([112, 112, 3]);
plt.axis('off')
plt.gray()
plt.imshow(img_orj)
plt.show()