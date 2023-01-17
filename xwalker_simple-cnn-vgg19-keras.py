import matplotlib.pyplot as plt

from glob import glob

from skimage.io import imread

from skimage.color import rgb2gray

from sklearn.model_selection import train_test_split

import keras

import numpy as np



from keras.models import Sequential

from keras.layers import (  Dense,

                            Flatten,

                            LeakyReLU

                         )

from keras.applications import  VGG19 

from keras.optimizers import Adam

from keras.utils.np_utils import to_categorical

from skimage.transform import resize

from tqdm import tqdm

from keras.preprocessing.image import ImageDataGenerator
def hist(History):

    fig, ax = plt.subplots(1, 2, figsize=(20, 5))



    ax[0].plot(History.history['loss'])

    ax[0].plot(History.history['val_loss'])

    ax[0].legend(['Training loss', 'Validation Loss'],fontsize=18)

    ax[0].set_xlabel('Epochs ',fontsize=16)

    ax[0].set_ylabel('Loss',fontsize=16)

    ax[0].set_title('Training loss x Validation Loss',fontsize=16)





    ax[1].plot(History.history['acc'])

    ax[1].plot(History.history['val_acc'])

    ax[1].legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)

    ax[1].set_xlabel('Epochs ',fontsize=16)

    ax[1].set_ylabel('Accuracy',fontsize=16)

    ax[1].set_title('Training Accuracy x Validation Accuracy',fontsize=16)
def plot_any(arr, title = ''):

    plt.figure(figsize = (15, 25))

    for i in range(len(arr)):

        plt.subplot(1,len(arr),i + 1)

        plt.title(title)

        plt.imshow(arr[i], cmap = 'gray');
path = '../input/skin-cancer-malignant-vs-benign/data/'

train_path = glob(path + 'train/*')

test_path = glob(path + 'test/*')
train_imgs = []

train_labels = []

test_imgs  = []

test_labels = []



x, y, z = 224, 224, 3



#train

for klass, folder in enumerate(tqdm(train_path)):

    for img in glob(folder + '/*'):

        

        img_resize = imread(img)

        img_resize = resize(img_resize, (x, y, z))



        train_imgs.append(img_resize)

        train_labels.append(klass)

        

#test

for klass, folder in enumerate(tqdm(test_path)):

    for img in glob(folder + '/*'):

        

        img_resize = imread(img)

        img_resize = resize(img_resize, (x, y, z))



        test_imgs.append(img_resize)

        test_labels.append(klass)
print('Treino: {} \nTeste: {}'.format(len(train_imgs), len(test_imgs)))
train_imgs = np.asarray(train_imgs)

test_imgs = np.asarray(test_imgs)



train_labels =  np.asarray(train_labels)

test_labels =  np.asarray(test_labels)
NUM_CLASSES = 1

EPOCHS = 15

BATCH_SIZE = 64

inputShape = (x, y, z)
X_train, X_test, y_train, y_test = train_test_split(

    train_imgs, 

    train_labels,

    test_size = 0.3, 

)
train_datagen = ImageDataGenerator( rescale = 1./255,

                                    rotation_range=90,

                                    width_shift_range=0.15,

                                    height_shift_range=0.15,

                                    horizontal_flip = True,

                                    vertical_flip = True,

                                    zoom_range=(0.9,1),

                                    fill_mode= 'nearest',

                                    brightness_range=(0.8, 1.2),

                                  )



train_generator = train_datagen.flow(X_train, y_train, batch_size = BATCH_SIZE)

val_generator = train_datagen.flow(X_test, y_test, batch_size = BATCH_SIZE, shuffle = True)
model = Sequential()

model.add(VGG19(include_top=False, weights='imagenet', input_shape= inputShape))

model.add(Flatten())

model.add(Dense(32))

model.add(LeakyReLU(0.001))

model.add(Dense(16))

model.add(LeakyReLU(0.001))

model.add(Dense(1, activation='sigmoid'))

model.layers[0].trainable = False



model.compile(loss=keras.losses.binary_crossentropy, optimizer='adam', metrics=['acc'])



History = model.fit_generator(train_generator,

    steps_per_epoch = len(train_imgs) // BATCH_SIZE,

    epochs = EPOCHS, 

    validation_data = val_generator,

    validation_steps = len(test_imgs) // BATCH_SIZE,

)
loss, accu = model.evaluate(test_imgs, test_labels)

print("%s: %.2f%%" % ('Accuracy...', accu))

print("%s: %.2f" % ('loss.......', loss))
hist(History)