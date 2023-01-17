import os, cv2, random

import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

from matplotlib import ticker

import seaborn as sns

%matplotlib inline 



from keras.models import Sequential

from keras.layers import Input, Dropout, Flatten, Convolution2D, MaxPooling2D, Dense, Activation

from keras.optimizers import RMSprop

from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping

from keras.utils import np_utils

from keras.preprocessing.image import ImageDataGenerator

from matplotlib import pyplot
TRAIN_DIR = '../input/train/'

TEST_DIR = '../input/test/'



ROWS = 32

COLS = 32

CHANNELS = 3



train_images = [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR)] # use this for full dataset

train_dogs =   [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR) if 'dog' in i]

train_cats =   [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR) if 'cat' in i]



test_images =  [TEST_DIR+i for i in os.listdir(TEST_DIR)]





# slice datasets for memory efficiency on Kaggle Kernels, delete if using full dataset

train_images = train_dogs[:1000] + train_cats[:1000]

random.shuffle(train_images)

test_images =  test_images[:30]



def read_image(file_path):

    img = cv2.imread(file_path, cv2.IMREAD_COLOR) #cv2.IMREAD_GRAYSCALE

    return cv2.resize(img, (ROWS, COLS), interpolation=cv2.INTER_CUBIC)





def prep_data(images):

    count = len(images)

    data = np.ndarray((count, CHANNELS, ROWS, COLS), dtype=np.float64)



   

    

    for i, image_file in enumerate(images):

        image = read_image(image_file)

        data[i] = image.T

        if i%250 == 0: print('Processed {} of {}'.format(i, count))

    

    return data



train = prep_data(train_images)

test = prep_data(test_images)



#train = train.reshape(train.shape[0], CHANNELS, ROWS, COLS)

#test = test.reshape(test.shape[0], CHANNELS, ROWS, COLS)

# convert from int to float

#train = train.astype('float32')

#test = test.astype('float32')



print("Train shape: {}".format(train.shape))

print("Test shape: {}".format(test.shape))
#labels = []

#for i in train_images:

 #   if 'dog' in i:

  #      labels.append(1)

  #  else:

   #     labels.append(0)



labels = np.ndarray(len(train), dtype=np.float64)

j = 0

for i in train_images:

    if 'dog' in i:

        labels[j] = 1

    else:

        labels[j] = 0

    j+=1
optimizer = RMSprop(lr=1e-4)

objective = 'binary_crossentropy'





def catdog():

    

    print("Make model")

    

    model = Sequential()



    model.add(Convolution2D(16, 4, 4, border_mode='same', input_shape=(3, ROWS, COLS), activation='relu', dim_ordering = 'th'))

    #model.add(Convolution2D(32, 3, 3, border_mode='same', activation='relu', dim_ordering = 'th'))

    model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering = 'th'))



    model.add(Convolution2D(32, 4, 4, border_mode='same', activation='relu', dim_ordering = 'th'))

    #model.add(Convolution2D(64, 4, 4, border_mode='same', activation='relu', dim_ordering = 'th'))

    model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering = 'th'))

    

    model.add(Convolution2D(32, 4, 4, border_mode='same', activation='relu', dim_ordering = 'th'))

    model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering = 'th'))

    

    #model.add(Convolution2D(16, 3, 3, border_mode='same', activation='relu', dim_ordering = 'th'))

    #model.add(Convolution2D(64, 4, 4, border_mode='same', activation='relu', dim_ordering = 'th'))

    #model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering = 'th'))

    

   # model.add(Convolution2D(256, 3, 3, border_mode='same', activation='relu'))

   # model.add(Convolution2D(256, 3, 3, border_mode='same', activation='relu'))

   # model.add(Convolution2D(256, 3, 3, border_mode='same', activation='relu'))

   # model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering = 'th'))



#     model.add(Convolution2D(256, 3, 3, border_mode='same', activation='relu'))

#     model.add(Convolution2D(256, 3, 3, border_mode='same', activation='relu'))

#     model.add(Convolution2D(256, 3, 3, border_mode='same', activation='relu'))

#     model.add(MaxPooling2D(pool_size=(2, 2)))



    model.add(Flatten())

    #model.add(Dense(75, activation='relu'))

    model.add(Dense(25, activation='relu'))

    model.add(Dropout(0.1))

    

    #model.add(Dense(256, activation='relu'))

    #model.add(Dropout(0.5))



    model.add(Dense(1))

    model.add(Activation('sigmoid'))



    model.compile(loss=objective, optimizer=optimizer, metrics=['accuracy'])

    

    print("Start training")

    

    return model





model = catdog()
nb_epoch = 5

batch_size = 50



## Callback for loss logging per epoch

class LossHistory(Callback):

    def on_train_begin(self, logs={}):

        self.losses = []

        self.val_losses = []

        

    def on_epoch_end(self, batch, logs={}):

        self.losses.append(logs.get('loss'))

        self.val_losses.append(logs.get('val_loss'))



early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='auto')        

        

def run_catdog():

    



    

    history = LossHistory()

    print("fitting")

    



    

    datagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=False, dim_ordering='th',

                                featurewise_center=True,

                                featurewise_std_normalization=True)

    test_datagen = ImageDataGenerator()

    validation_generator = test_datagen.flow(train, labels, batch_size=batch_size)

    datagen.fit(train)

    

    #for X_batch, y_batch in datagen.flow(train, labels, batch_size=9):

	# create a grid of 3x3 images

     #   for i in range(0, 9):

      #      pyplot.subplot(330 + 1 + i)

       #     pyplot.imshow(X_batch[i].reshape(28, 28), cmap=pyplot.get_cmap('gray'))

	# show the plot

       # pyplot.show()

       # break

    

    #X_batch, y_batch = datagen.flow(train, train, batch_size=32)

    

    #model.fit(train, labels, batch_size=batch_size, nb_epoch=nb_epoch,

          #    validation_split=0.25, verbose=0, shuffle=True, callbacks=[history, early_stopping])

    

    

    model.fit_generator(datagen.flow(train, labels, batch_size=batch_size), 

                        samples_per_epoch=len(train) * 0.75,nb_epoch=nb_epoch,

    #model.fit_generator(datagen, samples_per_epoch=len(train),nb_epoch=nb_epoch)

              #validation_split=0.25, shuffle=True,

              validation_data=validation_generator,

              nb_val_samples = len(train) * 0.25,

              verbose=1, callbacks=[history, early_stopping])

    



    predictions = model.predict(test, verbose=0)

    return predictions, history



predictions, history = run_catdog()
#loss = history.losses

val_loss = history.val_losses

#val_acc = history.val_acc



print(history.val_losses)



plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.title('VGG-16 Loss Trend')

#plt.plot(loss, 'blue', label='Training Loss')

plt.plot(val_loss, 'green', label='Validation Loss')

#plt.plot(val_acc, 'red', label='Validation acc')

plt.xticks(range(0,nb_epoch)[0::2])

plt.legend()

plt.show()
for i in range(0,10):

    if predictions[i, 0] >= 0.5: 

        print('I am {:.2%} sure this is a Dog'.format(predictions[i][0]))

    else: 

        print('I am {:.2%} sure this is a Cat'.format(1-predictions[i][0]))

        

    plt.imshow(test[i].T)

    plt.show()