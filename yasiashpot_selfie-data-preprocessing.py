import pandas as pd

import numpy as np

import tensorflow as tf

from tensorflow.keras.layers import Dropout, Dense, Conv2D, MaxPool2D, Flatten

from tensorflow.keras.callbacks import TensorBoard

import matplotlib.pyplot as plt

import seaborn as sns

import random

import time

print(tf.__version__)



import os

import csv

import sys



from keras.preprocessing.image import ImageDataGenerator

from keras.preprocessing.image import img_to_array

from keras.preprocessing.image import load_img

import cv2

from PIL import Image

from skimage.transform import resize



from sklearn.model_selection import train_test_split

from keras.utils import Sequence,to_categorical



from keras.models import Sequential

from keras.models import Model

from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau, TensorBoard

from keras import optimizers, losses, activations, models

from keras.layers import Convolution2D, Dense, Input, Flatten, Dropout, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D, Concatenate

from keras import applications



%matplotlib inline
!ls /kaggle/input/selfie-classification-basic-dataset
root_path = "/kaggle/input/selfie-classification-basic-dataset"

image_names_csv = os.path.join(root_path, "merged_dataset.csv")

images_folder = os.path.join(root_path, "Merged")



p = pd.read_csv(image_names_csv)

print(p.head(5),'\n_____________________________')



p = p.sample(frac=1, axis=0) # shuffling the content to ensure the model doesn't learn about the order of the items

#p = p.head(10000)



p['filename'] = p['filename'].apply(lambda x: os.path.join(images_folder,x))

print(p.head(5),'\n_____________________________')

p.reset_index(drop=True, inplace = True)

print(p.head(5))
def load_img_as_arr(img_path):

    #return np.array(Image.open(img_path).resize((299,299)))

    img = cv2.imread(img_path)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    #cv2.imshow("Resized",img.resize(300,300))

    return cv2.resize(img, (299, 299), interpolation = cv2.INTER_AREA)



def preprocess_img(img_path):

    img = load_img_as_arr(img_path)

    img = tf.cast(img, tf.float32)

    img = (img / 255.)

    img = (img - 0.5)

    img = (img * 2.)

    return img
#valid_idx = [i for i, img_path in X if load_image_as_arr(img_path).shape == (299,299,3)]

not_valid_idx = []

#valid_idx = []



for i, img_path in enumerate(p['filename']):

    

    if img_path[-3:].lower() == 'gif':

        not_valid_idx.append(i)

        continue

        

    img =  load_img_as_arr(img_path)

    #if img.shape == (299,299,3):

        #valid_idx.append(i)

    if img.shape == (299,299,4) or img.shape == (299,299):

        print(img_path, "woooooooooooooooops!")

        not_valid_idx.append(i)

        

print(len(not_valid_idx))

        

        
# valid_y = np.array([c for i, c in enumerate(y) if i in valid_idx])

# y = valid_y

# valid_x = np.array([x for i, x in enumerate(X) if i in valid_idx])

# X = valid_x

# print(y.shape, X.shape)





# X = np.array(X.drop(not_valid_idx))

# y = np.array(y.drop(not_valid_idx))



p.drop(not_valid_idx, inplace = True)

p.reset_index(drop=True, inplace = True)

X = np.array(p['filename'])

y = np.array(p['class'])



print(X,'\n_____________________________\n', y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, stratify=y, random_state=123, shuffle=True)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.176, stratify=y_train, random_state=123, shuffle=True)



print(X_train.shape, X_val.shape, X_test.shape, type(X_train))
partition = dict()



partition['train'] = X_train

partition['test'] = X_test

partition['validation'] = X_val



labels = { p['filename'][i] : p['class'][i] for i in range(p.shape[0]) }



# zipbObj = zip(p['filename'].tolist(), p['class'].tolist())

# labels = dict(zipbObj)
# def x_to_tensor(X):

#     #%%time

#     images_preprocessed = np.zeros((X.shape[0], 299, 299, 3), dtype=np.float32)



#     for i, img_path in enumerate(X):

#         img_resize = preprocess_img(img_path)

#         if img_resize.shape != (299,299,3):

#             print("woooooooooooooooops!")

#             print(img_resize.shape)

#             continue

#         images_preprocessed[i] = np.dstack([img_resize])

#     return images_preprocessed

    
# %%time

#train_images_preprocess = x_to_tensor(X_train)

#test_images_preprocess = x_to_tensor(X_test)
class DataGenerator(Sequence):

    'Generates data for Keras'

    def __init__(self, list_IDs, labels, batch_size=32, dim=(299,299), n_channels=3,

                 n_classes=3, shuffle=True):

        'Initialization'

        self.dim = dim

        self.batch_size = batch_size

        self.labels = labels

        self.list_IDs = list_IDs

        self.n_channels = n_channels

        self.n_classes = n_classes

        self.shuffle = shuffle

        self.on_epoch_end()



    def __len__(self):

        'Denotes the number of batches per epoch'

        return int(np.floor(len(self.list_IDs) / self.batch_size))



    def __getitem__(self, index):

        'Generate one batch of data'

        # Generate indexes of the batch

        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]



        # Find list of IDs

        list_IDs_temp = [self.list_IDs[k] for k in indexes]



        # Generate data

        X, y = self.__data_generation(list_IDs_temp)



        return X, y



    def on_epoch_end(self):

        'Updates indexes after each epoch'

        self.indexes = np.arange(len(self.list_IDs))

        if self.shuffle == True:

            np.random.shuffle(self.indexes)



    def __data_generation(self, list_IDs_temp):

        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)

        # Initialization

        # X = np.empty((self.batch_size, *self.dim, self.n_channels))

        X = np.zeros((self.batch_size, *self.dim, self.n_channels), dtype=np.float32)

        y = np.empty((self.batch_size), dtype=int)



        for i, ID in enumerate(list_IDs_temp):

            img_resize = preprocess_img(ID)

            if img_resize.shape != (299,299,3):

                continue

            # X[i,] = np.load('data/' + ID + '.npy')

            X[i] = np.dstack([img_resize])

            y[i] = self.labels[ID]

        return  X, to_categorical(y, num_classes=self.n_classes)
# Parameters

params = {'dim': (299, 299),

          'batch_size': 32,

          'n_classes': 3,

          'n_channels': 3}



# Datasets

#partition = # IDs

#labels = # Labels



# Generators

training_generator = DataGenerator(partition['train'], labels, **params,

          shuffle = True)

validation_generator = DataGenerator(partition['validation'], labels, **params,

          shuffle = False)





class myCallback(tf.keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs={}):

        if(logs.get('accuracy')>0.98):

            print("\nReached 98% accuracy so cancelling training!")

            self.model.stop_training = True

        

callbacks = myCallback()





#More about callbacks. Callbacks + tensorboard and something else



#callbacks = [tf.keras.callbacks.TensorBoard(log_dir='./log/transer_learning_model', update_freq='batch')]







# file_path="weights.best.hdf5"



# checkpoint = ModelCheckpoint(file_path, monitor='acc', verbose=1, save_best_only=True, mode='max')



# early = EarlyStopping(monitor="acc", mode="max", patience=15)



# callbacks_list = [checkpoint, early] #early



# history = model.fit(train_images_preprocess, 

#                               epochs=2, 

#                               shuffle=True, 

#                               verbose=True) #,

#                               #callbacks=callbacks_list)



pre_trained_model = applications.InceptionV3(input_shape = (299, 299, 3), # Shape of our images

                                include_top = False, # Leave out the last fully connected layer

                                weights = 'imagenet')

# for layer in pre_trained_model.layers:

#   layer.trainable = False

pre_trained_model.trainable = False

nclass = 3



add_model = Sequential()

add_model.add(pre_trained_model)

add_model.add(GlobalAveragePooling2D())

add_model.add(Dense(512, activation='relu'))

add_model.add(Dropout(0.5))

add_model.add(Dense(nclass, 

                    activation='softmax'))



model = add_model

model.compile(loss='categorical_crossentropy', 

#model.compile(loss='sparse_categorical_crossentropy',

#               optimizer=optimizers.SGD(lr=1e-4, 

#                                        momentum=0.9),

              optimizer=optimizers.RMSprop(learning_rate = 0.001),

              metrics=['accuracy'])

model.summary()
# Train model on dataset

history = model.fit_generator(generator=training_generator,

                    validation_data=validation_generator,

                    use_multiprocessing=False,

                    workers=2, epochs = 10,

                             callbacks=[callbacks])



#model.fit(train_images_preprocess, y_train, epochs=10)



# Now evaluate the model - note that we're evaluating on the new model, not the old one

#test_loss, test_acc = model.evaluate(test_images_preprocess, y_test)



#print('Test accuracy:', test_acc)
acc = history.history['accuracy']

val_acc = history.history['val_accuracy']



loss = history.history['loss']

val_loss = history.history['val_loss']



plt.figure(figsize=(8, 16))

plt.subplot(2, 1, 1)

plt.plot(acc, label='Training Accuracy')

plt.plot(val_acc, label='Validation Accuracy')

plt.legend(loc='lower right')

plt.ylabel('Accuracy')

plt.ylim([min(plt.ylim()),1])

plt.title('Training and Validation Accuracy')



plt.subplot(2, 1, 2)

plt.plot(loss, label='Training Loss')

plt.plot(val_loss, label='Validation Loss')

plt.legend(loc='upper right')

plt.ylabel('Cross Entropy')

plt.ylim([0,1.0])

plt.title('Training and Validation Loss')

plt.xlabel('epoch')

plt.show()
# Let's take a look to see how many layers are in the base model

print("Number of layers in the base model: ", len(pre_trained_model.layers))



# # Fine-tune from this layer onwards

# fine_tune_at = 100



# # Freeze all the layers before the `fine_tune_at` layer

# for layer in base_model.layers[:fine_tune_at]:

#   layer.trainable =  False
# predicted_batch = model.predict(image_batch)

# predicted_id = np.argmax(predicted_batch, axis=-1)

# predicted_label_batch = class_names[predicted_id]



# label_id = np.argmax(label_batch, axis=-1)



# plt.figure(figsize=(10,9))

# plt.subplots_adjust(hspace=0.5)

# for n in range(30):

#   plt.subplot(6,5,n+1)

#   plt.imshow(image_batch[n])

#   color = "green" if predicted_id[n] == label_id[n] else "red"

#   plt.title(predicted_label_batch[n].title(), color=color)

#   plt.axis('off')

# _ = plt.suptitle("Model predictions (green: correct, red: incorrect)")
# import time

# t = time.time()



# export_path = "/tmp/saved_models/{}".format(int(t))

# model.save(export_path, save_format='tf')



# export_path



# #Now confirm that we can reload it, and it still gives the same results:



# reloaded = tf.keras.models.load_model(export_path)



# result_batch = model.predict(image_batch)

# reloaded_result_batch = reloaded.predict(image_batch)



# abs(reloaded_result_batch - result_batch).max()

import sys

def sizeof_fmt(num, suffix='B'):

    ''' by Fred Cirera,  https://stackoverflow.com/a/1094933/1870254, modified'''

    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:

        if abs(num) < 1024.0:

            return "%3.1f %s%s" % (num, unit, suffix)

        num /= 1024.0

    return "%.1f %s%s" % (num, 'Yi', suffix)



for name, size in sorted(((name, sys.getsizeof(value)) for name, value in locals().items()),

                         key= lambda x: -x[1])[:10]:

    print("{:>30}: {:>8}".format(name, sizeof_fmt(size)))

    