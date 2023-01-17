import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from tensorflow import keras 

import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator # for data augmentation
sample_submission = pd.read_csv("../input/digit-recognizer/sample_submission.csv")

test = pd.read_csv("../input/digit-recognizer/test.csv")

train = pd.read_csv("../input/digit-recognizer/train.csv")



labels, imgs = train['label'], train.drop(['label'], axis = 1)



imgs.head()
from sklearn.model_selection import train_test_split

import seaborn as sns



# transform imgs to np.array with shape(sample_size, 28, 28, 1)

imgs_array = imgs.values.reshape(-1, 28, 28, 1)

print(imgs_array.shape)



# imgs/255 to normalise

train_imgs, test_imgs, train_y, test_y = train_test_split(imgs_array/255, labels, test_size = 0.3)



# label distribution plotting

sns.countplot(labels)

plt.title("label distribution")

plt.show()
# initialise a image generator 

data_gen = ImageDataGenerator(

    rescale = 1./255,

    rotation_range = 30,

    width_shift_range = 0.2,

    height_shift_range = 0.2,

    shear_range = 0.4,

    zoom_range = 0.4

)



# select the first image

plt.figure(figsize = (10, 10))

ax1 = plt.subplot(1, 2, 1)

temp_image = train_imgs[0:1]

ax1.set_title("original image")

ax1.axis('off')

ax1.imshow(temp_image.reshape(28, 28))

plt.show()



# generate images 

fig = plt.figure(figsize = (5, 2))

fig.suptitle("images generated")

for i in range(4):

    ax = fig.add_subplot(1, 4, i+1)

    ax.axis('off')

    data = data_gen.flow(temp_image, batch_size = 1)

    ax.imshow(data[0].reshape(28, 28))

plt.show()
def create_model():

    model = keras.Sequential([

        # first conv layer

        keras.layers.Conv2D(filters = 32, kernel_size = (5, 5), input_shape = (28, 28, 1), activation = 'relu'),

        keras.layers.MaxPooling2D(pool_size = (2, 2), strides = (2, 2)),

        keras.layers.Dropout(rate = 0.3),

        

        # second conv layer

        keras.layers.Conv2D(filters = 64, kernel_size = (3, 3), activation = 'relu'),

        keras.layers.MaxPooling2D(pool_size = (2, 2), strides = (2, 2)),

        keras.layers.Dropout(rate = 0.3),

        

        # softmax

        keras.layers.Flatten(),

        keras.layers.Dense(256, activation = 'relu'),

        keras.layers.Dropout(rate = 0.5),

        keras.layers.Dense(10, activation = 'softmax')

    ])

    return model



cnn = create_model()

cnn.summary()
# compile the model

cnn.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])



# define callbacks

earlystopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience = 2, verbose=0, mode='auto')



# define image generator 

batch_size = 256

data_gen = ImageDataGenerator(

    rotation_range = 40,  

    zoom_range = 0.2, 

    width_shift_range = 0.2,  

    height_shift_range = 0.2

) 



data_generator = data_gen.flow(train_imgs, train_y, batch_size = 16)



# ready to train

# steps_per_epoch = number of train samples//batch_size

# validation_steps = number of validation samples//batch_size 

# to prevent freezing when training

history = cnn.fit_generator(

    data_generator,

    steps_per_epoch = train_imgs.shape[0]//batch_size, 

    epochs = 10,

    validation_data = (test_imgs, test_y),

    validation_steps = test_imgs.shape[0]//batch_size,

    callbacks = [earlystopping]

)
figsize = (10, 2)

# Plot training & validation accuracy values

plt.figure(figsize = figsize)

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()



# Plot training & validation loss values

plt.figure(figsize = figsize)

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()
from sklearn.metrics import confusion_matrix, accuracy_score



preds = cnn.predict(test_imgs)

pred_y = np.argmax(preds, axis = 1)



print(accuracy_score(test_y, pred_y))

confusion_matrix(test_y, pred_y)
# test data preparation

test_array = test.values.reshape(-1, 28, 28, 1)



# predictions

preds = cnn.predict(test_array)

pred_test_y = np.argmax(preds, axis = 1)



# prepare submission file

Label = pd.Series(pred_test_y, name="Label")

ImageId = pd.Series(range(1, len(test)+1), name = "ImageId")

results = pd.concat([ImageId, Label], axis = 1)



results.to_csv('simple_cnn_mnist.csv', index = False)

results.head()