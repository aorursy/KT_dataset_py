import os

os.listdir("../input")
# Importing Keras libraries and packages

from keras.models import Sequential,load_model,Model

from keras.applications import InceptionV3

from keras.layers import Convolution2D

from keras.layers import MaxPooling2D,GlobalAveragePooling2D

from keras.layers import Flatten

from keras.layers import Dense

from keras.layers import Dropout

from keras.layers.normalization import BatchNormalization

model = Sequential()

#Layer 1

#Conv Layer 1

model.add(Convolution2D(filters = 6, 

                 kernel_size = 5, 

                 strides = 1, 

                 activation = 'relu', 

                 input_shape = (32,32,3)))

#Pooling layer 1

model.add(MaxPooling2D(pool_size = 2, strides = 2))

#Layer 2

#Conv Layer 2

model.add(Convolution2D(filters = 16, 

                 kernel_size = 5,

                 strides = 1,

                 activation = 'relu',

                 input_shape = (14,14,6)))

#Pooling Layer 2

model.add(MaxPooling2D(pool_size = 2, strides = 2))

#Flatten

model.add(Flatten())

#Layer 3

#Fully connected layer 1

model.add(Dense(units = 120, activation = 'relu'))

#Layer 4

#Fully connected layer 2

model.add(Dense(units = 84, activation = 'relu'))

#Layer 5

#Output Layer

model.add(Dense(units = 38, activation = 'softmax'))
model.summary()

#for layer in base_model.layers:

    #layer.trainable = False
model.compile(optimizer='rmsprop',

              loss='categorical_crossentropy',

              metrics=['accuracy'])
# image preprocessing

from keras.preprocessing.image import ImageDataGenerator



train_datagen = ImageDataGenerator(

    rotation_range=40,

    width_shift_range=0.2,

    height_shift_range=0.2,

    shear_range=0.2,

    zoom_range=0.2,

    horizontal_flip=True,

    fill_mode='nearest')



valid_datagen = ImageDataGenerator(rescale=1./255)



batch_size =32

base_dir = "../input/new-plant-diseases-dataset/new plant diseases dataset(augmented)/New Plant Diseases Dataset(Augmented)"



train_generator = train_datagen.flow_from_directory(base_dir+'/train',

                                                 target_size=(32, 32),

                                                 batch_size=batch_size,

                                                 class_mode='categorical')



validation_generator = valid_datagen.flow_from_directory(base_dir+'/valid',

                                            target_size=(32, 32),

                                            batch_size=batch_size,

                                            class_mode='categorical')
class_dict = train_generator.class_indices

print(class_dict)
li = list(class_dict.keys())

print(li)
EPOCHS = 50

BATCH_SIZE = 128

STEPS_PER_EPOCH = 550

VALIDATION_STEPS = 64

MODEL_FILE = 'LeNet.h5'

history = model.fit_generator(

    train_generator,

    epochs=EPOCHS,

    steps_per_epoch=STEPS_PER_EPOCH,

    validation_data=validation_generator,

    validation_steps=VALIDATION_STEPS)

  

model.save(MODEL_FILE)

#plotting training values

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()



acc = history.history['acc']

val_acc = history.history['val_acc']

loss = history.history['loss']

val_loss = history.history['val_loss']

epochs = range(1, len(loss) + 1)



#accuracy plot

plt.plot(epochs, acc, color='green', label='Training Accuracy')

plt.plot(epochs, val_acc, color='blue', label='Validation Accuracy')

plt.title('Training and Validation Accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend()



plt.figure()

#loss plot

plt.plot(epochs, loss, color='pink', label='Training Loss')

plt.plot(epochs, val_loss, color='red', label='Validation Loss')

plt.title('Training and Validation Loss')

plt.xlabel('Epoch')

plt.ylabel('Loss')

plt.legend()



plt.show()
# predicting an image

from keras.preprocessing import image

import numpy as np

image_path = "../input/new-plant-diseases-dataset/test/test/TomatoHealthy3.JPG"

new_img = image.load_img(image_path, target_size=(224, 224))

img = image.img_to_array(new_img)

img = np.expand_dims(img, axis=0)

img = img/255



print("Following is our prediction:")

prediction = model.predict(img)

# decode the results into a list of tuples (class, description, probability)

# (one such list for each sample in the batch)

d = prediction.flatten()

j = d.max()

for index,item in enumerate(d):

    if item == j:

        class_name = li[index]



##Another way

# img_class = classifier.predict_classes(img)

# img_prob = classifier.predict_proba(img)

# print(img_class ,img_prob )





#ploting image with predicted class name        

plt.figure(figsize = (4,4))

plt.imshow(new_img)

plt.axis('off')

plt.title(class_name)

plt.show()