import numpy as np # linear algebra

import os

from time import time

from keras.preprocessing.image import ImageDataGenerator, img_to_array, image

from keras.utils import np_utils

import json

from PIL import Image

import os

import tensorflow as tf
data_dir = "../input/coin-images/coins/data"



data_train_path =  data_dir + '/train'

data_valid_path = data_dir + '/validation'

data_test_path =  data_dir + '/test'



print(os.listdir("../input/coin-images/coins/data"))
with open('../input/coin-images/cat_to_name.json', 'r') as json_file:

    cat_2_name = json.load(json_file)



print(cat_2_name['200'])
batch_size=60



# Transforms

datagen_train = ImageDataGenerator(

    rescale=1./255,

    rotation_range=40,

    width_shift_range=0.1,  # randomly shift images horizontally 

    height_shift_range=0.1,  # randomly shift images vertically

    horizontal_flip=True,

    featurewise_std_normalization=True,

    samplewise_std_normalization=True)



datagen_valid = ImageDataGenerator(

    rescale=1./255,

    rotation_range=40,

    width_shift_range=0.1,  # randomly shift images horizontally

    height_shift_range=0.1,  # randomly shift images vertically

    horizontal_flip=True,

    featurewise_std_normalization=True,

    samplewise_std_normalization=True)



datagen_test = ImageDataGenerator(

    featurewise_std_normalization=True,

    samplewise_std_normalization=True,

    rescale=1./255)


train_generator = datagen_train.flow_from_directory(

        data_train_path,

        target_size=(224, 224),

        batch_size=batch_size,

        class_mode='categorical')



valid_generator = datagen_valid.flow_from_directory(

        data_valid_path,

        target_size=(224, 224),

        batch_size=batch_size,

        class_mode='categorical')



test_generator = datagen_test.flow_from_directory(

        data_test_path,

        target_size=(224, 224),

        batch_size=batch_size,

        class_mode='categorical')
import matplotlib.pyplot as plt





# Lets have a look at some of our images

images, labels = train_generator.next()



fig = plt.figure(figsize=(20,10))

fig.subplots_adjust(wspace=0.2, hspace=0.4)



# Lets show the first 32 images of a batch

for i, img in enumerate(images[:32]):

    ax = fig.add_subplot(4, 8, i + 1, xticks=[], yticks=[])

    ax.imshow(img)

    image_idx = np.argmax(labels[i])
int_to_dir = {v: k for k, v in train_generator.class_indices.items()}
from keras.applications import MobileNetV2

from keras.layers import Dense, Input, Dropout

from keras.models import Model





input_tensor = Input(shape=(224, 224, 3))

base_model = MobileNetV2(

    include_top=False,

    weights='imagenet',

    input_tensor=input_tensor,

    input_shape=(224, 224, 3),

    pooling='avg')



for layer in base_model.layers:

    layer.trainable = True  # trainable has to be false in order to freeze the layers



x = Dense(512, activation='relu')(base_model.output)

x = Dropout(.8)(x)



predictions = Dense(211, activation='softmax')(x)



model = Model(inputs=base_model.input, outputs=predictions)

from keras.optimizers import Adam



# compile the model (should be done *after* setting layers to non-trainable)

model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy',

             metrics=['accuracy'])
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping



num_train = len(train_generator.filenames)

num_valid = len(valid_generator.filenames)

num_test = len(train_generator.filenames)





# When to save the model

checkpointer = ModelCheckpoint(filepath='model.weights.best.hdf5', verbose=1, 

                               save_best_only=True)



# Reduce learning rate when loss doesn't improve after n epochs

scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.1,

                              patience=5, min_lr=1e-8, verbose=1)



# Stop early if model doesn't improve after n epochs

early_stopper = EarlyStopping(monitor='val_loss', patience=12,

                              verbose=0, restore_best_weights=True)



# Train the model

history = model.fit_generator(train_generator,

                    steps_per_epoch=num_train//batch_size,

                    epochs=100,

                    verbose=1,

                    callbacks=[checkpointer, scheduler, early_stopper],

                    validation_data=valid_generator,

                    validation_steps=num_valid//batch_size)
#model.load_weights('../input/mobilenetv2-weights/model.weights.best.hdf5')
model.save('mobilenet.h5')
score = model.evaluate_generator(test_generator, steps=num_test//1, verbose=1)

print('\n', 'Test accuracy:', score[1])
def normalizer(img):

    img_expand = np.expand_dims(img, axis=0)



    generator = ImageDataGenerator(

        featurewise_std_normalization=True,

        samplewise_std_normalization=True,

        rescale=1./255)

    image_flow = generator.flow(

        img_expand,

        y=None,

        batch_size=1

    )

    

    return image_flow.next()
def get_prediction(img, real_label):

    img = image.img_to_array(img)/255

    

    mean = [0.485, 0.456, 0.406]

    std = [0.229, 0.224, 0.225]

    img = (img - mean)/std

    

    img_expand = np.expand_dims(img, axis=0)



    prediction = model.predict(img_expand)

    prediction_int = np.argmax(prediction)



    dir_int = int_to_dir[prediction_int]

    label_name = cat_2_name[str(dir_int)]

    

    plt.imshow(img)

    

    print("Predicted: {}\nReal:      {}".format(label_name, cat_2_name[str(real_label)]))

    print()





for i in range(10):

    random_index = np.random.randint(0, len(test_generator.filenames))

    

    img = test_generator.filenames[random_index]

    img = image.load_img("../input/coin-images/coins/data/test/"+img, target_size=(224,224))

    real_label = test_generator.filenames[random_index].split("/")[0]



    get_prediction(img, real_label)