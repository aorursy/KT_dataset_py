!mkdir ../data

!mkdir ../data/dogs-vs-cats-redux-kernels-edition

!unzip -qq ../input/dogs-vs-cats-redux-kernels-edition/train.zip -d ../data/dogs-vs-cats-redux-kernels-edition/

!unzip -qq ../input/dogs-vs-cats-redux-kernels-edition/test.zip -d ../data/dogs-vs-cats-redux-kernels-edition/
!mkdir -p ../data/train/cats ../data/train/dogs ../data/validation/cats ../data/validation/dogs

!cp `ls ../data/dogs-vs-cats-redux-kernels-edition/train/cat* | grep '\.[0-9]\{1,3\}\.'` ../data/train/cats/  # copy first 1000 cats

!cp `ls ../data/dogs-vs-cats-redux-kernels-edition/train/dog* | grep '\.[0-9]\{1,3\}\.'` ../data/train/dogs/  # copy first 1000 dogs

!cp ../data/dogs-vs-cats-redux-kernels-edition/train/cat.1[0-3]??.jpg ../data/validation/cats/  # copy next 400 cats

!cp ../data/dogs-vs-cats-redux-kernels-edition/train/dog.1[0-3]??.jpg ../data/validation/dogs/  # copy next 400 dogs

!du ../data/ -h
import os

import matplotlib.pyplot as plt

from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

%matplotlib inline



files = [[path+f for f in os.listdir(path)[:10]] for path in [f'../data/train/{x}/' for x in ['cats', 'dogs']]]



fig, axs = plt.subplots(4, 5, figsize=(15,15), subplot_kw={'xticks': [], 'yticks': []})



for ax, img in zip(axs.flatten(), [item for sublist in files for item in sublist]):

    ax.imshow(load_img(img))

    ax.set_title(img.split('/')[-1])
# define image transformations

datagen = ImageDataGenerator(

        rotation_range=40,

        width_shift_range=0.2,

        height_shift_range=0.2,

        shear_range=0.2,

        zoom_range=0.2,

        horizontal_flip=True,

        fill_mode='nearest')



img_samples = 16

image_dir = '../preview/'

if not os.path.exists(image_dir):

    os.makedirs(image_dir)



img = load_img('../data/train/cats/cat.8.jpg')

x = img_to_array(img)

x = x.reshape((1,) + x.shape)



from itertools import islice

list(islice(datagen.flow(x, batch_size=1, save_to_dir=image_dir, save_prefix='cat', save_format='jpeg'), img_samples));
rows, cols = 2, img_samples // 2

fig, axs = plt.subplots(rows, cols, figsize=(16,3), subplot_kw={'xticks': [], 'yticks': []})



#for i, img in enumerate(os.listdir(image_dir)[:img_samples]):

#    axs[i//cols][i%cols].imshow(load_img(image_dir+img))



for ax, img in zip(axs.flatten(), os.listdir(image_dir)[:img_samples]):

    ax.imshow(load_img(image_dir+img))
import numpy as np

import pandas as pd

import keras, tensorflow

import time



print('Keras', keras.__version__)

print('TensorFlow', tensorflow.__version__)
from keras.models import Sequential

from keras.layers import Input, Conv2D, MaxPooling2D

from keras.layers import Activation, Dropout, Flatten, Dense



input_tensor = Input(shape=(150,150,3))  # input_shape for Theano should be (3, 150,150)



model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=(150,150,3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(32, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Flatten())

model.add(Dense(256, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(1, activation='sigmoid'))



print(model.summary())
model.compile(loss='binary_crossentropy',

              optimizer='rmsprop',

              metrics=['accuracy'])
from keras.preprocessing.image import ImageDataGenerator



train_dir      = '../data/train'

validation_dir = '../data/validation'

test_dir = "../data/test"



train_samples      = 2000

validation_samples = 800



target_size    = (150,150)  # all images will be resized to 150x150

batch_size     = 16



# rescale and augment training data

train_datagen = ImageDataGenerator(

        rescale=1./255,

        shear_range=0.2,

        zoom_range=0.2,

        horizontal_flip=True)



# rescale validation data

validation_datagen = ImageDataGenerator(

    rescale=1./255)



train_generator = train_datagen.flow_from_directory(

        train_dir,

        target_size=target_size,

        batch_size=batch_size,

        class_mode='binary')



validation_generator = validation_datagen.flow_from_directory(

        validation_dir,

        target_size=target_size,

        batch_size=batch_size,

        class_mode='binary')
epochs = 30



start_time = time.time()

history_simple = model.fit_generator(

    train_generator,

    steps_per_epoch=train_samples // batch_size,

    epochs=epochs,

    verbose=2,

    validation_data=validation_generator,

    validation_steps=validation_samples // batch_size)

print("--- took %d:%.2d minutes ---" % divmod(time.time() - start_time, 60))
def plot_history(history, acc_line=None, title=None, acc_lim=[0.5,1.0]):

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,5))

    if title:

        fig.suptitle(title, fontsize=16)

    

    ax1.plot(history.history['acc'])

    ax1.plot(history.history['val_acc'])

    if acc_line:

        ax1.axhline(y=acc_line, linewidth=2, linestyle='dashed', color='lightgrey')

    ax1.set_title('Model Accuracy')

    ax1.set_ylabel('Accuracy')

    ax1.set_xlabel('Epoch')

    #ax1.set_yticks(np.arange(0., 1.1, .1))

    ax1.set_ylim(acc_lim)

    ax1.legend(['Train', 'Test'])

    ax1.grid(b=True, which='major', color='lightgrey', linestyle='dotted')

    

    ax2.plot(history.history['loss'])

    ax2.plot(history.history['val_loss'])

    ax2.set_title('Model Loss')

    ax2.set_ylabel('Loss')

    ax2.set_xlabel('Epoch')

    ax2.set_ylim([0, 1.2])

    ax2.legend(['Train', 'Test'])

    ax2.grid(b=True, which='major', color='lightgrey', linestyle='dotted')
plot_history(history_simple, acc_line=0.8, title='Simple ConvNet')
from keras.applications.vgg16 import VGG16



weights_vgg16 = '../input/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'



base_model_vgg16 = VGG16(weights=weights_vgg16, include_top=False, input_shape=(150,150,3))
def get_bottleneck_features(model, image_dir, target_size, samples, batch_size=16):

    datagen   = ImageDataGenerator(rescale=1. / 255)

    generator = datagen.flow_from_directory(image_dir,

                                            target_size=target_size,

                                            batch_size=batch_size,

                                            class_mode=None,

                                            shuffle=False)

    return model.predict_generator(generator, samples // batch_size)



train_data_vgg16 = get_bottleneck_features(base_model_vgg16, train_dir, target_size, train_samples, batch_size)

print('created bottleneck features for training:', train_data_vgg16.shape)



validation_data_vgg16 = get_bottleneck_features(base_model_vgg16, validation_dir, target_size, validation_samples, batch_size)

print('created bottleneck features for validation:', validation_data_vgg16.shape)



train_labels = np.array([0] * (train_samples // 2) + [1] * (train_samples // 2))

validation_labels = np.array([0] * (validation_samples // 2) + [1] * (validation_samples // 2))
from keras.models import Model

from keras.layers import Input, Flatten, Dense, Dropout



def get_top_model(input_shape):

    input = Input(input_shape)

    x = Flatten()(input)

    x = Dense(256, activation='relu')(x)

    x = Dropout(0.5)(x)

    prediction = Dense(1, activation='sigmoid')(x)

    return Model(inputs=input, outputs=prediction)



top_model_vgg16 = get_top_model(train_data_vgg16.shape[1:])



top_model_vgg16.compile(loss='binary_crossentropy',

                        optimizer='rmsprop',

                        metrics=['accuracy'])
epochs = 30



start_time = time.time()

history_vgg16_top = top_model_vgg16.fit(train_data_vgg16, train_labels,

                                        verbose=2,

                                        epochs=epochs,

                                        batch_size=batch_size,

                                        validation_data=(validation_data_vgg16, validation_labels))

print("--- took %d:%.2d minutes ---" % divmod(time.time() - start_time, 60))
model_weigths_file = 'bottleneck_fc_model_vgg16.h5'

top_model_vgg16.save_weights(model_weigths_file)
plot_history(history_vgg16_top, acc_line=0.9, title='Pre-trained VGG16 with no augmentation', acc_lim=[0.75,1.0])
from keras.models import Sequential

from keras.layers import Input, Flatten, Dense, Dropout
# load pre-trained VGG16 network (without classfication layers)

base_model_vgg16 = VGG16(weights=weights_vgg16, include_top=False, input_shape=(150,150,3))

print('Model loaded.')



# set all but the last the last conv block to non-trainable (weights will not be updated)

for layer in base_model_vgg16.layers[:15]:

    layer.trainable = False



# create a classifier model to put on top of the convolutional model

top_model = Sequential()

top_model.add(Flatten(input_shape=base_model_vgg16.output_shape[1:]))

top_model.add(Dense(256, activation='relu'))

top_model.add(Dropout(0.5))

top_model.add(Dense(1, activation='sigmoid'))

top_model.load_weights(model_weigths_file)



# add the model on top of the convolutional base

model = Model(inputs = base_model_vgg16.input, outputs = top_model(base_model_vgg16.output))
from keras import optimizers



# compile model with a very slow learning rate

model.compile(loss='binary_crossentropy',

              #optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),

              #optimizer=optimizers.RMSprop(lr=1e-5),

              optimizer=optimizers.Adam(lr=1e-5),

              metrics=['accuracy'])
epochs = 30



start_time = time.time()

history_tuned = model.fit_generator(

    train_generator,

    steps_per_epoch=train_samples // batch_size,

    epochs=epochs,

    verbose=2,

    validation_data=validation_generator,

    validation_steps=validation_samples // batch_size)

print("--- took %d:%.2d minutes ---" % divmod(time.time() - start_time, 60))
plot_history(history_tuned, acc_line=0.93, title='Pre-trained VGG16 + pre-trained top model', acc_lim=[0.75,1.0])
from keras.models import Model

from keras.layers import Flatten, Dense, Dropout



# build the VGG16 network

base_model = VGG16(weights=weights_vgg16, include_top=False, input_shape=(150,150,3))



# freeze convolutional layers

for layer in base_model.layers:

    layer.trainable = False



x = Flatten()(base_model.output)

x = Dense(256, activation='relu')(x)

x = Dropout(0.5)(x)

predictions = Dense(1, activation='sigmoid')(x)



model_vgg16base = Model(inputs=base_model.input, outputs=predictions)



print(model_vgg16base.summary())
model_vgg16base.compile(loss='binary_crossentropy',

                        optimizer='rmsprop',

                        metrics=['accuracy'])
epochs = 30



start_time = time.time()

history_vgg16base = model_vgg16base.fit_generator(

    train_generator,

    steps_per_epoch=train_samples // batch_size,

    epochs=epochs,

    validation_data=validation_generator,

    validation_steps=validation_samples // batch_size)

print("--- took %d:%.2d minutes ---" % divmod(time.time() - start_time, 60))
plot_history(history_vgg16base, acc_line=0.9, title='Pre-trained VGG16', acc_lim=[0.75, 1.0])
from keras.applications.inception_v3 import InceptionV3



weights_incv3 = '../input/inceptionv3/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'



# load pre-trained weights and add global average pooling layer

base_model_incv3 = InceptionV3(weights=weights_incv3, input_shape=(150,150,3), include_top=False, pooling='avg')



# freeze convolutional layers

for layer in base_model_incv3.layers:

    layer.trainable = False



# define classification layers

#x = Dense(1024, activation='relu')(base_model_incv3.output)

#predictions = Dense(1, activation='sigmoid')(x)

x = Dense(256, activation='relu')(base_model_incv3.output)

x = Dropout(0.5)(x)

predictions = Dense(1, activation='sigmoid')(x)



model_incv3 = Model(inputs=base_model_incv3.input, outputs=predictions)

#print(model_incv3.summary())
model_incv3.compile(loss='binary_crossentropy',

                    optimizer=optimizers.RMSprop(lr=0.0001),

                    metrics=['accuracy'])
from keras.preprocessing.image import ImageDataGenerator

from keras.applications.inception_v3 import preprocess_input



def prep(image):

    # copy image to prevent overwriting

    return preprocess_input(image.copy())



train_datagen = ImageDataGenerator(

        shear_range=0.2,

        zoom_range=0.2,

        horizontal_flip=True,

        preprocessing_function=prep)



test_datagen = ImageDataGenerator(preprocessing_function=prep)



train_generator = train_datagen.flow_from_directory(

        train_dir,

        target_size=target_size,

        batch_size=batch_size,

        class_mode='binary')



validation_generator = test_datagen.flow_from_directory(

        validation_dir,

        target_size=target_size,

        batch_size=batch_size,

        class_mode='binary')
epochs = 30



start_time = time.time()

history_incv3 = model_incv3.fit_generator(

    train_generator,

    steps_per_epoch=train_samples // batch_size,

    epochs=epochs,

    validation_data=validation_generator,

    validation_steps=validation_samples // batch_size)

print("--- took %d:%.2d minutes ---" % divmod(time.time() - start_time, 60))
plot_history(history_incv3, acc_line=0.95, title="Pre-trained InceptionV3", acc_lim=[0.75,1.0])
from keras.applications.resnet50 import ResNet50, preprocess_input



weights_resnet50 = '../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'



base_model = ResNet50(weights = weights_resnet50, include_top = False, pooling = 'avg')
model = Sequential()

model.add(base_model)



model.add(Dense(256, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(1, activation='sigmoid'))



model.layers[0].trainable = False

print(model.summary())
model.compile(loss='binary_crossentropy',

                    optimizer=optimizers.RMSprop(lr=0.0001),

                    metrics=['accuracy'])
def prep(image):

    # copy image to prevent overwriting

    return preprocess_input(image.copy())



train_datagen = ImageDataGenerator(

        shear_range=0.2,

        zoom_range=0.2,

        horizontal_flip=True,

        preprocessing_function=prep)



test_datagen = ImageDataGenerator(preprocessing_function=prep)



train_generator = train_datagen.flow_from_directory(

        train_dir,

        target_size=target_size,

        batch_size=batch_size,

        class_mode='binary')



validation_generator = test_datagen.flow_from_directory(

        validation_dir,

        target_size=target_size,

        batch_size=batch_size,

        class_mode='binary')
epochs = 30



start_time = time.time()

history_resnet50 = model.fit_generator(

    train_generator,

    steps_per_epoch=train_samples // batch_size,

    epochs=epochs,

    validation_data=validation_generator,

    validation_steps=validation_samples // batch_size)

print("--- took %d:%.2d minutes ---" % divmod(time.time() - start_time, 60))
plot_history(history_resnet50, acc_line=0.97, title="Pre-trained ResNet-50", acc_lim=[0.85,1.0])
!mkdir ../output/

!ls ../output/
model.save("model_resnet50.h5")
import pandas as pd

sample_submission = pd.read_csv("../input/dogs-vs-cats-redux-kernels-edition/sample_submission.csv")
import pandas as pd

sample_submission = pd.read_csv("../input/dogs-vs-cats-redux-kernels-edition/sample_submission.csv")