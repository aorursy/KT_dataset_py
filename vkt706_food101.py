import keras

from keras.applications.densenet import DenseNet121, DenseNet169, DenseNet201

from keras.applications.xception import Xception

from keras.applications.inception_v3 import InceptionV3

from keras.applications.resnet50 import ResNet50

from keras.applications.vgg19 import VGG19

from keras.applications.vgg16 import preprocess_input

from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

from keras.layers import Dense, Flatten, Dropout, BatchNormalization, AveragePooling2D

from keras.models import Model, load_model, Sequential

from keras.optimizers import Adam, SGD

from keras.regularizers import l2

from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau



import numpy as np

import os

import random

from PIL import Image

from pathlib import Path

import seaborn as sns

from matplotlib.pyplot import imshow

import matplotlib.pyplot as plt

%matplotlib inline
import os

from pathlib import Path

path = Path('/kaggle/input/food41/images')

food_names = os.listdir(path)
files = os.listdir(path)

num_classes = len(files)

print ('Number of classes:', num_classes)



dist = dict()

# class distribution

for f in files:

    dist[f] = len(os.listdir(path/f))

plt.figure(figsize=(20, 5))

plt.bar(list(dist.keys()), dist.values(), color='y')

plt.xticks(rotation=90)

plt.show() # Plotting class distribution
import matplotlib.pyplot as plt

import random

from PIL import Image

def show_img(rows, cols, figsize=None):

    fig, ax = plt.subplots(rows, cols, figsize=figsize)

    for i in range(rows):

        for j in range(cols):

            random_folder = random.choice(files)

            random_file = random.choice(os.listdir(path/random_folder))

            im = Image.open(path/random_folder/random_file)

            ax[i, j].imshow(im)

            ax[i, j].set_xlabel(str(random_folder) + " " + str(im.size))

            ax[i, j].set_yticklabels([])

            ax[i, j].set_xticklabels([])

            ax[i, j].grid(False)



show_img(2, 5, (15, 5))
# Generate batches of image data with real-time data augmentation using ImageDataGenerator



batch_size = 64

datagen = ImageDataGenerator(rescale=1./255,

                             shear_range=0.3,

                             zoom_range=0.2,

                             preprocessing_function=preprocess_input,

                             validation_split=0.2,

                             horizontal_flip=True)



shape = (224, 224)

train_generator = datagen.flow_from_directory(path,

                                              target_size=shape,

                                              batch_size=batch_size,

                                              subset='training',

                                              class_mode='categorical')



val_generator = datagen.flow_from_directory(path,

                                            target_size=shape,

                                            batch_size=batch_size,

                                            subset='validation',

                                            class_mode='categorical')
!pip install efficientnet
!pip install wandb
import wandb

from wandb.keras import WandbCallback



# Wandb used to record and visualize every detail of model training.

# Hyperparameters



defaults=dict(

    lr = 0.01, 

    beta_1 = 0.9, 

    beta_2 = 0.999, 

    decay = 1e-8, 

    hidden_layers = 256, 

    dropout = 0.5,

    epoch = 1,

    )



name = "efficientnet"

config = defaults

wandb.init(project="food101", magic=True, config = defaults, name = name)

config = wandb.config
import efficientnet.keras as eff



class my_Models:

    def eff_net(hidden_layers, dropout, trainable= False):

        base_model = eff.EfficientNetB3(weights='imagenet', include_top=False, pooling='avg')

        for layer in base_model.layers:

            layer.trainable = trainable

        x = base_model.output

        x = Dense(hidden_layers, activation='relu')(x)

        x = BatchNormalization()(x)

        x = Dropout(dropout)(x)

        predictions = Dense(num_classes, activation='softmax')(x)

        model = Model(inputs=base_model.input, outputs=predictions)

        return model 

    

    def xception(hidden_layers, dropout, trainable= False):

        base_model = Xception(weights='imagenet', include_top=False, pooling='avg')

        for layer in base_model.layers:

            layer.trainable = False

        x = base_model.output

        x = Dense(hidden_layers, activation='relu')(x)

        x = BatchNormalization()(x)

        x = Dropout(dropout)(x)

        predictions = Dense(num_classes, activation='softmax')(x)

        model = Model(inputs=base_model.input, outputs=predictions)

    

        for layer in model.layers[:122]:

            layer.trainable = False

        for layer in model.layers[122:]:

            layer.trainable = trainable       

        return model 
from keras.metrics import top_k_categorical_accuracy



def top_2(y_true, y_pred):

    return top_k_categorical_accuracy(y_true, y_pred, k=2)

def top_3(y_true, y_pred):

    return top_k_categorical_accuracy(y_true, y_pred, k=3)

def top_5(y_true, y_pred):

    return top_k_categorical_accuracy(y_true, y_pred, k=5)



adam = Adam(lr=config.lr, beta_1=config.beta_1, beta_2=config.beta_2, epsilon=None, decay=config.decay, amsgrad=True)

nadam = keras.optimizers.Nadam(learning_rate=config.lr, beta_1=config.beta_1, beta_2=config.beta_2)

weights_path = '../working/eff_weights.hdf5'

mc = ModelCheckpoint(filepath=weights_path, monitor='val_acc', verbose=1, save_best_only=True)

es = EarlyStopping(monitor='val_acc', patience=3, verbose=1)



model = my_Models.xception(config.hidden_layers, config.dropout, trainable = True)

model.compile(loss='categorical_crossentropy', optimizer=nadam, metrics=['acc', top_3, top_5])
# Training the model

history = model.fit_generator(train_generator, epochs=8,

                              steps_per_epoch=train_generator.samples // batch_size,

                              validation_data=val_generator,

                              validation_steps=val_generator.samples // batch_size,

                              callbacks=[mc, es, WandbCallback(data_type="image", 

                                                               monitor='val_loss',

                                                               validation_data=val_generator, 

                                                               labels=food_names)])
print("EfficientNet")

# summarize history for accuracy

plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'val'], loc='upper left')

plt.show()



# summarize history for loss

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'val'], loc='upper left')

plt.show()

import keras



model = my_Models.xception(config.hidden_layers, config.dropout, trainable = True)

model.compile(loss='categorical_crossentropy', optimizer=nadam, metrics=['acc', top_3, top_5])



weights_path = '../working/eff_weights_tuned.hdf5'

model.load_weights(weights_path)



mc = ModelCheckpoint(filepath=weights_path, monitor='val_acc', verbose=1, save_best_only=True)

history = model.fit_generator(train_generator, epochs=6,

                              steps_per_epoch=train_generator.samples // batch_size,

                              validation_data=val_generator,

                              validation_steps=val_generator.samples // batch_size,

                              callbacks=[mc, es, WandbCallback(data_type="image", 

                                                               monitor='val_loss',

                                                               validation_data=val_generator, 

                                                               labels=food_names)])

model.save(os.path.join(wandb.run.dir, "eff_weights_end.h5"))
print("EfficientNet with Tuning: ")

# summarize history for accuracy

plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'val'], loc='upper left')

plt.show()



# summarize history for loss

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'val'], loc='upper left')

plt.show()
from keras.preprocessing.image import array_to_img

img, lbl = val_generator.next()



y_proba = model.predict_on_batch(img)

y_classes = y_proba.argmax(axis=-1)

print (y_proba.shape, lbl.shape, y_classes.shape)





images, predict = img, lbl

true_classes = np.argmax(lbl, axis=1)

pred_classes = y_classes

label_map = train_generator.class_indices



true_lbl = [dict((v,k) for k,v in label_map.items()).get(true_classes[i]) for i in range(len(true_classes))]

pred_lbl = [dict((v,k) for k,v in label_map.items()).get(pred_classes[i]) for i in range(len(pred_classes))]



fig = plt.figure(figsize=(15, 6))

for idx in np.arange(10):

    ax = fig.add_subplot(2, 10/2, idx+1, xticks=[], yticks=[])

    ax.imshow(array_to_img(images[idx]))

    ax.set_title("{}\n({})".format(str(pred_lbl[idx]), str(true_lbl[idx])),

                 color=("green" if true_lbl[idx]==pred_lbl[idx] else "red"))
