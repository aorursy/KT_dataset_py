## imports



import os

import numpy as np

from tensorflow.keras import layers

from tensorflow.keras.models import Sequential, load_model

from tensorflow.keras import callbacks

from tensorflow.keras.optimizers import SGD

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import GridSearchCV

import matplotlib.pyplot as plt
## preparing the image generators



train_generator = ImageDataGenerator(rescale=1/255, width_shift_range=0.25, height_shift_range=0.25,

                                     horizontal_flip=True, zoom_range=(1, 1.5))

test_generator = ImageDataGenerator(rescale=1/255)



main_dir='../input/intel-image-classification/'

train_data = train_generator.flow_from_directory(main_dir+'seg_train/seg_train', target_size=(150,150),

                                                 batch_size=64, class_mode='categorical')

test_data = test_generator.flow_from_directory(main_dir+'seg_test/seg_test', target_size=(150,150),

                                               batch_size=15, class_mode='categorical')
## after some trial & error I came up with this model



model = Sequential([

    layers.Conv2D(64, 3, input_shape=(150, 150, 3)),

    layers.BatchNormalization(),

    layers.Activation('relu'),

    layers.MaxPooling2D(2),

    layers.SeparableConv2D(128, 3),

    layers.BatchNormalization(),

    layers.Activation('relu'),

    layers.MaxPooling2D(2),

    layers.Conv2D(128, 3),

    layers.BatchNormalization(),

    layers.Activation('relu'),

    layers.MaxPooling2D(2),

    layers.Conv2D(256, 3),

    layers.BatchNormalization(),

    layers.Activation('relu'),

    layers.MaxPooling2D(2),

    layers.Flatten(),

    layers.Dense(48),

    layers.BatchNormalization(),

    layers.Activation('relu'),

    layers.Dropout(0.25),

    layers.Dense(24),

    layers.BatchNormalization(),

    layers.Activation('relu'),

    layers.Dropout(0.25),

    layers.Dense(6, activation='softmax')

])



sgd = SGD(lr=0.007, momentum=0.95, decay=0.0005)

loss = 'binary_crossentropy'

model.summary()

model.compile(optimizer=sgd, loss=loss, metrics=['acc'])
# the EarlyStopping callback in keras seems to not work properly with set baseline; 

# it just stops after the number of epochs set by patience parameter

# so I'll make my own version of it



class EarlyStopping(callbacks.Callback):

    def __init__(self, patience, metric, delta, baseline):

        self.baseline = baseline

        self.delta = delta

        self.patience = patience

        self.base_patience = patience

        self.base_reached = False

        self.metric = metric

        self.last_log = 0

        

    def on_epoch_end(self, epoch, logs={}):

        log = logs[self.metric]

        if not self.base_reached:

            self.base_reached = log >= self.baseline

            print('\nBaseline is%sreached' % (' ' if self.base_reached else ' not '))

        if(log < self.last_log+self.delta and self.base_reached):

            if self.patience:

                self.patience -= 1

                print('\nThe patience is now %d' % self.patience)

            else:

                print('\nStopped training with %s = %f' % (self.metric, log))

                self.model.stop_training = True

        else:

            self.patience = self.base_patience

        self.last_log = log

            

        

cb = [EarlyStopping(patience=3, metric='val_acc', delta=0.002, baseline=0.95),

      callbacks.ModelCheckpoint('model.h5', monitor='val_loss', save_best_only=True, verbose=1)]
## fitting the model



stats = model.fit_generator(train_data, validation_data=test_data, epochs=100,

                                  steps_per_epoch=256, validation_steps=200, callbacks=cb)
## let's load the best checkpoint and make some predictions!

## the only somewhat frequent mistake is misclassifying snowy mountains as glaciers; 

## besides it the classifier works properly with only some rare minor mistakes



checkpoint = load_model('model.h5')

pred_folder = '../input/intel-image-classification/seg_pred/seg_pred/'

pred_batch = [plt.imread(pred_folder+i) for i in np.random.choice(os.listdir(pred_folder), 16)]

labels = ['building', 'forest', 'glacier', 'mountain', 'sea', 'street']

fig, axs = plt.subplots(4, 4, figsize=(16,16))

for i in range(4):

    for j in range(4):

        img = pred_batch[i*4+j]

        axs[i, j].imshow(img)

        res = checkpoint.predict(img.reshape(1, 150, 150, 3)/255)

        axs[i, j].set_title(labels[np.argmax(res)])