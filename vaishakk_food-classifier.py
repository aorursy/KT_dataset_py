# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from IPython.display import clear_output

import copy
# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data_dir = '/kaggle/input/food-101/food-101/food-101'

imgs_dir = os.path.join(data_dir, 'images')

meta_dir = os.path.join(data_dir, 'meta')
train_meta = pd.read_csv(os.path.join(meta_dir, 'train.txt'), delimiter='/', names=['target', 'image'])

test_meta = pd.read_csv(os.path.join(meta_dir, 'test.txt'), delimiter='/', names=['target', 'image'])

train_meta['train'] = 1

test_meta['train'] = 0
all_data = pd.concat([train_meta, test_meta])

targets = all_data['target']

all_data = pd.get_dummies(all_data, columns=['target'])

all_data['target'] = targets

print(all_data.head())
labels = [label[7:] for label in all_data.columns.values[2:103]]

print(labels)
train_meta = all_data[all_data['train']==1]

test_meta = all_data[all_data['train']==0]

train_meta.drop('train', axis=1, inplace=True)

test_meta.drop('train', axis=1, inplace=True)

print(train_meta.head())

print(test_meta.head())
counts = train_meta.groupby('target').count()
counts.plot(kind='bar', stacked='True', figsize=(20, 10), color='blue', legend=False);
import tensorflow as tf

import random
image_generator = tf.keras.preprocessing.image.ImageDataGenerator(

            rotation_range=20,

            width_shift_range=0.2,

            height_shift_range=0.2,

            brightness_range=None,

            shear_range=0.2,

            zoom_range=0.2,

            channel_shift_range=0.0,

            fill_mode="nearest",

            horizontal_flip=True,

            vertical_flip=True,

            preprocessing_function=tf.keras.applications.resnet50.preprocess_input

        )
def generator(df, batch_size=32, target_size=224):

    num_imgs = len(df)

    while True:

        batch_indx = df.iloc[random.choices(range(0,num_imgs), k=batch_size)]

        #print(batch_indx)

        x = []

        y = []

        for _,row in batch_indx.iterrows():

            file_name = row['target'] + '/' + str(row['image']) + '.jpg'

            image = tf.keras.preprocessing.image.load_img(os.path.join(imgs_dir, file_name), target_size=(target_size, target_size))

            input_arr = tf.keras.preprocessing.image.img_to_array(image)

            x.append(input_arr)

            y.append(row.iloc[1:102].values)

        x = np.array(x, dtype=float)

        y = np.array(y, dtype=float)

        

        data_generator = image_generator.flow(x, y, batch_size=batch_size)

        xt, yt = data_generator.next()

        yield(np.array(xt), np.array(yt))

batch_size = 16

img_size = 224

dropout = 0.2

num_imgs = len(train_meta)

num_val_imgs = len(test_meta)

steps = int(num_imgs/batch_size)

val_steps = int(num_val_imgs/batch_size)

plt.rcParams["figure.figsize"] = (20, 10)
num_cols = 8

num_rows = np.ceil(batch_size/num_cols)

train_generator = generator(train_meta, batch_size=batch_size, target_size=img_size)

test_generator = generator(test_meta, batch_size=batch_size, target_size=img_size)

x, y = next(test_generator)

for i in range(0, batch_size):

    img = x[i,:,:,:]

    img[:,:,2] += 103.939

    img[:,:,1] += 116.779

    img[:,:,0] += 123.68

    plt.subplot(num_rows, num_cols, i+1)

    plt.imshow(img/255)
def get_model():

    base_model = tf.keras.applications.resnet50.ResNet50(include_top=False, weights='imagenet')

    #for layer in base_model.layers:

    #layer.trainable = False

    #base_model.summary()

    features = base_model.output

    x = tf.keras.layers.GlobalAveragePooling2D()(features)

    x = tf.keras.layers.Dense(4096, activation='relu')(x)

    x = tf.keras.layers.Dropout(dropout)(x)

    x = tf.keras.layers.Dense(4096, activation='relu')(x)

    x = tf.keras.layers.Dropout(dropout)(x)

    x = tf.keras.layers.Dense(101, activation='softmax')(x)

    model = tf.keras.models.Model(inputs=base_model.input, outputs=x)

    return model
class TheCallback(tf.keras.callbacks.Callback):

    def __init__(self, val_generator, val_steps, patience=5, lr_factor=0.1, model_file='model.h5'):

        self.val_generator = val_generator

        self.patience = patience

        self.best = np.Inf

        self.lr_factor = lr_factor

        self.model_file = model_file

        self.val_steps = val_steps

        

    def on_train_begin(self, logs={}):

        self.losses = []

        self.val_losses = []

        self.acc = []

        self.val_acc = []

        self.wait = 0

        self.best_weights = None

        print('Begin training with lr = {}'.format(tf.keras.backend.get_value(self.model.optimizer.lr)))

        

    def on_epoch_end(self, epoch, logs={}):

        

        # Append training loss and accuracy

        self.losses.append(logs['loss'])

        self.acc.append(logs['acc'])

        

        # Evaluate model

        model = self.model

        metrics = model.evaluate(

            self.val_generator, 

            steps=self.val_steps, 

            return_dict=True

        )

        

        # Append validation loss and accuracy

        self.val_losses.append(metrics['loss'])

        self.val_acc.append(metrics['acc'])

        

        # Plot learning curves

        plt.subplot(1,2,1)

        plt.plot(self.losses, label='Train Loss')

        plt.plot(self.val_losses, label='Validation Loss')

        plt.xlabel('Epoch')

        plt.ylabel('Loss')

        plt.legend()

        plt.subplot(1,2,2)

        plt.plot(self.acc, label='Train Accuracy')

        plt.plot(self.val_acc, label='Validation Accuracy')

        plt.xlabel('Epoch')

        plt.ylabel('Accuracy')

        plt.legend()

        clear_output(wait=True)

        plt.show()

        

        # Rate reduction on plateau

        if self.val_losses[-1] < self.best:

            self.best = self.val_losses[-1]

            self.best_weights = self.model.get_weights()

            self.wait = 0

        else:

            self.wait += 1

            # If ran out of patience reduce lr

            if self.wait > self.patience: 

                lr = tf.keras.backend.get_value(self.model.optimizer.lr) # Current lr

                tf.keras.backend.set_value(self.model.optimizer.lr, lr*self.lr_factor) # Reduce lr by lr_factor

                print('Reducing lr from {} to {}'.format(lr, lr*self.lr_factor))

                

    def on_train_end(self, logs={}):

        self.model.set_weights(self.best_weights)

        self.model.save(self.model_file) # Save the best model to file

        print('Training complete. Best model saved to {}. Final lr = {}.'.format(self.model_file, 

                                                                                 tf.keras.backend.get_value(self.model.optimizer.lr)))

            

        
def learn_rate_finder(model, start_lr=1e-6, end_lr=1, epochs=1, steps=1):

    class LearnRateFinder(tf.keras.callbacks.Callback):

        def __init__(self, start_lr, end_lr, lr_factor):

            self.start_lr = start_lr

            self.end_lr = end_lr

            self.lr_factor = lr_factor 

            

        def on_train_begin(self, logs={}):

            self.losses = []

            self.lrs = []

            self.batch_num = 1

            

        def on_batch_end(self, epoch, logs={}):

            self.batch_num += 1

            self.lrs.append(tf.keras.backend.get_value(self.model.optimizer.lr))

            self.losses.append(logs['loss'])

            tf.keras.backend.set_value(self.model.optimizer.lr, self.lrs[-1] * self.lr_factor)

            

        def get_best_lr_exp_weighted(self, beta=0.98, n_skip_beginning=10, n_skip_end=5):

            derivatives = self.exp_weighted_derivatives(beta)

            return min(zip(derivatives[n_skip_beginning:-n_skip_end], self.lrs[n_skip_beginning:-n_skip_end]))[1]



        def exp_weighted_losses(self, beta=0.98):

            losses = []

            avg_loss = 0.

            for batch_num, loss in enumerate(self.losses):

                avg_loss = beta * avg_loss + (1 - beta) * loss

                smoothed_loss = avg_loss / (1 - beta ** (batch_num+1))

                losses.append(smoothed_loss)

            return losses



        def exp_weighted_derivatives(self, beta=0.98):

            derivatives = [0]

            losses = self.exp_weighted_losses(beta)

            for i in range(1, len(losses)):

                derivatives.append((losses[i] - losses[i - 1]) / 1)

            return derivatives

                

        def on_train_end(self, logs={}):

            loss_derivatives = self.exp_weighted_derivatives()

            plt.plot(self.lrs, loss_derivatives, label='Derivative')

            plt.plot(self.lrs, self.losses, label='loss')

            plt.xscale('log')

            plt.legend()

            print(self.get_best_lr_exp_weighted())

            

    lr_factor = (float(end_lr) / float(start_lr)) ** (1. / float(steps * epochs))

    callback = LearnRateFinder(start_lr, end_lr, lr_factor)

    opt = tf.keras.optimizers.Adam(lr=start_lr)

    model.compile(loss='categorical_crossentropy', metrics=['acc'], optimizer=opt)

    model.fit(train_generator, epochs=epochs, steps_per_epoch=steps, verbose=1, callbacks=[callback])
model = get_model()

model.summary()

opt = tf.keras.optimizers.Adam(lr=1e-4)

model.compile(loss='categorical_crossentropy', metrics=['acc'], optimizer=opt)
callback = TheCallback(test_generator, 100, patience=3, model_file='resnet50_do0.2.h5')
history = model.fit_generator(train_generator, epochs=15, steps_per_epoch=steps, verbose=1, callbacks=[callback])
model = tf.keras.models.load_model('resnet50_do0.2.h5')

model.summary()
x = []

y = []

true_count = 0

total_count = 0

acc = 0

for _,row in test_meta.iterrows():

    file_name = row['target'] + '/' + str(row['image']) + '.jpg'

    image = tf.keras.preprocessing.image.load_img(

        os.path.join(imgs_dir, file_name), 

        target_size=(img_size, img_size)

    )

    x = np.expand_dims(tf.keras.preprocessing.image.img_to_array(image), axis=0)

    x = tf.keras.applications.resnet50.preprocess_input(x)

    pred = model.predict(x)

    y.append(labels[np.argmax(pred)])

    if y[-1] == row['target']:

        true_count += 1

    total_count += 1

    acc = true_count * 100 / total_count

    clear_output(wait=True)

    print('Tested {} images. Accuracy = {}'.format(total_count, acc))

clear_output(wait=True)

print('Test accuracy = {}'.format(acc))   
def random_crop_image(img, crop_size=224, crop_percent=75):

    width, height = img.size

    

    cut_size_x = int(width * crop_percent/100)

    cut_size_y = int(height * crop_percent/100)

    start_y = np.random.randint(0, height - cut_size_y)

    start_x = np.random.randint(0, width - cut_size_x)

    crop = img.crop(box=(start_x, start_y, start_x + cut_size_x, start_y + cut_size_y))

    crop = crop.resize((crop_size, crop_size))

    return crop
x = []

y = []

num_crops = 10

true_count = 0

total_count = 0

acc = 0

for _,row in test_meta.iterrows():

    x = np.zeros((num_crops, img_size, img_size, 3))

    file_name = row['target'] + '/' + str(row['image']) + '.jpg'

    image = tf.keras.preprocessing.image.load_img(

        os.path.join(imgs_dir, file_name), 

        target_size=(img_size, img_size)

    )

    for i in range(0, num_crops):

        x[i, :, :, :] = tf.keras.applications.resnet50.preprocess_input(

            tf.keras.preprocessing.image.img_to_array(

                random_crop_image(image, crop_percent=95)

            )

        )

        x[i, :, :, :] = (x[i, :, :, :])

    pred = model.predict(x)

    y.append(labels[np.argmax(np.sum(pred, axis=0))])

    if y[-1] == row['target']:

        true_count += 1

    total_count += 1

    acc = true_count * 100 / total_count

    clear_output(wait=True)

    print('Tested {} images. Accuracy = {}'.format(total_count, acc))

clear_output(wait=True)

print('Test accuracy = {}'.format(acc))   