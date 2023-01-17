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
counts = train_meta.groupby('target')['image'].count()
counts.plot(kind='bar', stacked='True', figsize=(20, 10), color='blue', legend=False);
import tensorflow as tf

import random
# Declare an ImageDataGenerator object with proper image transformations and preprocessing.

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

# A generator with no augmentation (for testing)

image_generator_no_aug = tf.keras.preprocessing.image.ImageDataGenerator(

            preprocessing_function=tf.keras.applications.resnet50.preprocess_input

        )
def generator(df, batch_size=32, target_size=224, test=False):

    """

    Takes a train/test metadata DataFrame and yields a batch of (x, y) for training/testing.

    Usage:

    (x, y) = generator(df, batch_size, target_size)

    Inputs:

    df - train/test metadata (DataFrame)

    batch_size - batch size for train/test (int)

    target_size - target size for input image (int)

    Output:

    A batch of training/testing data of size [batch_size, target_size, target_size, 3] (float)

    """

    num_imgs = len(df) # Number of images in the train/test set

    while True:

        batch_rows = df.iloc[random.choices(range(0,num_imgs), k=batch_size)] # Select batch_size number of random

                                                                              # rows from the meta data

        x = []

        y = []

        for _,row in batch_rows.iterrows():

            file_name = row['target'] + '/' + str(row['image']) + '.jpg' # Image file name

            image = tf.keras.preprocessing.image.load_img(

                os.path.join(imgs_dir, file_name), 

                target_size=(target_size, target_size)

            )                                                            # Read image

            input_arr = tf.keras.preprocessing.image.img_to_array(image) # Convert image to array

            x.append(input_arr)                                          # Append image array to x

            y.append(row.iloc[1:102].values)                             # Append categorical label to y

        # Convert x and y to numpy arrays

        x = np.array(x, dtype=float)

        y = np.array(y, dtype=float)

        # Perform transformations

        if test:

            data_generator = image_generator_no_aug.flow(x, y, batch_size=batch_size)

        else:

            data_generator = image_generator.flow(x, y, batch_size=batch_size) 

        xt, yt = data_generator.next()

        # Yield outputs

        yield(np.array(xt), np.array(yt))

batch_size = 16

img_size = 224

dropout = 0.5

num_imgs = len(train_meta)

num_val_imgs = len(test_meta)

steps = int(num_imgs/batch_size)

val_steps = int(num_val_imgs/batch_size)

model_file = 'resnet50_do_{}.h5'.format(dropout)

plt.rcParams["figure.figsize"] = (20, 10)
# For sub-plot

num_cols = 8 

num_rows = np.ceil(batch_size/num_cols)

# Declare train and test generators

train_generator = generator(train_meta, batch_size=batch_size, target_size=img_size)

test_generator = generator(test_meta, batch_size=batch_size, target_size=img_size, test=True)



x, y = next(test_generator) # Get a single batch of train data

# Plot all the images in the batch

for i in range(0, batch_size):

    img = x[i,:,:,:]

    # Inverse preprocessing

    img[:,:,2] += 103.939

    img[:,:,1] += 116.779

    img[:,:,0] += 123.68

    # Plot

    plt.subplot(num_rows, num_cols, i+1)

    plt.imshow(img/255)
def get_model():

    """

    Returns the network model.

    The base model is ResNet50 trained on imagenet dataset.

    The classifier contains 3 FC layers with 4096, 4096 and 101 nuerons respectively.

    """

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

    """

    A single callback class that combines all the best practises.

    1. Evaluates the validation loss and accuracy at the end of every epoch.

    2. Live plot of training curves (both loss and accuracy) updated after every epoch.

    3. Learning rate reduction on loss plateau.

    4. Saves best model to file.

    """

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

        print('Begin training with lr = {}'.format(self.model.optimizer.lr.numpy))

        

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

            self.model.save(self.model_file) # Save the best model to file

            print('Saving model to {} at epoch: {}.' 

                  'Validation Loss: {}. Validation Accuracy: {}'

                  .format(self.model_file, epoch, self.val_losses[-1], self.val_acc[-1])

                 )

            self.wait = 0

        else:

            self.wait += 1

            # If ran out of patience reduce lr

            if self.wait >= self.patience:

                lr = tf.keras.backend.get_value(self.model.optimizer.lr) # Current lr

                tf.keras.backend.set_value(self.model.optimizer.lr, lr*self.lr_factor) # Reduce lr by lr_factor

                print('Reducing lr from {} to {}'.format(lr, lr*self.lr_factor))

            

        
model = get_model()

#model = tf.keras.models.load_model('resnet50_do0.2.h5')

model.summary()

opt = tf.keras.optimizers.Adam(lr=1e-4) # Set initial lr = best lr

model.compile(loss='categorical_crossentropy', metrics=['acc'], optimizer=opt)
# Declare the callback

callback = TheCallback(test_generator, val_steps, patience=2, lr_factor=0.5, model_file=model_file)
# Run the model

history = model.fit_generator(train_generator, epochs=12, steps_per_epoch=steps, verbose=1, callbacks=[callback])
model = tf.keras.models.load_model(model_file)

model.summary()
x = []

y = []

true_count = 0 # Count of correctly classified

total_count = 0 # Total count of images tested

acc = 0 # Accuracy

# Run through all the test images

for _,row in test_meta.iterrows():

    file_name = row['target'] + '/' + str(row['image']) + '.jpg' # Test image file name

    image = tf.keras.preprocessing.image.load_img(

        os.path.join(imgs_dir, file_name), 

        target_size=(img_size, img_size)

    )                                                            # Load test image

    x = tf.keras.applications.resnet50.preprocess_input(

        tf.keras.preprocessing.image.img_to_array(image)

    ) # Apply pre-processing

    x = np.expand_dims(x, axis=0) # [224, 224, 3] => [1, 224, 224, 3]

    

    pred = model.predict(x) # Get probablities

    y.append(labels[np.argmax(pred)]) # Append class with max probablity to list of outputs

    # If prediction is correct increment true count

    if y[-1] == row['target']:

        true_count += 1

    total_count += 1

    acc = true_count * 100 / total_count # Calculate accuracy

    clear_output(wait=True)

    print('Tested {} images. Accuracy = {}'.format(total_count, acc))

clear_output(wait=True)

print('Test accuracy = {}'.format(acc))   
def random_crop_image(img, crop_size=224, crop_percent=75):

    """

    Returns a random crop of the input image. The crop will contain crop_percent% of the input and will be

    resized to [crop_size, crop_size, 3] before returning.

    Usage:

    random_crop_image(img, crop_size, crop_percent)

    Inputs:

    img - Input image

    crop_size - The required output size. Default is 224.

    crop_percent - Percent part of the input required in the output. Default is 75.

    """ 

    width, height = img.size # Size of input

    # Size of cropped image

    cut_size_x = int(width * crop_percent/100)

    cut_size_y = int(height * crop_percent/100)

    # Randomly select starting point

    start_y = np.random.randint(0, height - cut_size_y)

    start_x = np.random.randint(0, width - cut_size_x)

    # Crop and resize

    crop = img.crop(box=(start_x, start_y, start_x + cut_size_x, start_y + cut_size_y))

    crop = crop.resize((crop_size, crop_size))

    return crop
x = []

y = []

num_crops = 10

true_count = 0 # Count of correctly classified

total_count = 0 # Total count of images tested

acc = 0 # Accuracy

# Run through all the test images

for _,row in test_meta.iterrows():

    x = np.zeros((num_crops, img_size, img_size, 3))

    file_name = row['target'] + '/' + str(row['image']) + '.jpg' # Image file name

    image = tf.keras.preprocessing.image.load_img(

        os.path.join(imgs_dir, file_name), 

        target_size=(img_size, img_size)

    )                                                            # Read image

    # Take 10 random crops of each image

    for i in range(0, num_crops):

        x[i, :, :, :] = tf.keras.applications.resnet50.preprocess_input(

            tf.keras.preprocessing.image.img_to_array(

                random_crop_image(image, crop_percent=90)

            )

        )

    pred = model.predict(x) # Make predicition on all 10 crops

    y.append(labels[np.argmax(np.sum(pred, axis=0))]) # Take average and append to y

    # If prediction is correct increment true count

    if y[-1] == row['target']:

        true_count += 1

    total_count += 1

    acc = true_count * 100 / total_count # Calculate accuracy

    clear_output(wait=True)

    print('Tested {} images. Accuracy = {}'.format(total_count, acc))

clear_output(wait=True)

print('Test accuracy = {}'.format(acc))   