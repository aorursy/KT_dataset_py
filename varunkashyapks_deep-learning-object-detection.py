# Importing pandas and numpy for data importing and preprocessing

import numpy as np 

import pandas as pd 

# Importing OpenCV for the image reading

import cv2

import os, sys

from tqdm import tqdm
# Load the train and test for file this environment

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
# Checking the train csv file top 5 rows

train.head(5)
# Checking the test csv file top 5 rows

test.head(5)
# Printing the total image count in the Training Data and Test Data

print ('The traininng data has {} images.'.format(train.shape[0]))

print ('The test data has {} images.'.format(test.shape[0]))
# Now we are checkking the how many image labels are distributed so we use the seaborn,matplotlib

import matplotlib.pyplot as plt

import seaborn as sns

# This is for the training data

label_counts = train.label.value_counts()

plt.figure(figsize = (12,6))

sns.barplot(label_counts.index, label_counts.values, alpha = 0.9,color = 'slateblue')

plt.xticks(rotation = 'vertical')

plt.xlabel('Train Image Labels', fontsize =14)

plt.ylabel('Counts of Images', fontsize = 14)

plt.show()
# set path to read train and test image

TRAIN_PATH = '../input/train_img.tar.gz'

TEST_PATH = '../input/test_img.tar.gz'
# function to read images as arrays

def read_image(img_path):

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = cv2.resize(img, (128,128)) 

    return img
train_data = []

test_data = []

train_labels = train['label'].values



for img in tqdm(train['image_id'].values):

    train_data.append(read_image(TRAIN_PATH + '{}.png'.format(img)))

    

for img in tqdm(test['image_id'].values):

    test_data.append(read_image(TEST_PATH + '{}.png'.format(img)))
# normalize the images

x_train = np.array(train_data, np.float32) / 255.

x_test = np.array(test_data, np.float32) / 255.

# target variable - encoding numeric value

label_list = train['label'].tolist()

Y_train = {k:v+1 for v,k in enumerate(set(label_list))}

y_train = [Y_train[k] for k in label_list]
# check some images

def show_images(ix):

    image_train = read_image(TRAIN_PATH + train.image_id[ix] + '.png')

    image_test = read_image(TEST_PATH + test.image_id[ix] + '.png')

    

    pair = np.concatenate((image_train, image_test), axis=1)

    

    plt.figure(figsize = (6,6))

    plt.imshow(pair)

    

# first 4 images in train and test set

for idx in range(4):

    show_images(idx)
## just images doesn't help, lets see the images with their respective labels

plt.rc('axes', grid=False)



_, axs = plt.subplots(3,3, sharex = 'col', sharey='row', figsize = (7,7))

axs = axs.ravel()



# lets see first 8 images - you can increase i value to see more images

for i, (image_name, label) in enumerate(zip(train.image_id, train.label)):

    if i <= 8:

        img = read_image(TRAIN_PATH + image_name + '.png')

        axs[i].imshow(img)

        axs[i].set_title('{} - {}'.format(image_name, label))

    else:

        break
# lets train our first model, we'll use keras.



from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Convolution2D, MaxPooling2D

from keras.callbacks import EarlyStopping

from keras.utils import to_categorical
## keras accepts target variable as a ndarray so that we can set one output neuron per class

y_train = to_categorical(y_train)

## neural net architechture



model = Sequential()

model.add(Convolution2D(32, (3,3), activation='relu', padding='same',input_shape = (64,64,3))) # if you resize the image above, shape would be (128,128,3)

model.add(Convolution2D(32, (3,3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))



model.add(Convolution2D(64, (3,3), activation='relu', padding='same'))

model.add(Convolution2D(64, (3,3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.25))



model.add(Convolution2D(128, (3,3), activation='relu', padding='same'))

model.add(Convolution2D(128, (3,3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.25))



model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dense(256, activation='relu'))

model.add(Dropout(0.25))

model.add(Dense(y_train.shape[1], activation='softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
early_stops = EarlyStopping(patience=3, monitor='val_acc')

model.fit(x_train, y_train, batch_size=10, epochs=10, validation_split=0.3, callbacks=[early_stops])

# make prediction

predictions = model.predict(x_test)

predictions = np.argmax(predictions, axis= 1)

# get predicted labels

y_maps = dict()

y_maps = {v:k for k, v in Y_train.items()}

pred_labels = [y_maps[k] for k in predictions]
# make submission

sub1 = pd.DataFrame({'image_id':test.image_id, 'label':pred_labels})

sub1.to_csv('sub_one.csv', index=False)

## lets see what our classifier predicts on test images



# top 5 predictions

for i in range(5):

    print('I see this product is {}'.format(pred_labels[i]))

    plt.imshow(read_image(TEST_PATH +'{}.png'.format(test.image_id[i])))

    plt.show()