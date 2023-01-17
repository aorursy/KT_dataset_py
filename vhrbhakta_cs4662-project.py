import numpy as np

import pandas as pd

from keras.models import Sequential

from keras.layers import Convolution2D, MaxPooling2D

from keras.layers import Dense, Dropout, Activation, Flatten

from tensorflow.keras import utils

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import os
path = '../input/understanding_cloud_organization'

os.listdir(path)
train_df = pd.read_csv(f'{path}/train.csv')

sub = pd.read_csv(f'{path}/sample_submission.csv')
train_df.head()
n_train = len(os.listdir(f'{path}/train_images'))

n_test = len(os.listdir(f'{path}/test_images'))

print(f'There are {n_train} images in train dataset')

print(f'There are {n_test} images in test dataset')
value_counts = train_df.loc[train_df['EncodedPixels'].isnull() == False, 'Image_Label'].apply(lambda x: x.split('_')[1]).value_counts()



fish_height = value_counts['Fish']

gravel_height = value_counts['Gravel']

flower_height = value_counts['Flower']

sugar_height = value_counts['Sugar']



height = [fish_height, gravel_height, flower_height, sugar_height]

bars = ('Fish', 'Gravel', 'Flower', 'Sugar')

y_pos = np.arange(len(bars))

plt.bar(y_pos, height, color=(0.2, 0.4, 0.6, 0.6))



plt.xticks(y_pos, bars, color='orange', rotation=45, fontweight='bold', fontsize='17', horizontalalignment='right')

# Custom Axis title

plt.xlabel('Labels', fontweight='bold', color = 'orange', fontsize='17', horizontalalignment='center')

frequency_count = train_df.loc[train_df['EncodedPixels'].isnull() == False, 'Image_Label'].apply(lambda x: x.split('_')[0]).value_counts().value_counts()



once = frequency_count[1]

twice = frequency_count[2]

thrice = frequency_count[3]

four_times = frequency_count[4]



height = [once, twice, thrice, four_times]

bars = ('1', '2', '3', '4')

y_pos = np.arange(len(bars))

plt.bar(y_pos, height, color=(0.2, 0.4, 0.6, 0.6))



plt.xticks(y_pos, bars, color='orange', rotation=45, fontweight='bold', fontsize='17', horizontalalignment='right')

# Custom Axis title

plt.xlabel('Images with n number of labels', fontweight='bold', color = 'orange', fontsize='17', horizontalalignment='center')
train_df = train_df[~train_df['EncodedPixels'].isnull()]

train_df['Image'] = train_df['Image_Label'].map(lambda x: x.split('_')[0])

train_df['Class'] = train_df['Image_Label'].map(lambda x: x.split('_')[1])

classes = train_df['Class'].unique()

train_df = train_df.groupby('Image')['Class'].agg(set).reset_index()

for class_name in classes:

    train_df[class_name] = train_df['Class'].map(lambda x: 1 if class_name in x else 0)

train_df.head()
img_2_ohe_vector = {img:vec for img, vec in zip(train_df['Image'], train_df.iloc[:, 2:].values)}
train_imgs, val_imgs = train_test_split(train_df['Image'].values, 

                                        test_size=0.2, 

                                        stratify=train_df['Class'].map(lambda x: str(sorted(list(x)))),

                                        random_state=5)
from copy import deepcopy



class DataGenenerator(utils.Sequence):

    def __init__(self, images_list=None, folder_imgs=f'{path}/train_images', 

                 batch_size=32, shuffle=True, augmentation=None,

                 resized_height=224, resized_width=224, num_channels=3):

        self.batch_size = batch_size

        self.shuffle = shuffle

        self.augmentation = augmentation

        if images_list is None:

            self.images_list = os.listdir(folder_imgs)

        else:

            self.images_list = deepcopy(images_list)

        self.folder_imgs = folder_imgs

        self.len = len(self.images_list) // self.batch_size

        self.resized_height = resized_height

        self.resized_width = resized_width

        self.num_channels = num_channels

        self.num_classes = 4

        self.is_test = not 'train' in folder_imgs

        if not shuffle and not self.is_test:

            self.labels = [img_2_ohe_vector[img] for img in self.images_list[:self.len*self.batch_size]]



    def __len__(self):

        return self.len

    

    def on_epoch_start(self):

        if self.shuffle:

            random.shuffle(self.images_list)



    def __getitem__(self, idx):

        current_batch = self.images_list[idx * self.batch_size: (idx + 1) * self.batch_size]

        X = np.empty((self.batch_size, self.resized_height, self.resized_width, self.num_channels))

        y = np.empty((self.batch_size, self.num_classes))



        for i, image_name in enumerate(current_batch):

            path = os.path.join(self.folder_imgs, image_name)

            img = cv2.resize(cv2.imread(path), (self.resized_height, self.resized_width)).astype(np.float32)

            if not self.augmentation is None:

                augmented = self.augmentation(image=img)

                img = augmented['image']

            X[i, :, :, :] = img/255.0

            if not self.is_test:

                y[i, :] = img_2_ohe_vector[image_name]

        return X, y



    def get_labels(self):

        if self.shuffle:

            images_current = self.images_list[:self.len*self.batch_size]

            labels = [img_2_ohe_vector[img] for img in images_current]

        else:

            labels = self.labels

        return np.array(labels)

data_generator_train = DataGenenerator(train_imgs)

data_generator_train_eval = DataGenenerator(train_imgs, shuffle=False)

data_generator_val = DataGenenerator(val_imgs, shuffle=False)
model = Sequential()

model.add(Convolution2D(32, (3, 3), activation='relu', input_shape=(224,224,3)))

model.add(Convolution2D(32, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.25)) 

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(4, activation='softmax'))

model.compile(loss='categorical_crossentropy',metrics=['accuracy'], optimizer='adam')



print(model.summary())
from multiprocessing import cpu_count

import cv2



num_cores = cpu_count()



history = model.fit_generator(generator=data_generator_train,

                              validation_data=data_generator_val,

                              epochs=10,

                              workers=num_cores,

                              verbose=1

                             )
accuracy = history.history['accuracy']

loss = history.history['loss']

epochs = range(len(accuracy))

plt.plot(epochs, accuracy, 'b-', label='Training accuracy')

plt.title('Training and validation accuracy')

plt.legend()

plt.figure()

plt.plot(epochs, loss, 'r-', label='Training loss')

plt.title('Training and validation loss')

plt.legend()

plt.show()
from sklearn.metrics import multilabel_confusion_matrix



y_pred = model.predict_generator(data_generator_val, workers=num_cores)

y_true = data_generator_val.get_labels()



print(multilabel_confusion_matrix(y_true, y_pred))