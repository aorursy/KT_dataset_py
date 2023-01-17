# Load modules

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

import itertools



from keras.utils.np_utils import to_categorical

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

from keras.optimizers import RMSprop

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau



%matplotlib inline
%%time

# Load dataset

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

target = train.label

train = train.loc[:, 'pixel0':]
train.shape, target.shape
# Target balance checking

print(target.value_counts(normalize=True))

target.value_counts().plot.bar()
# Missing checking

train.isnull().any().describe()
# Scale data

train = train / 255.0

test = test / 255.0



# Reshape data

train = train.values.reshape(-1, 28, 28, 1)

test = test.values.reshape(-1, 28, 28, 1)



# Label encoding

target = to_categorical(target, num_classes=10)



# Train-test split

X_train, X_val, y_train, y_val = train_test_split(train, target, test_size=.1, random_state=2)
print('X_train shape: {}\ny_train shape: {}\nX_val shape: {}\ny_val shape: {}'.format(X_train.shape, y_train.shape, X_val.shape, y_val.shape))
# Xem thử data

plt.imshow(X_train[0][:, :, 0])
# Xem vài mẫu random ở mỗi class

label_df = pd.read_csv('../input/train.csv', usecols=['label'])



classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

sample_per_class = 6



plt.figure()

for j, cls in enumerate(classes):

    idxs = np.flatnonzero(label_df == int(cls))

    idxs = np.random.choice(idxs, sample_per_class, replace=False)

    for i, idx in enumerate(idxs):

        plt_idx = i * len(classes) + j + 1

        plt.subplot(sample_per_class, len(classes), plt_idx)

        plt.axis('off')

        plt.imshow(train[idx][:, :, 0])

        if i == 0:

            plt.title(cls)
# Set CNN model

model = Sequential()



model.add(Conv2D(filters=32, kernel_size=5, padding='same', activation='relu', input_shape=(28, 28, 1)))

model.add(Conv2D(filters=32, kernel_size=5, padding='same', activation='relu'))

model.add(MaxPool2D(pool_size=(2)))

model.add(Dropout(0.25))



model.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))

model.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))

model.add(MaxPool2D(pool_size=(2)))

model.add(Dropout(0.25))



model.add(Flatten())

model.add(Dense(256, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(10, activation='softmax'))
model.summary()
# CNN Optimizer

optimizer = RMSprop()



# Compile CNN model

model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])



# Set learning rate annealer

lr_reduction = ReduceLROnPlateau(monitor='val_acc', patience=3, factor=.5)
# Data augmentation

data_aug = ImageDataGenerator(rotation_range=10, width_shift_range=.1, height_shift_range=.1, zoom_range=.1)

data_aug.fit(X_train)
%%time

# Params

batch_size = 86 # For data_aug

epochs = 50 # For training model



# Model fitting

m = model.fit_generator(data_aug.flow(X_train, y_train, batch_size=batch_size),

                        epochs=epochs,

                        validation_data=(X_val, y_val),

                        verbose=2,

                        steps_per_epoch=X_train.shape[0] // batch_size,

                        callbacks=[lr_reduction])
epochs = range(1, len(m.history['acc']) + 1)



plt.figure(figsize=(22, 8))

plt.subplot(1,2,1)

plt.plot(epochs, m.history['loss'], 'r-', label='Training loss')

plt.plot(epochs, m.history['val_loss'], 'bo', label='Validation loss')

plt.xlabel('epoch'); plt.ylabel('loss'); plt.legend(loc='best')

plt.title('Training and validation loss')



plt.subplot(1,2,2)

plt.plot(epochs, m.history['acc'], 'r-', label='Training accuracy')

plt.plot(epochs, m.history['val_acc'], 'bo', label='Validation accuracy')

plt.xlabel('epoch'); plt.ylabel('accuracy'); plt.legend(loc='best')

plt.title('Training and validation accuracy')
# Confusion matrix

y_true = np.argmax(y_val, axis=1)

y_pred = model.predict_classes(X_val)

cm = confusion_matrix(y_true, y_pred)



plt.figure(figsize=(14, 10))

sns.heatmap(cm, annot=True, cbar=True, square=True, fmt='d', linewidths=.3, cmap="GnBu_r")
submission = pd.read_csv('../input/sample_submission.csv')

submission.loc[:, 'Label'] = model.predict_classes(test)

submission.to_csv('submission.csv', index=False)

submission.head(10)