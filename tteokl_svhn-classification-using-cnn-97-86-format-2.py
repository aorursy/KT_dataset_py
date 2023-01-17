import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import h5py

import keras

import cv2

from scipy.io import loadmat

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import OneHotEncoder

from sklearn.metrics import confusion_matrix

from keras.preprocessing.image import ImageDataGenerator

%matplotlib inline
def load_data(path):

    data = loadmat(path)

    return data['X'], data['y']
train_images, y_train = load_data('../input/svhndataset/train_32x32.mat')

test_images, y_test = load_data('../input/svhndataset/test_32x32.mat')

extra_images, y_extra = load_data('../input/svhndataset/extra_32x32.mat')
print("Training set:", train_images.shape, y_train.shape)

print("Test set:", test_images.shape, y_test.shape)

print("Extra training set:", extra_images.shape, y_extra.shape)
train_images, y_train = train_images.transpose((3,0,1,2)), y_train[:,0]

test_images, y_test = test_images.transpose((3,0,1,2)), y_test[:,0]

extra_images, y_extra = extra_images.transpose((3,0,1,2)), y_extra[:,0]
train_images = np.concatenate((train_images, extra_images))

y_train = np.concatenate((y_train, y_extra))



del extra_images, y_extra
print("Training set:", train_images.shape, y_train.shape)

print("Test set:", test_images.shape, y_test.shape)
def plot_images(images, labels, num_row=2, num_col=5):



    plt.rcParams['axes.grid'] = False

    fig, axes = plt.subplots(num_row, num_col, figsize=(2*num_col,2*num_row))

    for i in range(num_row * num_col):

        ax = axes[i//num_col, i%num_col]

        if images[i].shape[2] == 3:

          ax.imshow(images[i])

        else:

          ax.imshow(images[i].reshape(32,32), cmap = 'gray')

        ax.set_title(labels[i],weight='bold',fontsize=20)

    plt.tight_layout()

    

    plt.show()
plot_images(train_images, y_train)
plot_images(test_images, y_test)
y_train[y_train == 10] = 0

y_test[y_test == 10] = 0
def rgb2gray(image):

    return np.expand_dims(np.dot(image, [0.2125, 0.7154 ,0.0721]), axis=2)
X_train= np.zeros((train_images.shape[0],32,32,1))

for i in range(train_images.shape[0]):

  X_train[i] = rgb2gray(train_images[i])

X_test= np.zeros((test_images.shape[0],32,32,1))

for i in range(test_images.shape[0]):

  X_test[i] = rgb2gray(test_images[i])
del train_images, test_images
def minmax_normalize(image):

    return cv2.normalize(image, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F).reshape(32,32,1)
for i in range(X_train.shape[0]):

  X_train[i] = minmax_normalize(X_train[i])

for i in range(X_test.shape[0]):

  X_test[i] = minmax_normalize(X_test[i])
print("Training set:", X_train.shape, y_train.shape)

print("Test set:", X_test.shape, y_test.shape)
plot_images(X_train, y_train)
def plot_distribution(y1, y2, title1, title2):



    plt.rcParams['axes.facecolor'] = '#E6E6E6'

    plt.rcParams['axes.grid'] = True

    plt.rcParams['axes.axisbelow'] = True

    plt.rcParams['grid.color'] = 'w'

    plt.rcParams['figure.figsize'] = (12, 4)



    fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True)

    fig.suptitle('Class Distribution', fontsize=15, fontweight='bold', y=1.05)



    ax1.bar(np.arange(10),np.bincount(y1))

    ax1.set_title(title1)

    ax1.set_xlim(-0.5, 9.5)

    ax1.set_xticks(np.arange(10))

    ax2.bar(np.arange(10),np.bincount(y2),color='coral')

    ax2.set_title(title2)



    fig.tight_layout()
plot_distribution(y_train, y_test, "Training set", "Test set")
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=42)
plot_distribution(y_train, y_val, "Training set", "Validation set")
enc = OneHotEncoder().fit(y_train.reshape(-1, 1))
y_train = enc.transform(y_train.reshape(-1, 1)).toarray()

y_test = enc.transform(y_test.reshape(-1, 1)).toarray()

y_val = enc.transform(y_val.reshape(-1, 1)).toarray()
print("Training set", y_train.shape)

print("Validation set", y_val.shape)

print("Test set", y_test.shape)
datagen = ImageDataGenerator(rotation_range=8,

                             zoom_range=[0.95, 1.05],

                             height_shift_range=0.10,

                             shear_range=0.15)
keras.backend.clear_session()



model = keras.Sequential([

    keras.layers.Conv2D(32, (3, 3), padding='same', 

                           activation='relu',

                           input_shape=(32, 32, 1)),

    keras.layers.BatchNormalization(),

    keras.layers.Conv2D(32, (3, 3), padding='same', 

                        activation='relu'),

    keras.layers.MaxPooling2D((2, 2)),

    keras.layers.Dropout(0.3),

    



    keras.layers.Conv2D(64, (3, 3), padding='same', 

                           activation='relu'),

    keras.layers.BatchNormalization(),

    keras.layers.Conv2D(64, (3, 3), padding='same',

                        activation='relu'),

    keras.layers.MaxPooling2D((2, 2)),

    keras.layers.Dropout(0.3),

    



    keras.layers.Conv2D(128, (3, 3), padding='same', 

                           activation='relu'),

    keras.layers.BatchNormalization(),

    keras.layers.Conv2D(128, (3, 3), padding='same',

                        activation='relu'),

    keras.layers.MaxPooling2D((2, 2)),

    keras.layers.Dropout(0.3),

    

    

    keras.layers.Flatten(),

    keras.layers.Dense(128, activation='relu'),

    keras.layers.Dropout(0.4),    

    keras.layers.Dense(10,  activation='softmax')

])



early_stopping = keras.callbacks.EarlyStopping(patience=8)

optimizer = keras.optimizers.Adam(amsgrad=True)

model_checkpoint = keras.callbacks.ModelCheckpoint('best_cnn.h5', 

                   save_best_only=True)

model.compile(optimizer=optimizer,

              loss='categorical_crossentropy',

              metrics=['accuracy'])
model.summary()
history = model.fit_generator(datagen.flow(X_train, y_train, batch_size=128),

                              epochs=50, validation_data=(X_val, y_val),

                              callbacks=[early_stopping, model_checkpoint])
train_acc = history.history['accuracy']

val_acc = history.history['val_accuracy']



train_loss = history.history['loss']

val_loss = history.history['val_loss']
plt.figure(figsize=(20, 10))



plt.subplot(1, 2, 1)

plt.plot(train_acc, label='Training Accuracy')

plt.plot(val_acc, label='Validation Accuracy')

plt.legend()

plt.title('Epochs vs. Training and Validation Accuracy')

    

plt.subplot(1, 2, 2)

plt.plot(train_loss, label='Training Loss')

plt.plot(val_loss, label='Validation Loss')

plt.legend()

plt.title('Epochs vs. Training and Validation Loss')



plt.show()
test_loss, test_acc = model.evaluate(x=X_test, y=y_test, verbose=0)



print('Test accuracy is: {:0.4f} \nTest loss is: {:0.4f}'.

      format(test_acc, test_loss))
y_pred = model.predict(X_train)



y_pred = enc.inverse_transform(y_pred)

y_train = enc.inverse_transform(y_train)
plt.figure(dpi=300)

cm = confusion_matrix(y_train, y_pred)

plt.title('Confusion matrix for training set', weight='bold')

sns.heatmap(cm,annot=True,fmt='g',cmap='coolwarm',annot_kws={"size": 12})

plt.xlabel('Predicted')

plt.ylabel('Actual')

plt.show()