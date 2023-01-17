import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import random
%matplotlib inline
plt.style.use("seaborn")

from sklearn.cross_validation import train_test_split
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import regularizers
from keras import applications
from keras import backend as K
K.set_image_dim_ordering('tf')
TRAIN_DIR = '../input/train/'
TEST_DIR = '../input/test/'

def prepare_dataset(train_dir, test_dir, m_train=1000, m_test=20, m_validation=200):
    # load the train data
    cat_train_data = [TRAIN_DIR+cat for cat in os.listdir(TRAIN_DIR) if 'cat' in cat]
    dog_train_data = [TRAIN_DIR+dog for dog in os.listdir(TRAIN_DIR) if 'dog' in dog]
    train = cat_train_data[:m_train+m_validation] + dog_train_data[:m_train+m_validation]
    random.shuffle(train)
    
    train_data = []
    for image in train:
        temp = load_img(image, target_size=(150,150), interpolation="nearest")
        x = np.asarray(temp)
        train_data.append(x)
    
    # load the test data
    test = [TEST_DIR+i for i in os.listdir(TEST_DIR)]
    test = test[:m_test]
    
    test_data = []
    for image in test:
        temp = load_img(image, target_size=(150,150), interpolation="nearest")
        x = np.asarray(temp)
        test_data.append(x)
    
    # generating labels for training data
    label = []
    for img in train:
        if 'dog' in img:
            label.append(1)
        elif 'cat' in img:
            label.append(0)
    label = np.array(label)
    label = label.reshape((label.shape[0], 1))
    return np.array(train_data), label, np.array(test_data)

train, label, test = prepare_dataset(TRAIN_DIR, TEST_DIR)
print("Train shape: ", train.shape)
print("Labels shape: ", label.shape)
print("Test shape: ", test.shape)
print(label[0])
plt.imshow(train[0])
X_train, X_val, y_train, y_val = train_test_split(train, label, test_size=1/6, random_state=23)

X_train = X_train / 255
X_val = X_val / 255

print(X_train.shape, X_val.shape, y_train.shape, y_val.shape)
vgg = applications.VGG16(include_top=False, weights='imagenet', input_shape=X_train.shape[1:])
vgg_model = Sequential()
vgg_model.add(vgg)
vgg_model.add(Flatten())
vgg_model.add(Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
vgg_model.add(Dropout(0.5))
vgg_model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
vgg_model.add(Dropout(0.5))
vgg_model.add(Dense(1, activation='sigmoid'))

vgg_model.layers[0].trainable = False
vgg_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history_vgg = vgg_model.fit(X_train, y_train, epochs=15, batch_size=100, validation_data=(X_val, y_val))
plt.plot(history_vgg.history['loss'])
plt.plot(history_vgg.history['val_loss'])
plt.title("Train vs Validation Loss")
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
vgg19 = applications.VGG19(include_top=False, weights='imagenet', input_shape=X_train.shape[1:])
vgg19_model = Sequential()
vgg19_model.add(vgg19)
vgg19_model.add(Flatten())
vgg19_model.add(Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
vgg19_model.add(Dropout(0.5))
vgg19_model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
vgg19_model.add(Dropout(0.5))
vgg19_model.add(Dense(1, activation='sigmoid'))

vgg19_model.layers[0].trainable = False
vgg19_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history_vgg19 = vgg19_model.fit(X_train, y_train, epochs=15, batch_size=100, validation_data=(X_val, y_val))
plt.plot(history_vgg19.history['loss'])
plt.plot(history_vgg19.history['val_loss'])
plt.title("Train vs Validation Loss")
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
inception = applications.InceptionV3(include_top=False, weights='imagenet', input_shape=X_train.shape[1:])
inception_model = Sequential()
inception_model.add(vgg19)
inception_model.add(Flatten())
inception_model.add(Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
inception_model.add(Dropout(0.5))
inception_model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
inception_model.add(Dropout(0.5))
inception_model.add(Dense(1, activation='sigmoid'))

inception_model.layers[0].trainable = False
inception_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history_inception = inception_model.fit(X_train, y_train, epochs=15, batch_size=100, validation_data=(X_val, y_val))
plt.plot(history_inception.history['loss'])
plt.plot(history_inception.history['val_loss'])
plt.ylabel("loss")
plt.xlabel("epoch")
plt.title("Training vs Validation Loss")
plt.legend(['train', 'test'], loc='upper right')
f, ax = plt.subplots(2,1,figsize=(15,15))
ax[0].plot(history_vgg.history['val_loss'])
ax[0].plot(history_vgg19.history['val_loss'])
ax[0].plot(history_inception.history['val_loss'])
ax[0].set_ylabel("loss")
ax[0].set_xlabel("epoch")
ax[0].set_title("Validation Loss")
ax[0].legend(['VGG16', 'VGG19', 'InceptionV3'], loc='upper right')

ax[1].plot(history_vgg.history['val_acc'])
ax[1].plot(history_vgg19.history['val_acc'])
ax[1].plot(history_inception.history['val_acc'])
ax[1].set_ylabel("accuracy")
ax[1].set_xlabel("epoch")
ax[1].set_title("Validation Accuracy")
ax[1].legend(['VGG16', 'VGG19', 'InceptionV3'], loc='upper right')
num_test = 5
predictions = vgg_model.predict(test[:num_test,:,:,:])
for i in range(num_test):
    if predictions[i] == 0:
        print("Predicted: cat ", predictions[i])
    else:
        print("predicted: dog", predictions[i])
    plt.imshow(test[i])
    plt.show()