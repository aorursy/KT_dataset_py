import numpy as np 
import pandas as pd 
import os
import matplotlib.pyplot as plt
import matplotlib.image as mplimg
from matplotlib.pyplot import imshow

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

import tensorflow as tf

from keras import layers
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from keras.layers import Input, Dense, Activation, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout
from keras.models import Model

import keras.backend as K
from keras.models import Sequential

import warnings
warnings.simplefilter("ignore", category=DeprecationWarning)
SEED = 42
np.random.seed(SEED)
tf.set_random_seed(SEED)
train_df = pd.read_csv("../input/train.csv")
train_df.head()
def prepareImages(data, m, dataset):
    print("Preparing images")
    X_train = np.zeros((m, 100, 100, 3))
    count = 0
    
    for fig in data['Image']:
        #load images into images of size 100x100x3
        img = image.load_img("../input/"+dataset+"/"+fig, target_size=(100, 100, 3))
        x = image.img_to_array(img)
        x = preprocess_input(x)

        X_train[count] = x
        if (count%500 == 0):
            print("Processing image: ", count+1, ", ", fig)
        count += 1
    
    return X_train
def prepare_labels(y):
    values = np.array(y)
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    # print(integer_encoded)

    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    # print(onehot_encoded)

    y = onehot_encoded
    # print(y.shape)
    return y, label_encoder
X = prepareImages(train_df, train_df.shape[0], "train")
X /= 255
y, label_encoder = prepare_labels(train_df['Id'])
train_df["Id"].value_counts()
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    rotation_range=90,
    featurewise_center=True,
    width_shift_range=0.2,
    height_shift_range=0.2)

train_datagen.fit(X)
BATCH_SIZE = 32
EPOCHS = 40
from keras.applications import MobileNet
from keras.applications.mobilenet import preprocess_input
from keras.optimizers import Adam
from keras.metrics import categorical_accuracy, top_k_categorical_accuracy, categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

train_datagen.fit(X)
def valid_generator(batch_size, data, target):
    while True:
        for kk in range(0, data.shape[0], batch_size):
            start = kk
            end = min(start + batch_size, data.shape[0])
            x = data[start:end]
            y = target[start:end]
            yield x, y
val_set_number = np.random.choice(X.shape[0], 2000)
valid_datagen = valid_generator(BATCH_SIZE, X[val_set_number], y[val_set_number])
def top_5_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=5)
best_save_model_file = '../working/mymodel.h5'
callbacks = [EarlyStopping(monitor='val_loss',
                           patience=20,
                           verbose=1,
                           min_delta=0.00001,
                           mode='min'),
             ReduceLROnPlateau(monitor='val_loss',
                               factor=0.1,
                               patience=2,
                               verbose=1,
                               min_delta=0.0001,
                               mode='min'),
             ModelCheckpoint(monitor='val_loss',save_weights_only=True,
                             filepath=best_save_model_file,
                             save_best_only=True,
                             mode='min') ,
             ]
model = MobileNet(input_shape=(100, 100, 3), alpha=1., weights=None, classes=5005)
model.compile(optimizer=Adam(lr=0.002), loss='categorical_crossentropy',
              metrics=[categorical_crossentropy, categorical_accuracy, top_5_accuracy])
print(model.summary())
history = model.fit_generator(train_datagen.flow(X, y, batch_size=BATCH_SIZE), validation_data=valid_datagen, validation_steps=X.shape[0] // BATCH_SIZE,
                   steps_per_epoch = X.shape[0] // BATCH_SIZE,
                   epochs=EPOCHS, verbose=1, callbacks=callbacks)
plt.plot(history.history['categorical_accuracy'])
plt.title('Model categorical accuracy')
plt.ylabel('categorical accuracy')
plt.xlabel('Epoch')
plt.show()
test = os.listdir("../input/test/")
print(len(test))
col = ['Image']
test_df = pd.DataFrame(test, columns=col)
test_df['Id'] = ''
X = prepareImages(test_df, test_df.shape[0], "test")
X /= 255
predictions = model.predict(np.array(X), verbose=1)
for i, pred in enumerate(predictions):
    test_df.loc[i, 'Id'] = ' '.join(label_encoder.inverse_transform(pred.argsort()[-5:][::-1]))
test_df.head(10)
test_df.to_csv('submission.csv', index=False)


