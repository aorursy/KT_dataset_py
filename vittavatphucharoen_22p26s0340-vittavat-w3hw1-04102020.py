import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from skimage.transform import rotate, resize
import tensorflow as tf
## Create Function for Sampling and add data to DataFrame to fix imbalance category
def sampling_add_Xy(X, y):
  diff = y[y == 0].shape[0] - y[y == 1].shape[0]
  ind_one = np.where(y == 1)[0]
  sampling_ind = np.random.choice(ind_one, diff)
  X_to_add = X[sampling_ind]
  y_to_add = y[sampling_ind]
  X = np.append(X, X_to_add, axis = 0)
  y = np.append(y, y_to_add, axis = 0)
  return X, y
## Import Dataset
trn_set = pd.read_csv("../input/super-ai-image-classification/train/train/train.csv")
## Read Image
path = '../input/super-ai-image-classification/train/train/images/'
fn = trn_set['id']
trn_set['path_fn'] = path + fn
trn_set['img'] = trn_set['path_fn'].apply(lambda x: plt.imread(x))
trn_set['img'] = trn_set['img'].apply(lambda x: resize(x, (224,224)))
trn_set.groupby(by = 'category').count()
X = np.array(trn_set['img'].to_list())
y = np.array(trn_set['category'].to_list())
#y = tf.keras.utils.to_categorical(y, num_classes = 2)
## Train, Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1973)
print(X_train.shape, y_train.shape)
## Oversampling to balance classes
X_train, y_train = sampling_add_Xy(X_train, y_train)
print(X_train.shape, y_train.shape)
## DenseNet121 - 3 epoch = 20
densenet = tf.keras.applications.densenet.DenseNet121(include_top = False, weights = 'imagenet', input_shape = (224, 224, 3),
                                                     pooling = 'max')
# fit output
x = tf.keras.layers.Flatten()(densenet.output)
x = tf.keras.layers.Dropout(rate = 0.8)(x)
x = tf.keras.layers.Dense(3715, activation='relu')(x)
x = tf.keras.layers.Dropout(rate = 0.7)(x)
x = tf.keras.layers.Dense(2700, activation='relu')(x)
x = tf.keras.layers.Dropout(rate = 0.6)(x)
x = tf.keras.layers.Dense(300, activation='relu')(x)
x = tf.keras.layers.Dropout(rate = 0.5)(x)
x = tf.keras.layers.Dense(2, activation='softmax')(x)
model = tf.keras.Model(densenet.input, x)
model.compile(loss='sparse_categorical_crossentropy', optimizer= tf.keras.optimizers.Adam(lr=0.0001),metrics=['accuracy'])
model.summary()
## Data Augmentation
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    #featurewise_center=True,
    #featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

datagen.fit(X_train)
## fits the model on batches with real-time data augmentation:
history = model.fit(datagen.flow(X_train, y_train, batch_size = 32),
          steps_per_epoch = len(X_train) / 32, epochs = 42, validation_data = (X_test, y_test))
## version1
#model.compile(loss='sparse_categorical_crossentropy', optimizer= tf.keras.optimizers.Adam(lr=1e-5),metrics=['accuracy'])
#history = model.fit(X_train, y_train, epochs= 50, validation_data = (X_test,y_test))

# summarize history for accuracy
#plt.plot(history.history['accuracy'])
#plt.plot(history.history['val_accuracy'])
#plt.title('model accuracy')
#plt.ylabel('accuracy')
#plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='upper left')
#plt.show()
# summarize history for loss
#plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
#plt.title('model loss')
#plt.ylabel('loss')
#plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='upper left')
#plt.show()
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
y_hat = model.predict(X_test)

from sklearn.metrics import f1_score
#y_hat_trnf = y_hat.round()
y_hat_trnf = y_hat.argmax(axis=1)

F_measure = f1_score(y_test, y_hat_trnf)
print("F1 :", F_measure)
## Save Model
import pickle

Pkl_Filename = "DensNet121_1.pkl"  
with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(model, file)

## Import Test Set
import pathlib

test_set = []
fnames = []
path = pathlib.Path("../input/super-ai-image-classification/val/val/images")
path = path.glob("*.jpg")

for imagepath in path:
        fname = os.path.basename(str(imagepath))
        fnames.append(fname)
        img = plt.imread(str(imagepath))
        img = resize(img,(224,224))
        test_set.append(img)

test_set = np.array(test_set)
test_set.shape
y_hat_submission = model.predict(test_set)
y_hat_submission = y_hat_submission.argmax(axis=1)
## Create Submission DataFrame
submission_df = pd.DataFrame({'id': fnames,
              'category':y_hat_submission})
## Export .csv file
submission_df.to_csv('val_final.csv', index = False)

## VGG16
vgg = tf.keras.applications.vgg16.VGG16(include_top = False, weights = 'imagenet', input_shape = (224, 224, 3))
# fit output
x = tf.keras.layers.Flatten()(vgg.output)
x = tf.keras.layers.Dropout(rate = 0.8)(x)
x = tf.keras.layers.Dense(3715, activation='relu')(x)
x = tf.keras.layers.Dropout(rate = 0.5)(x)
x = tf.keras.layers.Dense(2700, activation='relu')(x)
x = tf.keras.layers.Dropout(rate = 0.5)(x)
x = tf.keras.layers.Dense(300, activation='sigmoid')(x)
x = tf.keras.layers.Dropout(rate = 0.5)(x)
x = tf.keras.layers.Dense(2, activation='softmax')(x)
model = tf.keras.Model(vgg.input, x)
model.compile(loss='sparse_categorical_crossentropy', optimizer= tf.keras.optimizers.Adam(lr=0.0001),metrics=['accuracy'])
model.summary()
## DenseNet121 - F1_submision0.84, F1 notebook 0.82
densenet = tf.keras.applications.densenet.DenseNet121(include_top = False, weights = 'imagenet', input_shape = (224, 224, 3),
                                                     pooling = 'max')
# fit output
x = tf.keras.layers.Flatten()(densenet.output)
x = tf.keras.layers.Dropout(rate = 0.8)(x)
x = tf.keras.layers.Dense(3715, activation='relu')(x)
x = tf.keras.layers.Dropout(rate = 0.5)(x)
x = tf.keras.layers.Dense(2700, activation='relu')(x)
x = tf.keras.layers.Dropout(rate = 0.5)(x)
x = tf.keras.layers.Dense(300, activation='sigmoid')(x)
x = tf.keras.layers.Dropout(rate = 0.5)(x)
x = tf.keras.layers.Dense(2, activation='softmax')(x)
model = tf.keras.Model(densenet.input, x)
model.compile(loss='sparse_categorical_crossentropy', optimizer= tf.keras.optimizers.Adam(lr=0.0001),metrics=['accuracy'])
model.summary()
## DenseNet121 - F1 notebook 81
densenet = tf.keras.applications.densenet.DenseNet121(include_top = False, weights = 'imagenet', input_shape = (224, 224, 3),
                                                     pooling = 'max')
# fit output
x = tf.keras.layers.Flatten()(densenet.output)
x = tf.keras.layers.Dropout(rate = 0.8)(x)
x = tf.keras.layers.Dense(3715, activation='relu')(x)
x = tf.keras.layers.Dropout(rate = 0.5)(x)
x = tf.keras.layers.Dense(2700, activation='relu')(x)
x = tf.keras.layers.Dropout(rate = 0.5)(x)
x = tf.keras.layers.Dense(300, activation='relu')(x)
x = tf.keras.layers.Dropout(rate = 0.5)(x)
x = tf.keras.layers.Dense(2, activation='softmax')(x)
model = tf.keras.Model(densenet.input, x)
model.compile(loss='sparse_categorical_crossentropy', optimizer= tf.keras.optimizers.Adam(lr=0.0001),metrics=['accuracy'])
model.summary()
## use epoch = 40
## DenseNet121 - 3 epoch = 20 F1 notebook 
densenet = tf.keras.applications.densenet.DenseNet121(include_top = False, weights = 'imagenet', input_shape = (224, 224, 3),
                                                     pooling = 'max')
# fit output
x = tf.keras.layers.Flatten()(densenet.output)
x = tf.keras.layers.Dropout(rate = 0.8)(x)
x = tf.keras.layers.Dense(3715, activation='relu')(x)
x = tf.keras.layers.Dropout(rate = 0.7)(x)
x = tf.keras.layers.Dense(2700, activation='relu')(x)
x = tf.keras.layers.Dropout(rate = 0.6)(x)
x = tf.keras.layers.Dense(300, activation='relu')(x)
x = tf.keras.layers.Dropout(rate = 0.5)(x)
x = tf.keras.layers.Dense(2, activation='softmax')(x)
model = tf.keras.Model(densenet.input, x)
model.compile(loss='sparse_categorical_crossentropy', optimizer= tf.keras.optimizers.Adam(lr=0.0001),metrics=['accuracy'])
model.summary()