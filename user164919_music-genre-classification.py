# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import librosa 
import matplotlib.pyplot as plt
import librosa.display


N_FFT = 2048
N_MELS = 128
HOP_LEN = 512
y, sfr = librosa.load('/kaggle/input/gtzan-dataset-music-genre-classification/Data/genres_original/hiphop/hiphop.00016.wav')

melSpec = librosa.feature.melspectrogram(y=y, sr=sfr, n_mels=N_MELS,hop_length=HOP_LEN, n_fft=N_FFT)
melSpec_dB = librosa.power_to_db(melSpec, ref=np.max)

librosa.display.specshow(melSpec_dB)
path='../input/gtzan-dataset-music-genre-classification/Data/images_original'
data_gen=tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255., validation_split=0.2)
(img_height, img_width)=(299,299)
batch_size=32
train_generator = data_gen.flow_from_directory(
    path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training') # set as training data

validation_generator = data_gen.flow_from_directory(
    path, # same directory as training data
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation') # set as validation data
from tensorflow.keras.applications import Xception
base_model=Xception(include_top=False,input_shape=(299,299,3))

# Freeze the base_model
base_model.trainable = False

# Create new model on top
inputs = tf.keras.Input(shape=(299,299,3))

x = base_model(inputs, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x)  # 1st improove Regularize with dropout
x = tf.keras.layers.Dense(512,activation='relu')(x)
x = tf.keras.layers.Dropout(0.2)(x) # 2st improove Regularize with dropout
x = tf.keras.layers.Dense(64,activation='relu')(x)
outputs = tf.keras.layers.Dense(10,activation='softmax')(x)

model = tf.keras.Model(inputs, outputs)
model.summary()
model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['acc'])
history_warmup = model.fit(train_generator, validation_data=validation_generator,epochs=20)
base_model.trainable = True
model.summary()
optimizer = tf.keras.optimizers.Adam(1e-5)
model.compile(loss='categorical_crossentropy',optimizer=optimizer,metrics=['acc'])
history_final = model.fit(train_generator, validation_data=validation_generator,epochs=20)
plt.plot(range(0,20), history_warmup.history['acc'], label='Warmup Accuracy (training data)')
plt.plot(range(0,20), history_warmup.history['val_acc'], label='Warmup Accuracy (validation data)')

plt.plot(range(20,40), history_final.history['acc'], label='Final Accuracy (training data)')
plt.plot(range(20,40), history_final.history['val_acc'], label='Final Accuracy (validation data)')

plt.ylabel('Accuracy')
plt.xlabel('No. epoch')
#plt.legend(loc="upper left")
plt.show()
import itertools

#http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
import os
import cv2 

IMG = '../input/gtzan-dataset-music-genre-classification/Data/images_original'
#IMG = './dataset/'
img_dataset = []
genre_target = []
genres = {}
classes = []
i = 0
for root, dirs, files in os.walk(IMG):
    for name in files:
        filename = os.path.join(root, name)
        img_dataset.append(filename)
        genre = filename.split('/')[-2]
        genre_target.append(genre)
        
        if(genre not in genres):
            classes.append(genre)
            genres[genre] = i
            i+=1

img = cv2.imread(img_dataset[0],1)
def crop_borders(img,x1=35,x2=252,y1=54,y2=389):
    cropped = img[x1:x2,y1:y2]
    return cropped

def get_y():
    '''Convierte los generos en un array de targets y'''
    y = []
    for genre in genre_target:
        n = genres[genre]
        y.append(n)
    return np.array(y)

def get_x(shape=[999,217,335], flag=1):
    x = np.empty(shape, np.uint8)
    for i in range(len(img_dataset)):
        img = cv2.imread(img_dataset[i],flag)
        img = crop_borders(img)
        x[i] = img
    return np.array(x)
img = cv2.imread(img_dataset[0])
img = crop_borders(img)

img.shape
X = get_x(shape=[999,img.shape[0], img.shape[1], img.shape[2]]) #Imagenes en color, RGB -> 3 canales
y = get_y()

m = len(y)
num_labels = 10 #estilos de musica diferente

print(X.shape, y.shape)
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2)

y = to_categorical(y)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
base_model=Xception(include_top=False,input_shape=(217, 335, 3))

# Freeze the base_model
base_model.trainable = False

# Create new model on top
inputs = tf.keras.Input(shape=(217, 335, 3))

x = base_model(inputs, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x)  # 1st improove Regularize with dropout
x = tf.keras.layers.Dense(512,activation='relu')(x)
x = tf.keras.layers.Dropout(0.2)(x) # 2st improove Regularize with dropout
x = tf.keras.layers.Dense(64,activation='relu')(x)
outputs = tf.keras.layers.Dense(10,activation='softmax')(x)

model = tf.keras.Model(inputs, outputs)
model.summary()
model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['acc'])
history_warmup = model.fit(X_train,y_train,
                           validation_data=(X_test, y_test),
                           epochs=20)
base_model.trainable = True
model.summary()
optimizer = tf.keras.optimizers.Adam(1e-5)
model.compile(loss='categorical_crossentropy',optimizer=optimizer,metrics=['acc'])
history_final = model.fit(X_train,y_train,
                           validation_data=(X_test, y_test),
                           epochs=20)
plt.plot(range(0,20), history_warmup.history['acc'], label='Warmup Accuracy (training data)')
plt.plot(range(0,20), history_warmup.history['val_acc'], label='Warmup Accuracy (validation data)')

plt.plot(range(20,40), history_final.history['acc'], label='Final Accuracy (training data)')
plt.plot(range(20,40), history_final.history['val_acc'], label='Final Accuracy (validation data)')

plt.ylabel('Accuracy')
plt.xlabel('No. epoch')
#plt.legend(loc="upper left")
plt.show()
from sklearn.metrics import confusion_matrix

preds = np.argmax(model.predict(X_test), axis = 1)
y_orig = np.argmax(y_test, axis = 1)
cm = confusion_matrix(preds, y_orig)
#keys = OrderedDict(sorted(genres.items(), key=lambda t: t[1])).keys()
plt.figure(figsize=(10,10))
plot_confusion_matrix(cm, classes, normalize=True)
model.save('my_h5_model.h5')
model = tf.keras.models.load_model('my_h5_model.h5')
def build_model(input_shape):
    """Generates CNN model
    :param input_shape (tuple): Shape of input set
    :return model: CNN model
    """

    # build network topology
    model = tf.keras.Sequential()

    # 1st conv layer
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
    model.add(tf.keras.layers.BatchNormalization())

    # 2nd conv layer
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
    model.add(tf.keras.layers.BatchNormalization())

    # 3rd conv layer
    model.add(tf.keras.layers.Conv2D(32, (2, 2), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same'))
    model.add(tf.keras.layers.BatchNormalization())

    # flatten output and feed it into dense layer
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.4))

    # output layer
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    return model
# create network
input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])
model = build_model(input_shape)

# compile model
optimiser = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=optimiser,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()


# train model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=32, epochs=30)

plt.plot(history.history['accuracy'], label='Accuracy (training data)')
plt.plot(history.history['val_accuracy'], label='Accuracy (validation data)')
plt.plot(history.history["loss"], label="training loss")
plt.plot(history.history["val_loss"], label="validation loss")

plt.ylabel('Accuracy')
plt.xlabel('No. epoch')
plt.legend(loc="upper left")
plt.show()
from sklearn.metrics import confusion_matrix

preds = np.argmax(model.predict(X_test), axis = 1)
y_orig = np.argmax(y_test, axis = 1)
cm = confusion_matrix(preds, y_orig)
plt.figure(figsize=(10,10))
plot_confusion_matrix(cm, classes, normalize=True)