import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sklearn
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import keras
from keras.callbacks import EarlyStopping
from keras.layers import *
from keras.optimizers import *
from keras.preprocessing.image import ImageDataGenerator
from keras.models import *

N_CLASSES = 10

test_df = pd.read_csv("../input/fashion-mnist_test.csv")
train_df = pd.read_csv("../input/fashion-mnist_train.csv")

Y_test = test_df['label'].values
X_test = test_df.drop(['label'], axis=1).values.reshape((-1, 28, 28, 1))/255
Y_train = train_df['label'].values
X_train = train_df.drop(['label'], axis=1).values.reshape((-1, 28, 28, 1))/255
model = Sequential()
model.add(InputLayer((28,28,1)))
for n, k in [(64,3),(64,3),(128,3)]:
    model.add(Conv2D(n, kernel_size=k, strides=1, padding='same'))
    model.add(LeakyReLU())
    model.add(BatchNormalization())
    model.add(Conv2D(n, kernel_size=k, strides=1, padding='same'))
    model.add(LeakyReLU())
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.1))
model.add(Flatten())
model.add(Dense(N_CLASSES))
model.add(Activation('softmax'))
model.summary()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
train_gen = ImageDataGenerator(rotation_range=10.0,
                                       width_shift_range=2, 
                                       height_shift_range=2,
                                       shear_range=0.1, 
                                       zoom_range=0.1, 
                                       data_format='channels_last',
                                       validation_split=0.0)

epochs=30
batch_size=30
history = model.fit_generator(train_gen.flow(X_train, Y_train, batch_size=batch_size),
                    epochs=epochs,
                    steps_per_epoch=len(X_train)//batch_size,
                    validation_data=(X_test, Y_test),
                    callbacks=[EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=0, mode='auto')],
                    verbose=2)
plt.subplot(1,2,2)
plt.plot(history.history['loss'], 'r--')
plt.plot(history.history['val_loss'], 'b')
plt.subplot(1,2,2)
plt.plot(history.history['acc'], 'r--')
plt.plot(history.history['val_acc'], 'b');
predicted = np.argmax(model.predict(X_test), axis=1)
print(classification_report(Y_test, predicted))
sns.heatmap(confusion_matrix(Y_test, predicted));