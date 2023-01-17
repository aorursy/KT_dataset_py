import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from tensorflow.keras.datasets import cifar10
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
X_train = X_train/255
X_test = X_test/255
X_train.shape
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
fig = plt.figure(figsize=(8,8))
for i in range(25):
    plt.subplot(5,5, i+1)
    plt.tight_layout()
    plt.imshow(X_train[i], cmap = 'gray', interpolation=None)
    plt.xlabel(class_names[int(y_train[i])])
    plt.xticks([])
    plt.yticks([])
from keras.utils import to_categorical
y_cat_train = to_categorical(y_train,10)
y_cat_test = to_categorical(y_test,10)
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Activation
model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(4,4), input_shape = (32,32,3)))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(filters=32, kernel_size=(4,4), input_shape = (32,32,3)))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor='val_loss', patience=2)
model.fit(X_train, y_cat_train, epochs=50, validation_data=(X_test,y_cat_test), callbacks=[early_stop])
metrics = pd.DataFrame(model.history.history)
metrics
metrics[['accuracy', 'val_accuracy']].plot()
metrics[['loss', 'val_loss']].plot()
model.evaluate(X_test,y_cat_test,verbose=0)
from sklearn.metrics import classification_report, confusion_matrix
predictions = model.predict_classes(X_test)
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))
import seaborn as sns

plt.figure(figsize=(10,10))
sns.heatmap(confusion_matrix(y_test, predictions), annot=True)
fig = plt.figure(figsize=(8,8))
for i in range(25):
    plt.subplot(5,5, i+1)
    plt.tight_layout()
    plt.imshow(X_test[i], cmap = 'gray', interpolation=None)
    plt.xlabel('Prediction= {}\nTrue={}'.format(class_names[int(predictions[i])],class_names[int(y_test[i])]))
    plt.xticks([])
    plt.yticks([])