import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn')
%matplotlib inline

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from keras.layers import Conv2D, Flatten, MaxPooling2D, BatchNormalization, Activation, Dense, Dropout
from keras.models import Sequential
from keras.utils import to_categorical
from keras.applications import InceptionV3
import keras.backend as K
train = pd.read_csv('../input/sign_mnist_train.csv')
print(train.shape)
train.head()
labels = train.pop('label')
labels = to_categorical(labels)
labels.shape
train = train.values
train = np.array([np.reshape(i, (28,28)) for i in train])
train = train / 255
train.shape
plt.imshow(train[0])
X_train, X_val, y_train, y_val = train_test_split(train, labels, test_size=0.2, random_state=23)
print(X_train.shape, X_val.shape, y_train.shape, y_val.shape)
X_train = X_train.reshape(X_train.shape[0], 28,28,1)
X_val = X_val.reshape(X_val.shape[0], 28,28,1)
print(X_train.shape, X_val.shape)
model = Sequential()
model.add(Conv2D(4, (5,5), input_shape=(28,28,1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=2))
model.add(Conv2D(12, (5,5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=2))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(25, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=40, batch_size=512, validation_data=(X_val, y_val))
model.save_weights('model1.h5')
f, ax = plt.subplots(1, 2, figsize=(20,10))
ax[0].set_title("Model Accuracy")
ax[0].plot(history.history['acc'])
ax[0].plot(history.history['val_acc'])
ax[0].set_xlabel("Epochs")
ax[0].set_ylabel("Accuracy")
ax[0].legend(['training', 'validation'])

ax[1].set_title("Model Loss")
ax[1].plot(history.history['loss'])
ax[1].plot(history.history['val_loss'])
ax[1].set_xlabel("Epochs")
ax[1].set_ylabel("Loss")
ax[1].legend(['training', 'validation'])
test = pd.read_csv("../input/sign_mnist_test.csv")
test.shape
# generate label data
y_test = test.pop('label')
y_test = to_categorical(y_test)
y_test.shape
# generate test data
X_test = test.values
X_test = np.array([np.reshape(i, (28,28)) for i in X_test])
X_test = X_test / 255
X_test = X_test.reshape(X_test.shape[0], 28,28,1)
X_test.shape
predictions = model.predict(X_test)
test_accuracy = accuracy_score(np.argmax(y_test, axis=1), np.argmax(predictions, axis=1))
test_accuracy
num_preview = 5
for i in range(num_preview):
    predicted_class = np.argmax(predictions[i])
    prob = predictions[i][predicted_class]
    print("Predicted: {} with the probability of: {}".format(chr(predicted_class+65), prob))
    print("Actual class: ", chr(np.argmax(y_test[i])+65))
    plt.imshow(X_test[i].reshape(28,28))
    plt.show()
