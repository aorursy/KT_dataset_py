# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import tensorflow as tf
import matplotlib.pyplot as plt
train_data = pd.read_csv("../input/digit-recognizer/train.csv")
test_data = pd.read_csv("../input/digit-recognizer/test.csv")

print(train_data.shape, test_data.shape)

X = np.array(train_data.drop("label", axis=1)).astype('float32')
y = np.array(train_data['label']).astype('float32')
plt.figure()
plt.imshow(X[0].reshape(28, 28))
plt.colorbar()
plt.grid(False)
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(X[i].reshape(28, 28), cmap=plt.cm.binary)
    plt.xlabel(y[i])
plt.show()
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

X = X / 255.0
X = X.reshape(-1, 28, 28, 1)
y = to_categorical(y)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2) #To create validation data = 20% of total training data

X_train.shape, X_val.shape
X_test = np.array(test_data).astype('float32')
X_test = X_test / 255.0
X_test = X_test.reshape(-1, 28, 28, 1)

X_test.shape
from tensorflow.keras import callbacks

lr_schedule = callbacks.LearningRateScheduler(lambda epoch: 1e-8 * 10 ** (epoch/10))
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.MaxPool2D(2, 2),
    
    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.MaxPool2D(2, 2),
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(512, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])
model.summary()
from tensorflow.keras.optimizers import Adam

model.compile(optimizer=Adam(learning_rate=1e-8), loss="categorical_crossentropy", metrics=["accuracy"])

history = model.fit(X_train, y_train, epochs=50, batch_size=525, validation_data=(X_val, y_val), callbacks=[lr_schedule])
accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.axis([10, 60, 0.98, 1])
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
lrs = 1e-8 * (10 ** (np.arange(50)/10))
plt.semilogx(lrs, loss)
plt.axis([1e-8, 1e-3, 0, 0.01])
model.compile(optimizer=Adam(learning_rate=1e-4), loss="categorical_crossentropy", metrics=["accuracy"])
history = model.fit(X_train, y_train, epochs=50, batch_size=525, validation_data=(X_val, y_val))
accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.axis([10, 60, 0.98, 1])
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
model.evaluate(X_val, y_val)
predictions = model.predict_classes(X_val)
y_true = np.argmax(y_val, axis=1)

correct = np.nonzero(predictions==y_true)[0]
incorrect = np.nonzero(predictions!=y_true)[0]

print(incorrect.shape)
plt.figure(figsize=(10, 10))
for i, incorrect in enumerate(incorrect[0:25]):
    plt.subplot(5,5,i+1)
    plt.imshow(X_val[incorrect].reshape(28,28), cmap='gray', interpolation='none')
    plt.title("Predicted {}, Class {}".format(predictions[incorrect], y_true[incorrect]))
    plt.tight_layout()
model.compile(optimizer=Adam(learning_rate=1e-4), loss="categorical_crossentropy", metrics=["accuracy"])
history = model.fit(X, y, epochs=50, batch_size=525)
final_predictions = model.predict_classes(X_test)

submit = pd.DataFrame(final_predictions,columns=["Label"])
submit["ImageId"] = pd.Series(range(1,(len(final_predictions)+1)))
submission = submit[["ImageId","Label"]]
submission.shape
submission.to_csv("submission.csv",index=False)
