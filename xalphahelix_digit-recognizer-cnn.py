import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math
import matplotlib.pyplot as plt
%matplotlib inline

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
from keras.optimizers import Adam, RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
print('Train shape: {}'.format(train.shape))
print('Test shape: {}'.format(test.shape)) # Test data does not contain the "label" column
train.head()
Y_train = train['label']
X_train = train.drop(labels=['label'], axis=1)
plt.hist(Y_train)
plt.title('Frequency Histogram of Digits')
plt.xlabel('Digit')
plt.ylabel('Frequency')
fig, ax = plt.subplots(3, 3, figsize=(10, 6))
# Plot the first 9 digits in the training set
for i in range(9):
    data = X_train.iloc[i].values
    n = math.ceil((i+1)/3) - 1
    m = [0, 1, 2] * 3
    ax[m[i], n].imshow(data.reshape(28, 28), cmap='gray')
# Normalize the data
X_train = X_train.astype('float32') / 255
test = test.astype('float32') / 255
# One-hot-encoding
Y_train = to_categorical(Y_train, num_classes=10)
# Reshape image in 3 dimensions
# Keras requires an extra dimension, which is a channel
# Gray-scaled images use only 1 channel, while RGB images use 3 channels
X_train = X_train.values.reshape(-1, 28, 28, 1)
test = test.values.reshape(-1, 28, 28, 1)
# Split the data into a training and a validation set to evaluate model performance
seed = 42
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.1, 
                                                  random_state=seed)
model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', 
                 input_shape=(28, 28, 1)))
model.add(BatchNormalization())
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(strides=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters=64, kernel_size=(5, 5), activation='relu', 
                 input_shape=(28, 28, 1)))
model.add(BatchNormalization())
model.add(Conv2D(filters=64, kernel_size=(5, 5), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(strides=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
batch_size = 32
epochs = 20
optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
#optimizer = Adam()
model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
annealer = ReduceLROnPlateau(monitor='val_acc', patience=3, factor=0.5, min_lr=1e-5)
# Data augmentation
# Generates more training data by applying small transformations to images
datagen = ImageDataGenerator(featurewise_center=False,
                            samplewise_center=False,
                            featurewise_std_normalization=False,
                            samplewise_std_normalization=False,
                            rotation_range=10,
                            width_shift_range=0.1,
                            height_shift_range=0.1,
                            zoom_range=0.1)
# Compute quantities required for featurewise normalization
datagen.fit(X_train)
# Fits the model on batches with real-time data augmentation
history = model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size),
                    epochs=epochs, 
                    verbose = 2,
                    callbacks=[annealer],
                    validation_data = (X_val, Y_val),
                    steps_per_epoch=X_train.shape[0] // batch_size)
# Check performance on entire validation set
final_loss, final_accuracy = model.evaluate(X_val, Y_val)
print('Final Loss: {:.4f}, Final Accuracy: {:.4f}'.format(final_loss, final_accuracy))
plt.plot(history.history['loss'], color='b')
plt.plot(history.history['val_loss'], color='r')
plt.title('Training vs. Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['Training Loss', 'Validation Loss'])
plt.show()
plt.plot(history.history['acc'], color='b')
plt.plot(history.history['val_acc'], color='r')
plt.title('Training vs. Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(['Training Accuracy', 'Validation Accuracy'])
plt.show()
Y_pred = model.predict(X_val)
# Convert predictions to one-hot vectors
Y_pred = np.argmax(Y_pred, axis=1)
Y_true = np.argmax(Y_val, axis=1)
cm = confusion_matrix(Y_true, Y_pred)
print(cm)
predictions = model.predict(test)
predictions = np.argmax(predictions, axis=1)
results = [[i, predictions[i-1]] for i in range(1, 28001)]
submission = pd.DataFrame(results, columns=['ImageId', 'Label'])
submission.to_csv('submission.csv', index=False)