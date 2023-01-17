import numpy as np 

import pandas as pd



import seaborn as sns

import matplotlib.pyplot as plt



from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation, Flatten, Convolution2D, MaxPooling2D

from keras.utils import np_utils

from keras.preprocessing.image import ImageDataGenerator

from keras.optimizers import Adam,RMSprop

from keras.layers.normalization import BatchNormalization

from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler



from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix
# uploading data

train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
train.head()
test.head()
y_train = train['label'].astype('int32')

X_train = (train.drop(['label'], axis = 1)).values.astype('float32')

X_test = test.values.astype('float32')



batch_size, img_rows, img_cols = 64, 28, 28

X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)

X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)



X_train.shape, X_test.shape
# ploting countplot for train data

plt.figure(figsize=(8,6))

plt.title('Countplot for train data')

sns.countplot(y_train)
# images 

plt.figure(figsize=(10,10))

for i in range(20):

    plt.subplot(4, 5, i+1)

    plt.imshow(X_train[i].reshape((28, 28)))

plt.show()
# normalize the data

X_train /= 255

X_test /= 255
# one-hot encoding for y_train

y_train = np_utils.to_categorical(y_train, 10)
# split train data

X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, 

                                                      test_size = 0.1, random_state = 12345)
input_shape = (img_rows, img_cols, 1)

# using early stopping

callback_es = EarlyStopping(monitor = 'val_accuracy', patience = 3)

def first_cnn_model_keras(optimizer):

    model = Sequential()

    # CNN layers

    model.add(Convolution2D(64, 5, 5, padding = 'same', kernel_initializer = 'he_uniform', 

                            input_shape = input_shape))

    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'same'))

    model.add(Convolution2D(128, 5, 5, padding = 'same', kernel_initializer = 'he_uniform'))

    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'same'))

    # Fully connected layers

    model.add(Flatten())

    model.add(Dense(1024))

    model.add(Activation('relu'))

    model.add(Dropout(0.5))

    model.add(Dense(10))

    model.add(Activation('softmax'))

    # compile model

    model.compile(optimizer, loss='categorical_crossentropy', metrics = ['accuracy'])

    return model
# compile and fit first model (Adam optimizer)

model1 = first_cnn_model_keras(Adam(learning_rate = 0.001, amsgrad = True))

h1 = model1.fit(X_train, y_train, batch_size = batch_size, epochs = 20, verbose = 1,

          validation_data = (X_valid, y_valid), callbacks = [callback_es])

final_loss_first_adam, final_acc_first_adam = model1.evaluate(X_valid, y_valid, verbose=0)

print("Final loss: {0:.4f}, final accuracy: {1:.4f}".format(final_loss_first_adam, final_acc_first_adam))
# compile and fit first model (RMSprop optimizer)

model2 = first_cnn_model_keras(RMSprop(lr=0.001))

h2 = model2.fit(X_train, y_train, batch_size = batch_size, epochs = 20, verbose = 1,

          validation_data = (X_valid, y_valid),callbacks = [callback_es])

final_loss_first_rmsprop, final_acc_first_rmsprop = model2.evaluate(X_valid, y_valid, verbose=0)

print("Final loss: {0:.4f}, final accuracy: {1:.4f}".format(final_loss_first_rmsprop, final_acc_first_rmsprop))
# data augmentation

datagen = ImageDataGenerator(rotation_range = 10, 

                             zoom_range = 0.1, 

                             width_shift_range = 0.1,

                             height_shift_range = 0.1)

datagen.fit(X_train)

train_batches = datagen.flow(X_train, y_train, batch_size = batch_size)
# compile and fit first model (Adam optimizer and data augmentation)

model3 = first_cnn_model_keras(Adam(learning_rate = 0.001, amsgrad = True))

h3 = model3.fit_generator(train_batches, epochs = 40, verbose = 1,

          validation_data = (X_valid, y_valid), callbacks = [callback_es])

final_loss_first_adam_aug, final_acc_first_adam_aug = model3.evaluate(X_valid, y_valid, verbose=0)

print("Final loss: {0:.4f}, final accuracy: {1:.4f}".format(final_loss_first_adam_aug, final_acc_first_adam_aug))
# compile and fit first model (RMSprop optimizer and data augmentation)

model4 = first_cnn_model_keras(RMSprop(lr=0.001))

h4 = model4.fit_generator(train_batches, epochs = 40, verbose = 1,

          validation_data = (X_valid, y_valid), callbacks = [callback_es])

final_loss_first_rmsprop_aug, final_acc_first_rmsprop_aug = model4.evaluate(X_valid, y_valid, verbose=0)

print("Final loss: {0:.4f}, final accuracy: {1:.4f}".format(final_loss_first_rmsprop_aug, final_acc_first_rmsprop_aug))
# second model

def second_cnn_model_keras(optimizer):

    model = Sequential()

    # CNN layers

    model.add(Convolution2D(64, kernel_size = (5, 5), input_shape = input_shape, kernel_initializer = 'he_uniform'))

    model.add(Activation('relu'))

    model.add(Convolution2D(64, kernel_size = (5, 5), kernel_initializer = 'he_uniform'))

    model.add(Activation('relu'))

    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'same'))

    model.add(Dropout(0.25))

    model.add(Convolution2D(128, kernel_size = (3, 3), kernel_initializer = 'he_uniform'))

    model.add(Activation('relu'))

    model.add(BatchNormalization())

    model.add(Convolution2D(128, kernel_size = (3, 3), kernel_initializer = 'he_uniform'))

    model.add(Activation('relu'))

    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'same'))

    model.add(Dropout(0.25))

    # Fully connected layers

    model.add(Flatten())

    model.add(Dense(256))

    model.add(Activation('relu'))

    model.add(BatchNormalization())

    model.add(Dropout(0.25))

    model.add(Dense(10))

    model.add(Activation('softmax'))

    # compile model

    model.compile(loss = 'categorical_crossentropy', optimizer = optimizer, metrics=['accuracy'])

    return model
# compile and fit second model (Adam optimizer)

model5 = second_cnn_model_keras(Adam(learning_rate = 0.001, amsgrad = True))

h5 = model5.fit(X_train, y_train, batch_size = batch_size, epochs = 20, verbose = 1,

          validation_data = (X_valid, y_valid), callbacks = [callback_es])

final_loss_second_adam, final_acc_second_adam = model5.evaluate(X_valid, y_valid, verbose=0)

print("Final loss: {0:.4f}, final accuracy: {1:.4f}".format(final_loss_second_adam, final_acc_second_adam))
# compile and fit second model (RMSprop optimizer)

model6 = second_cnn_model_keras(RMSprop(lr=0.001))

h6 = model6.fit(X_train, y_train, batch_size = batch_size, epochs = 20, verbose = 1,

          validation_data = (X_valid, y_valid), callbacks = [callback_es])

final_loss_second_rmsprop, final_acc_second_rmsprop = model6.evaluate(X_valid, y_valid, verbose=0)

print("Final loss: {0:.4f}, final accuracy: {1:.4f}".format(final_loss_second_rmsprop, final_acc_second_rmsprop))
# compile and fit second model (Adam optimizer and data augmentation)

model7 = second_cnn_model_keras(Adam(learning_rate = 0.001, amsgrad = True))

h7 = model7.fit_generator(train_batches, epochs = 20, verbose = 1,

          validation_data = (X_valid, y_valid),callbacks = [callback_es])

final_loss_second_adam_aug, final_acc_second_adam_aug = model7.evaluate(X_valid, y_valid, verbose=0)

print("Final loss: {0:.4f}, final accuracy: {1:.4f}".format(final_loss_second_adam_aug, final_acc_second_adam_aug))
# compile and fit second model (RMSprop optimizer and data augmentation)

model8 = second_cnn_model_keras(RMSprop(lr=0.001))

h8 = model8.fit_generator(train_batches, epochs = 20, verbose = 1,

          validation_data = (X_valid, y_valid), callbacks = [callback_es])

final_loss_second_rmsprop_aug, final_acc_second_rmsprop_aug = model8.evaluate(X_valid, y_valid, verbose=0)

print("Final loss: {0:.4f}, final accuracy: {1:.4f}".format(final_loss_second_rmsprop_aug, final_acc_second_rmsprop_aug))
models = ['first_cnn_adam', 'first_cnn_rmsprop', 'first_cnn_adam_aug', 'first_cnn_rmsprop_aug', 

          'second_cnn_adam', 'second_cnn_rmsprop', 'second_cnn_adam_aug', 'second_cnn_rmsprop_aug']

dict_values = {'loss': [final_loss_first_adam, final_loss_first_rmsprop, final_loss_first_adam_aug, 

                   final_loss_first_rmsprop_aug, final_loss_second_adam, final_loss_second_rmsprop,

                   final_loss_second_adam_aug, final_loss_second_rmsprop_aug],

           'accuracy': [final_acc_first_adam, final_acc_first_rmsprop, final_acc_first_adam_aug, 

                   final_acc_first_rmsprop_aug, final_acc_second_adam, final_acc_second_rmsprop,

                   final_acc_second_adam_aug, final_acc_second_rmsprop_aug]}



df = pd.DataFrame(dict_values, index = models, columns = ['loss', 'accuracy'])

df
accuracy = h5.history['accuracy']

val_accuracy = h5.history['val_accuracy']

loss = h5.history['loss']

val_loss = h5.history['val_loss']

epochs = range(len(accuracy))



f, ax = plt.subplots(1, 2, figsize=(18, 8))

ax[0].plot(epochs, accuracy, 'r--', label='Training accuracy')

ax[0].plot(epochs, val_accuracy, 'b', label='Validation accuracy')

ax[0].set_title('Train and validation accuracy')

ax[0].set_xlabel('epoch')

ax[0].set_ylabel('accuracy')

ax[0].grid()

ax[1].plot(epochs, loss, 'r--', label='Training loss')

ax[1].plot(epochs, val_loss, 'b', label='Validation loss')

ax[1].set_title('Train and validation loss')

ax[1].set_xlabel('epoch')

ax[1].set_ylabel('loss')

ax[1].grid()

plt.show()
# predict the values from validation data

y_predict = model5.predict(X_valid)

# convert predict and validation data to one-hot vectors

y_predict_class = np.argmax(y_predict, axis = 1)

y_true = np.argmax(y_valid, axis = 1)



plt.subplots(figsize = (12, 10))

sns.heatmap(confusion_matrix(y_true, y_predict_class), annot=True, 

            linewidths = 0.5, fmt = '.0f', cmap = 'Reds', linecolor = 'black')

plt.xlabel('Predicted Label')

plt.ylabel("True Label")

plt.title('Confusion Matrix')

plt.show()
# create ensemble models

model = [0]*10

for i in range(10):  

    model[i] = Sequential()

    # CNN layers

    model[i].add(Convolution2D(64, kernel_size = (3, 3), input_shape = input_shape, kernel_initializer = 'he_uniform'))

    model[i].add(Activation('relu'))

    model[i].add(BatchNormalization())

    model[i].add(Convolution2D(64, kernel_size = (3, 3), kernel_initializer = 'he_uniform'))

    model[i].add(Activation('relu'))

    model[i].add(BatchNormalization())

    model[i].add(Convolution2D(64, kernel_size = (5, 5), kernel_initializer = 'he_uniform'))

    model[i].add(Activation('relu'))

    model[i].add(BatchNormalization())

    model[i].add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))

    model[i].add(Dropout(0.45))

    model[i].add(Convolution2D(128, kernel_size = (3, 3), kernel_initializer = 'he_uniform'))

    model[i].add(Activation('relu'))

    model[i].add(BatchNormalization())

    model[i].add(Convolution2D(128, kernel_size = (3, 3), kernel_initializer = 'he_uniform'))

    model[i].add(Activation('relu'))

    model[i].add(BatchNormalization())

    model[i].add(Convolution2D(128, kernel_size = (5, 5), kernel_initializer = 'he_uniform'))

    model[i].add(Activation('relu'))

    model[i].add(BatchNormalization())

    model[i].add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))

    model[i].add(Dropout(0.45))

    # Fully connected layers

    model[i].add(Flatten())

    model[i].add(Dense(512))

    model[i].add(Activation('relu'))

    model[i].add(Dropout(0.45))

    model[i].add(Dense(1024))

    model[i].add(Activation('relu'))

    model[i].add(Dropout(0.45))

    model[i].add(Dense(10))

    model[i].add(Activation('softmax'))

    # compile model

    model[i].compile(loss = 'categorical_crossentropy', optimizer = Adam(lr = 0.0005, amsgrad = True), metrics=['accuracy'])
# edit early stopping and learning models

callback_lrs = LearningRateScheduler(lambda x: 1e-3 * 0.95 ** x)

epochs = 40

history = [0]*10

for j in range(10):

    X_train1, X_valid1, y_train1, y_valid1 = train_test_split(X_train, y_train, test_size = 0.1)

    history[j] = model[j].fit_generator(datagen.flow(X_train1, y_train1, batch_size = batch_size), epochs = epochs, verbose = 0,

          validation_data = (X_valid1, y_valid1), callbacks = [callback_lrs])

    print('CNN:', j+1, 'Epochs =', epochs, 'Train accuracy:', max(history[j].history['accuracy']), 'Validation accuracy:', max(history[j].history['val_accuracy']))
# predict the values from validation data

results_valid = np.zeros((X_valid.shape[0],10)) 

# convert predict and validation data to one-hot vectors

for j in range(10):

    results_valid = results_valid + model[j].predict(X_valid)

y_valid_class = np.argmax(results_valid, axis = 1)

y_true = np.argmax(y_valid, axis = 1)



plt.subplots(figsize = (12, 10))

sns.heatmap(confusion_matrix(y_true, y_valid_class), annot=True, 

            linewidths = 0.5, fmt = '.0f', cmap = 'Reds', linecolor = 'black')

plt.xlabel('Predicted Label')

plt.ylabel("True Label")

plt.title('Confusion Matrix')

plt.show()
# create zero matrix

results = np.zeros((X_test.shape[0],10)) 

# predict the values for test dataset

for j in range(10):

    results = results + model[j].predict(X_test)

# convert predict to one-hot vectors

y_test_class = np.argmax(results, axis = 1)

# create and save predict dataframe

submission = pd.DataFrame({'ImageId': list(range(1, len(y_test_class)+1)), 'Label': np.array(y_test_class)})

submission.to_csv('submission.csv', index=False)

print(submission)