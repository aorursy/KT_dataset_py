import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

import matplotlib.pyplot as plt

import os



from sklearn.model_selection import train_test_split

from tensorflow.keras.optimizers import RMSprop,Adam

from keras.utils.np_utils import to_categorical

# from keras.models import Sequential

# from keras.optimizers import Adam, RMSprop



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

train_data = pd.read_csv("../input/digit-recognizer/train.csv")

test_data = pd.read_csv("../input/digit-recognizer/test.csv")
train_data.head()
train_data.shape
y_train = train_data['label']



x_train = train_data.drop('label', axis=1 )



x_train.shape
x_train.head()
pd.unique(train_data.label)
y_train.value_counts()
# 784 = 28*28 pixel

x_train = x_train.values.reshape((x_train.shape[0],28,28))



print('training data shape : ',x_train.shape)



x_test = test_data.values.reshape((test_data.shape[0],28,28))



print('test data shape : ',x_test.shape)


for index in range(0,8):

    plt.subplot(2, 4, index+1)

    plt.axis('off')

#     plt.imshow(x_train[index], cmap=plt.cm.gray_r, interpolation='nearest')

    plt.imshow(x_train[index], cmap = plt.cm.binary, interpolation='nearest')

    plt.title('Image: %i' % index)
for index in range(0,8):

    plt.subplot(2, 4, index+1)

#     plt.axis('off')

    plt.imshow(x_test[index], cmap = plt.cm.binary, interpolation='nearest')

    plt.title('Image: %i' % index)
x_train[1]
X_train = tf.keras.utils.normalize(x_train, axis=1)

X_test = tf.keras.utils.normalize(x_test, axis=1)



print("Training set images visulaization post normalisation")



for index in range(0,8):

    plt.subplot(2, 4, index+1)

    plt.axis('off')

    plt.imshow(X_train[index], cmap = plt.cm.binary, interpolation='nearest')

    plt.title('Image: %i' % index)

print("Test set images visulaization post normalisation")



for index in range(0,8):

    plt.subplot(2, 4, index+1)

    plt.axis('off')

    plt.imshow(X_test[index], cmap = plt.cm.binary, interpolation='nearest')

    plt.title('Image: %i' % index)
X_train_2, X_val_1, Y_train_2, Y_val_1 = train_test_split(X_train, y_train, test_size = 0.1, random_state= 20)
print(X_train_2.shape)

print(Y_train_2.shape)
Y_train_2 = np.ravel(Y_train_2)
model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Flatten()) # Input layer

model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu)) # Hidden layer 1

model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu)) # Hidden layer 2

model.add(tf.keras.layers.Dense(10, activation = tf.nn.softmax)) # Output layer


optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, decay=0.0, amsgrad=False)





model.compile(optimizer = optimizer , loss = "sparse_categorical_crossentropy", metrics=["accuracy"])




history = model.fit(X_train_2,Y_train_2,

                    batch_size=32,

                    epochs=200,

                    verbose = 1,

                    # We pass some validation for

                    # monitoring validation loss and metrics

                    # at the end of each epoch

                    validation_data=(X_val_1,Y_val_1))



# model.fit(X_train_2,Y_train_2,epochs = 100, batch_size = 10)
# X_train_2, X_val_1, Y_train_2, Y_val_1

val_loss, val_acc = model.evaluate(X_val_1,Y_val_1)

print(val_loss, val_acc)
loss = history.history['loss']

epochs = range(len(loss))

plt.plot(epochs, loss, 'b', label='Training Loss')

plt.show()
# Plot all but the first 10

loss = history.history['loss']

epochs = range(10, len(loss))

plot_loss = loss[10:]

# print(plot_loss)

plt.plot(epochs, plot_loss, 'b', label='Training Loss')

plt.show()
# Further zoom into the loss plot

# Plot from 70 epoch...

loss = history.history['loss']

epochs = range(70, len(loss))

plot_loss = loss[70:]

# print(plot_loss)

plt.plot(epochs, plot_loss, 'b', label='Training Loss')

plt.show()
# X_train_2, X_val_1, Y_train_2, Y_val_1

lr_schedule = tf.keras.callbacks.LearningRateScheduler(

    lambda epoch: 1e-8 * 10**(epoch / 20))



optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, decay=0.0, amsgrad=False)

model.compile(optimizer = optimizer , loss = "sparse_categorical_crossentropy", metrics=["accuracy"])

history_new = model.fit(X_train_2,Y_train_2,

                    batch_size=32,

                    epochs=100,

                    verbose = 1,

                    callbacks=[lr_schedule],

                    validation_data=(X_val_1,Y_val_1))

lrs = 1e-8 * (10 ** (np.arange(100) / 20))

plt.semilogx(lrs, history_new.history["loss"])

plt.axis([1e-8, 1e-3, 0, 300])

plt.ylim(0,1e-09)
print(history_new.history["val_loss"])
lrs = 1e-8 * (10 ** (np.arange(100) / 20))

plt.semilogx(lrs, history_new.history["val_loss"])

plt.axis([1e-8, 1e-3, 0, 300])

plt.ylim(0.3,0.5)
# X_train_2, X_val_1, Y_train_2, Y_val_1



optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, decay=0.0, amsgrad=False)





model.compile(optimizer = optimizer , loss = "sparse_categorical_crossentropy", metrics=["accuracy"])



history_final = model.fit(X_train_2,Y_train_2,

                    batch_size=32,

                    epochs=100,

                    verbose = 1,

                    # We pass some validation for

                    # monitoring validation loss and metrics

                    # at the end of each epoch

                    validation_data=(X_val_1,Y_val_1))
val_loss, val_acc = model.evaluate(X_val_1,Y_val_1)

print(val_loss, val_acc)
#X_train_2, X_val_1, Y_train_2, Y_val_1 = train_test_split(X_train, y_train)



optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, decay=0.0, amsgrad=False)

model.compile(optimizer = optimizer , loss = "sparse_categorical_crossentropy", metrics=["accuracy"])



history_sub = model.fit(X_train, y_train,

                    batch_size=32,

                    epochs=100,

                    verbose = 1)
predictions = model.predict([X_test], verbose = 0)   #predict always takes a list

print(predictions)
print(np.argmax(predictions[2]))

plt.imshow(X_test[2], cmap = plt.cm.binary)

plt.show()
print(np.argmax(predictions[4]))

plt.imshow(X_test[4], cmap = plt.cm.binary)

plt.show()
submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),

                         "Label": np.argmax(predictions,axis = 1)})

submissions.head()
submissions.to_csv("DNN_digit_Recognition_submission.csv", index=False, header=True)