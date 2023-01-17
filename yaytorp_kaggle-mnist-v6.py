import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPool2D, Dropout, BatchNormalization
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
train = pd.read_csv('../input/digit-recognizer/train.csv')
test = pd.read_csv('../input/digit-recognizer/test.csv')
train.head()
y_train = train['label']
X_train = train.drop(['label'], axis=1)
X_train.isnull().any().describe()
test.isnull().any().describe()
X_train.shape
X_train = X_train/255.
test = test/255.
X_train.shape
X_train = X_train.values.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)
y_train.head()
y_train = to_categorical(y_train, num_classes = 10)  #num_classes is basically the number of variables we have in our y_train dataframe. This case we have numbers from 0 to 9.
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=48) #random_state = 48 is just any random seed you can set.
plt.imshow(X_train[65][:,:,0])
model = tf.keras.models.Sequential()

model.add(Conv2D(filters=32, kernel_size=(3,3), padding='Same', activation='relu', input_shape = (28,28,1)))
model.add(BatchNormalization())

model.add(Conv2D(filters=32, kernel_size=(3,3), padding='Same', activation='relu'))
model.add(BatchNormalization())

model.add(Conv2D(filters=32, kernel_size=(5,5), padding='Same', activation='relu', strides=2))
model.add(BatchNormalization())
model.add(Dropout(rate=0.4))

model.add(Conv2D(filters=64, kernel_size=(3,3), padding='Same', activation='relu'))
model.add(BatchNormalization())

model.add(Conv2D(filters=64, kernel_size=(3,3), padding='Same', activation='relu'))
model.add(BatchNormalization())

model.add(Conv2D(filters=64, kernel_size=(5,5), padding='Same', activation='relu', strides=2))
model.add(BatchNormalization())
model.add(Dropout(rate=0.4))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(rate=0.4))

model.add(Dense(10, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
imageDataGen = ImageDataGenerator(width_shift_range=0.1, # randomly shift images horizontally
                                  height_shift_range=0.1, # randomly shift images vertically
                                  rotation_range=10,   # randomly rotate images in the range provided.
                                  zoom_range=0.1) # Randomly zoom image 

imageDataGen.fit(X_train)
reduce_lr = LearningRateScheduler(lambda x: 1e-3 * 0.95 ** x, verbose=0)
history = model.fit_generator(imageDataGen.flow(X_train, y_train, batch_size=64), 
                    verbose = 2,
                    validation_data=(X_val, y_val), 
                    epochs=50, 
                    steps_per_epoch=X_train.shape[0] // 64,
                    callbacks= [reduce_lr])
from sklearn.metrics import confusion_matrix
# Predict the values from the validation dataset
y_pred = model.predict(X_val)
# Convert predictions classes to one hot vectors 
y_pred_classes = np.argmax(y_pred,axis = 1) 
# Convert validation observations to one hot vectors
y_true = np.argmax(y_val,axis = 1) 
# compute the confusion matrix
confusion_mtx = confusion_matrix(y_true, y_pred_classes) 
# plot the confusion matrix
f,ax = plt.subplots(figsize=(8, 8))
sns.heatmap(confusion_mtx, annot=True, linewidths=0.01,cmap="Blues",linecolor="gray", fmt= '.1f',ax=ax)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()
pred = model.predict(test)
pred = np.argmax(pred, axis=1)
pred = pd.Series(pred, name="Label")
final_res = pd.concat([pd.Series(range(1,28001),name = "ImageId"),pred],axis = 1)
final_res.to_csv("MNIST_try_v8.csv",index=False)
