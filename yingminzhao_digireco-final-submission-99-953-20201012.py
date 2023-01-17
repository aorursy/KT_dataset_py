import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
import tensorflow
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from PIL import Image
from tensorflow import keras
from sklearn.utils import resample

import keras
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ReduceLROnPlateau
from keras.models import Model
from keras import datasets, layers, models
from keras.layers import Dense , Conv2D,MaxPooling2D,Flatten,Dropout 

# Load train data set
dfdg = pd.read_csv('train.csv')
print(dfdg.shape)

y = dfdg['label']
X = dfdg.drop('label',axis=1)
print(X.shape, y.shape)
# Split train and test data
train_images_1, test_images_1, train_labels_1, test_labels_1 = train_test_split(X, y, test_size=0.1, random_state=11)
print(train_images_1.shape, test_images_1.shape)
print(train_labels_1.shape, test_labels_1.shape)
train_images_1 = train_images_1.values
train_labels_1 = train_labels_1.values
test_images_1 = test_images_1.values
test_labels_1 = test_labels_1.values
# reshape & Normalization
train_images_1 = train_images_1.reshape((37800, 28, 28, 1))
test_images_1 = test_images_1.reshape((4200, 28, 28, 1))
train_images_1, test_images_1 = train_images_1 / 255.0, test_images_1 / 255.0 
# Load MINST data
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
# reshape and Normalization
train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))
train_images, test_images = train_images / 255.0, test_images / 255.0 
print(train_images.shape, test_images.shape)
print(train_labels.shape, test_labels.shape)
print(train_images_1.shape, test_images_1.shape)
print(train_labels_1.shape, test_labels_1.shape)
train_images = np.concatenate((train_images, train_images_1), axis=0)
test_images = np.concatenate((test_images, test_images_1), axis=0)
train_labels = np.concatenate((train_labels, train_labels_1), axis=0)
test_labels = np.concatenate((test_labels, test_labels_1), axis=0)
print(train_images.shape, test_images.shape)
print(train_labels.shape, test_labels.shape)
w1 = pd.read_csv('wrong1_normalized.csv')
w2 = pd.read_csv('wrong2_normalized_2.csv')
w3 = pd.read_csv('wrong3_normalized_2.csv')
w4 = pd.read_csv('wrong4_normalized.csv')
w5 = pd.read_csv('wrong5_normalized.csv')
w6 = pd.read_csv('wrong6_normalized.csv')
w7 = pd.read_csv('wrong7_normalized.csv')
w8 = pd.read_csv('wrong8_normalized.csv')
w1 = pd.concat([w1,w2])
w3 = pd.concat([w3,w4])
w5 = pd.concat([w5,w6])
w7 = pd.concat([w7,w8])
wrong = pd.concat([w1,w3])
wrong = pd.concat([wrong,w5])
wrong = pd.concat([wrong,w7])
# Up-Sampling
df_upsampling = pd.DataFrame()

# Upsampling of
df_upsampling = resample(wrong[wrong['label']==0], replace=True, n_samples=2000, random_state=11)
df_upsampling1 = resample(wrong[wrong['label']==1], replace=True, n_samples=2000, random_state=11)
df_upsampling = pd.concat([df_upsampling, df_upsampling1])
df_upsampling1 = resample(wrong[wrong['label']==2], replace=True, n_samples=2000, random_state=11)
df_upsampling = pd.concat([df_upsampling, df_upsampling1])
df_upsampling1 = resample(wrong[wrong['label']==3], replace=True, n_samples=2000, random_state=11)
df_upsampling = pd.concat([df_upsampling, df_upsampling1])
df_upsampling1 = resample(wrong[wrong['label']==4], replace=True, n_samples=2000, random_state=11)
df_upsampling = pd.concat([df_upsampling, df_upsampling1])
df_upsampling1 = resample(wrong[wrong['label']==5], replace=True, n_samples=2000, random_state=11)
df_upsampling = pd.concat([df_upsampling, df_upsampling1])
df_upsampling1 = resample(wrong[wrong['label']==6], replace=True, n_samples=2000, random_state=11)
df_upsampling = pd.concat([df_upsampling, df_upsampling1])
df_upsampling1 = resample(wrong[wrong['label']==7], replace=True, n_samples=2000, random_state=11)
df_upsampling = pd.concat([df_upsampling, df_upsampling1])
df_upsampling1 = resample(wrong[wrong['label']==8], replace=True, n_samples=2000, random_state=11)
df_upsampling = pd.concat([df_upsampling, df_upsampling1])
df_upsampling1 = resample(wrong[wrong['label']==9], replace=True, n_samples=2000, random_state=11)
df_upsampling = pd.concat([df_upsampling, df_upsampling1])

# Display all
df_upsampling['label'].value_counts()
yw = df_upsampling['label']
Xw = df_upsampling.drop(['label','pred','correct'],axis=1)
# Split train and test data
train_images_1, test_images_1, train_labels_1, test_labels_1 = train_test_split(Xw, yw, test_size=0.1, random_state=11)
print(train_images_1.shape, test_images_1.shape)
print(train_labels_1.shape, test_labels_1.shape)
train_images_1 = train_images_1.values
train_labels_1 = train_labels_1.values
test_images_1 = test_images_1.values
test_labels_1 = test_labels_1.values
# reshape
train_images_1 = train_images_1.reshape((18000, 28, 28, 1))
test_images_1 = test_images_1.reshape((2000, 28, 28, 1))
print(train_images_1.shape, test_images_1.shape)
print(train_labels_1.shape, test_labels_1.shape)
train_images = np.concatenate((train_images, train_images_1), axis=0)
test_images = np.concatenate((test_images, test_images_1), axis=0)
train_labels = np.concatenate((train_labels, train_labels_1), axis=0)
test_labels = np.concatenate((test_labels, test_labels_1), axis=0)
print(train_images.shape, test_images.shape)
print(train_labels.shape, test_labels.shape)
input_shape = (28, 28, 1)
model = models.Sequential()
model.add(Conv2D(64, kernel_size=(3, 3),activation='relu',input_shape=input_shape))  # Convolution Layer -1 
model.add(Conv2D(64, kernel_size=(3, 3),activation='relu'))                          # Convolution Layer -2 
model.add(MaxPool2D((2, 2)))                                                         # Pooling Layer -1
model.add(BatchNormalization())                                                      # Normalization 
model.add(Dropout(0.25))                                                             # Set 25% to 0 to prevent overfitting 

model.add(Conv2D(filters = 128, kernel_size = (3,3),activation ='relu'))             # Convolution Layer -3 
model.add(Conv2D(filters = 128, kernel_size = (3,3),activation ='relu'))             # Convolution Layer -4 
model.add(MaxPool2D(pool_size=(2,2)))                                                # Pooling Layer -2 
model.add(BatchNormalization())                                                      # Normalization
model.add(Conv2D(filters = 256, kernel_size = (3,3),activation ='relu'))             # Convolution Layer -5 
model.add(MaxPool2D(pool_size=(2,2)))                                                # Pooling Layer -3 
model.add(Dropout(0.25))                                                             # Set 25% to 0 to prevent overfitting

model.add(Flatten())                                                                 # Flattens input
model.add(BatchNormalization())                                                      # Normalization
model.add(Dense(512, activation = "relu"))                                           # Connection Layer -1 
model.add(Dropout(0.5))                                                              #  Set 50% to 0 to prevent overfitting
model.add(Dense(10, activation = "softmax"))                                         # Connection Layer -2 
# print a useful summary of the model
model.summary()
# Model Plot
keras.utils.plot_model(model, "final_model_with_shape_info.png", show_shapes=True)
# Compile all layers
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.RMSprop(),
              metrics=['accuracy'])

# Define Callback function for reducting learning rate when no more improvement of accuracy
learning_rate_reduction = ReduceLROnPlateau(monitor='accuracy', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.3, 
                                            min_lr=0.0001)

# Fit train&test data
epochs_range = 60
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_images, train_labels,batch_size=256,epochs=epochs_range,validation_data=(test_images, test_labels),callbacks=learning_rate_reduction )
# Evaluate the model on test data
test_loss, test_acc = model.evaluate(test_images, test_labels,verbose=2)
print("Test Accuracy: ", test_acc)
print("Test loss:", test_loss)
# Accuracy plot
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss=history.history['loss']
val_loss=history.history['val_loss']

e_range = range(epochs_range)

plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
plt.plot(e_range, acc, label='Training Accuracy')
plt.plot(e_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.ylim([0.98, 1])

plt.subplot(1, 2, 2)
plt.plot(e_range, loss, label='Training Loss')
plt.plot(e_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Loss')
plt.ylim([0.00, 0.05])
plt.show()
# Save trained Model to lib
model.save('trained_cnn_model_20201012_01_099919.model')
# Predict and generate Submission file
df_t = pd.read_csv('test.csv')
X_t = df_t.values
X_t = X_t.reshape((28000, 28, 28, 1))
X_t = X_t/255.0
predictions = model.predict_classes(X_t, verbose=0)
submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),
                         "Label": predictions})
submissions.to_csv("submission21_predictions_cnn.csv", index=False, header=True)
