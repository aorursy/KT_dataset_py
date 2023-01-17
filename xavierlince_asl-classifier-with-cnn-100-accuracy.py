from IPython.display import Image
Image(filename='../input/sign-language-mnist/amer_sign2.png')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
df_tr = pd.read_csv('../input/sign-language-mnist/sign_mnist_train.csv')
df_tr.head()
# load test set
df_te = pd.read_csv('../input/sign-language-mnist/sign_mnist_test.csv')

# Create train/test

X_test = df_te.iloc[:,1:].values
y_test = df_te[['label']].values

print('X_tr shape', X_test.shape, X_test.dtype)
print('y_te shape', y_test.shape, y_test.dtype)

X_test = (X_test - 128)/255
def show_img(img, df):
    
    # Take the label
    label = df['label'][img]
    
    # Take the pixels
    pixels = df.iloc[img, 1:]

    # The pixel intensity values are integers from 0 to 255
    pixels = np.array(pixels, dtype='uint8')

    # Reshape the array into 28 x 28 array (2-dimensional array)
    pixels = pixels.reshape((28, 28))

    # Plot
    plt.title('Label is {label}'.format(label=label))
    plt.imshow(pixels, cmap='gray')
    plt.show()

show_img(90, df_tr)
list_data = [df_tr, df_te]
fig,axes = plt.subplots(nrows=1,ncols=2,figsize=(15,6))

for data, ax, names in zip(list_data, axes.ravel(), ['train', 'test']):
    sns.countplot(data['label'], palette='rocket', ax=ax)
    ax.set_title("Frequency for each letter in the {} dataset".format(names))
    ax.set_xlabel('Letters')
    ax.set_ylabel('Frequency')
    ax.set_xticklabels(['A','B','C','D','E','F','G','H','I','K','L','M','N','O','P','Q','R','S',
                            'T','U','V','W','X','Y'])

plt.tight_layout()
# Create train/test

X = df_tr.iloc[:,1:].values
y = df_tr[['label']].values

from sklearn.model_selection import train_test_split

X_tr, X_v, y_tr, y_v = train_test_split(X, y, test_size=0.2, random_state=14)

print('X_tr shape', X_tr.shape, X_tr.dtype)
print('X_v shape', X_v.shape, X_v.dtype)
print('y_tr shape', y_tr.shape, y_tr.dtype)
print('y_v shape', y_v.shape, y_v.dtype)

X_tr = (X_tr - 128)/255
X_v = (X_v - 128)/255
# X_tr and y_tr to right shape for CNN

train_x = X_tr.reshape(-1,28,28,1) 
train_y = y_tr.reshape(-1,1) 

# val_x and val_y
val_x = X_v.reshape(-1,28,28,1)
val_y = y_v.reshape(-1,1)

X_test = X_test.reshape(-1, 28, 28, 1)

print(train_x.shape)
print(val_x.shape)
from sklearn.preprocessing import LabelBinarizer

lb=LabelBinarizer()
y_tr= lb.fit_transform(y_tr)
y_v= lb.fit_transform(y_v)
y_test= lb.fit_transform(y_test)
# With data augmentation to prevent overfitting
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
        featurewise_center=False,  
        samplewise_center=False,  
        featurewise_std_normalization=False,  
        samplewise_std_normalization=False,  
        zca_whitening=False,  
        rotation_range=10,  
        zoom_range = 0.1, 
        width_shift_range=0.1, 
        height_shift_range=0.1,  
        horizontal_flip=False, 
        vertical_flip=False)  


datagen.fit(train_x)
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout , BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# Convolutional network 
model = keras.Sequential()

model.add(keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, activation='relu', input_shape=(28, 28, 1)))
model.add(keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, activation='relu'))

model.add(keras.layers.Conv2D(filters=64, kernel_size=2, strides=1, activation='relu'))
model.add(keras.layers.Conv2D(filters=64, kernel_size=2, strides=1, activation='relu'))
model.add(keras.layers.MaxPool2D(pool_size=2))

model.add(keras.layers.Conv2D(filters=128, kernel_size=2, strides=1, activation='relu'))
model.add(keras.layers.Conv2D(filters=128, kernel_size=2, strides=1, activation='relu'))
model.add(keras.layers.MaxPool2D(pool_size=2))
model.add(Dropout(0.2))

model.add(keras.layers.Conv2D(filters=256, kernel_size=2, strides=1, activation='relu'))
model.add(keras.layers.Conv2D(filters=256, kernel_size=2, strides=1, activation='relu'))
model.add(keras.layers.MaxPool2D(pool_size=2))

model.add(keras.layers.Flatten())

model.add(keras.layers.Dense(512,activation='relu'))
model.add(Dropout(0.25))

model.add(keras.layers.Dense(256,activation='relu'))

model.add(keras.layers.Dense(24, activation='softmax'))

print(model.summary())

# Compile the model
model.compile(optimizer=keras.optimizers.Adam(), loss='categorical_crossentropy', metrics=['acc'])


early_stopping = [EarlyStopping(patience=3, monitor='val_loss'), ReduceLROnPlateau(patience=2), 
                  ModelCheckpoint(filepath='ASL_MNIST_CNN_temp.h5', save_best_only=True)]
history = model.fit(datagen.flow(train_x, y_tr, batch_size = 250), epochs = 100, validation_data = (val_x, y_v) , callbacks = early_stopping)

model.save('ASL_MNIST_CNN.h5') # Saves architecture and weights
print('Model Saved')
test_loss, test_acurracy = model.evaluate(X_test, y_test)
print('Test loss: {:.2f}, accuracy: {:.2f}%'.format(test_loss, test_acurracy*100))
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np

# Create two plots: one for the loss value, one for the accuracy
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

# Plot accuracy values
ax1.plot(history.history['loss'], label='train loss')
ax1.plot(history.history['val_loss'], label='val loss')
ax1.set_title('Validation loss {:.3f} (mean last 3)'.format(
    np.mean(history.history['val_loss'][-3:]) # last three values
))
ax1.set_xlabel('epoch')
ax1.set_ylabel('loss value')
ax1.legend()

# Plot accuracy values
ax2.plot(history.history['acc'], label='train acc')
ax2.plot(history.history['val_acc'], label='val acc')
ax2.set_title('Validation accuracy {:.3f} (mean last 3)'.format(
    np.mean(history.history['val_acc'][-3:]) # last three values
))
ax2.set_xlabel('epoch')
ax2.set_ylabel('accuracy')
ax2.legend()
plt.show()
predictions = model.predict_classes(X_test)
predictions
# Remove one hot encoding from the target
y_test_=np.argmax(y_test, axis=1)
y_test_[1]
from sklearn.metrics import classification_report

print(classification_report(y_test_, predictions, target_names = ['A','B','C','D','E','F','G','H','I','K','L','M','N','O','P','Q','R','S',
                        'T','U','V','W','X','Y']))
from sklearn.metrics import confusion_matrix
import seaborn as sns

matrix = confusion_matrix(y_true=y_test_, y_pred=predictions)

plt.figure(figsize = (20,15))
ax = sns.heatmap(matrix,cmap= "Blues", linecolor = 'black' , linewidth = 0, annot = True, fmt='', xticklabels=['A','B','C','D','E','F','G','H','I','K','L','M','N','O','P','Q','R','S',
                        'T','U','V','W','X','Y'], yticklabels=['A','B','C','D','E','F','G','H','I','K','L','M','N','O','P','Q','R','S',
                        'T','U','V','W','X','Y']);
ax.set(xlabel='Classified as', ylabel='True label')
plt.show()