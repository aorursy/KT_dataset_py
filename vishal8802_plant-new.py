import keras

from keras.models import Sequential

from keras.layers import Dense,Activation, Dropout, Flatten

from keras.layers import Conv2D, MaxPooling2D

from keras.utils import to_categorical

from keras.preprocessing import image

import numpy as np

import pandas as pd

from sklearn.preprocessing import LabelBinarizer

import pickle

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from tqdm import tqdm
root="../input/tomato-leaves-as-nparray"

image_array=np.load(f'{root}/image_as_array.npy')
label_array=np.load(f'{root}/label_as_array.npy')
label_binarizer = LabelBinarizer()

image_labels = label_binarizer.fit_transform(label_array)

pickle.dump(label_binarizer,open('label_transform.pkl', 'wb'))

n_classes = len(label_binarizer.classes_)
x,y=image_array,image_labels
x.shape,y.shape
X_train, X_test, y_train, y_test = train_test_split(x,y, random_state=42, test_size=0.2)
model = Sequential() 

model.add(Conv2D(32, (2, 2), input_shape=(256,256,3))) 

model.add(Activation('relu')) 

model.add(MaxPooling2D(pool_size=(2, 2))) 

  

model.add(Conv2D(32, (2, 2))) 

model.add(Activation('relu')) 

model.add(MaxPooling2D(pool_size=(2, 2))) 

  

model.add(Conv2D(64, (2, 2))) 

model.add(Activation('relu')) 

model.add(MaxPooling2D(pool_size=(2, 2))) 

  

model.add(Flatten()) 

model.add(Dense(64)) 

model.add(Activation('relu')) 

model.add(Dropout(0.5)) 

model.add(Dense(10)) 

model.add(Activation('sigmoid'))
model.summary()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), batch_size=64)
print("[INFO] Calculating model accuracy")

scores = model.evaluate(x_test, y_test)

print(f"Test Accuracy: {scores[1]*100}")



# save the model to disk

print("[INFO] Saving model...")

pickle.dump(model,open('cnn_model.pkl', 'wb'))

print("[INFO] Model saved")