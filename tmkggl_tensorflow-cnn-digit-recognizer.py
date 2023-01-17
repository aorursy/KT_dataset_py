import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import tensorflow as tf 

from keras.utils.np_utils import to_categorical 

from keras.preprocessing.image import ImageDataGenerator
#Training data

train_data = pd.read_csv('../input/train.csv') # Import the dataset

train_y = train_data["label"] # Create label vector

train_data.drop(["label"], axis=1, inplace=True) # Remove the label vector from the pixel column matrix

train_X = train_data

train_X = train_X.values.reshape(-1, 28, 28, 1)

train_y = train_y.values

train_y = to_categorical(train_y)

train_X = train_X/255.00 # Normalization

#Test data

test_X = pd.read_csv('../input/test.csv')

test_X = test_X.values.reshape(-1,28,28,1)

test_X = test_X / 255.0 # Normalization
model = tf.keras.Sequential([

tf.keras.layers.Conv2D(32, kernel_size = (5,5), padding = 'same', activation ='relu', input_shape = (28,28,1)),

tf.keras.layers.Conv2D(32, kernel_size = (5,5), padding = 'same', activation ='relu'),

tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=2), 

tf.keras.layers.Dropout(0.2),

tf.keras.layers.Flatten(),

tf.keras.layers.Dense(512, activation = "relu"),

tf.keras.layers.Dense(10, activation = "softmax")

])
model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
datagen = ImageDataGenerator(rotation_range=5, zoom_range=0.09) # try  

datagen.fit(train_X)

batch =512

model.fit_generator(datagen.flow(train_X, train_y, batch_size=batch), epochs=45)
predictions = model.predict(test_X)

predictions[354]

pred = np.argmax(predictions, axis=1)



plt.imshow(test_X[354][:,:,0],cmap='gray')

plt.show()



pred[354]



pred_digits = pd.DataFrame({'ImageId': range(1,len(test_X)+1) ,'Label':pred })

pred_digits.to_csv("pre_digits.csv",index=False)

pred_digits.head()