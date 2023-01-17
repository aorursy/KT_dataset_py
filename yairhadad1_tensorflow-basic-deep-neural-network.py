# basic lib

import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt



# tensorflow

import tensorflow as tf



# keras 

from keras.models import Sequential

from keras.layers import Dense,Flatten



# train test split

from sklearn.model_selection import train_test_split



#ignore warning messages 

import warnings

warnings.filterwarnings('ignore') 
# Get the train and test data 

train = pd.read_csv("../input/train.csv")



# check the columns in the data

print(train.columns)
# Splite data the X - Our data , and y - the prdict label

X = train.drop('label',axis = 1)

y = train['label']
# Check the data shape (columns , rows)

print(X.shape)



print("We have 42000 row - images , and 784 columns - pixels")

print("Every image has 784 pixels")

print("\n784 = 28X28")



# let look at the data

X.head()
# let look at the data 



# print 1 image by row number

row_number = 5

plt.imshow(X.iloc[row_number].values.reshape(28,28),interpolation='nearest', cmap='Greys')

plt.show()



# print 4X4 first images

# plt.figure(figsize = (12,10))

# row, colums = 4, 4

# for i in range(16):  

#     plt.subplot(colums, row, i+1)

#     plt.imshow(X.iloc[i].values.reshape(28,28),interpolation='nearest', cmap='Greys')

# plt.show()
# splite the data

X_train, X_test, y_train, y_test = train_test_split(X,y)



# scale data

X_train = X_train.apply(lambda X: X/255)

X_test = X_test.apply(lambda X: X/255)
print("Data after scaler")

row_number = 2

plt.imshow(X_train.iloc[row_number].values.reshape(28,28),interpolation='nearest', cmap='Greys')

plt.show()
# reshape

X_train = tf.reshape(X_train, [-1, 28, 28,1])

X_test = tf.reshape(X_test, [-1, 28, 28,1])
input_shape = (28,28)

output_count = 10



model = Sequential([

    Flatten(input_shape=input_shape),

    Dense(128, activation=tf.nn.relu),

    Dense(output_count, activation=tf.nn.softmax)

])



model.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, steps_per_epoch = 32,epochs=15, validation_data = (X_test, y_test), validation_steps = 10)
test_loss, test_acc = model.evaluate(X_test, y_test, steps = 10)

print("loss",test_loss)

print("acc",test_acc)
input_shape = (28,28)

output_count = 10





model = Sequential([

    Flatten(input_shape=input_shape),

    Dense(128, activation=tf.nn.relu),

    Dense(64, activation=tf.nn.relu),

    Dense(32, activation=tf.nn.relu),

    Dense(16, activation=tf.nn.relu),

    Dense(output_count, activation=tf.nn.softmax)

])



model.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, steps_per_epoch = 32,epochs=30, validation_data = (X_test, y_test), validation_steps = 10)