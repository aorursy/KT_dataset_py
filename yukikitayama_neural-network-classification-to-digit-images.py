import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



import tensorflow as tf
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
print("Training set dimension: {}".format(train.shape))

print("Test set dimension: {}".format(test.shape))

print(train.info())

print(train.head())
test.head()
X_train = train.loc[:,'pixel0':'pixel783']

y_train = train.loc[:,'label']

X_test = test.copy()

print(X_train.shape)

print(y_train.shape)

print(X_test.shape)
print("Number of categories: {}".format(len(y_train.value_counts())))

print(y_train.value_counts().sort_index())
temp = X_train.loc[0,:]

temp = temp.values.reshape((28, 28)) 



label = y_train[0]



plt.figure()

plt.imshow(temp)

plt.colorbar()

plt.grid(False)

plt.title("True label is {}".format(label))

plt.show()
print("Maximum pixel value is {}".format(max(X_train.max())))

print("Minimum pixel value is {}".format(min(X_train.min())))
X_train = X_train / 255.0

X_test = X_test / 255.0
print("Maximum pixel value is {}".format(max(X_train.max())))

print("Minimum pixel value is {}".format(min(X_train.min())))
# build

model = tf.keras.Sequential()

model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu, input_shape = (28*28,)))

model.add(tf.keras.layers.Dense(10, activation = tf.nn.softmax))
# compile

model.compile(optimizer = 'adam',

              loss = 'sparse_categorical_crossentropy',

              metrics = ['accuracy'])
EPOCH = 5



model.fit(X_train, y_train, epochs = EPOCH, verbose = 1, validation_split = 0.3)
pred_train = model.predict(X_train)
pred_train[0]
np.argmax(pred_train[0])
# true label

y_train[0]
pred_train[1]
np.argmax(pred_train[1])
# true label

y_train[1]
pred_test = model.predict(X_test)
pred_test.shape
pred_test[0]
np.argmax(pred_test[0])
test_id = np.arange(1, X_test.shape[0]+1,1)

test_id
predictions = np.argmax(pred_test, axis = 1)
print(test_id.shape)

print(predictions.shape)
sub = pd.DataFrame(data = {'ImageId':test_id,

                           'Label':predictions})
sub.head()