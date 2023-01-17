import numpy as np

import pandas as pd



x_train = pd.read_csv("../input/train.csv")

x_val = pd.read_csv("../input/test.csv")
#Checking data for nulls

#defining a function to re-use

def check_null(dataframe):

    null_count = dataframe.isnull().sum()

    for index, val in enumerate(null_count):

        if (val != 0):

            print(null_count.index.values[index], ":", val)



print("train set")

check_null(x_train)

print("-" * 15)

print("validation set")

check_null(x_val)
# Extract y from data

y_train = x_train.iloc[:, 0]

x_train = x_train.drop("label", axis=1)



#Check for coherent values

print("x_train max :", x_train.max().max())

print("x_train min :", x_train.min().min())

print("x_val max :", x_val.max().max())

print("x_val min :", x_val.min().min())

print("y_train values :", np.unique(y_train))
# Scale to 0-1 instead of 0-255

x_train = x_train / 255

x_val = x_val / 255



# Check range of values

#print("x_train max :", x_train.max().max())

#print("x_train min :", x_train.min().min())

#print("x_val max :", x_val.max().max())

#print("x_val min :", x_val.min().min())
# Reshape to suitable form for Keras

# -1 let numpy guess the number of this axis

x_train = x_train.values.reshape(-1, 28, 28, 1)

x_val = x_val.values.reshape(-1, 28, 28, 1)



# Check the shape

#print("x_train shape : ", x_train.shape)

#print("x_val shape : ", x_val.shape)
from sklearn.model_selection import train_test_split

import seaborn as sb



# Check data repartition in train set

#y_train.value_counts()

sb.countplot(y_train)



# Split into train & test set

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size = 0.1)
from keras.utils.np_utils import to_categorical



# Encode y_train

y_train = to_categorical(y_train, num_classes = 10)

y_test_dummies = to_categorical(y_test, num_classes = 10)



# Check encoding

#print(y_train.shape)

#print(y_test_dummies.shape)

#print(np.unique(y_train))

#print(np.unique(y_test_dummies))
from keras.models import Sequential

from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, Dropout

from keras.preprocessing.image import ImageDataGenerator



model = Sequential()

model.add(Convolution2D(32, kernel_size = (3, 3), activation = "relu", input_shape = (28, 28, 1)))

model.add(Dropout(0.25))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Convolution2D(32, kernel_size = (3, 3), activation = "relu", input_shape = (28, 28, 1)))

model.add(Dropout(0.25))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Flatten())

model.add(Dense(10, activation = "softmax"))



model.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])



#imagedatagenerator

datagen = ImageDataGenerator(rescale = 0.1, shear_range = 0.2, zoom_range = 0.2, rotation_range = 0.2)

model.fit_generator(datagen.flow(x_train, y_train), epochs=10, validation_data = (x_test, y_test_dummies), steps_per_epoch = 1182)
from sklearn.metrics import confusion_matrix

#testing result



y_pred = model.predict(x_test)

#convering predictions from format [0, 0, 1, ..] to format 0-9

y_pred_classes = np.argmax(y_pred,axis = 1)



cm = confusion_matrix(y_test, y_pred_classes)



correct_pred = 0

for i in range(0, 10):

    correct_pred += cm[i][i]

print("Accuracy : ", correct_pred / y_pred_classes.size)

#Predicting official results

y_val = model.predict(x_val)

y_val_classes = np.argmax(y_val,axis = 1)



#Added imageId indexes

indexes = np.empty([28000, 1])

index = 0

for val in indexes:

    indexes[index] = index + 1

    index += 1



result = np.hstack((indexes, y_val_classes.reshape(28000, 1))).astype(int)



df = pd.DataFrame(result)

df.columns = ["ImageId", "Label"]

df.to_csv("output.csv", index=False)