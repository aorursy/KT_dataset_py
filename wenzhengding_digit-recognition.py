import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



from keras.utils import plot_model # plot the model for visulization

from sklearn.model_selection import train_test_split # split train and test set 

from keras.models import Sequential # sequential model to build CNN model

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D # layer units that will be used to build model

from keras.optimizers import Adam # optimizer that will be used





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# The output is the path that the data is stored
# Read train and test data

train_data = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

test_data = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")



# Let's have a look at how the data looks like by listing 10 data row from train data

train_data.head(10)
# Split the train data into X(features) and y(value)

X_train = train_data.drop(labels='label', axis=1)

y_train = train_data['label']
# Check for null values in train data

X_train.isnull().any().describe()
# Check for null values in test data

test_data.isnull().any().describe()
# Split train and test data

# Set the random seed

random_seed = 1



X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.2, random_state=random_seed)
# Convert X(features) to image-like form

X_train = X_train.values.reshape(-1, 28, 28, 1)

X_val = X_val.values.reshape(-1, 28, 28, 1)

test_data = test_data.values.reshape(-1, 28, 28, 1)





# Do one hot encoding for both training and validation y(value)

y_train = pd.get_dummies(y_train)

y_val = pd.get_dummies(y_val)

# Implement CNN model and fit 

model = Sequential()



model.add(Conv2D(filters = 32, kernel_size = (4,4),padding = 'Same', 

                 activation ='relu', input_shape = (28,28,1)))

model.add(Conv2D(filters = 32, kernel_size = (4,4),padding = 'Same', 

                 activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2)))





model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))

model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))





model.add(Flatten())

model.add(Dense(128, activation = "relu"))

model.add(Dense(10, activation = "softmax"))



optimizer = Adam()



model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])



epochs = 3 

batch_size = 100



history = model.fit(X_train, y_train, batch_size = batch_size, epochs = epochs, \

          validation_data = (X_val, y_val))
plot_model(model, to_file='model.png', show_shapes=True)
fig, ax = plt.subplots(2,1)

ax[0].plot(history.history['loss'], color='pink', label="Training loss")

ax[0].plot(history.history['val_loss'], color='green', label="validation loss",axes =ax[0])





ax[1].plot(history.history['accuracy'], color='pink', label="Training accuracy")

ax[1].plot(history.history['val_accuracy'], color='green',label="Validation accuracy")
# For each data row, pick the one with the largest probability to be the result

prob_result = model.predict(test_data)

result = np.argmax(prob_result, axis=1)

# Rename the column

id_column = pd.Series(range(1,28001), name="ImageId")

result_column = pd.Series(result, name="Label")

submission  = pd.concat([id_column, result_column], axis=1)

submission.to_csv("output.csv",index=False)