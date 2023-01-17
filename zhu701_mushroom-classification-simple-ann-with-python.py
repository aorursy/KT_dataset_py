import numpy as np # linear algebra

import pandas as pd # data processing



# Importing the dataset

data= pd.read_csv('/kaggle/input/mushroom-classification/mushrooms.csv')

data.shape
data.head(5)
# Encoding categorical data

from sklearn.preprocessing import LabelEncoder

labelencoder=LabelEncoder()

for col in data.columns:

    data[col] = labelencoder.fit_transform(data[col])

data.head(5)
x = data.iloc[:, 1:23].values

y = data.iloc[:, 0].values
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

X_train.shape
X_test.shape
import keras

from keras.models import Sequential

from keras.layers import Dense



# Initialising the ANN

classifier = Sequential()



# Adding the input layer 

classifier.add(Dense(units = 11, kernel_initializer = 'uniform', activation = 'relu', input_dim = 22))



# Adding the second hidden layer

classifier.add(Dense(units = 11, kernel_initializer = 'uniform', activation = 'relu'))



# Adding the output layer

classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))



# Compiling the ANN

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])



classifier.summary()
# Fitting the ANN to the Training set

history = classifier.fit(X_train, y_train, batch_size = 10, epochs = 50)
import matplotlib.pyplot as plt

# Checking the key names

print(history.history.keys())
# Summarizing history for accuracy

plt.plot(history.history['accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
# Summarizing history for loss

plt.plot(history.history['loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper right')

plt.show()
# Predicting the test set results

y_pred = classifier.predict(X_test) #these are probabilities

y_pred = (y_pred > 0.5) #we need to convert it to binary



# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)



print(cm)