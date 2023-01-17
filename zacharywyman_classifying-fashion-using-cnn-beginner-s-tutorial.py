# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv('/kaggle/input/fashionmnist/fashion-mnist_train.csv')

test = pd.read_csv('/kaggle/input/fashionmnist/fashion-mnist_test.csv')
train.dtypes, test.dtypes
#splitting data into labels

y_train = train['label']

y_test = test['label']

X_train = train.drop(labels = ['label'], axis=1)

X_test = test.drop(labels = ['label'], axis = 1)
X_train = np.array(X_train, dtype = 'float32')

X_test = np.array(X_test, dtype = 'float32')

y_train = np.array(y_train, dtype = 'float32')

y_test = np.array(y_test, dtype = 'float32')
train.values.min(), train.values.max()
X_train, X_test = X_train / 255, X_test / 255
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size = 0.2, random_state = 42)
import matplotlib.pyplot as plt

class_names = ['T_shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 

               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']



plt.imshow(X_train[0].reshape(28,28))

label = int(y_train[0])

plt.title(class_names[label])
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)

X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

X_valid = X_valid.reshape(X_valid.shape[0], 28, 28, 1)

X_train.shape
from keras.models import Sequential

from keras.layers import Flatten, MaxPooling2D, Dropout, Dense, Conv2D



model = Sequential()



model.add(Conv2D(filters = 32, kernel_size = 3, activation = 'relu', input_shape = [28,28, 1]))



model.add(MaxPooling2D(pool_size = (2)))

model.add(Dropout(0.25)) # reduces overfitting

model.add(Flatten())

model.add(Dense(32, activation = 'relu'))

model.add(Dense(10, activation = 'softmax'))



model.summary()
from keras.optimizers import Adam

model.compile(loss ='sparse_categorical_crossentropy', optimizer=Adam(lr=0.001),metrics =['accuracy'])
#if we want to implement early stopping to save training time we can modify below as model.fit(X_train, y_train, batch_size = 128, epochs = 50, verbose = 1,

#validation_data = (X_valid, y_valid), callbacks = stop)



from keras.callbacks import EarlyStopping

stop = EarlyStopping(patience = 15) # patience stands for 15 epochs 
history = model.fit(X_train, y_train, batch_size = 400, epochs = 75, verbose = 1, validation_data = (X_valid, y_valid))
fig, ax = plt.subplots(1, 2, figsize = (15,5))

ax[0].plot(history.history['accuracy'], label = 'Accuracy')

ax[0].plot(history.history['val_accuracy'], label = 'Validation Accuracy')

ax[0].set_title('Accuracy')

ax[0].legend()



ax[1].plot(history.history['loss'], label = 'Loss')

ax[1].plot(history.history['val_loss'], label = 'Validation Loss')

ax[1].set_title('Loss')

ax[1].legend()
score = model.evaluate(X_test, y_test, verbose = 1)

print('Accuracy: ' + str(score[1]))

print('Loss: ' + str(score[0]))
predicted_classes = model.predict_classes(X_test)

predicted_classes = predicted_classes.reshape(-1,1)

y_true = test['label']



pred_df = pd.DataFrame(predicted_classes)
pred_df['Actual'] = y_true
cols = ['Predictions', 'Actual']

pred_df.columns = [i for i in cols]

pred_df
from sklearn.metrics import classification_report
classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']



print(classification_report(y_true, predicted_classes, target_names = classes))