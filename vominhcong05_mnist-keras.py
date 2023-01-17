# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
from keras.utils.np_utils import to_categorical
training_dataset = pd.read_csv("../input/train.csv")

testing_dataset = pd.read_csv("../input/test.csv")
training_dataset.head(5)
training_dataset['label'].unique()
training_dataset['label'].count()
Y_train = training_dataset['label']
X_train = training_dataset.drop(labels=['label'], axis=1)
X_test = testing_dataset.values
X_test.shape
X_train.isnull().any().describe()
Y_train.isnull().any()
X_train = X_train/255.0
X_test = X_test/ 255.0
X_train.shape
X_train.values
X_test.shape
# Reshape image in 3 dimensions (height = 28px, width = 28px, canal = 1)

X_train = X_train.values.reshape(-1, 28, 28, 1)
X_test = X_test.reshape([-1, 28, 28, 1])
Y_train = to_categorical(Y_train, num_classes=10)
random_seed = 2
from sklearn.model_selection import train_test_split
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, random_state=random_seed, test_size = 0.3)
# Model Architecture

# Input -> [Conv2d (5, 5) -> Maxpooling (2, 2) -> Dropout] x2 -> [Conv2d (3, 3) -> Maxpooling (2, 2) -> Dropout] x2 -> Flatten -> Classify
from keras import Sequential

from keras.layers import Dense, MaxPooling2D, Dropout, Flatten, Conv2D
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu', padding='same', input_shape=(28, 28, 1)))

model.add(MaxPooling2D(padding='same', pool_size=(2, 2)))

model.add(Dropout(0.25))



model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu', padding='same'))

model.add(MaxPooling2D(padding='same', pool_size=(2, 2)))

model.add(Dropout(0.25))



model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))

model.add(MaxPooling2D(padding='same', pool_size=(2, 2)))

model.add(Dropout(0.25))



model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))

model.add(MaxPooling2D(padding='same', pool_size=(2, 2)))

model.add(Dropout(0.25))



model.add(Flatten())

model.add(Dense(256, activation='relu'))

model.add(Dropout(0.25))

model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'] )

model.fit(batch_size=200, epochs=12, x=X_train, y=Y_train, validation_data=[X_val, Y_val])
prediction = model.predict(X_test)
label = np.argmax(prediction, axis=1)
test_id = np.reshape(range(1, len(prediction) + 1), label.shape)
my_submission = pd.DataFrame({'ImageId': test_id, 'Label': label})

# you could use any filename. We choose submission here

my_submission.to_csv('submission.csv', index=False)