import numpy as np # linear algebra

import matplotlib.pyplot as plt

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
dataset = pd.read_csv('../input/voice.csv')

dataset.head()
dataset.describe()
dataset.corr()
num_columns = dataset.shape[1]

x = dataset.iloc[:,:20].values

y = dataset.iloc[:,20].values
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

gender_labels = LabelEncoder()

y = gender_labels.fit_transform(y)

# lets see which is 0 and which is 1

print(list(gender_labels.inverse_transform([0,1])))
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state = 0)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

x_train = sc.fit_transform(x_train)

x_test = sc.transform(x_test)
import keras

from keras.models import Sequential

from keras.layers import Dense



classifier = Sequential()

classifier.add(Dense(units = 11, activation = 'relu', kernel_initializer = 'uniform', input_shape = (20,)))

classifier.add(Dense(units = 11, activation = 'relu', kernel_initializer = 'uniform'))

classifier.add(Dense(units = 1, activation = 'sigmoid', kernel_initializer = 'uniform', input_shape = (20,)))

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

classifier.fit(x_train, y_train, batch_size = 10, epochs = 100)
y_pred = classifier.predict(x_test)

y_pred = (y_pred > 0.5)
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print(cm)
import matplotlib.pyplot as plt

plt.matshow(cm)

plt.colorbar()