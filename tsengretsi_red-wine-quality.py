import pandas

import numpy



from keras.layers import Dropout

from keras.models import Sequential

from keras.layers import Dense

data = pandas.read_csv('../input/winequality-red.csv')
data.shape
data.dtypes
data.sample(30)
data.isnull()
data = data.values



X = data[:,0:11]

Y = data[:,11]



X.shape
from keras.utils import np_utils

Y = np_utils.to_categorical(Y)

Y.shape
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1 ,random_state = 0)



from sklearn.model_selection import train_test_split

X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.11 ,random_state = 0)
X_train.shape
X_test.shape
X_val.shape
model = Sequential()

model.add(Dense(30, input_dim = 11, kernel_initializer = 'uniform', activation = 'relu'))

model.add(Dense(40, kernel_initializer='uniform', activation='relu'))

model.add(Dense(10, kernel_initializer='uniform', activation='relu'))

model.add(Dropout(0.3))

model.add(Dense(50, kernel_initializer='uniform', activation='relu'))

model.add(Dropout(0.1))

model.add(Dense(30, kernel_initializer='uniform', activation='relu'))

model.add(Dense(20, kernel_initializer='uniform', activation='relu'))

model.add(Dense(9, kernel_initializer='uniform', activation='relu'))

model.add(Dense(9, kernel_initializer='uniform', activation='softmax'))

model.summary()
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
model.fit(X_train, Y_train, epochs = 100, batch_size = 10)
scores = model.evaluate(X_val, Y_val)

print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
from keras.models import model_from_json



model_json = model.to_json()

with open(r'red_wine_quality_model.json', "w") as json_file:

    json_file.write(model_json)



model.save_weights(r'red_wine_quality_model.h5')