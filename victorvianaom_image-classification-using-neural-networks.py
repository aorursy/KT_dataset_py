import os

print(os.listdir('../input'))

print(os.listdir('../input/digit-recognizer'))
import pandas as pd

train_set = pd.read_csv('../input/digit-recognizer/train.csv')

train_set
y = train_set['label']

y
X = train_set.drop('label', axis=1, inplace=True)

X = train_set

X
y = pd.get_dummies(y)

y
#The first hidden layer will have 600 neurons, second: 400, third: 200, fourth: 10.

#from tensorflow.keras import Sequential

#from tensorflow.keras.layers import Dense



from keras.models import Sequential

from keras.layers import Dense



classifier = Sequential()



#Now addindg the hidden layers:

classifier.add(Dense(units=600, kernel_initializer='uniform', activation='relu', input_dim=784))

classifier.add(Dense(units=400, kernel_initializer='uniform', activation='relu'))

classifier.add(Dense(units=200, kernel_initializer='uniform', activation='relu'))

classifier.add(Dense(units=10, kernel_initializer='uniform', activation='sigmoid'))
classifier.compile(optimizer='sgd', loss='mean_squared_error', metrics=['accuracy'])
classifier.fit(X, y, batch_size=10, epochs=10)
X_test = pd.read_csv("../input/digit-recognizer/test.csv")

y_pred = classifier.predict(X_test)

y_pred