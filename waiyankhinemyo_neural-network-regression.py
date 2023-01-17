from keras.layers.core import Dense
from keras.models import Sequential
import numpy as np
import math
from sklearn.datasets import load_boston
X, Y = load_boston(return_X_y=True)
print(X)
#prepare training features and training labels
X_train = X[0:500]
Y_train = Y[0:500]
#prepare testing features and testing labels
X_test = X[500:]
Y_test = Y[500:]
#construct neural network model
model = Sequential()
model.add(Dense(500, input_shape=(X_train.shape[1],), activation='sigmoid'))
model.add(Dense(500, activation='sigmoid'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse', metrics=None)
model.fit(X_train, Y_train, epochs=500, verbose=1)
#perform auto-evaluation
loss = model.evaluate(X_test, Y_test, verbose=1)
print('Loss = ', loss)
#perform prediction (let's eye-ball the results)
predictions = model.predict(X_test)
acc = 0
for i in np.arange(len(predictions)):
    acc += math.pow(Y_test[i] - predictions[i], 2)
    print('Actual: ', Y_test[i], ', Predicted: ', predictions[i])

print('Computed loss: ', acc/len(Y_test))
