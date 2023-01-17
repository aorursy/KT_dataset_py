from keras.layers.core import Dense, Activation
from keras.models import Sequential
import numpy as np
X_train = np.random.randint(-100,100,size=(300, 3)) #300 lines of values in 3 dimensions, generated values will be between -100 & 100
Y_train = np.array([[sum(x)] for x in X_train]) #sum of these values
print(X_train)
print(Y_train)
X_test = np.random.randint(-1000,1000,size=(20, 3))
Y_test = np.array([[sum(x)] for x in X_test])
model = Sequential()

model.add(Dense(500, input_shape=(X_train.shape[1],), activation='relu'))

model.add(Dense(1)) #no activation - whatever output - will be output

model.compile(optimizer='adam', loss='mse', metrics=None)
model.fit(X_train, Y_train, epochs=500, verbose=1)
loss = model.evaluate(X_test, Y_test, verbose=1)
print('Loss = ', loss )
predictions = model.predict(X_test)
print(predictions)
for i in np.arange(len(predictions)):
    print('Data: ', X_test[i], ', Actual: ', Y_test[i], ', Predicted: ', predictions[i])