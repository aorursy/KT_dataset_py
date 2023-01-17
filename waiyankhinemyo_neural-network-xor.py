from keras.layers.core import Dense, Activation
from keras.models import Sequential
import numpy as np
X_train = [[x, y] for x in np.random.randint(0,2,300) for y in np.random.randint(0,2,1)]
Y_train = [[x ^ y] for x, y in X_train]
X_train = np.array(X_train)
Y_train = np.array(Y_train)
print(X_train[0:10])
print(Y_train[0:10])
Y_train_ohe = np.array([[1,0] if x[0] == 0 else [0,1] for x in Y_train])
print(Y_train)
print(Y_train_ohe)
X_test = np.array([[x, y] for x in np.random.randint(0,2,100) for y in np.random.randint(0,2,1)])
Y_test = np.array([[x ^ y] for x, y in X_test])
print(Y_test)
Y_test_ohe = [[1,0] if x[0] == 0 else [0,1] for x in Y_test]
Y_test_ohe = np.array(Y_test_ohe)
print(Y_test)
print(Y_test_ohe)
model = Sequential() #constructing sequencial neural network - empty

#add a layer, 500 is no. of nodes/neurons we want to use in a hidden layer
#input_shape (2,) means in this case: input X_train is [0,0] etc. i.e. 2 features
#actication='sigmoid' cuz we are doing classification, non-linear function
model.add(Dense(500, input_shape=(2,), activation='sigmoid'))
#model.add(Dense(500, activation='sigmoid')) --> uncomment this to add another hidden layer and improve accuracy

#this is output layer - 2 cuz we are expecting 2 outputs in this case: 0 or 1
#activation='softmax' due to probability
model.add(Dense(2, activation='softmax'))

#loss='categorical_crossentropy' is recommended for 'softmax' (another type - 'binomial_corssentropy' is also possible)
#accuracy is how many is predicted correctly
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=["accuracy"])

#fitting data to model
#batch_size is how many rows of data the model will take each time - in this case 64 rows of data
#epochs means how many time you want to train the model with the same training data i.e. in this case - 500 times training
model.fit(X_train, Y_train_ohe, batch_size=64, epochs=500, verbose=1)
loss, accuracy = model.evaluate(X_test, Y_test_ohe, verbose=1)
print('Loss = ', loss, ', Accuracy = ', accuracy)
predictions = model.predict(X_test)
for i in np.arange(len(predictions)):
    print('Actual: ', Y_test_ohe[i], ', Predicted: ', predictions[i])