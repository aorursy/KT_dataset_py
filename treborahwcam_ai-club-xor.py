#Numpy is a library that lets you create lists of numbers and manipulate them in all sorts of cool ways!

import numpy 



#Keras is the library that we will be using to make the ML models

import keras

print("This code just ran!")
#This is the data we will be using to train the model.

Xs = numpy.array([ #Input data

    [0, 0], 

    [0, 1], 

    [1, 0], 

    [1, 1]

])



Ys = numpy.array([ #Expected output data

    [0],

    [1],

    [1],

    [0]

])
model = keras.models.Sequential()
#Time to add our first layer!  The input_dim tells the network how many inputs we'll be giving it.

#The units is the number of neurons in the layer 

#Activation tells the layer what activation function to use! (things like sin, cos, tanh, sigmoid, ect)

model.add(keras.layers.Dense(input_dim=2, units=4, activation="relu"))
model.add(keras.layers.Dense(units=4, activation="relu")) #We only need to specify number of inputs in the first layer

model.add(keras.layers.Dense(units=1, activation="sigmoid"))
model.compile(optimizer="sgd", loss="mean_squared_error")
print(model.summary())
print(model.predict(Xs))
#Train the model by calling the fit function (the epochs is the # of times the model will train on the data)

model.fit(Xs, Ys, epochs=1)

print(model.predict(Xs))
for i in range(10):

    model.fit(Xs, Ys, epochs=2000, verbose=0) #Setting the verbos to 0 means that keras won't tell us the loss.  That's ok!

    

    #Now get the model to tell us what it thinks!

    print(model.predict(Xs))

    print("================")