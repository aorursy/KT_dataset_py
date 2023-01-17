from keras.models import Sequential

from keras.layers import Dense, Activation
# Sequential model with one Dense layer with 2 neurons.



# Create the layers and add them in the order that they should be connected.

model = Sequential()

model.add(Dense(2))



# OR



# Create an array of layers and pass it to the constructor of the Sequential class.



layers = [Dense(2)]

model = Sequential(layers)
# MLP with 2 inputs in the visible layer, 5 neurons in the hidden layer and one neuron in the output layer.

# Activation functions are defined separately from layers.



model = Sequential()

model.add(Dense(5, input_dim=2))

model.add(Activation('relu'))

model.add(Dense(1))

model.add(Activation('sigmoid'))
from keras import optimizers



model.compile(optimizer='sgd', loss='mean_squared_error')



# OR



algorithm = optimizers.SGD(lr=0.1, momentum=0.3)

model.compile(optimizer=algorithm, loss='mean_squared_error')
# Defining metrics when compiling the model.



model.compile(optimizer='sgd', loss='mean_squared_error', metrics=['accuracy'])
# history = model.fit(X, y, batch_size=10, epochs=100)
# loss, accuracy = model.evaluate(test_X, test_y)
# predictions = model.predict(test_X)
# predictions = model.predict_classes(test_X)
from keras.layers import Input



visible = Input(shape=(2,))
# Create the input layer, then create a hidden layer as a Dense that receives input only from the input layer.

# (visible) after the creation of the Dense layer connects the input layer’s output as the input to the Dense hidden layer.



from keras.layers import Input

from keras.layers import Dense



visible = Input(shape=(2,))

hidden = Dense(2)(visible)
from keras.models import Model

from keras.layers import Input

from keras.layers import Dense



visible = Input(shape=(2,))

hidden = Dense(2)(visible)

model = Model(inputs=visible, outputs=hidden)
# Multilayer Perceptron



from keras.utils import plot_model

from keras.models import Model

from keras.layers import Input

from keras.layers import Dense



# 10 inputs

visible = Input(shape=(10,))



# 3 hidden layers with 10, 20, and 10 neurons

hidden1 = Dense(10, activation='relu')(visible)

hidden2 = Dense(20, activation='relu')(hidden1)

hidden3 = Dense(10, activation='relu')(hidden2)



# Output layer with 1 output

output = Dense(1, activation='sigmoid')(hidden3)



model = Model(inputs=visible, outputs=output)



# summarize layers

model.summary()



# plot graph

plot_model(model, to_file='multilayer_perceptron_graph.png')
# Convolutional Neural Network

from keras.utils import plot_model

from keras.models import Model

from keras.layers import Input

from keras.layers import Dense

from keras.layers.convolutional import Conv2D

from keras.layers.pooling import MaxPooling2D



# Model receives black and white 64 x 64 images as input

visible = Input(shape=(64,64,1))



# Sequence of two convolutional and pooling layers as feature extractors

conv1 = Conv2D(32, kernel_size=4, activation='relu')(visible)

pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

conv2 = Conv2D(16, kernel_size=4, activation='relu')(pool1)

pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)



# Fully connected layer to interpret the features

hidden1 = Dense(10, activation='relu')(pool2)



# Output layer with a sigmoid activation for two-class predictions

output = Dense(1, activation='sigmoid')(hidden1)



model = Model(inputs=visible, outputs=output)



# summarize layers

model.summary()



# plot graph

plot_model(model, to_file='convolutional_neural_network.png')
# Recurrent Neural Network

from keras.utils import plot_model

from keras.models import Model

from keras.layers import Input

from keras.layers import Dense

from keras.layers.recurrent import LSTM



# 100 time steps of one feature as input

visible = Input(shape=(100,1))



# Single LSTM hidden layer to extract features from the sequence

hidden1 = LSTM(10)(visible)



# Fully connected layer to interpret the LSTM output

hidden2 = Dense(10, activation='relu')(hidden1)



# Output layer for making binary predictions

output = Dense(1, activation='sigmoid')(hidden2)



model = Model(inputs=visible, outputs=output)



# summarize layers

model.summary()



# plot graph

plot_model(model, to_file='recurrent_neural_network.png')
