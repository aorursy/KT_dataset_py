from keras.utils import np_utils 

from keras.datasets import mnist 

import seaborn as sns

from keras.initializers import RandomNormal

%matplotlib notebook

import matplotlib.pyplot as plt

import numpy as np

import time
# https://gist.github.com/greydanus/f6eee59eaf1d90fcb3b534a25362cea4

# https://stackoverflow.com/a/14434334

# this function is used to update the plots for each epoch and error

def plt_dynamic(x, vy, ty, ax, colors=['b']):

    ax.plot(x, vy, 'b', label="Validation Loss")

    ax.plot(x, ty, 'r', label="Train Loss")

    plt.legend()

    plt.grid()

    fig.canvas.draw()
# the data, shuffled and split between train and test sets 

(X_train, y_train), (X_test, y_test) = mnist.load_data()
type(X_train)
plt.imshow(X_train[218])
print("Number of training examples :", X_train.shape[0], "and each image is of shape (%d, %d)"%(X_train.shape[1], X_train.shape[2]))

print("Number of training examples :", X_test.shape[0], "and each image is of shape (%d, %d)"%(X_test.shape[1], X_test.shape[2]))
# if you observe the input shape its 2 dimensional vector

# for each image we have a (28*28) vector

# we will convert the (28*28) vector into single dimensional vector of 1 * 784 



X_train = X_train.reshape(X_train.shape[0], X_train.shape[1]*X_train.shape[2]) 

X_test = X_test.reshape(X_test.shape[0], X_test.shape[1]*X_test.shape[2]) 
# after converting the input images from 3d to 2d vectors



print("Number of training examples :", X_train.shape[0], "and each image is of shape (%d)"%(X_train.shape[1]))

print("Number of training examples :", X_test.shape[0], "and each image is of shape (%d)"%(X_test.shape[1]))
# An example data point

print(X_train[0])
# if we observe the above matrix each cell is having a value between 0-255

# before we move to apply machine learning algorithms lets try to normalize the data

# X => (X - Xmin)/(Xmax-Xmin) = X/255



X_train = X_train/255

X_test = X_test/255
# example data point after normlizing

print(X_train[0])
# here we are having a class number for each image

print("Class label of first image :", y_train[0])



# lets convert this into a 10 dimensional vector

# ex: consider an image is 5 convert it into 5 => [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]

# this conversion needed for MLPs 



Y_train = np_utils.to_categorical(y_train, 10) 

Y_test = np_utils.to_categorical(y_test, 10)



print("After converting the output into a vector : ",Y_train[0])
from keras.models import Sequential 

from keras.layers import Dense, Activation 
# some model parameters



output_dim = 10

input_dim = X_train.shape[1]



batch_size = 128 

nb_epoch = 20
# start building a model

model = Sequential()

model.add(Dense(output_dim, input_dim=input_dim, activation='softmax'))
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, Y_train, batch_size = batch_size, epochs = nb_epoch, verbose = 1, validation_data = (X_test, Y_test)) 
print(history.history.keys())
score = model.evaluate(X_test, Y_test, verbose=1) 

print('Test score:', score[0]) 

print('Test accuracy:', score[1])
fig,ax = plt.subplots(1,1)

ax.set_xlabel('epoch') ; ax.set_ylabel('Categorical Crossentropy Loss')



# list of epoch numbers

x = list(range(1,nb_epoch+1))



vy = history.history['val_loss']

ty = history.history['loss']

plt_dynamic(x, vy, ty, ax)
# Multilayer perceptron



model_sigmoid = Sequential()

model_sigmoid.add(Dense(512, activation='sigmoid', input_shape=(input_dim,)))

model_sigmoid.add(Dense(128, activation='sigmoid'))

model_sigmoid.add(Dense(output_dim, activation='softmax'))



model_sigmoid.summary()
model_sigmoid.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])



history = model_sigmoid.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1, validation_data=(X_test, Y_test))
score = model_sigmoid.evaluate(X_test, Y_test, verbose=0) 

print('Test score:', score[0]) 

print('Test accuracy:', score[1])



fig,ax = plt.subplots(1,1)

ax.set_xlabel('epoch') ; ax.set_ylabel('Categorical Crossentropy Loss')



# list of epoch numbers

x = list(range(1,nb_epoch+1))



vy = history.history['val_loss']

ty = history.history['loss']

plt_dynamic(x, vy, ty, ax)
w_after = model_sigmoid.get_weights()



h1_w = w_after[0].flatten().reshape(-1,1)

h2_w = w_after[2].flatten().reshape(-1,1)

out_w = w_after[4].flatten().reshape(-1,1)





fig = plt.figure(figsize=(10,6))

plt.title("Weight matrices after model trained")

plt.subplot(1, 3, 1)

plt.title("Trained model Weights")

ax = sns.violinplot(y=h1_w,color='b')

plt.xlabel('Hidden Layer 1')



plt.subplot(1, 3, 2)

plt.title("Trained model Weights")

ax = sns.violinplot(y=h2_w, color='r')

plt.xlabel('Hidden Layer 2 ')



plt.subplot(1, 3, 3)

plt.title("Trained model Weights")

ax = sns.violinplot(y=out_w,color='y')

plt.xlabel('Output Layer ')

plt.show()
model_sigmoid = Sequential()

model_sigmoid.add(Dense(512, activation='sigmoid', input_shape=(input_dim,)))

model_sigmoid.add(Dense(128, activation='sigmoid'))

model_sigmoid.add(Dense(output_dim, activation='softmax'))



model_sigmoid.summary()
model_sigmoid.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])



history = model_sigmoid.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1, validation_data=(X_test, Y_test))
score = model_sigmoid.evaluate(X_test, Y_test, verbose=1) 

print('Test score:', score[0]) 

print('Test accuracy:', score[1])



fig,ax = plt.subplots(1,1)

ax.set_xlabel('epoch') ; ax.set_ylabel('Categorical Crossentropy Loss')



# list of epoch numbers

x = list(range(1,nb_epoch+1))



vy = history.history['val_loss']

ty = history.history['loss']

plt_dynamic(x, vy, ty, ax)
w_after = model_sigmoid.get_weights()



h1_w = w_after[0].flatten().reshape(-1,1)

h2_w = w_after[2].flatten().reshape(-1,1)

out_w = w_after[4].flatten().reshape(-1,1)





fig = plt.figure(figsize=(10,6))

plt.title("Weight matrices after model trained")

plt.subplot(1, 3, 1)

plt.title("Trained model Weights")

ax = sns.violinplot(y=h1_w,color='b')

plt.xlabel('Hidden Layer 1')



plt.subplot(1, 3, 2)

plt.title("Trained model Weights")

ax = sns.violinplot(y=h2_w, color='r')

plt.xlabel('Hidden Layer 2 ')



plt.subplot(1, 3, 3)

plt.title("Trained model Weights")

ax = sns.violinplot(y=out_w,color='y')

plt.xlabel('Output Layer ')

plt.show()
model_relu = Sequential()

model_relu.add(Dense(512, activation='relu', input_shape=(input_dim,), kernel_initializer=RandomNormal(mean=0.0, stddev=0.062, seed=None)))

model_relu.add(Dense(128, activation='relu', kernel_initializer=RandomNormal(mean=0.0, stddev=0.125, seed=None)) )

model_relu.add(Dense(output_dim, activation='softmax'))



model_relu.summary()
model_relu.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])



history = model_relu.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1, validation_data=(X_test, Y_test))
score = model_relu.evaluate(X_test, Y_test, verbose=1) 

print('Test score:', score[0]) 

print('Test accuracy:', score[1])



fig,ax = plt.subplots(1,1)

ax.set_xlabel('epoch') ; ax.set_ylabel('Categorical Crossentropy Loss')



# list of epoch numbers

x = list(range(1,nb_epoch+1))



vy = history.history['val_loss']

ty = history.history['loss']

plt_dynamic(x, vy, ty, ax)
w_after = model_relu.get_weights()



h1_w = w_after[0].flatten().reshape(-1,1)

h2_w = w_after[2].flatten().reshape(-1,1)

out_w = w_after[4].flatten().reshape(-1,1)





fig = plt.figure(figsize=(10,6))

plt.title("Weight matrices after model trained")

plt.subplot(1, 3, 1)

plt.title("Trained model Weights")

ax = sns.violinplot(y=h1_w,color='b')

plt.xlabel('Hidden Layer 1')



plt.subplot(1, 3, 2)

plt.title("Trained model Weights")

ax = sns.violinplot(y=h2_w, color='r')

plt.xlabel('Hidden Layer 2 ')



plt.subplot(1, 3, 3)

plt.title("Trained model Weights")

ax = sns.violinplot(y=out_w,color='y')

plt.xlabel('Output Layer ')

plt.show()
model_relu = Sequential()

model_relu.add(Dense(512, activation='relu', input_shape=(input_dim,), kernel_initializer=RandomNormal(mean=0.0, stddev=0.062, seed=None)))

model_relu.add(Dense(128, activation='relu', kernel_initializer=RandomNormal(mean=0.0, stddev=0.125, seed=None)) )

model_relu.add(Dense(output_dim, activation='softmax'))



print(model_relu.summary())
model_relu.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])



history = model_relu.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1, validation_data=(X_test, Y_test))
score = model_relu.evaluate(X_test, Y_test, verbose=1) 

print('Test score:', score[0]) 

print('Test accuracy:', score[1])



fig,ax = plt.subplots(1,1)

ax.set_xlabel('epoch') ; ax.set_ylabel('Categorical Crossentropy Loss')



# list of epoch numbers

x = list(range(1,nb_epoch+1))



vy = history.history['val_loss']

ty = history.history['loss']

plt_dynamic(x, vy, ty, ax)
w_after = model_relu.get_weights()



h1_w = w_after[0].flatten().reshape(-1,1)

h2_w = w_after[2].flatten().reshape(-1,1)

out_w = w_after[4].flatten().reshape(-1,1)





fig = plt.figure(figsize=(10,6))

plt.title("Weight matrices after model trained")

plt.subplot(1, 3, 1)

plt.title("Trained model Weights")

ax = sns.violinplot(y=h1_w,color='b')

plt.xlabel('Hidden Layer 1')



plt.subplot(1, 3, 2)

plt.title("Trained model Weights")

ax = sns.violinplot(y=h2_w, color='r')

plt.xlabel('Hidden Layer 2 ')



plt.subplot(1, 3, 3)

plt.title("Trained model Weights")

ax = sns.violinplot(y=out_w,color='y')

plt.xlabel('Output Layer ')

plt.show()
from keras.layers.normalization import BatchNormalization



model_batch = Sequential()



model_batch.add(Dense(512, activation='sigmoid', input_shape=(input_dim,), kernel_initializer=RandomNormal(mean=0.0, stddev=0.039, seed=None)))

model_batch.add(BatchNormalization())



model_batch.add(Dense(128, activation='sigmoid', kernel_initializer=RandomNormal(mean=0.0, stddev=0.55, seed=None)) )

model_batch.add(BatchNormalization())



model_batch.add(Dense(output_dim, activation='softmax'))





model_batch.summary()
model_batch.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])



history = model_batch.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1, validation_data=(X_test, Y_test))
score = model_batch.evaluate(X_test, Y_test, verbose=1) 

print('Test score:', score[0]) 

print('Test accuracy:', score[1])



fig,ax = plt.subplots(1,1)

ax.set_xlabel('epoch') ; ax.set_ylabel('Categorical Crossentropy Loss')



# list of epoch numbers

x = list(range(1,nb_epoch+1))



vy = history.history['val_loss']

ty = history.history['loss']

plt_dynamic(x, vy, ty, ax)
w_after = model_batch.get_weights()



h1_w = w_after[0].flatten().reshape(-1,1)

h2_w = w_after[2].flatten().reshape(-1,1)

out_w = w_after[4].flatten().reshape(-1,1)





fig = plt.figure(figsize=(10,6))

plt.title("Weight matrices after model trained")

plt.subplot(1, 3, 1)

plt.title("Trained model Weights")

ax = sns.violinplot(y=h1_w,color='b')

plt.xlabel('Hidden Layer 1')



plt.subplot(1, 3, 2)

plt.title("Trained model Weights")

ax = sns.violinplot(y=h2_w, color='r')

plt.xlabel('Hidden Layer 2 ')



plt.subplot(1, 3, 3)

plt.title("Trained model Weights")

ax = sns.violinplot(y=out_w,color='y')

plt.xlabel('Output Layer ')

plt.show()
# https://stackoverflow.com/questions/34716454/where-do-i-call-the-batchnormalization-function-in-keras



from keras.layers import Dropout



model_drop = Sequential()



model_drop.add(Dense(512, activation='sigmoid', input_shape=(input_dim,), kernel_initializer=RandomNormal(mean=0.0, stddev=0.039, seed=None)))

model_drop.add(BatchNormalization())

model_drop.add(Dropout(0.5))



model_drop.add(Dense(128, activation='sigmoid', kernel_initializer=RandomNormal(mean=0.0, stddev=0.55, seed=None)) )

model_drop.add(BatchNormalization())

model_drop.add(Dropout(0.5))



model_drop.add(Dense(output_dim, activation='softmax'))





model_drop.summary()
model_drop.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])



history = model_drop.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1, validation_data=(X_test, Y_test))
score = model_drop.evaluate(X_test, Y_test, verbose=1) 

print('Test score:', score[0]) 

print('Test accuracy:', score[1])



fig,ax = plt.subplots(1,1)

ax.set_xlabel('epoch') ; ax.set_ylabel('Categorical Crossentropy Loss')



# list of epoch numbers

x = list(range(1,nb_epoch+1))



vy = history.history['val_loss']

ty = history.history['loss']

plt_dynamic(x, vy, ty, ax)
w_after = model_drop.get_weights()



h1_w = w_after[0].flatten().reshape(-1,1)

h2_w = w_after[2].flatten().reshape(-1,1)

out_w = w_after[4].flatten().reshape(-1,1)





fig = plt.figure(figsize=(10,6))

plt.title("Weight matrices after model trained")

plt.subplot(1, 3, 1)

plt.title("Trained model Weights")

ax = sns.violinplot(y=h1_w,color='b')

plt.xlabel('Hidden Layer 1')



plt.subplot(1, 3, 2)

plt.title("Trained model Weights")

ax = sns.violinplot(y=h2_w, color='r')

plt.xlabel('Hidden Layer 2 ')



plt.subplot(1, 3, 3)

plt.title("Trained model Weights")

ax = sns.violinplot(y=out_w,color='y')

plt.xlabel('Output Layer ')

plt.show()
from keras.optimizers import Adam,RMSprop,SGD

def best_hyperparameters(activ):



    model = Sequential()

    model.add(Dense(512, activation=activ, input_shape=(input_dim,), kernel_initializer=RandomNormal(mean=0.0, stddev=0.062, seed=None)))

    model.add(Dense(128, activation=activ, kernel_initializer=RandomNormal(mean=0.0, stddev=0.125, seed=None)) )

    model.add(Dense(output_dim, activation='softmax'))





    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

    

    return model
# https://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/



activ = ['sigmoid','relu']



from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import GridSearchCV



model = KerasClassifier(build_fn=best_hyperparameters, epochs=nb_epoch, batch_size=batch_size, verbose=0)

param_grid = dict(activ=activ)



# if you are using CPU

# grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)

# if you are using GPU dont use the n_jobs parameter



grid = GridSearchCV(estimator=model, param_grid=param_grid)

grid_result = grid.fit(X_train, Y_train)
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

means = grid_result.cv_results_['mean_test_score']

stds = grid_result.cv_results_['std_test_score']

params = grid_result.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):

    print("%f (%f) with: %r" % (mean, stdev, param))