# Using a densely layered neural network with no activation function to fit a linear regression
# Expected outcome- perfect fit since the NN is a linear operation
import pandas as pd
import random

def calculate_y(POW, COEFF, CONSTANT):
    def c(x):
        return COEFF * x**POW + CONSTANT
    return c

def gen_data(range_start, range_end, n_data_points, calculate_y):
    # Generate samples of x in increments of 0.01
    x = [random.randint(range_start * 100,range_end * 100)/100 for _ in range(n_data_points)]
    y = [calculate_y(n) for n in x]
    data = {'x': x, 'y': y}
    return pd.DataFrame(data=data)
data = gen_data(1,100,100, calculate_y(1,2,17))
data.head()
import matplotlib.pyplot as plt

# Split the data into training and validation data sets
training_dataset = data.sample(frac=0.8, random_state=0)
testing_dataset = data.drop(training_dataset.index)
training_dataset.plot(x='x', y='y', kind='scatter', title='Generated Training Data Points')
testing_dataset.plot(x='x', y='y', kind='scatter', title='Generated Testing Data Points')

training_labels = training_dataset.pop('y')
testing_labels = testing_dataset.pop('y')
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
print(tf.__version__)

# Use 2 densely connected layers with linear activation functions
def build_model():
  model = keras.Sequential([
    layers.Dense(10, activation=None, input_shape=[1]),
    layers.Dense(10, activation=None),
    layers.Dense(1)
  ])

  optimizer = tf.train.RMSPropOptimizer(0.001)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
  return model

model = build_model()
model.summary()
# Display training progress by printing a single dot for each completed epoch
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

# The patience parameter is the amount of epochs to check for improvement
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=50)

EPOCHS = 1000

history = model.fit(
  training_dataset, training_labels,
  epochs=EPOCHS, validation_split = 0.2, verbose=0,
  callbacks=[early_stop, PrintDot()])
def plot_history(history):
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error')
  plt.plot(hist['epoch'], hist['mean_absolute_error'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
           label = 'Val Error')
  plt.legend()
  plt.ylim([0,5])
  
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Square Error')
  plt.plot(hist['epoch'], hist['mean_squared_error'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_squared_error'],
           label = 'Val Error')
  plt.legend()
  plt.ylim([0,20])

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch

plot_history(history)
hist.tail()
# Generate some data points outside of the range
bigger_dataset = gen_data(100,200,50, calculate_y(1,2,17))
smaller_dataset = gen_data(-100,0,50, calculate_y(1,2,17))
test_x = pd.Series([]).append(smaller_dataset['x'], ignore_index=True).append(testing_dataset['x'], ignore_index=True).append(bigger_dataset['x'], ignore_index=True)
real_y = pd.Series([]).append(smaller_dataset['y'], ignore_index=True).append(testing_labels, ignore_index=True).append(bigger_dataset['y'], ignore_index=True)
predicted_y = model.predict(test_x)
# Plot all the results
plt.scatter(test_x, real_y)
plt.scatter(test_x, predicted_y)
plt.xlabel('x values')
plt.ylabel('y values')
# As shown linear regression is fit perfectly with a densely connected neural network
# Try fitting a parabola

# Generate data
data = gen_data(1,100,100, calculate_y(2,7,-37))
data.head()

# Split the data into training and validation data sets
training_dataset = data.sample(frac=0.8, random_state=0)
testing_dataset = data.drop(training_dataset.index)
training_dataset.plot(x='x', y='y', kind='scatter', title='Generated Training Data Points')
testing_dataset.plot(x='x', y='y', kind='scatter', title='Generated Testing Data Points')

training_labels = training_dataset.pop('y')
testing_labels = testing_dataset.pop('y')

# Rebuild the model
model = build_model()
model.summary()
# Fit the model on the data
history = model.fit(
  training_dataset, training_labels,
  epochs=EPOCHS, validation_split = 0.2, verbose=0,
  callbacks=[early_stop, PrintDot()])

loss, mae, mse = model.evaluate(training_dataset, training_labels, verbose=0)
print("\n Training set Mean Abs Error: {:5.2f}".format(mae))
# Extrapolate the test points
# Generate some data points outside of the range
bigger_dataset = gen_data(100,200,50, calculate_y(2,7,-37))
smaller_dataset = gen_data(-100,0,50, calculate_y(2,7,-37))
test_x = pd.Series([]).append(smaller_dataset['x'], ignore_index=True).append(testing_dataset['x'], ignore_index=True).append(bigger_dataset['x'], ignore_index=True)
real_y = pd.Series([]).append(smaller_dataset['y'], ignore_index=True).append(testing_labels, ignore_index=True).append(bigger_dataset['y'], ignore_index=True)
predicted_y = model.predict(test_x)
# Plot all the results
plt.scatter(test_x, real_y)
plt.scatter(test_x, predicted_y)
plt.xlabel('x values')
plt.ylabel('y values')
# As demonstrated a linear fit is drawn
# Lets see what adding activation functions do to this

# Use 2 densely connected layers with linear activation functions
def build_relu_model():
  model = keras.Sequential([
    layers.Dense(100, activation=tf.nn.relu, input_shape=[1]),
    layers.Dense(100, activation=tf.nn.relu),
    layers.Dense(100, activation=tf.nn.relu),
    layers.Dense(100, activation=tf.nn.relu),
    layers.Dense(100, activation=tf.nn.relu),
    layers.Dense(1)
  ])

  optimizer = tf.train.RMSPropOptimizer(0.001)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
  return model

# Lets try using relu activation functions and see how good the fit becomes
model = build_relu_model()

# Fit the model on the data
history = model.fit(
  training_dataset, training_labels,
  epochs=EPOCHS, validation_split = 0.2, verbose=0,
  callbacks=[early_stop, PrintDot()])

loss, mae, mse = model.evaluate(training_dataset, training_labels, verbose=0)
print("\n Training set Mean Abs Error: {:5.2f}".format(mae))
# Extrapolate the test points
# Generate some data points outside of the range
predicted_y = model.predict(test_x)
# Plot all the results
plt.scatter(test_x, real_y)
plt.scatter(test_x, predicted_y)
plt.xlabel('x values')
plt.ylabel('y values')
# As you can see relu introduces bends in the linear fit but only within the training data range
# Lets try using sigmoid
def build_sigmoid_model():
  model = keras.Sequential([
    layers.Dense(1000, activation=tf.nn.sigmoid, input_shape=[1]),
    layers.Dense(1000, activation=tf.nn.sigmoid),
    layers.Dense(1000, activation=tf.nn.sigmoid),
    layers.Dense(1000, activation=tf.nn.sigmoid),
    layers.Dense(1000, activation=tf.nn.sigmoid),
    layers.Dense(1)
  ])
  optimizer = tf.train.RMSPropOptimizer(0.001)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
  return model
# Lets try using relu activation functions and see how good the fit becomes
model = build_sigmoid_model()

# Fit the model on the data
history = model.fit(
  training_dataset, training_labels,
  epochs=EPOCHS, validation_split = 0.2, verbose=0,
  callbacks=[early_stop, PrintDot()])

loss, mae, mse = model.evaluate(training_dataset, training_labels, verbose=0)
print("\n Training set Mean Abs Error: {:5.2f}".format(mae))
# Extrapolate the test points
# Generate some data points outside of the range
predicted_y = model.predict(test_x)
# Plot all the results
plt.scatter(test_x, real_y)
plt.scatter(test_x, predicted_y)
plt.xlabel('x values')
plt.ylabel('y values')
# Sigmoid does not work at all for regression, 
# because sigmoid limits the output values of each neuron from 0 to 1 
# therefore the overall y values cannot become too big
# Lets try something that might work, 
# use sigmoid in the first hidden layer to introduce continuous non-linear features
# then use linear to scale the values
def build_mixed_model():
  model = keras.Sequential([
    layers.Dense(100, activation=tf.nn.sigmoid, input_shape=[1]),
    layers.Dense(100, activation=tf.nn.sigmoid, input_shape=[1]),
      layers.Dense(100, activation=tf.nn.sigmoid, input_shape=[1]),
    layers.Dense(100, activation=None),
    layers.Dense(100, activation=None),
    layers.Dense(100, activation=None),
    layers.Dense(1)
  ])
  optimizer = tf.train.RMSPropOptimizer(0.001)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
  return model

# Lets try using relu activation functions and see how good the fit becomes
model = build_mixed_model()

# Fit the model on the data
history = model.fit(
  training_dataset, training_labels,
  epochs=EPOCHS, validation_split = 0.2, verbose=0,
  callbacks=[early_stop, PrintDot()])

loss, mae, mse = model.evaluate(training_dataset, training_labels, verbose=0)
print("\n Training set Mean Abs Error: {:5.2f}".format(mae))
# Extrapolate the test points
# Generate some data points outside of the range
predicted_y = model.predict(test_x)
# Plot all the results
plt.scatter(test_x, real_y)
plt.scatter(test_x, predicted_y)
plt.xlabel('x values')
plt.ylabel('y values')
# A much better fit but still not perfect
# This is because any combination of the sigmoid function cannot produce a polynomial function
# Sigmoid in the end will still produce a curve that has a limited Y value no matter how much the
# linear matrix transforms scale it later on. Since there is no data extrapolating beyond x=100
# using any other segmented-linear activation function such as relu will not improve this model much
# Another activation function is needed, one which is capable of producing a polynomial fit
# https://ml-cheatsheet.readthedocs.io/en/latest/activation_functions.html
# So lets try using a custom made activation function which we know for sure will mathematically produce
# a parabolic fit, multiply the output by itself o^2
# This produces a (ax+k)^2 output of the neuron where a is the weight and k is the neuron's bias
#which when expanded is ax^2+2akx+k, 
# a sum of these neurons in a single dense layer produces a parabolic output

from keras.layers import Activation
from keras.utils.generic_utils import get_custom_objects

def square_x(x):
    return x**2

def build_parabolic_model():
  model = keras.Sequential([
    layers.Dense(100, activation=square_x, input_shape=[1]),
    layers.Dense(100, activation=None),
    layers.Dense(100, activation=None),
    layers.Dense(100, activation=None),
    layers.Dense(1)
  ])
  optimizer = tf.train.RMSPropOptimizer(0.001)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
  return model

# Lets try using squared activation functions and see how good the fit becomes
model = build_parabolic_model()

# Fit the model on the data
history = model.fit(
  training_dataset, training_labels,
  epochs=EPOCHS, validation_split = 0.2, verbose=0,
  callbacks=[early_stop, PrintDot()])

loss, mae, mse = model.evaluate(training_dataset, training_labels, verbose=0)
print("\n Training set Mean Abs Error: {:5.2f}".format(mae))
# Extrapolate the test points
# Generate some data points outside of the range
predicted_y = model.predict(test_x)
# Plot all the results
plt.scatter(test_x, real_y)
plt.scatter(test_x, predicted_y)
plt.xlabel('x values')
plt.ylabel('y values')
# Great! this is a perfect fit able to be extrapolated even beyond the range of the dataset
# However, using the parabolic activation function we are already assuming our dataset has the relation of 
# y = x^2
# Is there an activation function which is able to fit any binomial expansion of 
# y = (x+k)^n where n could be any value?
# In fourier transform, any repeating wave can be broken down into a combination of sine and cosine waves,
# What is the fourier transform equivilent of polynomial functions?