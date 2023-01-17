import os
import shutil
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras import backend as K

print(tf.__version__)
# using directly available datasets for simplicity
boston_housing = tf.keras.datasets.boston_housing
(train_data, train_labels), (test_data, test_labels)= boston_housing.load_data()
# Shuffle the training set
order = np.argsort(np.random.random(train_labels.shape))
train_data = train_data[order]
train_labels = train_labels[order]
print("Training set: {}".format(train_data.shape))  # 404 examples, 13 features
print("Testing set:  {}".format(test_data.shape))   # 102 examples, 13 features
print(train_data[0]) # display sample features, notice the different scales
column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']

df = pd.DataFrame(train_data, columns=column_names)
df.head()
# The labels are the house proces in thousands of dollars
print(train_labels[0:10])  # Display first 10 entries
# Normalize features
# For each feature, subtract the mean of the feature and divide by the standard deviation

# Test data is *not* used when calculating the mean and std
mean = train_data.mean(axis=0)
std = train_data.std(axis=0)
train_data = (train_data - mean) / std
test_data = (test_data - mean) / std

print(train_data[0])  # First normalized training sample
# We will build a sequential model with 2 densely connected hidden layers and an output layer which returns a continuous value
def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation=tf.nn.relu, input_shape=(train_data.shape[1],)),
        tf.keras.layers.Dense(64, activation=tf.nn.relu),
        tf.keras.layers.Dense(1)
    ])
    
    optimizer = tf.train.RMSPropOptimizer(0.001)
    model.compile(loss='mse', optimizer=optimizer, metrics=['mae'])
    return model

model = build_model()
model.summary()
# Display training progress by printing a single dot for each completed epoch
class PrintDot(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0:
            print('')
        print('.', end='')

EPOCHS = 500

# Store training stats
history = model.fit(train_data, train_labels, epochs=EPOCHS, validation_split=0.2, verbose=0, callbacks=[PrintDot()])
# let's visualize the model progress using stats in history object
def plot_history(history):
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [1000$]')
    plt.plot(history.epoch, np.array(history.history['mean_absolute_error']), label='Train Loss')
    plt.plot(history.epoch, np.array(history.history['val_mean_absolute_error']), label = 'Val loss')
    plt.legend()
    plt.ylim([0, 5])

plot_history(history)
# Let's automatically stop training when validation score doesn't improve.
# We will use a callback that tests a training condition for every epoch. If set amounts of epochs elapse without showing improvements, then we automatically stop training.
model = build_model()

# The patience parameter is the amount of epochs to check for improvement
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)

history = model.fit(train_data, train_labels, epochs=EPOCHS,
                    validation_split=0.2, verbose=0,
                    callbacks=[early_stop, PrintDot()])

plot_history(history)
# Let's see how did the model perform using test set
[loss, mae] = model.evaluate(test_data, test_labels, verbose=0)
print("Testing set Mean Absolute Error: ${:7.2f}".format(mae * 1000))

test_predictions = model.predict(test_data).flatten()

plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [1000$]')
plt.ylabel('Predictions [1000$]')
plt.axis('equal')
plt.xlim(plt.xlim())
plt.ylim(plt.ylim())
_ = plt.plot([-100, 100], [-100, 100])
error = test_predictions - test_labels
plt.hist(error, bins = 50)
plt.xlabel("Prediction Error [1000$]")
_ = plt.ylabel("Count")
print('Starting model export process')

# take these as program arguments
MODEL_OUTPUT_PATH = './'
MODEL_VERSION = 1


export_path = os.path.join(tf.compat.as_bytes(MODEL_OUTPUT_PATH), tf.compat.as_bytes(str(MODEL_VERSION)))
print('Cleaning directory and exporting model to', export_path)
shutil.rmtree(export_path)
builder = tf.saved_model.builder.SavedModelBuilder(export_path)

