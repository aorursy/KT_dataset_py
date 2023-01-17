import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import itertools
warnings.filterwarnings("ignore")
%matplotlib inline
from PIL import Image
data = pd.read_csv(r"../input/predicting-a-pulsar-star/pulsar_stars.csv")
data.head(10)
data.info()
data.describe()
data.shape
data.isnull().sum()
dt = data['target_class'].value_counts()
print(dt)
sns.pairplot(data, hue='target_class',palette='cubehelix',kind = 'reg')
plt.figure(figsize=(16,10))

plt.subplot(2,2,1)
sns.violinplot(data=data,y=" Mean of the integrated profile",x="target_class")
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
data_x = data.drop(["target_class"],axis=1)

scaler = MinMaxScaler(feature_range=(0, 1))
X = scaler.fit_transform(data_x)

X_train, X_val_and_test, Y_train, Y_val_and_test = train_test_split(X,data["target_class"].values,random_state = 42,test_size= 0.15)
X_val, X_test, Y_val, Y_test = train_test_split(X_val_and_test, Y_val_and_test, test_size=0.5)
print(X_train.shape, X_val.shape, X_test.shape, Y_train.shape, Y_val.shape, Y_test.shape)

print("X_train shape: {}, X_test: {}".format(X_train.shape,X_test.shape))
print("Y_train shape: {}, Y_test: {}".format(Y_train.shape,Y_test.shape))
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, input_dim=X_train.shape[1], activation=tf.nn.relu, name='Input'),
    tf.keras.layers.Dense(16, activation=tf.nn.relu),
    tf.keras.layers.Dense(32, activation=tf.nn.relu),
    tf.keras.layers.Dense(64, activation=tf.nn.relu),
    tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)])

#Once the model is created, you can config the model with 
#losses and metrics with model.compile(), train the model 
#with model.fit(), or use the model to do prediction with model.predict().

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#Here we are first feeding the training data(Xtrain) and training labels(Ytrain).
#We then use Keras to allow our model to train for 200 epochs on a batch_size of 32.
hist = model.fit(X_train, Y_train, epochs=200,validation_data=(X_val,Y_val),batch_size=32)
model.evaluate(X_test, Y_test)
prediction = model.predict(X_test)
prediction = (prediction > 0.5).astype('int')
print(prediction)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, prediction)
print(cm)
sns.heatmap(cm,annot=True,fmt="d")
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model Loss')
plt.legend(['Train', 'Val'])
plt.show()
import tensorflow as tf
model_1 = tf.keras.Sequential([
    tf.keras.layers.Dense(16, input_dim=X_train.shape[1], activation=tf.nn.relu, name='Input'),
    tf.keras.layers.Dense(16, activation=tf.nn.relu),
    tf.keras.layers.Dense(32, activation=tf.nn.relu),
    tf.keras.layers.Dense(64, activation=tf.nn.relu),
    tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)])

model_1.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#Train for 100 epochs on a batch_size of 32.
hist_1 = model_1.fit(X_train, Y_train, epochs=100,validation_data=(X_val,Y_val),batch_size=32)
model_1.evaluate(X_test, Y_test)
prediction_1 = model_1.predict(X_test)
prediction_1 = (prediction_1 > 0.5).astype('int')
print(prediction_1)
from sklearn.metrics import confusion_matrix
cm_1 = confusion_matrix(Y_test, prediction_1)
print(cm_1)
sns.heatmap(cm_1,annot=True,fmt="d")
plt.plot(hist_1.history['loss'])
plt.plot(hist_1.history['val_loss'])
plt.title('Model Loss')
plt.legend(['Train', 'Val'])
plt.show()
model_2 = tf.keras.Sequential([
    tf.keras.layers.Dense(16, input_dim=X_train.shape[1], activation=tf.nn.relu, name='Input'),
    tf.keras.layers.Dense(16, activation=tf.nn.relu),
    tf.keras.layers.Dense(32, activation=tf.nn.relu),
    tf.keras.layers.Dense(64, activation=tf.nn.relu),
    tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)])

model_2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#Train for 100 epochs on a batch_size of 10.
hist_2 = model_2.fit(X_train, Y_train, epochs=100,validation_data=(X_val,Y_val),batch_size=10)
model_2.evaluate(X_test, Y_test)
prediction_2 = model_2.predict(X_test)
prediction_2 = (prediction_2 > 0.5).astype('int')
print(prediction_2)
from sklearn.metrics import confusion_matrix
cm_2 = confusion_matrix(Y_test, prediction_2)
print(cm_2)
sns.heatmap(cm_2,annot=True,fmt="d")
plt.plot(hist_2.history['loss'])
plt.plot(hist_2.history['val_loss'])
plt.title('Model Loss')
plt.legend(['Train', 'Val'])
plt.show()