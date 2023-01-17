import numpy as np 
import pandas as pd 
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()
X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] / 255.0
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                   "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
model = keras.Sequential()
model.add(keras.layers.Flatten(input_shape=[28,28]))
model.add(keras.layers.Dense(300,activation='relu'))
model.add(keras.layers.Dense(100,activation='relu'))
model.add(keras.layers.Dense(10,activation='softmax'))
model.summary()
model.get_weights()
keras.utils.plot_model(model)
model.compile(loss='sparse_categorical_crossentropy',optimizer='sgd',metrics='accuracy')
history = model.fit(X_train,y_train,epochs=30,validation_data=(X_valid,y_valid))
pd.DataFrame(history.history).plot(figsize=(8,5))
plt.grid(True)
plt.gca().set_ylim(0,1)
plt.show()
model.evaluate(X_test,y_test)
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
housing = fetch_california_housing()
X_train_full, X_test, y_train_full, y_test = train_test_split(
        housing.data, housing.target)
X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_full, y_train_full)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)
X_test_scaled = scaler.transform(X_test)
model = keras.models.Sequential([
        keras.layers.Dense(30, activation="relu", input_shape=(X_train_scaled.shape[1],)),
        keras.layers.Dense(50, activation="relu"),
        keras.layers.Dense(1)
    ])
keras.utils.plot_model(model)
model.compile(loss='mse',optimizer='sgd',metrics=['mse'])
history = model.fit(X_train_scaled, y_train, epochs=20,validation_data=(X_valid_scaled, y_valid))
pd.DataFrame(history.history).plot(figsize=(10,10))
plt.grid(True)
plt.show()
model.evaluate(X_test_scaled,y_test)
