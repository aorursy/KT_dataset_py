import tensorflow as tf

import numpy as np

from tensorflow import keras

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



data=pd.read_excel("../input/datmodel/model.xlsx")

data=data.dropna()

print("Vstupni data")

print(data)



X=np.array(data["X"].values.tolist()).ravel()

y=np.round(np.array(data["Y"].values.tolist()).ravel()).astype("int32")

Start=keras.Input(shape=(1) ,name="main")

Model1=Start

Model1=keras.layers.Dense(200)(Model1)

Model1 = keras.activations.relu(Model1)

Model1 = tf.keras.layers.Dropout(0.5)(Model1)

Model1=keras.layers.Dense(200)(Model1)

Model1 = keras.activations.relu(Model1)

Model1 = tf.keras.layers.Dropout(0.5)(Model1)

Model1=keras.layers.Dense(100)(Model1)

Model1 = keras.activations.relu(Model1)

Model1 = tf.keras.layers.Dropout(0.5)(Model1)

Model1=keras.layers.Dense(100)(Model1)

Model1 = keras.activations.relu(Model1)

Model1 = tf.keras.layers.Dropout(0.5)(Model1)

Model1=keras.layers.Dense(1)(Model1)

Model1 = keras.activations.relu(Model1)

mc = keras.Model(inputs=Start, outputs=Model1)

optmain=tf.keras.optimizers.Adam()

lomain = tf.keras.losses.MeanSquaredError()

mc.compile(loss=lomain, optimizer=optmain,metrics=['accuracy'])

mc.load_weights("../input/datmodel/outmodel.hdf5")

#mc.fit(X,y,epochs=20000)

plt.figure(figsize=(15, 10))

print("model out")

print(X)

print(mc.predict(X))

print("graf")

X_test = np.arange(0.0, 500, 1)[:, np.newaxis]

y_1 = mc.predict(X_test)

plt.scatter(X_test, y_1 ,s=10, edgecolor="black",

            c="blue", label="pred")

plt.scatter(X, y, s=35, edgecolor="black",

            c="darkorange", label="data")

plt.xlabel("data")

plt.ylabel("target")

plt.title("model")

plt.legend()

plt.show()

#mc.save("outmodel.hdf5")


