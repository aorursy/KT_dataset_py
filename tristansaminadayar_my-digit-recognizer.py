import numpy as np

import pandas as pd

import tensorflow as tf

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import random
dataset = pd.read_csv("../input/digit-recognizer/train.csv")

dataset.head()
y = dataset.label.to_numpy()

X = dataset[["pixel" + str(i) for i in range(784)]].to_numpy().reshape(-1, 28, 28, 1) / 255

X.shape
X_train, X_test, y_train, y_test = train_test_split(X, y)
fig = plt.figure(figsize=(10,10))

for i in range(10):

  for j in range(10):

    num = random.randint(0, 31500)

    sub = plt.subplot(10,10, i*10+j+1)

    sub.imshow(X_train[num].reshape(28, 28), cmap='binary')

    sub.set_title(y_train[num], fontsize=10)

    sub.axis("off")

plt.tight_layout()

plt.savefig("plt.png")

plt.show()
model = tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(32, 3, activation="relu",input_shape=[28, 28, 1]),

    tf.keras.layers.MaxPooling2D(2),

    tf.keras.layers.Conv2D(64, 3, activation="relu", padding="SAME"),

    tf.keras.layers.MaxPooling2D(2),

    tf.keras.layers.Conv2D(128, 3, activation="relu", padding="SAME"),

    tf.keras.layers.MaxPooling2D(2),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(64, activation="relu"),

    tf.keras.layers.Dense(32, activation="relu"),

    tf.keras.layers.Dense(10, activation = "softmax")    

])
model.compile(loss="sparse_categorical_crossentropy",

              optimizer="nadam", 

              metrics=["accuracy"])

early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True)

history = model.fit(X_train.reshape(-1, 28, 28, 1) , y_train, epochs=100, validation_split = 0.2, 

                    callbacks=[early_stopping_cb])
model.evaluate(X_train, y_train)
dataset = pd.read_csv("../input/digit-recognizer/test.csv")

X_final = dataset[["pixel" + str(i) for i in range(784)]].to_numpy().reshape(-1, 28, 28, 1)

y_final = model.predict(X_final)



ret = np.zeros(shape=(28000), dtype=np.int8)

for i in range(28000):

    ret[i] = np.where(y_final[i] == np.max(y_final[i]))[0]

ret
fig = plt.figure(figsize=(10,10))

for i in range(10):

  for j in range(10):

    num = random.randint(0, 2033)

    sub = plt.subplot(10,10, i*10+j+1)

    sub.imshow(X_final[num].reshape(28, 28), cmap='binary')

    sub.set_title(str(ret[num]), fontsize=10)

    sub.axis("off")

plt.tight_layout()

plt.savefig("plt_predict.png")

plt.show()
pd.DataFrame(zip([i for i in range(1, y_final.shape[0] + 1)],ret), columns=["ImageId", "Label"]).to_csv("out.csv", index=False)