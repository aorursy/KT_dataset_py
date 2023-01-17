import numpy as np
import pandas as pd
import tensorflow as tf
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
print(train.shape)
print(test.shape)
train.head()
test.head()
y = train['label'].values
y.shape
X = train.iloc[:, 1:].values
X.shape
X = X / 255
model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(128, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(128, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(X, y, epochs=10)
prediction = model.predict(test.values)
prediction.shape
single = [np.argmax(_) for _ in prediction]
result = pd.DataFrame({
    "ImageId": list(range(1,len(prediction)+1)), 
    "Label": single
})
result.to_csv("submission.csv", index=False, header=True)
