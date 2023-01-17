import pandas as pd

import numpy as np 

import tensorflow as tf

from sklearn.model_selection import train_test_split
train = pd.read_csv("../input/titanic/train.csv")

test = pd.read_csv("../input/titanic/test.csv")

train.head()
X = pd.get_dummies(train[["Sex", "Embarked"]]).join(train[["Pclass","SibSp","Parch","Age"]].replace(np.nan, 50))

y = train["Survived"].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y)

X_train.head()
model = tf.keras.models.Sequential([

    tf.keras.layers.Dense(50),

    tf.keras.layers.Dense(50, activation="relu"),

    tf.keras.layers.Dense(1, activation="sigmoid")

])
model.compile(

    optimizer='nadam', 

    metrics=[tf.keras.metrics.BinaryAccuracy()], 

    loss=tf.keras.losses.BinaryCrossentropy()

             )
early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=20)

model_checkpoint_cb = tf.keras.callbacks.ModelCheckpoint("Titanic_model.h5", save_best_only=True)

model.fit(X_train, y_train, epochs=100, validation_split=0.2, callbacks=[early_stopping_cb, model_checkpoint_cb])
model = tf.keras.models.load_model("Titanic_model.h5")

model.evaluate(X_test, y_test)
X_final_test = pd.get_dummies(test[["Sex", "Embarked"]]).join(test[["Pclass","SibSp","Parch", "Age"]].replace(np.nan, 25)).to_numpy()
y_final_test = np.round(model.predict(X_final_test))
return_ = pd.DataFrame(test["PassengerId"]).join(pd.DataFrame(data=y_final_test, columns=["Survived"], dtype=np.int8)).to_csv("out.csv", index=False)