import numpy as np

import pandas as pd

import tensorflow as tf



train = pd.read_csv("../input/titanic/train.csv")

test = pd.read_csv("../input/titanic/test.csv")
train["Sex"][train["Sex"] == "male"] = 0

train["Sex"][train["Sex"] == "female"] = 1

test["Sex"][test["Sex"] == "male"] = 0

test["Sex"][test["Sex"] == "female"] = 1

train["Age"] = train["Age"].fillna(train["Age"].median())

test["Age"] = test["Age"].fillna(test["Age"].median())

test["Fare"] = test["Fare"].fillna(test["Fare"].median())
train.head(5)
train_target = train.pop('Survived')

train.pop('PassengerId')

train.pop('Name')

train.pop('Parch')

train.pop('Ticket')

train.pop('Cabin')

train.pop('Embarked')

print(train)

print(train_target)
train_array = np.asarray(train.values)

target_array = np.asarray(train_target.values)

print(train_array)

print(target_array)
dataset = tf.data.Dataset.from_tensor_slices((train_array.astype('float32'), target_array))
model = tf.keras.Sequential([

    tf.keras.layers.Dense(10, activation='relu'),

    tf.keras.layers.Dense(10, activation='relu'),

    tf.keras.layers.Dense(1, activation='sigmoid')

])



model.compile(optimizer='adam',

              loss='binary_crossentropy',

              metrics=['accuracy'])



model.fit(train_array.astype('float32'), target_array, epochs=15)
test.pop('PassengerId')

test.pop('Name')

test.pop('Parch')

test.pop('Ticket')

test.pop('Cabin')

test.pop('Embarked')

test_array = np.asarray(test.values)

y = model.predict(test_array.astype('float32'))

z = []

for p in y:

    if p[0] < 0.5:

        z.append(0)

    else:

        z.append(1)

Z = np.array(z)

print(Z.shape)

print(Z)
test = pd.read_csv("../input/titanic/test.csv")

PassengerId = np.array(test["PassengerId"]).astype(int)

my_solution = pd.DataFrame(Z, PassengerId, columns = ["Survived"])

my_solution.to_csv("my_prediction.csv", index_label = ["PassengerId"])