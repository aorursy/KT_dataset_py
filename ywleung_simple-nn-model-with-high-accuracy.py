import os
print(os.listdir("../input"))
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
# load datasets
X_train = pd.read_csv("../input/train.csv")
X_train, Y_train = X_train.iloc[:, 2:], X_train.iloc[:, 1]

# convert text labels into one-hot vector
lb = preprocessing.LabelEncoder()
lb.fit(Y_train)
Y_oh = lb.transform(Y_train)
Y_oh = np.eye(len(np.unique(Y_oh)))[Y_oh]

# split datasets into random train and dev subset
X_train, X_dev, Y_train, Y_dev = train_test_split(X_train, Y_oh, test_size = 0.3)
from keras.layers import Input, Dense, Activation, Dropout, BatchNormalization
from keras.models import Model
input_shape = X_train.shape[-1]
output_shape = Y_oh.shape[-1]

X_input = Input((input_shape, ))
X = Dense(128, activation = "relu")(X_input)
X = BatchNormalization()(X)
X = Dropout(0.5)(X)
X_output = Dense(output_shape, activation = "softmax")(X)

model = Model(inputs = X_input, outputs = X_output)
model.summary()
model.compile(loss = "categorical_crossentropy",
              optimizer = "adam",
              metrics = ["accuracy"])
train_history = model.fit(X_train, Y_train, epochs = 100)
plt.subplot(1, 2, 1)
plt.plot(train_history.history["acc"])
plt.title("Accuracy")
plt.xlabel("epoch")

plt.subplot(1, 2, 2)
plt.plot(train_history.history["loss"])
plt.title("Loss")
plt.xlabel("epoch")
model.evaluate(X_dev, Y_dev)
# load test set
X_test = pd.read_csv("../input/test.csv")
X_test, X_id = X_test.iloc[:, 1:], X_test.iloc[:, 0]
# prediction
pred = model.predict(X_test)
# convert prediction into required format
df = pd.DataFrame(pred)
df.columns = lb.classes_
df = pd.concat([X_id, df], axis = 1)
# save as csv
df.to_csv("submission.csv", index = False)
