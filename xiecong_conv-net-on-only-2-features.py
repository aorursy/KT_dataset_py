%matplotlib inline

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib

from sklearn import metrics

from sklearn.neighbors import NearestNeighbors



from keras.models import Sequential, Model

from keras.layers import Dense, Dropout, Convolution2D, MaxPooling2D, Flatten, Input

from keras.optimizers import adam

from keras.utils.np_utils import to_categorical



import seaborn as sns



%config InlineBackend.figure_format = 'retina'
train = pd.read_csv("../input/train.csv")
X_train = train.iloc[:,1:].values

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1) #reshape to rectangular

X_train = X_train/255 #pixel values are 0 - 255 - this makes puts them in the range 0 - 1



y_train = train["label"].values
y_ohe = to_categorical(y_train)
model = Sequential()

model.add(Convolution2D(32, 3, 3, input_shape = (28, 28, 1), activation="relu"))

model.add(Convolution2D(32, 3, 3, activation="relu"))

model.add(MaxPooling2D(pool_size=(2,2)))



model.add(Convolution2D(32, 3, 3, activation="relu"))

#model.add(Convolution2D(32, 3, 3, activation="relu"))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Flatten())



model.add(Dense(128, activation = "relu"))

model.add(Dense(16, activation = "relu"))

model.add(Dense(2))

model.add(Dense(10, activation="softmax"))
model.compile(loss='categorical_crossentropy', 

              optimizer = adam(lr=0.001), metrics = ["accuracy"])
hist = model.fit(X_train, y_ohe,

          validation_split = 0.05, batch_size = 128, nb_epoch = 7)
#getting the 2D output:

output = model.get_layer("dense_3").output

extr = Model(model.input, output)
matplotlib.rcParams['figure.figsize'] = (10.0, 10.0)



X_proj = extr.predict(X_train[:10000])

X_proj.shape



proj = pd.DataFrame(X_proj[:,:2])

proj.columns = ["comp_1", "comp_2"]

proj["labels"] = y_train[:10000]



sns.lmplot("comp_1", "comp_2",hue = "labels", data = proj, fit_reg=False)