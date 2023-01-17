import pandas

import keras

from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation

from keras.optimizers import SGD

from keras.preprocessing import text

import numpy as np
g = pandas.read_csv("../input/kc_house_data.csv",

                    encoding = "ISO-8859-1")

g["price"] = g["price"]/1000





X = g[["sqft_above","sqft_basement","sqft_lot","sqft_living","floors",

       "bedrooms","yr_built","lat","long","bathrooms"]].values

Y = g["price"].values

zipcodes        = pandas.get_dummies(g["zipcode"]).values

condition       = pandas.get_dummies(g["condition"]).values

grade           = pandas.get_dummies(g["grade"]).values



X = np.concatenate((X,zipcodes),axis=1)

X = np.concatenate((X,condition),axis=1)

X = np.concatenate((X,grade),axis=1)
model = Sequential()

model.add(Dense(50, input_dim=97, init='normal', activation='relu'))

model.add(Dense(5, init='normal',activation="relu"))

model.add(Dropout(0.05))

model.add(Dense(5, init='normal',activation="relu"))

model.add(Dense(1, init='normal'))





sgd = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)



model.compile(loss='mae',

              optimizer=sgd,  

              metrics=["mae"])  

 

model.fit(X,Y,nb_epoch=300,verbose=2)