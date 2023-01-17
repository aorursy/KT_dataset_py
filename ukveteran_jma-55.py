import pandas as pd

import numpy as np

from keras.utils import np_utils

from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation, Flatten

from sklearn.model_selection import train_test_split





d     = pd.read_csv("../input/mushrooms.csv")

index = 0

Q     = 0
for names in list(d):

    print(names)

    if (index > 0):

        X = pd.get_dummies(d[names]).values

        if (index == 1):

            Q = X

        else:

            Q = np.concatenate((Q,X),axis=1)

    index = index + 1



mushroom_class = pd.get_dummies(d["class"]).values

mushroom_class = mushroom_class[:,0]



X_train, X_test, y_train, y_test = train_test_split(Q, mushroom_class, test_size=0.332)





print(X_train.shape)



model = Sequential()

model.add(Dense(32, input_dim=117, init='uniform', activation='relu'))

model.add(Dropout(0.1))

model.add(Dense(15, activation='relu'))

model.add(Dropout(0.1))

model.add(Dense(1, activation='sigmoid'))

 

model.compile(loss='binary_crossentropy',

              optimizer='sgd',  

              metrics=["accuracy"])  



model.fit(X_train,y_train,nb_epoch=200,verbose=2,validation_data = (X_test,y_test))



results = model.evaluate(X_test,y_test)

print(results)