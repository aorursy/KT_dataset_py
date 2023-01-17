# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from keras.models import Sequential

from keras.layers.core import Dense, Dropout, Activation

from keras.utils import np_utils, generic_utils

from keras.callbacks import EarlyStopping



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

df = pd.read_table("../input/train.csv", sep=',')

x = df.iloc[:,1:].values

y = df.iloc[:,:1].values



pre = pd.read_table("../input/test.csv", sep=',')

test = np.array(pre)



#Keras

X = np.array(x)

label = np_utils.to_categorical(y, 10)





model = Sequential()

model.add(Dense(400, init='uniform', input_dim=784))

model.add(Activation("relu"))

model.add(Dropout(0.5))



model.add(Dense(400, init='uniform', input_dim=300))

model.add(Activation("relu"))

model.add(Dropout(0.5))





model.add(Dense(10))

model.add(Activation("softmax"))



early_stopping = EarlyStopping(monitor='categorical_accuracy', patience=50)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=["categorical_accuracy"])

model.fit(X, label, nb_epoch=500, batch_size=1000, validation_split=0.3, callbacks=[early_stopping])



preds = model.predict_classes(test)



final = pd.DataFrame({"ImageId":range(1, len(list(preds))+1), "Label":list(preds)})       



final.to_csv("keras.csv", index=None)



# Any results you write to the current directory are saved as output.