import pandas as pd

import matplotlib.pyplot as plt

from keras.models import Sequential

from keras import layers

from sklearn.model_selection import train_test_split

from keras import regularizers
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

print(train.shape, test.shape)
from sklearn.preprocessing import scale



y = train['target']

train = train.drop(['target', 'ID_code'], axis=1)

id_test = test['ID_code']

test = test.drop(['ID_code'], axis=1)



# Scaling the data:

train = scale(train)

test = scale(test)



x_train, x_test, y_train, y_test = train_test_split(train, y, test_size=0.25,

                                                    random_state=42)
from keras.models import Sequential

from keras import layers



input_dim = train.shape[1]

print('Input dimension =', input_dim)



model = Sequential()

model.add(layers.Dense(16, input_dim=input_dim, activation='relu', kernel_regularizer=regularizers.l2(0.005)))

model.add(layers.Dense(16, input_dim=input_dim, activation='relu', kernel_regularizer=regularizers.l2(0.005)))

model.add(layers.Dense(1, activation='sigmoid'))



model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

model.summary()
history = model.fit(x_train, y_train, epochs=3, validation_data=(x_test, y_test))
# save our prediction

prediction = model.predict(test)

pd.DataFrame({"ID_code":id_test,"target":prediction[:,0]}).to_csv('result_keras.csv',index=False,header=True)