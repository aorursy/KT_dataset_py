import pandas as pd

import numpy as np
train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')
train_df = train_df.replace(["male", "female"], [0, 1]).fillna(0)

test_df = test_df.replace(["male", "female"], [0, 1]).fillna(0)
train_y = train_df[["Survived"]]

interest_columns = ["Sex", "Age", "Fare"]

col_num = len(interest_columns)

train_x = train_df[interest_columns]

test_x = test_df[interest_columns]
y = train_y.astype(np.float32).values

x = train_x.astype(np.float32).values



x_test = test_x.astype(np.float32).values
print(x.shape)

print(y.shape)

print(x_test.shape)
from keras.models import Sequential, load_model

from keras.layers import Dense, Activation

from sklearn.model_selection import train_test_split
nn_in_train, nn_in_test, nn_out_train, nn_out_test = train_test_split(x, y, test_size=.5)
print(nn_in_train.shape)

print(nn_in_test.shape)
np.random.seed(2)

model = Sequential()



model.add(Dense(2, input_shape=(col_num,)))

model.add(Activation("linear"))



model.add(Dense(2))

model.add(Activation("relu"))



output_num = 1 # One value representing if the passenger survived

model.add(Dense(output_num))

model.add(Activation("sigmoid"))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(nn_in_train, nn_out_train, nb_epoch=100, batch_size=50)
prediction = np.round(model.predict(nn_in_test))
np.sum(nn_out_test == prediction)/nn_out_test.shape[0]

to_kaggle = pd.DataFrame(np.round(model.predict(x_test)))
result = pd.concat([test_df[['PassengerId']], to_kaggle], axis=1)

result.columns = ["PassengerId", "Survived"]

result.Survived = result.Survived.astype(int)
result.to_csv("result.csv", index=False) # If we save the index, it adds an additional column