import numpy as np
import pandas as pd 
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# for preprocessing the data:
from sklearn.preprocessing import LabelEncoder, StandardScaler
from category_encoders.cat_boost import CatBoostEncoder
from sklearn.model_selection import train_test_split

# importing the neural network libraries:
from keras.optimizers import *
from keras.losses import binary_crossentropy
from keras.models import Sequential
from keras.layers import Dense

# importing a classifier from xgboost:
from xgboost import XGBClassifier

# importing metrics to measure our accuracy:
from sklearn.metrics import accuracy_score
data = pd.read_csv("../input/tictactoe/tic-tac-toe.csv")
data.head()
data.isnull().sum()
y = data['class']
data.drop(['class'], inplace=True, axis=1)
label = LabelEncoder()

y = label.fit_transform(y)
cbe = CatBoostEncoder()
data = cbe.fit_transform(data, y)
train, test, ytrain, ytest = train_test_split(data, y,
                                              test_size=0.4, train_size=0.6)
model = Sequential([
    Dense(256, activation='relu', input_shape=(9,)),
    Dense(128, activation='relu'),
    Dense(128, activation='relu'),
    Dense(32, activation='relu'),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])
model.compile(metrics=['accuracy'], loss='binary_crossentropy', optimizer='Adam')
model.fit(train, ytrain, epochs=40,
          validation_data=(test, ytest))
xg = XGBClassifier(n_estimators=350)

xg.fit(train, ytrain)

xgPreds = xg.predict(test)
accuracy_score(xgPreds, ytest)