# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
veri=pd.read_csv("../input/0205veriler/veriler.csv")
veri
df2 = pd.DataFrame([["tr",127,33,10,"e"], ["tr",160,53,10,"k"],["tr",190,93,10,"e"]], columns=veri.columns)
veri=veri.append(df2)
veri = veri.reset_index(drop=True)
veri
x=veri.iloc[:,1:3].values
y=veri.iloc[:,4].values
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer

sc = StandardScaler()

x_sd = sc.fit_transform(x)

import matplotlib.pyplot as plt
yatay=veri.iloc[:,2].values
dikey=veri.iloc[:,1].values
labl = veri.iloc[:,4].values
color= ['red' if l == "k" else 'blue' for l in labl]
color[22:24]=["green","green","green"]
plt.scatter(yatay, dikey, color=color)
plt.show()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_yeni = le.fit_transform(y)

from keras.utils import np_utils

dummy_y=np_utils.to_categorical(y_yeni)
from sklearn.model_selection import train_test_split
#x_train, x_test,y_train,y_test = train_test_split(x,dummy_y,test_size=0.33, random_state=0, stratify=dummy_y)
x_train=x_sd[0:22,:]
x_test=x_sd[22:,:]
y_train=dummy_y[0:22,:]
y_test=dummy_y[22:,:]
from keras.models import Sequential

model = Sequential()
from keras.layers import Dense

model.add(Dense(units=2, activation='relu', input_dim=2))
model.add(Dense(units=8, activation='relu'))
model.add(Dense(units=2, activation='softmax'))
from keras import optimizers
optimizer=optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True)
optimizer2=optimizers.Adam(lr=0.01)
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer2,
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=200, batch_size=256)
y_pred=model.predict(x_test, batch_size=128)
y_pred
for i in y_pred:
    if (i[0]>i[1]):
        print("1 0")
    else:
        print("0 1")
y_test
model.summary()
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=1000, criterion = 'entropy')
rfc.fit(x_train,y_train)
rfc.predict(x_test)