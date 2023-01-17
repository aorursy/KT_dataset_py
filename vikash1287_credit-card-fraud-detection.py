import pandas as pd
import numpy as np
import keras
dataset = pd.read_csv("../input/creditcard.csv")
dataset.head(2)
dataset.shape
from sklearn.preprocessing import  StandardScaler
dataset['NormalizedAmount']=StandardScaler().fit_transform(dataset["Amount"].values.reshape(-1,1))
dataset.head(1)
dataset = dataset.drop(["Amount"],axis=1)
dataset = dataset.drop(["Time"],axis=1)
x =dataset.iloc[:,dataset.columns !="Class"].values
y = dataset.iloc[:, dataset.columns == "Class"].values
x
y
from sklearn.model_selection import train_test_split
x_train ,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20,random_state=0)
x_train = np.array(x_train)
x_test = np.array(x_test)
y_train = np.array(y_train)
y_test = np.array(y_test)
from keras.models import  Sequential
from keras.layers import Dense
from keras.layers import Dropout
x.shape
classifier = Sequential([
    Dense(units =16,input_dim=29,activation='relu'),
    Dense(units =24,activation='relu'),
    Dense(units =20,activation='relu'),
    Dropout(0.5),
    Dense(units =21,activation='relu'),
    Dense(units =24,activation='relu'),
    Dense(1,activation='sigmoid')
])
classifier.summary()
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
classifier.fit(x_train,y_train,batch_size=30,epochs=5)

score = classifier.evaluate(x_test,y_test)
print(score)
#99% accuracy,great!






