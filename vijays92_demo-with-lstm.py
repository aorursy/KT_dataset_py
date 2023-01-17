import numpy as np

import matplotlib.pyplot as plt

import keras

from sklearn.model_selection import train_test_split
from keras.models import Sequential

from keras.layers import Dense,LSTM

from keras.losses import categorical_crossentropy,mean_absolute_error

from keras.optimizers import SGD, adam
data = [[[(i+j)/100] for i in range(5)] for j in range(100)]
target= [(i+5)/100 for i in range(100)]
data= np.array(data,dtype=float)

target= np.array(target,dtype=float)
data.shape,target.shape
x_train,x_test,y_train,y_test= train_test_split(data,target,test_size=0.2,random_state=4)
model= Sequential()

model.add(LSTM((1),batch_input_shape=(None,5,1),return_state=False))
model.summary()
model.compile(optimizer=SGD(),loss=mean_absolute_error,metrics=['acc'])
model_history1= model.fit(data,target,epochs=50)
result=model.predict(x_test)
plt.scatter(range(20),result,c='g')

plt.scatter(range(20),y_test,c='r')
plt.plot(model_history1.history['loss'],c='g',label='Loss')

plt.legend()

model_history= model.fit(data,target,epochs=1000)
result1=model.predict(x_test)
plt.scatter(range(20),result1,c='g')

plt.scatter(range(20),y_test,c='r')
plt.plot(model_history.history['loss'],c='g',label='Loss')

plt.legend()