import tensorflow as tf
from tensorflow.keras import layers, models
import pandas as pd
import numpy as np
df=pd.read_csv("../input/diabetes/diabetes.csv")
x=df.drop("Outcome",axis=1)
x
from sklearn.preprocessing import scale
x=scale(x)
y=df.Outcome.values
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2)
model = tf.keras.Sequential([
  tf.keras.layers.Dense(12, activation="relu", input_shape=(8,)),  
  tf.keras.layers.Dense(8, activation="relu"),
  tf.keras.layers.Dense(1,activation="sigmoid")
])
from tensorflow.keras.optimizers import Adam
model.compile(optimizer = Adam(lr=0.001), 
              loss = 'binary_crossentropy', 
              metrics=['accuracy'])
history = model.fit(xtrain, 
                    ytrain,
                   epochs =100,batch_size=40)
acc=model.evaluate(xtest,ytest)
print(acc)
#model.save("../working/")
import matplotlib.pyplot as plt
acc = history.history['accuracy']
loss1 =  history.history['loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.title('Training  accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss1, 'r', label='Training Loss')
plt.title('Training  loss')
plt.legend()

plt.show()
#import keras
#model1 = keras.models.load_model('../working/')
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2)
model1=tf.keras.Sequential([
      tf.keras.layers.Dense(12,activation='relu',input_shape=(8,)),
      tf.keras.layers.Dense(8,activation='relu'),
      tf.keras.layers.Dense(1,activation='sigmoid')
])
from tensorflow.keras.optimizers import SGD
model1.compile(optimizer = SGD(lr=0.001), 
              loss = 'binary_crossentropy', 
              metrics=['accuracy'])
history = model1.fit(xtrain, 
                    ytrain,
                   epochs =100,batch_size=15)

acc=model1.evaluate(xtest,ytest)
print(acc)
import matplotlib.pyplot as plt
acc = history.history['accuracy']
loss2 =  history.history['loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.title('Training  accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss2, 'r', label='Training Loss')
plt.title('Training  loss')
plt.legend()

plt.show()
model2=tf.keras.Sequential([
      tf.keras.layers.Dense(12,activation='relu',input_shape=(8,)),
      tf.keras.layers.Dense(8,activation='relu'),
      tf.keras.layers.Dense(1,activation='sigmoid')
])
from tensorflow.keras.optimizers import RMSprop
model2.compile(optimizer = RMSprop(lr=0.001), 
              loss = 'binary_crossentropy', 
              metrics=['accuracy'])
history = model2.fit(xtrain, 
                    ytrain,
                   epochs =100,batch_size=40)
acc1=model2.evaluate(xtest,ytest)
print(acc1)
import matplotlib.pyplot as plt
acc = history.history['accuracy']
loss3 =  history.history['loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.title('Training  accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss3, 'r', label='Training Loss')
plt.title('Training  loss')
plt.legend()

plt.show()
import matplotlib.pyplot as plt
plt.plot(epochs, loss1, 'r', label='Adam')
plt.plot(epochs, loss2, 'g', label='SGD')
plt.plot(epochs, loss3, 'b', label='RMSprop')
plt.legend()
plt.show()