import tensorflow as tf
import pandas as pd
import numpy as np


df_train = pd.read_csv("../input/digit-recognizer/train.csv")
df_test = pd.read_csv("../input/digit-recognizer/test.csv")

df_train

X = df_train.drop(labels=['label'],axis =1)
y = df_train['label'] 

from sklearn.model_selection import train_test_split
X_train , X_test , y_train ,y_test = train_test_split(X,y,test_size =.2)
X_train = tf.keras.utils.normalize(X_train,axis=1)
X_test = tf.keras.utils.normalize(X_test,axis=1)
model = tf.keras.models.Sequential()  # a basic feed-forward model
model.add(tf.keras.layers.Flatten()) 
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu)) 
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu)) 
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))



model.compile( optimizer='adam',
              loss= 'sparse_categorical_crossentropy' ,
              metrics = ['accuracy']
              )
model.fit(X_train.values,y_train.values,epochs=40)
model.evaluate(X_test,y_test)
prediction  = model.predict(df_test.values)

prediction = [np.argmax(prediction[i]) for i in range(prediction.shape[0])]

submit = pd.DataFrame({"ImageId": [i for i in range(1,28001)],
                      "Label":prediction})
submit.to_csv("submit2.csv",index=False)
import keras
from keras import metrics

prediction1  = model.predict(X_test.values)
prediction1 = [np.argmax(prediction1[i]) for i in range(prediction1.shape[0])]

keras.metrics.accuracy(prediction1, y_test)

