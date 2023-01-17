import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
train_df = pd.read_csv("../input/fashion-mnist_train.csv")
test_df = pd.read_csv("../input/fashion-mnist_test.csv")
train_df.head()
train_df.shape
test_df.head()
test_df.shape
train_df.info()
training = np.array(train_df,dtype='float32')
testing = np.array(test_df,dtype='float32')
plt.imshow(training[10,1:].reshape(28,28))
import random
i = random.randint(1,60000)
plt.imshow(training[i,1:].reshape(28,28))
label = training[i,0]
label
#Train the model
x_train = training[:,1:]/255
y_train = training[:,0]

x_test = testing[:,1:]/255
y_test = testing[:,0]
x_train
from sklearn.model_selection import  train_test_split
x1_train,x_validate,y1_train,y_validate = train_test_split(x_train,y_train,test_size =0.2,random_state = 6)
x1_train
x1_train = x1_train.reshape(x1_train.shape[0],*(28,28,1))
x_test = x_test.reshape(x_test.shape[0],*(28,28,1))
x_validate = x_validate.reshape(x_validate.shape[0],*(28,28,1))
x1_train
import keras
from keras.models import  Sequential
from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
classifier = Sequential()
classifier.add(Conv2D(128,3,3,input_shape=(28,28,1),activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Flatten())
classifier.add(Dense(output_dim=64,activation='relu'))
classifier.add(Dense(output_dim=10,activation='sigmoid'))

#compile
classifier.compile(loss='sparse_categorical_crossentropy',optimizer=Adam(lr=0.01),metrics=['accuracy'])
epochs =50

classifier.fit(x1_train,y1_train,
              batch_size=512,
              nb_epoch=epochs,
              verbose =1,
              validation_data = (x_validate,y_validate))
evaluation = classifier.evaluate(x_test,y_test)
print("Test Accuracy :{:0.3f}".format(evaluation[1]))
predicted_classes = classifier.predict_classes(x_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,predicted_classes)
plt.figure(figsize=(14,10))
sns.heatmap(cm,annot=True)
