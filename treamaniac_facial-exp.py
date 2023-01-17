import tensorflow as tf
import pandas as pd
import numpy as np
import os
#import seaborn as sn
import matplotlib.pyplot as pl
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D,MaxPool2D,Flatten,Dropout,BatchNormalization
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
print(len(tf.config.experimental.list_physical_devices('GPU')))
for dirname,_,filenames in os.walk(r'C:\Users\aman7\OneDrive\Desktop\Coding ninjas assignment 1\assignments\kaggle facial exp',topdown=True):
    for filename in filenames:
        print(os.path.join(dirname,filename))
#image is 48x48x1
train_data=pd.read_csv('/kaggle/input/challenges-in-representation-learning-facial-expression-recognition-challenge/train.csv')
test_data= pd.read_csv('/kaggle/input/challenges-in-representation-learning-facial-expression-recognition-challenge/test.csv')
icml=pd.read_csv('/kaggle/input/challenges-in-representation-learning-facial-expression-recognition-challenge/icml_face_data.csv')
#combined dataset with training, public test, private test
icml.tail()
icml[' Usage'].value_counts()
train_data.head()
#consist of both private and public test
test_data.tail()
x_train,x_test,y_train,y_test=[],[],[],[]
for row in train_data.values:
    value=row[0]
    pixels=row[1].split(' ')
    x_train.append(np.array(pixels,'float32'))
    y_train.append(value)

for row in test_data.values:
    pixels=row[0].split(' ')
    x_test.append(np.array(pixels,'float32'))
y_test=icml[(icml[' Usage']=='PublicTest')|(icml[' Usage']=='PrivateTest')]['emotion'].values
x_train=np.array(x_train,'float32')
y_train=np.array(y_train,'float32')
x_test=np.array(x_test,'float32')
y_test=np.array(y_test,'float32')
x_test.shape,x_train.shape,y_train.shape,y_test.shape
x_train_f=x_train.reshape(x_train.shape[0],48,48,1)
x_test_f=x_test.reshape(x_test.shape[0],48,48,1)
#final values
x_train_f/=255
x_test_f/=255
y_test=to_categorical(y_test)
y_train=to_categorical(y_train)
y_test
pl.figure(figsize=(5,5))
pl.imshow(x_train_f[1].reshape(48,48))
pl.show()
input_width=48
input_height=48
n_channels=1
input_pixels=2304
n_cov1=64
n_cov2=128
cov1_k=10
cov2_k=10
stride_cov1=1
stride_cov2=1
max_pool_k1=2
max_pool_k2=2
n_hidden=1024
n_out=7
epochs = 40
input_shape = (48,48,1)
batch=100
#step 2 architecture
model=Sequential()
cv1=Conv2D(filters=n_cov1,kernel_size=(3,3),strides=(stride_cov1,stride_cov1),padding='same',activation='relu',input_shape=input_shape)
model.add(cv1)
model.add(MaxPool2D((max_pool_k1,max_pool_k1)))
model.add(Dropout(0.25))

model.add(Conv2D(filters=n_cov2,kernel_size=(3,3),strides=(stride_cov2,stride_cov2),padding='same',activation='relu'))
model.add(MaxPool2D((max_pool_k2,max_pool_k2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters=256,kernel_size=(3,3),strides=(stride_cov2,stride_cov2),padding='same',activation='relu'))
model.add(MaxPool2D((max_pool_k2,max_pool_k2)))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(units=n_hidden,activation='relu'))
model.add(Dropout(0.3))


model.add(Dense(units=n_out,activation='softmax'))



model.compile(optimizer=keras.optimizers.Adam(),loss=keras.losses.categorical_crossentropy,metrics=['accuracy'])

print(model.summary())

es=keras.callbacks.EarlyStopping(monitor='val_accuracy',patience=5,min_delta=0.001)

history3=model.fit(x_train_f,y_train,epochs=epochs,batch_size=batch,validation_data=(x_test_f,y_test),callbacks=[es])
history3
accuracy = history3.history['accuracy']
val_accuracy = history3.history['val_accuracy']
loss = history3.history['loss']
val_loss = history3.history['val_loss']
epochs = range(len(accuracy))

pl.plot(epochs, accuracy, 'bo', label='Training accuracy')
pl.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
pl.title('Training and validation accuracy')
pl.legend()
pl.figure()

pl.plot(epochs, loss, 'bo', label='Training loss')
pl.plot(epochs, val_loss, 'b', label='Validation loss')
pl.title('Training and validation loss')
pl.legend()
pl.show()
y_pred=model.predict(x_test_f)
y_pred_class=model.predict_classes(x_test_f)
Exp = ["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"]
for index in range(50):
    print('predicted result:',Exp[y_pred_class[index]])
    print('actual result:',Exp[np.argmax(y_test[index])])
    pl.figure(figsize=(3,3))
    pl.imshow(x_test_f[index].reshape(48,48))
    pl.show()
from sklearn.metrics import classification_report as cr,confusion_matrix as cm
print(cr(np.argmax(y_test,axis=1),y_pred_class,target_names=Exp))
print(cm(np.argmax(y_test,axis=1),y_pred_class))
print(Exp)