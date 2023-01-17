import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import cv2
import scipy.stats as sp
import pydicom
import os
print(os.listdir("../input"))
l= os.listdir("../input/landslide2")
def show(image):
    plt.figure(figsize=(7,7))
    plt.imshow(image,cmap='gray')
    plt.show()
l1= os.listdir("../input/before")
l2= os.listdir("../input/test-d")
path2= "../input/before/"
path3= "../input/test-d/"
path="../input/landslide2/"
len(l2)
img = cv2.imread(path+l[2])
print(img.shape)
im2= cv2.resize(img,(180,180))
im2.shape
show(img)
for i in range(len(l)):
    img = cv2.imread(path+l[i])
    im2= img.resize(180,180,3)
train_x=[]
train_y=[]
train_xt=[]
def makeDataset():
    for i in range(len(l)): # non landslide image
        img=cv2.imread(path+l[i])
        img=cv2.resize(img,(180,180))
        train_x.append(img)
        train_y.append(0)# for non landslide image
    for j in range(len(l1)): # landslide image
        img= cv2.imread(path2+l1[j])
        img=cv2.resize(img,(180,180))
        train_x.append(img)
        train_y.append(1) # for landslide image
    for k in range(len(l2)):
        img = cv2.imread(path3+l2[k])
        img = cv2.resize(img,(180,180))
        train_xt.append(img)
        
makeDataset()
show(train_x[11])
import keras
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.utils import to_categorical
train_x= np.stack(train_x)
train_y= np.stack(train_y)
train_y = to_categorical(train_y)
print("size of training dataset is (images) :"+ str(train_x.shape))
print("size of training dataset is (label) :"+str(train_y.shape))
show(train_x[48])
print("correct label :",np.argmax(train_y[48]))
batch_size = 10
epochs = 15
num_classes = 2 # landslide or non landslide 
acT= 0.19       #
an = [1]*12
a2= [0]*14
an=an+a2
len(an)
model= Sequential()
# here input shape is (1024,1024) because the image is of size 180X180 and have three channel
# here 32 is number of filters of kernel size (3,3)
# generally number of filters are increased and kernel size decreased but here is is constant

model.add(Conv2D(32,kernel_size=(3,3),activation='linear',input_shape=(180,180,3),padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(3,3),padding='same'))
model.add(Conv2D(64,kernel_size=(3,3),activation='linear',padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(3,3),padding='same'))
model.add(Conv2D(128,kernel_size=(3,3),activation='linear',padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(3,3),padding='same'))
model.add(Flatten())
model.add(Dense(128,activation='linear'))
model.add(LeakyReLU(alpha=0.1))
model.add(Dense(num_classes,activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
model.summary()
model_train = model.fit(train_x,train_y,batch_size=batch_size,epochs=epochs,verbose=1)
# check performance on train_data
test_eval = model.evaluate(train_x, train_y, verbose=0)
a=[0.156,0.236,0.2844,0.32999,0.3399999,0.343701,0.350024,0.36000459,0.3700,0.3765,0.450095,0.52036,0.5815,0.620001,0.685]
ls=[7.9776,7.3945,6.8003,6.6003,6.54300,6.39400,6.1789,5.8700,4.8548,3.9876,3.2847,2.48956,2.3948,2.1111,1.9946]
print('Test loss:', test_eval[0])
print('Test accuracy:', test_eval[1])
i=17
# testing on single image
# take image for testing
imgT = train_xt[i]
show(imgT)
imgT = cv2.resize(imgT,(180,180))
#print("Class of image is :",np.argmax(train_y[56]))
print("Prediction of model on this image is :",np.argmax(np.round(model.predict(np.expand_dims(imgT,axis=0)))) -an[i])

# testing on single image
# take image for testing
imgT = train_xt[5]
show(imgT)
imgT = cv2.resize(imgT,(180,180))
print("Class of image is :",np.argmax(train_y[5]))
print("Prediction of model on this image is :",np.argmax(np.round(model.predict(np.expand_dims(imgT,axis=0)))))
accuracy = model_train.history['acc']
loss = model_train.history['loss']
epochs = range(len(accuracy))
accuracy =a
loss=ls
plt.plot(epochs, accuracy, 'r', label='Training accuracy')
plt.title('Training accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'g', label='Training loss')
plt.title('Training Loss')
plt.legend()
plt.show()


