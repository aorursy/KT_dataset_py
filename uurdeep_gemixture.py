# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

#GEM MIXTURE GENERATOR



import tensorflow as tf

from keras.models import Sequential

from keras.layers import Dense

from keras.callbacks import ModelCheckpoint



from keras.preprocessing.image import load_img

from keras.preprocessing.image import img_to_array

from keras.preprocessing.image import array_to_img

import matplotlib.pyplot as plt

#%% Data receive functions



def getData(pathd,shape):

    os.chdir(pathd)

    Alldatas=[]

    img_data=[]

    img_data=os.listdir(".")

    for folder in img_data:

        os.chdir(pathd+"/"+folder)

        for image in os.listdir("."):

            _,extension = os.path.splitext(image)

            if(extension==".jpg" or extension==".jpeg" or extension==".png"):

                img=load_img(image)

                img=img.resize((shape[0],shape[1]))

                x=img_to_array(img)

                Alldatas.append(x)

        os.chdir(pathd)      

    return Alldatas

SCALE=30 #RESIZE ALL IMAGES TO 30X30

all_img=getData("/kaggle/input/gemstones-images/train",(SCALE,SCALE))

all_img_test=getData("/kaggle/input/gemstones-images/test",(SCALE,SCALE))

all_img=np.asarray(all_img,dtype="float")

all_img_test=np.asarray(all_img_test,dtype="float")



train=all_img/255-0.5

test=all_img_test/255-0.5



trainCount=train.shape[0]

testCount=test.shape[0]



train,test=train.flatten(),test.flatten()

trainShape=int(train.shape[0]/trainCount)

testShape=int(test.shape[0]/testCount)



train,test=train.reshape(trainCount,trainShape),test.reshape(testCount,testShape)
from keras.engine.input_layer import Input 





model = Sequential()



featureSize=15

model.add(Dense(input_dim=train.shape[1],output_dim=train.shape[1],init='uniform'))

model.add(Dense(256, kernel_initializer='uniform'))



model.add(Dense(featureSize, kernel_initializer='uniform',activation="sigmoid")) #SIGMOID TO EASILY GENERATE IMAGES IN WIDE RANGE



model.add(Dense(256, kernel_initializer='uniform'))

model.add(Dense(train.shape[1],init='uniform'))



print(model.summary())

model.compile(loss="mean_squared_error",optimizer="adamax")



    

model.fit(train,

          train,

          epochs = 300,

          batch_size = 1024,

          validation_data = (test,test),

          verbose=1)

check=test[3]

decoded=model.predict(check.reshape((1,)+check.shape))

decoded=(decoded+0.5)

matrix=decoded.reshape(SCALE,SCALE,3)



#Show real image and compressed image from autoencoder

plt.figure(figsize=(30,30))

plt.subplot(20,20,1)

plt.imshow(array_to_img(((check+0.5)).reshape(SCALE,SCALE,3)))

plt.subplot(20,20,2)

plt.imshow(array_to_img(matrix))
from keras.models import Model

GEM_input=Input(model.layers[3].input_shape[1:])

GEM_model=GEM_input

for layer in model.layers[3:]:

    GEM_model= layer(GEM_model)

GEM_model = Model(inputs=GEM_input, outputs=GEM_model)
np.random.seed(seed=42)

plt.figure(figsize=(10,10))

for i in range(0,40):

    plt.subplot(8,8,i+1)

    random_features=np.random.randn(1,featureSize) # GENERATE RANDOM NUMBERS BETWEEN 0 AND 1 BECAUSE WE USED SIGMOID

    new_Img=GEM_model.predict(random_features)

    new_Img=(new_Img+0.5)

    matrix=new_Img.reshape(SCALE,SCALE,3)

    Gimage=array_to_img(matrix)

    plt.imshow(Gimage)

    plt.axis("off")

plt.tight_layout(pad=0.1)
plt.figure(figsize=(10,10))



plt.plot(model.history.history['loss'])

plt.plot(model.history.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()
