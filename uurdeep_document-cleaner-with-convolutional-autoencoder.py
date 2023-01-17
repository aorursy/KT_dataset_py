# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np 

import os

import tensorflow as tf

from keras.models import Sequential

from keras.layers import Dense,Conv2D,Flatten,MaxPooling2D,UpSampling2D,InputLayer,Reshape



from keras.preprocessing.image import load_img

from keras.preprocessing.image import img_to_array

from keras.preprocessing.image import array_to_img

import matplotlib.pyplot as plt



#%% Data receive functions, receive datas from folders



def getData(pathd,shape):

    #file i/o çalışılmalı

    os.chdir(pathd)

    Alldatas=[]

    img_data=[]

    img_data=os.listdir(".")

    for image in img_data:

        _,extension = os.path.splitext(image)

        if(extension==".jpg" or extension==".jpeg" or extension==".png"):

            img=load_img(image)

            img=img.resize((shape[0],shape[1]))

            x=img_to_array(img)

           # x=x.reshape((1,) + x.shape)

            Alldatas.append(x)

    return Alldatas

scale=(540,258)

all_img=getData("/kaggle/input/cleaning-dirty-documents-unzipped/train",scale)

all_img_y=getData("/kaggle/input/cleaning-dirty-documents-unzipped/train_cleaned",scale)

#%%

all_img=tf.image.rgb_to_grayscale(all_img)

all_img_y=tf.image.rgb_to_grayscale(all_img_y)

#%%

def prepare(arr,flatten=True):

    arr=np.asarray(arr,dtype="float32")

    arr2=arr/255-0.5

    if(flatten):

        Count=arr2.shape[0]

        arr2=arr2.flatten()

        shap=int(arr2.shape[0]/Count)

        arr2=arr2.reshape(Count,shap)

    return arr2

        

# EDIT DATASET AND RESHAPE

train = prepare(all_img,flatten=False)

train_y = prepare(all_img_y,flatten=False)

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(train,train_y,test_size=0.1,random_state=42)
#CREATE AUTOENCODER



from keras.layers import Activation

from keras import optimizers

def cust(x):

    return tf.keras.backend.sigmoid(x)-0.5



opt = optimizers.adamax(learning_rate=0.001)

model = Sequential()

model.add(Conv2D(128, (3, 3), padding='same',input_shape=(258,540,1),data_format="channels_last"))

model.add(MaxPooling2D((2, 2), padding='same')) 

model.add(UpSampling2D((2, 2))) #SIGMOID TO EASILY GENERATE IMAGES IN WIDE RANGE

model.add(Conv2D(1, (3, 3), activation=cust, padding='same'))

model.compile(loss="mean_squared_error",optimizer=opt)

print(model.summary())



    

model.fit(x_train,

          y_train,

          epochs = 600,

          batch_size = 7,

          validation_data = (x_test,y_test),

          verbose=1)



#%% Check difference between test images

for i in range(0,9):

    check=x_test[i]

    matrix=model.predict(check.reshape((1,)+check.shape)).reshape(258,540,1)

    

    #Show real image and generated image from autoencoder

    plt.figure(figsize=(50,50))

    plt.subplot(10,10,1)

    plt.imshow(array_to_img(check+0.5),cmap="gray")

    plt.subplot(10,10,2)

    plt.imshow(array_to_img(matrix+0.5),cmap="gray")



#%%

#Plot Loss

plt.figure(figsize=(10,10))



plt.plot(model.history.history['loss'])

plt.plot(model.history.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()

#%% Vısualizing Model

from keras.utils.vis_utils import plot_model

plot_model(model, show_shapes=True, show_layer_names=True)