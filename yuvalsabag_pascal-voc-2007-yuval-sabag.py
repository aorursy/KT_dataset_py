# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import cv2
import imageio
import matplotlib.pyplot as plt

def load_my_image(img_name):
    return imageio.imread(img_name)


def load_my_images(_max=5011):
    directory='../input/pascal-voc-2007/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages/'
    dir_list=os.listdir(directory)
    images=[]
    for index,path in enumerate(dir_list):
        if index>_max:
            return np.array(images)
        full_path=directory+path
        image=load_my_image(full_path)
        images.append(np.array(image))
    return np.array(images)

        
   
def plot_losses_with_psnr(titles,history,kind):
    plt.figure(figsize=(20,9))
    for title in titles:
        plt.plot(history.history[title])
    plt.title('Model '+kind)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(titles, loc='upper left')
    plt.show()

def plot_psnr(history):
    psnr = [x for x in history.history.keys() if 'PSNR' in x]
    losses  = [x for x in history.history.keys() if 'PSNR' not in x]
    plot_losses_with_psnr(losses,history,'loss')
    plot_losses_with_psnr(psnr,history,'psnr')
from keras import backend as K
def PSNR(y_true, y_pred):
    max_pixel = 1.0
    return (10.0 * K.log((max_pixel ** 2) / (K.mean(K.square(y_pred - y_true), axis=-1)))) / 2.303
imgs_bad=[]
imgs_mid=[]
imgs_good=[]
images=load_my_images(100)
for img in images:
    img_bad=cv2.resize(img,(72,72))
    img_mid=cv2.resize(img,(144,144))
    img_good=cv2.resize(img,(288,288))
    imgs_bad.append(img_bad/255.0)
    imgs_mid.append(img_mid/255.0)
    imgs_good.append(img_good/255.0)

imgs_bad=np.array(imgs_bad)
imgs_mid=np.array(imgs_mid)
imgs_good=np.array(imgs_good)
imgs_bad=[]
imgs_mid=[]
imgs_good=[]
images=load_my_images()
for img in images:
    img_bad=cv2.resize(img,(72,72))
    img_mid=cv2.resize(img,(144,144))
    img_good=cv2.resize(img,(288,288))
    imgs_bad.append(img_bad)
    imgs_mid.append(img_mid)
    imgs_good.append(img_good)

imgs_bad=np.array(imgs_bad)
imgs_mid=np.array(imgs_mid)
imgs_good=np.array(imgs_good)
def next_images(images1,images2,images3,start,end,num_of_epochs=1):
    for e in range(num_of_epochs):
        for i,(image1,image2,image3) in enumerate(zip(images1[start:end],images2[start:end],images3[start:end])):
            yield np.expand_dims((image1/255.0),axis=0),[np.expand_dims((image2/255.0),axis=0),np.expand_dims((image3/255.0),axis=0)]
    
x_part3_train=next_images(imgs_bad,imgs_mid,imgs_good,1000,5011,5)
x_part3_val=next_images(imgs_bad,imgs_mid,imgs_good,0,1000,5)
x_part3_test=next_images(imgs_bad,imgs_mid,imgs_good,0,1000,5)
def plot_imgs(rows,cols,imgs_list):
    fig,ax=plt.subplots(rows,cols,figsize=(20,9))
    for i in range(len(imgs_list)):
        if len(imgs_list)//cols >1:
            ax[i//cols,i%cols].imshow(imgs_list[i])
        else:
            ax[i].imshow(imgs_list[i])
current_img=4;
plot_imgs(1,3,[imgs_bad[current_img],imgs_mid[current_img],imgs_good[current_img]])
from keras.layers import Conv2D,UpSampling2D
from keras import Input,Model

def my_first_model():
    my_input=Input(shape=(72,72,3))
    x_lay=Conv2D(64,3,activation='relu',padding='same')(my_input)
    x_lay=Conv2D(64,3,activation='relu',padding='same')(x_lay)
    x_lay=UpSampling2D()(x_lay)
    x_lay=Conv2D(3,1,activation='sigmoid',padding='same',name='output')(x_lay)
    return Model(my_input,x_lay)

model=my_first_model()
model.compile(loss='mse',metrics=[PSNR],optimizer='adam')

history=model.fit(imgs_bad[:80],imgs_mid[:80],validation_data=[imgs_bad[80:],imgs_mid[80:]],epochs=20)
plot_psnr(history)
preds=model.predict(imgs_bad[80:])
for i in range(0,20):
    plot_imgs(1,4,[imgs_bad[i+80],imgs_mid[i+80],imgs_good[i+80],preds[i]])
from keras.layers import Conv2D,UpSampling2D,LeakyReLU
from keras import Input,Model
def my_second_model():
    my_input=Input(shape=(None,None,3), name='input')
    x_lay=Conv2D(64,3,activation=LeakyReLU(0.2),padding='same')(my_input)
    x_lay=Conv2D(64,3,activation=LeakyReLU(0.2),padding='same')(x_lay)
    x_lay=UpSampling2D()(x_lay)
    up_sample2=UpSampling2D()(x_lay)
    up_sample2=Conv2D(3,1,activation='sigmoid',padding='same',name='288')(up_sample2)
    x_lay=Conv2D(3,1,activation='sigmoid',padding='same',name='144')(x_lay)
    return Model(my_input,outputs=[x_lay,up_sample2])

model2=my_second_model()
model2.compile(loss='mse',metrics=[PSNR],optimizer='adam')

history2=model2.fit_generator(x_part3_train,steps_per_epoch=4011,epochs=5,validation_data=x_part3_val,validation_steps=1000)
preds=model2.predict_generator(x_part3_test,1000)
for i in range(0,20):
    plot_imgs(1,5,[imgs_bad[i],imgs_mid[i],imgs_good[i],preds[0][i],preds[1][i]])
plot_psnr(history2)
from keras.layers import Conv2D,UpSampling2D,Activation,Add
from keras import Input,Model
def residual_bolck(my_shape):
    _input=Input(shape=(None,None,my_shape))
    x_lay=Conv2D(my_shape,3,activation=LeakyReLU(0.2),padding='same')(_input)
    x_lay=Conv2D(my_shape,3,activation=LeakyReLU(0.2),padding='same')(x_lay)
    x_lay= Add()([_input,x_lay])
    return Model(_input,Activation('relu')(x_lay))
from keras.layers import Conv2D,UpSampling2D
from keras import Input,Model
def my_third_model():
    my_input=Input(shape=(None,None,3))
    x_lay=Conv2D(32,3,activation=LeakyReLU(0.2),padding='same')(my_input)
    x_lay=residual_bolck(32)(x_lay)
    x_lay=residual_bolck(32)(x_lay)
    x_lay=UpSampling2D()(x_lay)
    up_sample2=residual_bolck(32)(x_lay)
    up_sample2=UpSampling2D()(up_sample2)
    up_sample2=Conv2D(3,1,activation='sigmoid',padding='same',name='288')(up_sample2)
    x_lay=Conv2D(3,1,activation='sigmoid',padding='same',name='144')(x_lay)
    return Model(my_input,outputs=[x_lay,up_sample2])

model3=my_third_model()
model3.compile(loss='mse',metrics=[PSNR],optimizer='adam')

history3=model3.fit_generator(x_part3_train,steps_per_epoch=4011,epochs=5,validation_data=x_part3_val,validation_steps=1000)
preds=model3.predict_generator(x_part3_test,1000)
for i in range(0,20):
    plot_imgs(1,5,[imgs_bad[i],imgs_mid[i],imgs_good[i],preds[0][i],preds[1][i]])
plot_psnr(history3)
from keras.layers import Conv2D,UpSampling2D,Activation,Concatenate
from keras import Input,Model
def dilated_bolck(my_shape):
    _input=Input(shape=(None,None,my_shape))
    x_lay1=Conv2D(my_shape,3,activation='relu',dilation_rate=(1,1),padding='same')(_input)
    x_lay2=Conv2D(my_shape,3,activation='relu',dilation_rate=(2,2),padding='same')(_input)
    x_lay3=Conv2D(my_shape,3,activation='relu',dilation_rate=(4,4),padding='same')(_input)
    x_lay=Concatenate()([x_lay1,x_lay2,x_lay3])
    x_lay=Activation('relu')(x_lay)
    x_lay=Conv2D(my_shape,3,activation='relu',padding='same')(x_lay)
    return Model(_input,x_lay)
from keras.layers import Conv2D,UpSampling2D,LeakyReLU
from keras import Input,Model
def my_forth_model():
    my_input=Input(shape=(None,None,3))
    x_lay=Conv2D(32,3,activation=LeakyReLU(0.2),padding='same')(my_input)
    x_lay=dilated_bolck(32)(x_lay)
    x_lay=dilated_bolck(32)(x_lay)
    x_lay=UpSampling2D()(x_lay)
    up_sample2=dilated_bolck(32)(x_lay)
    up_sample2=UpSampling2D()(up_sample2)
    up_sample2=Conv2D(3,1,activation='sigmoid',padding='same',name='288')(up_sample2)
    x_lay=Conv2D(3,1,activation='sigmoid',padding='same',name='144')(x_lay)
    return Model(my_input,outputs=[x_lay,up_sample2])

model4=my_forth_model()
model4.compile(loss='mse',metrics=[PSNR],optimizer='adam')

history4=model4.fit_generator(x_part3_train,steps_per_epoch=4011,epochs=5,validation_data=x_part3_val,validation_steps=1000)
preds=model4.predict_generator(x_part3_test,1000)
for i in range(0,20):
    plot_imgs(1,5,[imgs_bad[i],imgs_mid[i],imgs_good[i],preds[0][i],preds[1][i]])
plot_psnr(history4)
from tensorflow.keras.layers import Conv2D,UpSampling2D,Concatenate,LeakyReLU
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras import Input,Model
vgg_lay = VGG16(weights='imagenet', include_top=False, input_shape = (None,None, 3))
vgg_lay = Model(inputs=vgg_lay.input,outputs=vgg_lay.get_layer("block1_conv2").output)
def my_fifth_model():
    my_input=Input(shape=(None,None,3))
    x_lay=Conv2D(64,3,activation=LeakyReLU(0.2),padding='same')(my_input)
    x_lay=Conv2D(64,3,activation=LeakyReLU(0.2),padding='same')(x_lay)
    x_lay2=vgg_lay(my_input)
    x_lay2=Concatenate()([x_lay,x_lay2])
    x_lay2=UpSampling2D()(x_lay2)
    x_lay2=Conv2D(3,1,activation='sigmoid',padding='same',name='144')(x_lay2)
    up_sample2=UpSampling2D()(x_lay2)
    up_sample2=Conv2D(3,1,activation='sigmoid',padding='same',name='288')(up_sample2)
    return Model(my_input,outputs=[x_lay2,up_sample2])

model5=my_fifth_model()
model5.compile(loss='mse',metrics=[PSNR],optimizer='adam')

history5=model5.fit_generator(x_part3_train,steps_per_epoch=4011,epochs=5,validation_data=x_part3_val,validation_steps=1000)
preds=model5.predict_generator(x_part3_test,1000)
for i in range(0,20):
    plot_imgs(1,5,[imgs_bad[i],imgs_mid[i],imgs_good[i],preds[0][i],preds[1][i]])
plot_psnr(history5)
from tensorflow.keras.layers import Conv2D,UpSampling2D,Concatenate,Lambda,LeakyReLU
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras import Input,Model
vgg_lay = VGG16(weights='imagenet', include_top=False, input_shape = (None,None, 3))
vgg_lay = Model(inputs=vgg_lay.input,outputs=vgg_lay.get_layer("block1_conv2").output)
def my_sixth_model():
    my_input=Input(shape=(None,None,3))
    x_lay=Conv2D(64,3,activation=LeakyReLU(0.2),padding='same')(my_input)
    x_lay=Conv2D(64,3,activation=LeakyReLU(0.2),padding='same')(x_lay)
    x_lay2=vgg_lay(my_input)
    x_lay2=Concatenate()([x_lay,x_lay2])
    x_lay2 = Lambda(lambda x:tf.nn.depth_to_space(x,2),name="lambda")(x_lay2)
    x_lay2=Conv2D(3,1,activation='sigmoid',padding='same',name='144')(x_lay2)
    
    up_space2 = Conv2D(64,3,activation='sigmoid',padding='same')(x_lay2)
    up_space2 = Lambda(lambda x:tf.nn.depth_to_space(x,2),name="lambda_b")(up_space2)
    up_space2=Conv2D(3,1,activation='sigmoid',padding='same',name='288')(up_space2)
    return Model(my_input,outputs=[x_lay2,up_space2])

model6=my_sixth_model()
model6.compile(loss='mse', metrics=[PSNR],optimizer='adam')

history6=model6.fit_generator(x_part3_train,steps_per_epoch=4011,epochs=5,validation_data=x_part3_val,validation_steps=1000)
preds=model6.predict_generator(x_part3_test,1000)
for i in range(0,20):
    plot_imgs(1,5,[imgs_bad[i],imgs_mid[i],imgs_good[i],preds[0][i],preds[1][i]])
plot_psnr(history6)