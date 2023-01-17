import cv2

import numpy as np

from glob import glob

from matplotlib import pylab as plt
def load_train():

  path1=sorted(glob('../input/data_set/data_set/train/*'))

  path2=sorted(glob('../input/data_set/data_set/label/*'))

  

  train_images=[]

  train_labels=[]

  for filename1,filename2 in zip(path1,path2):

    img1=cv2.imread(filename1,-1)

    img2=cv2.imread(filename2,-1)

    img1=img1[...,::-1]

    img1=img1.reshape(512,512,1)

    img2=img2[...,::-1]

    img2=img2.reshape(512,512,1)

    train_images.append(img1)

    train_labels.append(img2)

    

  train_images=np.array(train_images)/255

  train_labels=np.array(train_labels)/255

  

  return train_images,train_labels
train_images,train_labels=load_train()

print(train_images.shape)
fig=plt.figure()

for i in range(2):

  plt.subplot(1,2,i+1)

  plt.imshow(train_labels[i].reshape(512,512),cmap='gray')
def load_test(batch_size):

  path=glob('../input/data_set/data_set/test/*')

  batch=np.random.choice(path,size=batch_size)

  test_images=[]

  for filename in batch:

    img=cv2.imread(filename,-1)

    img=img[...,::-1]

    img=img.reshape(512,512,1)

    test_images.append(img)

  

  test_images=np.array(test_images)/255

  

  return test_images
test_images=load_test(2)

print(test_images.shape)
fig=plt.figure()

for i in range(2):

  plt.subplot(1,2,i+1)

  plt.imshow(test_images[i].reshape(512,512),cmap='gray')
import keras.layers as layers

from keras.models import Model

from keras.optimizers import Adam

from keras import backend as K
class attention_unet():

  def __init__(self,img_rows=512,img_cols=512):

    self.img_rows=img_rows

    self.img_cols=img_cols

    self.img_shape=(self.img_rows,self.img_cols,1)

    self.df=64

    self.uf=64

    

  def build_unet(self):

    def conv2d(layer_input,filters,dropout_rate=0,bn=False):

      d=layers.Conv2D(filters,kernel_size=(3,3),strides=(1,1),padding='same')(layer_input)

      if bn:

        d=layers.BatchNormalization()(d)

      d=layers.Activation('relu')(d)

      

      d=layers.Conv2D(filters,kernel_size=(3,3),strides=(1,1),padding='same')(d)

      if bn:

        d=layers.BatchNormalization()(d)

      d=layers.Activation('relu')(d)

      

      if dropout_rate:

        d=layers.Dropout(dropout_rate)(d)

      

      return d

    

    def deconv2d(layer_input,filters,bn=False):

      u=layers.UpSampling2D((2,2))(layer_input)

      u=layers.Conv2D(filters,kernel_size=(3,3),strides=(1,1),padding='same')(u)

      if bn:

        u=layers.BatchNormalization()(u)

      u=layers.Activation('relu')(u)

      

      return u

    

    def attention_block(F_g,F_l,F_int,bn=False):

      g=layers.Conv2D(F_int,kernel_size=(1,1),strides=(1,1),padding='valid')(F_g)

      if bn:

        g=layers.BatchNormalization()(g)

      x=layers.Conv2D(F_int,kernel_size=(1,1),strides=(1,1),padding='valid')(F_l)

      if bn:

        x=layers.BatchNormalization()(x)

#       print(g.shape)

#       print(x.shape)

      psi=layers.Add()([g,x])

      psi=layers.Activation('relu')(psi)

      

      psi=layers.Conv2D(1,kernel_size=(1,1),strides=(1,1),padding='valid')(psi)

      

      if bn:

        psi=layers.BatchNormalization()(psi)

      psi=layers.Activation('sigmoid')(psi)

      

      return layers.Multiply()([F_l,psi])

    

    inputs=layers.Input(shape=self.img_shape)

    

    conv1=conv2d(inputs,self.df)

    pool1=layers.MaxPooling2D((2,2))(conv1)

    

    conv2=conv2d(pool1,self.df*2,bn=True)

    pool2=layers.MaxPooling2D((2,2))(conv2)

    

    conv3=conv2d(pool2,self.df*4,bn=True)

    pool3=layers.MaxPooling2D((2,2))(conv3)

    

    conv4=conv2d(pool3,self.df*8,dropout_rate=0.5,bn=True)

    pool4=layers.MaxPooling2D((2,2))(conv4)

    

    conv5=conv2d(pool4,self.df*16,dropout_rate=0.5,bn=True)

    

    up6=deconv2d(conv5,self.uf*8,bn=True)

    conv6=attention_block(up6,conv4,self.uf*8,bn=True)

    up6=layers.Concatenate()([up6,conv6])

    conv6=conv2d(up6,self.uf*8)

    

    up7=deconv2d(conv6,self.uf*4,bn=True)

    conv7=attention_block(up7,conv3,self.uf*4,bn=True)

    up7=layers.Concatenate()([up7,conv7])

    conv7=conv2d(up7,self.uf*4)

    

    up8=deconv2d(conv7,self.uf*2,bn=True)

    conv8=attention_block(up8,conv2,self.uf*2,bn=True)

    up8=layers.Concatenate()([up8,conv8])

    conv8=conv2d(up8,self.uf*2)

    

    up9=deconv2d(conv8,self.uf,bn=True)

    conv9=attention_block(up9,conv1,self.uf,bn=True)

    up9=layers.Concatenate()([up9,conv9])

    conv9=conv2d(up9,self.uf)

    

    outputs=layers.Conv2D(1,kernel_size=(1,1),strides=(1,1),activation='sigmoid')(conv9)

    

    model=Model(inputs=inputs,outputs=outputs)

    

    return model
a=attention_unet()

unet=a.build_unet()

#unet.summary()
def dice_coef_loss(y_true,y_pred):

  y_true_f=K.flatten(y_true)

  y_pred_f=K.flatten(y_pred)

  intersection=K.sum(y_true_f*y_pred_f)

  return 1-(2*intersection)/(K.sum(y_true_f*y_true_f)+K.sum(y_pred_f*y_pred_f))
unet.compile(loss=dice_coef_loss,

             optimizer=Adam(1e-4),

             metrics=['accuracy'])
unet.fit(train_images,train_labels,validation_split=0.1,batch_size=1,epochs=20,verbose=1,shuffle=True)
result=unet.predict(test_images[0:2])

fig=plt.figure()

for i in range(2):

  plt.subplot(2,2,i+1)

  plt.imshow(test_images[i].reshape(512,512),cmap='gray')

  plt.subplot(2,2,i+3)

  plt.imshow(result[i].reshape(512,512),cmap='gray')
!nvidia-smi