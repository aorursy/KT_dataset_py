from glob import glob

import numpy as np

from matplotlib import pylab as plt

import cv2

import tensorflow as tf

print(tf.__version__)

import tensorflow.keras.layers as layers

from tensorflow.keras.models import Model

import time

import os
def load_data(batch_size):

    path1=sorted(glob('../input/building/facade/test_picture/*'))

    path2=sorted(glob('../input/building/facade/test_label/*'))

    i=np.random.randint(0,27)

    batch1=path1[i*batch_size:(i+1)*batch_size]

    batch2=path2[i*batch_size:(i+1)*batch_size]

    

    img_A=[]

    img_B=[]

    for filename1,filename2 in zip(batch1,batch2):

        img1=cv2.imread(filename1)

        img2=cv2.imread(filename2)

        img1=img1[...,::-1]

        img2=img2[...,::-1]

        img1=cv2.resize(img1,(256,256),interpolation=cv2.INTER_AREA)

        img2=cv2.resize(img2,(256,256),interpolation=cv2.INTER_AREA)

        img_A.append(img1)

        img_B.append(img2)

      

    img_A=np.array(img_A)/127.5-1

    img_B=np.array(img_B)/127.5-1

    

    return img_A,img_B 
def load_batch(batch_size):

    path1=sorted(glob('../input/building/facade/train_picture/*'))

    path2=sorted(glob('../input/building/facade/train_label/*'))

    n_batches=int(len(path1)/batch_size)

  

    for i in range(n_batches):

        batch1=path1[i*batch_size:(i+1)*batch_size]

        batch2=path2[i*batch_size:(i+1)*batch_size]

        img_A,img_B=[],[]

        for filename1,filename2 in zip(batch1,batch2):

            img1=cv2.imread(filename1)

            img2=cv2.imread(filename2)

            img1=img1[...,::-1]

            img2=img2[...,::-1]

            img1=cv2.resize(img1,(256,256),interpolation=cv2.INTER_AREA)    

            img2=cv2.resize(img2,(256,256),interpolation=cv2.INTER_AREA)

            img_A.append(img1)#picture

            img_B.append(img2)#label

      

        img_A=np.array(img_A)/127.5-1

        img_B=np.array(img_B)/127.5-1

    

        yield img_A,img_B #return generator
class pix2pix():

    def __init__(self):

        self.img_rows=256

        self.img_cols=256

        self.channels=3

        self.img_shape=(self.img_rows,self.img_cols,self.channels)

    

        patch=int(self.img_rows/(2**4)) # 16

        self.disc_patch=(patch,patch,1)

    

        self.gf=64

        self.df=64

    

        optimizer=tf.keras.optimizers.Adam(0.0002,0.5)

    

        self.discriminator=self.build_discriminator()

        #self.discriminator.summary()

        self.discriminator.compile(loss='binary_crossentropy',

                              optimizer=optimizer)

    

        self.generator=self.build_generator()

        #self.generator.summary()

    

        img_A=layers.Input(shape=self.img_shape)#picture--label

        img_B=layers.Input(shape=self.img_shape)#label--real

    

        img=self.generator(img_A)

    

        self.discriminator.trainable=False

    

        valid=self.discriminator([img,img_A])

    

        self.combined=Model(img_A,valid)

        self.combined.compile(loss='binary_crossentropy',

                              optimizer=optimizer)

    

    def build_generator(self):

        def conv2d(layer_input,filters,f_size=(4,4),bn=True):

            d=layers.Conv2D(filters,kernel_size=f_size,strides=(2,2),padding='same')(layer_input)

            d=layers.LeakyReLU(0.2)(d)

            if bn:

                d=layers.BatchNormalization()(d)

            return d

    

        def deconv2d(layer_input,skip_input,filters,f_size=(4,4),dropout_rate=0):

            u=layers.UpSampling2D((2,2))(layer_input)

            u=layers.Conv2D(filters,kernel_size=f_size,strides=(1,1),padding='same',activation='relu')(u)

            if dropout_rate:

                u=layers.Dropout(dropout_rate)(u)

            u=layers.BatchNormalization()(u)

            u=layers.Concatenate()([u,skip_input])

            return u

    

        d0=layers.Input(shape=self.img_shape)

    

        d1=conv2d(d0,self.gf,bn=False) 

        d2=conv2d(d1,self.gf*2)         

        d3=conv2d(d2,self.gf*4)         

        d4=conv2d(d3,self.gf*8)         

        d5=conv2d(d4,self.gf*8)         

        d6=conv2d(d5,self.gf*8)        

    

        d7=conv2d(d6,self.gf*8)         

    

        u1=deconv2d(d7,d6,self.gf*8,dropout_rate=0.5)   

        u2=deconv2d(u1,d5,self.gf*8,dropout_rate=0.5)   

        u3=deconv2d(u2,d4,self.gf*8,dropout_rate=0.5)   

        u4=deconv2d(u3,d3,self.gf*4)   

        u5=deconv2d(u4,d2,self.gf*2)   

        u6=deconv2d(u5,d1,self.gf)     

        u7=layers.UpSampling2D((2,2))(u6)

    

        output_img=layers.Conv2D(self.channels,kernel_size=(4,4),strides=(1,1),padding='same',activation='tanh')(u7)

    

        return Model(d0,output_img)

  

    def build_discriminator(self):

        def d_layer(layer_input,filters,f_size=(4,4),bn=True):

            d=layers.Conv2D(filters,kernel_size=f_size,strides=(2,2),padding='same')(layer_input)

            d=layers.LeakyReLU(0.2)(d)

            if bn:

                d=layers.BatchNormalization()(d)

            return d

    

        img_A=layers.Input(shape=self.img_shape)

        img_B=layers.Input(shape=self.img_shape)

    

        combined_imgs=layers.Concatenate(axis=-1)([img_A,img_B])

    

        d1=d_layer(combined_imgs,self.df,bn=False)

        d2=d_layer(d1,self.df*2)

        d3=d_layer(d2,self.df*4)

        d4=d_layer(d3,self.df*8)

    

        validity=layers.Conv2D(1,kernel_size=(4,4),strides=(1,1),padding='same',activation='sigmoid')(d4)

    

        return Model([img_A,img_B],validity)

  

    def train(self,epochs,batch_size=1):

        valid=np.ones((batch_size,)+self.disc_patch)

        fake=np.zeros((batch_size,)+self.disc_patch)

    

        for epoch in range(epochs):

            start=time.time()

            for batch_i,(img_A,img_B) in enumerate(load_batch(1)):

                gen_imgs=self.generator.predict(img_A)

        

                d_loss_real = self.discriminator.train_on_batch([img_B, img_A], valid)

                d_loss_fake = self.discriminator.train_on_batch([gen_imgs, img_A], fake)

                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        

                g_loss = self.combined.train_on_batch(img_A,valid)



                if batch_i % 50 == 0:

                    print ("[Epoch %d] [Batch %d] [D loss: %f] [G loss: %f]" % (epoch,

                                                                                batch_i,

                                                                                d_loss,

                                                                                g_loss))

            

            self.sample_images(epoch)

            print('Time for epoch {} is {} sec'.format(epoch,time.time()-start))

      

    def sample_images(self, epoch):

        r, c = 3, 3

        img_A, img_B =load_data(3)

        fake_A = self.generator.predict(img_A)



        gen_imgs = np.concatenate([img_A, fake_A, img_B])



        # Rescale images 0 - 1

        gen_imgs = 0.5 * gen_imgs + 0.5



        titles = ['Condition', 'Generated', 'Original']

        fig, axs = plt.subplots(r, c)

        cnt = 0

        for i in range(r):

            for j in range(c):

                axs[i,j].imshow(gen_imgs[cnt])

                axs[i,j].set_title(titles[i])

                axs[i,j].axis('off')

                cnt += 1

        fig.savefig("./%d.png" % (epoch))

        plt.show()
if __name__ == '__main__':

    gan = pix2pix()

    gan.train(epochs=50, batch_size=1)

    