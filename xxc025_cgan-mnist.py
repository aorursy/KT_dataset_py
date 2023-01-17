from __future__ import print_function,division
import keras

from keras.models import Model

from matplotlib import pylab as plt

import numpy as np

import time
class CGAN():

    def __init__(self):

        self.rows=28

        self.cols=28

        self.channels=1

        self.img_shape=(self.rows,self.cols,self.channels)

        self.num_classes=10

        self.latent_dim=100

        

        optimizer=keras.optimizers.Adam(0.0002,0.5)

        

        self.discriminator=self.build_discriminator()

        self.discriminator.compile(loss='binary_crossentropy',

                                  optimizer=optimizer)

        

        self.generator=self.build_generator()

        noise=keras.layers.Input(shape=(self.latent_dim,))

        label=keras.layers.Input(shape=(1,))

        img=self.generator([noise,label])

        self.discriminator.trainable=False

        valid=self.discriminator([img,label])

        

        self.combined=Model([noise,label],valid)

        self.combined.compile(loss='binary_crossentropy',

                             optimizer=optimizer)

    

    def build_generator(self):

        model=keras.models.Sequential()

        

        model.add(keras.layers.Dense(256,input_dim=self.latent_dim))

        model.add(keras.layers.LeakyReLU(0.2))

        model.add(keras.layers.BatchNormalization())

        

        model.add(keras.layers.Dense(512))

        model.add(keras.layers.LeakyReLU(0.2))

        model.add(keras.layers.BatchNormalization())

        

        model.add(keras.layers.Dense(1024))

        model.add(keras.layers.LeakyReLU(0.2))

        model.add(keras.layers.BatchNormalization())

        

        model.add(keras.layers.Dense(np.prod(self.img_shape),activation='tanh'))#784

        model.add(keras.layers.Reshape(self.img_shape))

        

        model.summary()

        

        noise=keras.layers.Input(shape=(self.latent_dim,))

        label=keras.layers.Input(shape=(1,),dtype='int32')

        # keras.layers.Embedding(input_dim, output_dim)

        # input_dim: int > 0. Size of the vocabulary.

        # output_dim: int >= 0. Dimension of the dense embedding.

        # return:shape=(batch,input_length,output_dim)

        label_embedding=keras.layers.Flatten()(keras.layers.Embedding(self.num_classes,self.latent_dim)(label))#(1,100)-->(100*1,)

        model_input=keras.layers.multiply([noise,label_embedding])#(100,)

        

        img=model(model_input)

        

        return Model([noise,label],img) # multi-input

    

    def build_discriminator(self):

        model=keras.models.Sequential()

        

        model.add(keras.layers.Dense(512,input_dim=np.prod(self.img_shape)))

        model.add(keras.layers.LeakyReLU(0.2))

        

        model.add(keras.layers.Dense(512))

        model.add(keras.layers.LeakyReLU(0.2))

        model.add(keras.layers.Dropout(0.4))

        

        model.add(keras.layers.Dense(512))

        model.add(keras.layers.LeakyReLU(0.2))

        model.add(keras.layers.Dropout(0.4))

        

        model.add(keras.layers.Dense(1,activation='sigmoid'))

        

        model.summary()

        

        img=keras.layers.Input(shape=self.img_shape)

        label=keras.layers.Input(shape=(1,),dtype='int32')

        

        label_embedding=keras.layers.Flatten()(keras.layers.Embedding(self.num_classes,

                                                                            np.prod(self.img_shape))(label))

        flat_img=keras.layers.Flatten()(img)

        

        model_input=keras.layers.multiply([flat_img,label_embedding])#784

        validity=model(model_input)

        

        return Model([img,label],validity)

    

    def train(self,epochs,batch_size=128,sample_interval=50):

        (x_train,y_train),(_,_)=keras.datasets.mnist.load_data()

        x_train=(x_train.astype('float32')-127.5)/127.5 #(-1,1)

        x_train=np.expand_dims(x_train,axis=3) #(batch,28,28)-->(batch,28,28,1)

        

        valid=np.ones((batch_size,1))

        fake=np.zeros((batch_size,1))

        

        for epoch in range(epochs):

            #---------

            # Train D  

            #---------

            idx=np.random.randint(0,x_train.shape[0],batch_size)

            imgs,labels=x_train[idx],y_train[idx]

            

            noise=np.random.normal(0,1,(batch_size,100))

            gen_imgs=self.generator.predict([noise,labels])

            

            d_loss_real=self.discriminator.train_on_batch([imgs,labels],valid)

            d_loss_fake=self.discriminator.train_on_batch([gen_imgs,labels],fake)

            d_loss=0.5*np.add(d_loss_real,d_loss_fake)

            

            #---------

            # Train G  

            #---------

            #sampled_labels=np.random.randint(0,9,batch_size).reshape(-1,1)

            g_loss=self.combined.train_on_batch([noise,labels],valid)

            

            if epoch % sample_interval == 0:

                self.sample_images(epoch)

                print('{} [D loss:{}] [G loss:{}]'.format(epoch,d_loss,g_loss))

                

    def sample_images(self, epoch):

        r,c=2,5

        noise=np.random.normal(0,1,(r*c,100))

        sampled_labels=np.arange(0,10).reshape(-1,1)

        

        gen_imgs=self.generator.predict([noise,sampled_labels])

        

        gen_imgs=0.5*gen_imgs+0.5

        

        fig,axs=plt.subplots(r,c)

        cnt=0

        for i in range(r):

            for j in range(c):

                axs[i,j].imshow(gen_imgs[cnt,:,:,0],cmap='gray')

                axs[i,j].set_title('{}'.format(sampled_labels[cnt]))

                axs[i,j].axis('off')

                cnt+=1

            

        plt.show()

        
a=CGAN()

a.train(epochs=25000,batch_size=32,sample_interval=200)