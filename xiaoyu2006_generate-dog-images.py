import gc

gc.enable()
import os

import numpy as np

from PIL import Image

import keras



def imageToMatrix(filename,w=64,h=64):

    im=Image.open(filename)

    im=im.resize((w,h),Image.ANTIALIAS)

    width,height=im.size

    data=im.getdata()

    data=np.array(data,dtype='float32')/255.0

    new_data=np.reshape(data,(width,height,3))

    del data,im

    return new_data



import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
imgs=os.listdir('../input/all-dogs/all-dogs/')

# print(imgs)



d=[]

i=0



for img in imgs:

    d.append(imageToMatrix('../input/all-dogs/all-dogs/'+img))

    if i%2000==0:

        print("Reading the "+str(i)+"th photo.")

    i+=1



del imgs

    

print("\nend")



d=np.array(d,dtype='float32')
from keras.layers import Input,Dense,Reshape,Flatten,Dropout,BatchNormalization,Activation,ZeroPadding2D,GaussianNoise

from keras.layers.advanced_activations import LeakyReLU

from keras.layers.convolutional import UpSampling2D,Conv2D

from keras.models import Sequential,Model

from keras.optimizers import Adam



class DCGAN(object):

    def __init__(self,img_rows=64,img_cols=64,img_channels=1,latent_dim=100):

        self.img_rows=img_rows

        self.img_cols=img_cols

        self.channels=img_channels

        self.img_shape=(img_rows,img_cols,img_channels)

        self.latent_dim=latent_dim

        

        optimizer=Adam(0.0002,0.5)

        self.discriminator=self._build_discriminator()

        self.discriminator.compile(

            loss='binary_crossentropy',

            optimizer=optimizer,

            metrics=['accuracy']

        )

        self.generator=self._build_generator()

        z=Input(shape=(100,))

        img=self.generator(z)

        self.discriminator.trainable=False

        vaild=self.discriminator(img)

        self.combined=Model(z,vaild)

        self.combined.compile(

            loss='binary_crossentropy',

            optimizer=optimizer

        )

    

    def _build_generator(self):

        model = Sequential()

        model.add(Dense(64 * 8 * 8, activation="relu", input_dim=self.latent_dim))

        model.add(Reshape((8, 8, 64)))

        #model.add(GaussianNoise(0.5))

        model.add(UpSampling2D())

        model.add(Conv2D(128, kernel_size=3, padding="same"))

        model.add(BatchNormalization(momentum=0.8))

        model.add(Activation("relu"))

        #model.add(GaussianNoise(0.5))

        model.add(UpSampling2D())

        model.add(Conv2D(64, kernel_size=3, padding="same"))

        model.add(BatchNormalization(momentum=0.8))

        model.add(Activation("relu"))

        #model.add(GaussianNoise(0.1))

        model.add(UpSampling2D())

        model.add(Conv2D(64, kernel_size=3, padding="same"))

        model.add(BatchNormalization(momentum=0.8))

        model.add(Activation("relu"))

        model.add(Conv2D(self.channels, kernel_size=3, padding="same"))

        model.add(Activation("tanh"))

        model.summary()

        noise = Input(shape=(self.latent_dim,))

        img = model(noise)

        

        return Model(noise, img)

    

    def _build_discriminator(self):

        model = Sequential()

        model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))

        model.add(LeakyReLU(alpha=0.2))

        model.add(Dropout(0.25))

        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))

        model.add(ZeroPadding2D(padding=((0,1),(0,1))))

        model.add(BatchNormalization(momentum=0.8))

        model.add(LeakyReLU(alpha=0.2))

        model.add(Dropout(0.25))

        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))

        model.add(BatchNormalization(momentum=0.8))

        model.add(LeakyReLU(alpha=0.2))

        model.add(Dropout(0.25))

        model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))

        model.add(BatchNormalization(momentum=0.8))

        model.add(LeakyReLU(alpha=0.2))

        model.add(Dropout(0.25))

        model.add(Flatten())

        model.add(Dense(1, activation='sigmoid'))

        model.summary()

        img = Input(shape=self.img_shape)

        validity = model(img)

        

        return Model(img, validity)

    

    def train(self,data_set,epochs,batch_size=128):

        X_train = data_set

        # X_train = np.expand_dims(X_train, axis=3)

        valid = np.ones((batch_size, 1))

        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            idx = np.random.randint(0, X_train.shape[0], batch_size)

            imgs = X_train[idx]

            

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            gen_imgs = self.generator.predict(noise)

            

            d_loss_real = self.discriminator.train_on_batch(imgs, valid)

            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)

            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            

            g_loss = self.combined.train_on_batch(noise, valid)

            

            if epoch%100==0:

                print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))



            #if epoch % save_interval == 0:

            #    self.save_imgs(epoch,save_name)



    def show_img(self):

        r, c = 5, 5

        noise = np.random.normal(0, 1, (r * c, self.latent_dim))

        gen_imgs = self.generator.predict(noise)*255

    

        fig, axs = plt.subplots(r, c)

        cnt = 0

        for i in range(r):

            for j in range(c):

                axs[i,j].imshow(gen_imgs[cnt, :,:,0])

                axs[i,j].axis('off')

                cnt += 1

        plt.show()

        plt.close()



    def getImg(self):

        r, c = 5, 5

        noise = np.random.normal(0, 1, (r * c, self.latent_dim))

        gen_imgs = self.generator.predict(noise)[0]*255

        return Image.fromarray(gen_imgs.astype('uint8'))



network=DCGAN(64,64,3)



network.train(d,100000)
# import pickle

import zipfile

# fobj=open('network.gan','w')

# pickle.dump(fobj,network)



z = zipfile.PyZipFile('images.zip', mode='w')



network.getImg()



for k in range(10000):

    img = network.getImg()

    f = str(k)+'.png'

    img.save(f,'PNG'); z.write(f); os.remove(f);

    if k%500==0:

        print("Saving the "+str(k)+"th img.")

z.close()
%matplotlib inline

network.getImg()