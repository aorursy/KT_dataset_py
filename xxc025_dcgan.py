from __future__ import absolute_import,division,print_function
import tensorflow as tf

tf.enable_eager_execution()

print(tf.__version__)
import matplotlib.pyplot as plt

import numpy as np

import tensorflow.keras.layers as layers

import os

import cv2

import time

import warnings

warnings.simplefilter('ignore',DeprecationWarning)

from IPython import display
array_of_img = [] 

def read_directory():

    # this loop is for read each image in this foder,directory_name is the foder name with images.

    for filename in os.listdir(r'../input/face/face'):

        #print(filename)

        img = cv2.imread('../input/face/face' + "/" + filename)# 返回numpy.ndarray

        img=img[...,::-1]

        array_of_img.append(img)

    return array_of_img
a=read_directory()
train_images=np.array(a)

print(train_images.shape)
train_images=train_images.astype('float32')#默认float64

train_images=(train_images-127.5)/127.5

#train_images=train_images/255

print(type(train_images))

#print(train_images[0])

plt.imshow(train_images[0])
BUFFER_SIZE=7000

BATCH_SIZE=64
train_dataset=tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

print(type(train_dataset))

print(train_dataset)
def make_generator_model():

  model=tf.keras.Sequential()

  model.add(layers.Dense(12*12*256,use_bias=False,input_shape=(100,)))

  model.add(layers.BatchNormalization())

  model.add(layers.LeakyReLU(0.2))

  

  model.add(layers.Reshape((12,12,256)))

  

  model.add(layers.Conv2DTranspose(128,(5,5),strides=(1,1),padding='same',use_bias=False))

  model.add(layers.BatchNormalization())

  model.add(layers.LeakyReLU(0.2))

  

  model.add(layers.Conv2DTranspose(64,(5,5),strides=(2,2),padding='same',use_bias=False))

  model.add(layers.BatchNormalization())

  model.add(layers.LeakyReLU(0.2))

  

  model.add(layers.Conv2DTranspose(32,(5,5),strides=(2,2),padding='same',use_bias=False))

  model.add(layers.BatchNormalization())

  model.add(layers.LeakyReLU(0.2))

  

  model.add(layers.Conv2DTranspose(3,(5,5),strides=(2,2),padding='same',use_bias=False,activation='tanh'))

  

  return model  
generator=make_generator_model()

generator.summary()
def make_discriminator_model():

  model=tf.keras.Sequential()

  

  model.add(layers.Conv2D(64,(5,5),strides=(2,2),padding='same',input_shape=[96,96,3]))

  model.add(layers.LeakyReLU(0.2))

  model.add(layers.Dropout(0.3))

  

  model.add(layers.Conv2D(128,(5,5),strides=(2,2),padding='same'))

  model.add(layers.LeakyReLU(0.2))

  model.add(layers.Dropout(0.3))

  

  model.add(layers.Conv2D(256,(5,5),strides=(2,2),padding='same'))

  model.add(layers.LeakyReLU(0.2))

  model.add(layers.Dropout(0.3))

  

  model.add(layers.Flatten())

  model.add(layers.Dense(1))

  

  return model
discriminator=make_discriminator_model()

discriminator.summary()
def discriminator_loss(real_output,fake_output):

    real_loss=tf.losses.sigmoid_cross_entropy(tf.ones_like(real_output),real_output)

    fake_loss=tf.losses.sigmoid_cross_entropy(tf.zeros_like(fake_output),fake_output)

    total_loss=real_loss+fake_loss

    return total_loss



def generator_loss(fake_output):

    return tf.losses.sigmoid_cross_entropy(tf.ones_like(fake_output),fake_output)
generator_optimizer=tf.train.AdamOptimizer(0.0002,0.5)

discriminator_optimizer=tf.train.AdamOptimizer(0.0002,0.5)
checkpoint_dir='./training_checkpoints'

checkpoint_prefix=os.path.join(checkpoint_dir,'ckpt')

checkpoint=tf.train.Checkpoint(generator_optimizer=generator_optimizer,

                                 discriminator_optimizer=discriminator_optimizer,

                                 generator=generator,

                                 discriminator=discriminator)
EPOCHS=300

noise_dim=100

num_examples_to_generate=16

random_vector_for_generation=tf.random.normal([num_examples_to_generate,noise_dim])
def train_step(images):

    noise=tf.random.normal([BATCH_SIZE,noise_dim])

    with tf.GradientTape() as gen_tape,tf.GradientTape() as disc_tape:

        generated_images=generator(noise,training=True)

    

        real_output=discriminator(images,training=True)

        fake_output=discriminator(generated_images,training=True)

    

        gen_loss=generator_loss(fake_output)

        disc_loss=discriminator_loss(real_output,fake_output)

    

    gradients_of_generator=gen_tape.gradient(gen_loss,generator.trainable_variables)

    gradients_of_discriminator=disc_tape.gradient(disc_loss,discriminator.trainable_variables)

  

    generator_optimizer.apply_gradients(zip(gradients_of_generator,generator.trainable_variables))

    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator,discriminator.trainable_variables))
train_step=tf.contrib.eager.defun(train_step)
def train(dataset,epochs):

    for epoch in range(epochs):

        start=time.time()

        for image_batch in dataset:

            #print(image_batch) (64,96,96,3)

            train_step(image_batch)

    

        display.clear_output(wait=True)

        generate_and_save_images(generator,

                            epoch+1,

                            random_vector_for_generation)

    

        if (epoch+1)%20==0:

            checkpoint.save(file_prefix=checkpoint_prefix)

      

        print('Time for epoch {} is {} sec'.format(epoch+1,time.time()-start))

    

    display.clear_output(wait=True)

    generate_and_save_images(generator,

                           epochs,

                           random_vector_for_generation)
def generate_and_save_images(model,epoch,test_input):

    predictions=model(test_input,training=False)

  

    fig=plt.figure(figsize=(4,4))

  

    for i in range(predictions.shape[0]):

        plt.subplot(4,4,i+1)

        #plt.imshow(predictions[i,:,:,:]*127.5+127.5)

        plt.imshow(tf.cast(predictions[i,:,:,:]*127.5+127.5,np.uint8))

        plt.axis('off')

  

    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))

    plt.show()
%%time

train(train_dataset,EPOCHS)