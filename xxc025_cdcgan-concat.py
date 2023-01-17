from __future__ import absolute_import,division,print_function



#!pip install tensorflow-gpu==2.0.0-alpha0

import tensorflow as tf

tf.enable_eager_execution()

print(tf.__version__)
import matplotlib.pyplot as plt

import numpy as np

import tensorflow.keras.layers as layers

import os

import PIL

import time

from tensorflow.keras.models import Model

from IPython import display

import warnings

warnings.simplefilter('ignore',DeprecationWarning)
(train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()

print(train_images.shape)

plt.imshow(train_images[0],cmap='gray')

plt.grid('off')
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')

train_images = (train_images-127.5) /127.5 # Normalize the images to [-1, 1]

a=np.zeros((train_labels.shape[0],10))

a[np.arange(train_labels.shape[0]),train_labels]=1

train_labels=a

train_labels=train_labels.astype('float32')

print(train_labels.shape)

print(train_labels[0:5])

plt.imshow(train_images[0].reshape(28,28),cmap='gray')

plt.grid('off')
BUFFER_SIZE=60000

BATCH_SIZE=256



train_dataset = tf.data.Dataset.from_tensor_slices(train_images).batch(BATCH_SIZE,drop_remainder=True)

train_label=tf.data.Dataset.from_tensor_slices(train_labels).batch(BATCH_SIZE,drop_remainder=True)

print(type(train_dataset))

print(train_dataset)

print(train_label)
def make_generator_model():

    model = tf.keras.Sequential()

    model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(110,)))

    model.add(layers.BatchNormalization())

    model.add(layers.LeakyReLU())

      

    model.add(layers.Reshape((7, 7, 256)))

    assert model.output_shape == (None, 7, 7, 256) # Note: None is the batch size

    

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))

    assert model.output_shape == (None, 7, 7, 128)  

    model.add(layers.BatchNormalization())

    model.add(layers.LeakyReLU())



    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))

    assert model.output_shape == (None, 14, 14, 64)    

    model.add(layers.BatchNormalization())

    model.add(layers.LeakyReLU())



    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))

    assert model.output_shape == (None, 28, 28, 1)

    

#     noise=layers.Input(shape=(100,))

#     label=layers.Input(shape=(1,))

#     label_embedding=layers.Flatten()(layers.Embedding(10,100)(label))

#     model_input=layers.multiply([noise,label_embedding])

#     img=model(model_input)

    noise=layers.Input(shape=(100,))

    label=layers.Input(shape=(10,))

    model_input=layers.concatenate([noise,label],axis=1)

    print(model_input.shape)

    img=model(model_input)

  

    return Model([noise,label],img)
generator=make_generator_model()

generator.summary()

# noise = tf.random.normal([1, 100])

# label=tf.constant(2,shape=(1,))

noise=tf.random.normal([10,100])

label=tf.eye(10)

gen_imgs= generator([noise,label], training=False)#training=True-->training  training=False-->inference

plt.grid('off')

plt.imshow(gen_imgs[0, :, :, 0], cmap='gray')
# It's similar to a regular CNN-based image classifier.

def make_discriminator_model():

    model=tf.keras.Sequential()

    

#     model.add(layers.Dense(28*28*1,input_shape=(784,)))

#     model.add(layers.LeakyReLU())

    

#     model.add(layers.Reshape((28,28,1)))

        

    model.add(layers.Conv2D(64,(5,5),strides=(2,2),padding='same'))

    model.add(layers.LeakyReLU())

    model.add(layers.Dropout(0.3))

  

    model.add(layers.Conv2D(128,(5,5),strides=(2,2),padding='same'))

    model.add(layers.LeakyReLU())

    model.add(layers.Dropout(0.3))

  

    model.add(layers.Flatten())

    model.add(layers.Dense(1))

    

#     img = layers.Input(shape=(28,28,1))

#     label = layers.Input(shape=(1,))



#     label_embedding = layers.Flatten()(layers.Embedding(10, np.prod((28,28,1)))(label))

#     flat_img = layers.Flatten()(img)



#     model_input = layers.multiply([flat_img, label_embedding])

    img=layers.Input(shape=(28,28,1))

    label=layers.Input(shape=(10,))

    y=layers.Dense(28*28)(label)

    y=layers.Reshape((28,28,1))(y)

    model_input=layers.concatenate([img,y])

    print(model_input.shape)

    validity=model(model_input)

    

    return Model([img,label],validity)
discriminator=make_discriminator_model()

discriminator.summary()

decision=discriminator([gen_imgs,label])

print(decision)
def discriminator_loss(real_output,fake_output):

    # [1,1,...,1] with real output since it is true and we want our generated examples to look like it

    real_loss=tf.losses.sigmoid_cross_entropy(tf.ones_like(real_output),real_output)

    fake_loss=tf.losses.sigmoid_cross_entropy(tf.zeros_like(fake_output),fake_output)

    total_loss=real_loss+fake_loss

    return total_loss



def generator_loss(fake_output):

    return tf.losses.sigmoid_cross_entropy(tf.ones_like(fake_output), fake_output)



generator_optimizer = tf.train.AdamOptimizer(0.0002,0.5)

discriminator_optimizer = tf.train.AdamOptimizer(0.0002,0.5)
checkpoint_dir = './training_checkpoints'

checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,

                                 discriminator_optimizer=discriminator_optimizer,

                                 generator=generator,

                                 discriminator=discriminator)
EPOCHS=100

noise_dim=100

num_to_generate=10

random_generation=tf.random.normal([num_to_generate,noise_dim])

con_label=tf.eye(10)
def train_step(images,labels):

    noise=tf.random.normal([BATCH_SIZE,noise_dim])

    with tf.GradientTape() as gen_tape,tf.GradientTape() as disc_tape:

         generated_images=generator([noise,labels],training=True)

    

         real_output=discriminator([images,labels],training=True)

         fake_output=discriminator([generated_images,labels],training=True)

    

         gen_loss=generator_loss(fake_output)

         disc_loss=discriminator_loss(real_output,fake_output)

    

    gradients_of_generator=gen_tape.gradient(gen_loss,generator.trainable_variables)

    gradients_of_discriminator=disc_tape.gradient(disc_loss,discriminator.trainable_variables)

  

    generator_optimizer.apply_gradients(zip(gradients_of_generator,generator.trainable_variables))

    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator,discriminator.trainable_variables))
train_step = tf.contrib.eager.defun(train_step) # 将Python函数编译为可调用的TensorFlow图
def train(dataset,labelset,epochs):

    for epoch in range(epochs):

        start=time.time()

        for image_batch,label_batch in zip(dataset,labelset):

            train_step(image_batch,label_batch)

    

        display.clear_output(wait=True)

        generate_and_save_images(generator,

                                 epoch+1,

                                 [random_generation,con_label])

    

        if (epoch+1)%15==0:

            checkpoint.save(file_prefix=checkpoint_prefix)

      

        print('Time for epoch {} is {} sec'.format(epoch+1,time.time()-start))

    

    display.clear_output(wait=True)

    generate_and_save_images(generator,

                           epochs,

                           [random_generation,con_label])
def generate_and_save_images(model,epoch,test_input):

    predictions=model(test_input,training=False)

    r,c=4,4

    fig=plt.figure(figsize=(r,c))

  

    for i in range(predictions.shape[0]):

        plt.subplot(r,c,i+1)

        #plt.title('{}'.format(i))

        plt.imshow(predictions[i,:,:,0]*0.5+0.5,cmap='gray')

        plt.axis('off')

  

    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))

    plt.show()

%%time

train(train_dataset,train_label,EPOCHS)