import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from PIL import Image
import matplotlib.pyplot as plt
import os
from keras import preprocessing
from keras.models import Sequential
from tensorflow.keras import layers
#from keras.layers import Conv2D,Dropout,Dense,Flatten,Conv2DTranspose,BatchNormalization,LeakyReLU,Reshape
import tensorflow as tf
from tensorflow.keras.utils import Progbar

# Any results you write to the current directory are saved as output.
TEST_SET_SIZE = 50000
BUFFER_SIZE = TEST_SET_SIZE
BATCH_SIZE = 128
NOISE_DIM = 100
NUM_EXAMPLES_TO_GENERATE = 16
EXAMPLE_SEED = tf.random.normal([NUM_EXAMPLES_TO_GENERATE, NOISE_DIM])
EPOCHS = 300  #set epoch according to your training dataset size,i had chosen 50k images hence epochs are high as 300...
BATCH_SIZE = 128
def decode_img(file_path):
    file = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(file, channels=3)
    return img   

def process_path(file_path):
    img = decode_img(file_path)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.central_crop(img, 0.7)
    
    img = tf.image.crop_to_bounding_box(
        img, 30, 10, 115, 115
    )
    img = tf.image.resize_with_pad(img, 64,64)
    return img

img_path = "/kaggle/input/celeba-dataset/img_align_celeba/img_align_celeba/"
ds_train_paths = tf.data.Dataset.list_files(str(img_path + '*.jpg'))
ds_train_paths = ds_train_paths.take(50000)

ds_train = ds_train_paths.map(process_path).cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(8*8*256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((8, 8, 256)))
    assert model.output_shape == (None, 8, 8, 256) # Note: None is the batch size

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 16, 16, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 32, 32, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 64, 64, 3)

    return model

def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[64, 64, 3]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1,activation='sigmoid'))

    return model

generator = make_generator_model()
discriminator = make_discriminator_model()

gan = tf.keras.models.Sequential([generator, discriminator])

discriminator.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.Adam())
discriminator.trainable = False
gan.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.Adam())
gan.summary()
def generate_and_save_images(generator, epoch, test_input=EXAMPLE_SEED, samples=NUM_EXAMPLES_TO_GENERATE):
    
    fake_faces = generator.predict(test_input)
    
    fig = plt.figure(figsize=(10,10))

    for k in range(samples):
        plt.subplot(4, 4, k+1)
        plt.imshow(fake_faces[k])
        plt.axis('off')

    plt.tight_layout()
    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()
history = []

with tf.device('/gpu:0'):
    for epoch in range(EPOCHS):

        progress_bar = Progbar(TEST_SET_SIZE, stateful_metrics=['gen_loss','disc_loss'])
        print("\nepoch {}/{}".format(epoch+1,EPOCHS))

        for X_batch in ds_train:
            #train the disceriminator
            train_label=tf.ones([BATCH_SIZE], tf.int32)
            discriminator.trainable = True
            discriminator_real_loss = discriminator.train_on_batch(X_batch,train_label)

            noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])
            fake_images = generator.predict_on_batch(noise)
            train_label=tf.zeros([BATCH_SIZE], tf.int32)
            discriminator_fake_loss = discriminator.train_on_batch(fake_images,train_label)

            #train the generator
            noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])
            train_label=np.ones(shape=(BATCH_SIZE,1))
            discriminator.trainable = False
            generator_loss = gan.train_on_batch(noise, train_label)

            values=[('gen_loss',generator_loss), ('disc_loss',discriminator_real_loss+discriminator_fake_loss)]
            progress_bar.add(BATCH_SIZE, values=values)

        history.append({'gen_loss': generator_loss, 
                        'disc_loss_fake': discriminator_fake_loss, 
                        'disc_loss_real': discrimfinator_real_loss})
        
        if epoch % 5 == 0:
             generate_and_save_images(generator,epoch)
generate_and_save_images(generator,1)
import imageio
import glob

anim_file = 'dcgan.gif'

with imageio.get_writer(anim_file, mode='I') as writer:
    filenames = glob.glob('image*.png')
    filenames = sorted(filenames)
    last = -1
    for i,filename in enumerate(filenames):
        frame = 2*(i**0.5)
        if round(frame) > round(last):
            last = frame
        else:
            continue
        image = imageio.imread(filename)
        writer.append_data(image)
    image = imageio.imread(filename)
    writer.append_data(image)

import IPython
if IPython.version_info > (6,2,0,''):
    display.Image(filename=anim_file)