# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



from __future__ import absolute_import

from __future__ import division

from __future__ import print_function



import math

import numpy as np

from PIL import Image

import keras



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

output_path = "/kaggle/working"
def generator_model():

    model = keras.Sequential()

    model.add(keras.layers.Dense(input_dim=100, output_dim=1024))

    model.add(keras.layers.Activation('tanh'))

    model.add(keras.layers.Dense(128 * 7 * 7))

    model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.Activation('tanh'))

    model.add(keras.layers.Reshape((7, 7, 128), input_shape=(128 * 7 * 7,)))

    model.add(keras.layers.UpSampling2D(size=(2, 2)))

    model.add(keras.layers.Conv2D(64, (5, 5), padding='same'))

    model.add(keras.layers.Activation('tanh'))

    model.add(keras.layers.UpSampling2D(size=(2, 2)))

    model.add(keras.layers.Conv2D(1, (5, 5), padding='same'))

    model.add(keras.layers.Activation('tanh'))

    return model
def discriminator_model():

    model = keras.Sequential()

    model.add(

        keras.layers.Conv2D(64, (5, 5),

                               padding='same',

                               input_shape=(28, 28, 1))

    )

    model.add(keras.layers.Activation('tanh'))

    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(keras.layers.Conv2D(128, (5, 5)))

    model.add(keras.layers.Activation('tanh'))

    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(keras.layers.Flatten())

    model.add(keras.layers.Dense(1024))

    model.add(keras.layers.Activation('tanh'))

    model.add(keras.layers.Dense(1))

    model.add(keras.layers.Activation('sigmoid'))

    return model
def generator_containing_discriminator(g, d):

    model = keras.Sequential()

    model.add(g)

    d.trainable = False

    model.add(d)

    return model
def combine_images(generated_images):

    # 生成图片拼接

    num = generated_images.shape[0]

    width = int(math.sqrt(num))

    height = int(math.ceil(float(num)/width))

    shape = generated_images.shape[1:3]

    image = np.zeros((height*shape[0], width*shape[1]), dtype=generated_images.dtype)

    for index, img in enumerate(generated_images):

        i = int(index/width)

        j = index % width

        image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] = img[:, :, 0]

    return image
def train(BATCH_SIZE):

    # 下载的地址为：https://s3.amazonaws.com/img-datasets/mnist.npz

    # (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

    (X_train, y_train), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()

    X_train = (X_train.astype(np.float32) - 127.5) / 127.5

    X_train = X_train[:, :, :, None]  # None将3维的X_train扩展为4维

    X_test = X_test[:, :, :, None]



    d = discriminator_model()

    g = generator_model()

    d_on_g = generator_containing_discriminator(g, d)



    d_optim = keras.optimizers.SGD(lr=0.001, momentum=0.9, nesterov=True)

    g_optim = keras.optimizers.SGD(lr=0.001, momentum=0.9, nesterov=True)



    g.compile(loss='binary_crossentropy', optimizer="SGD")

    d_on_g.compile(loss='binary_crossentropy', optimizer=g_optim)



    d.trainable = True

    d.compile(loss='binary_crossentropy', optimizer=d_optim)



    for epoch in range(30):

        print("Epoch is", epoch)



        print("Number of batches", int(X_train.shape[0] / BATCH_SIZE))



        for index in range(int(X_train.shape[0] / BATCH_SIZE)):



            noise = np.random.uniform(-1, 1, size=(BATCH_SIZE, 100))



            image_batch = X_train[index * BATCH_SIZE:(index + 1) * BATCH_SIZE]



            generated_images = g.predict(noise, verbose=0)



            X = np.concatenate((image_batch, generated_images))

            y = [1] * BATCH_SIZE + [0] * BATCH_SIZE



            d_loss = d.train_on_batch(X, y)  # (2*BATCH_SIZE,28,28,1) -> (2*BATCH_SIZE,1)



            # 随机生成的噪声服从均匀分布

            noise = np.random.uniform(-1, 1, (BATCH_SIZE, 100))



            # 固定判别器

            d.trainable = False



            g_loss = d_on_g.train_on_batch(noise, [1] * BATCH_SIZE)  # (BATCH_SIZE,100) -> (BATCH_SIZE,28,28,1) -> (BATCH_SIZE,1)



            # 令判别器可训练

            d.trainable = True

            # 每经过100次迭代输出一张生成的图片

            if index % 300 == 0:

                image = combine_images(generated_images)

                image = image * 127.5 + 127.5

                Image.fromarray(image.astype(np.uint8)).save(f"{output_path}/" + str(epoch) + "_" + str(index) + ".png")

                print("batch %d d_loss : %f g_loss : %f" % (index, d_loss, g_loss))



            # 每100次迭代保存一次生成器和判别器的权重

            if index % 300 == 9:

                g.save_weights(f'{output_path}/generator.h5', True)

                d.save_weights(f'{output_path}/discriminator.h5', True)


def generate(BATCH_SIZE, nice=False):

    # 训练完模型后，可以运行该函数生成图片

    g = generator_model()

    g.compile(loss='binary_crossentropy', optimizer="SGD")

    g.load_weights(f'{output_path}/generator.h5')

    if nice:

        d = discriminator_model()

        d.compile(loss='binary_crossentropy', optimizer="SGD")

        d.load_weights(f'{output_path}/discriminator.h5')

        noise = np.random.uniform(-1, 1, (BATCH_SIZE*20, 100))

        generated_images = g.predict(noise, verbose=1)

        d_pret = d.predict(generated_images, verbose=1)

        index = np.arange(0, BATCH_SIZE*20)

        index.resize((BATCH_SIZE*20, 1))

        pre_with_index = list(np.append(d_pret, index, axis=1))

        pre_with_index.sort(key=lambda x: x[0], reverse=True)

        nice_images = np.zeros((BATCH_SIZE,) + generated_images.shape[1:3], dtype=np.float32)

        nice_images = nice_images[:, :, :, None]

        for i in range(BATCH_SIZE):

            idx = int(pre_with_index[i][1])

            nice_images[i, :, :, 0] = generated_images[idx, :, :, 0]

        image = combine_images(nice_images)

    else:

        noise = np.random.uniform(-1, 1, (BATCH_SIZE, 100))

        generated_images = g.predict(noise, verbose=0)

        image = combine_images(generated_images)

    image = image*127.5+127.5

    Image.fromarray(image.astype(np.uint8)).save(f"{output_path}/generated_image_{BATCH_SIZE}.png")
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'

d = discriminator_model()

g = generator_model()

d_on_g = generator_containing_discriminator(g, d)

# print(g.summary())

# print(d.summary())

print(d_on_g.summary())

train(100)  # 100为batch大小，可以随意指定。

generate(1)  # 132为batch大小，可以随意指定。该值大小也决定了生成的图片中含有多少个数字。

generate(32)  # 32为batch大小，可以随意指定。该值大小也决定了生成的图片中含有多少个数字。