import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
def show_img(path, title, figsize=(12,6)):
    plt.figure(figsize=figsize)
    img = mpimg.imread(path)
    imgplot = plt.imshow(img)
    plt.axis('off')
    plt.title(title)
    plt.show()
show_img('../input/introductiongan/introduction-gan/cumulative_gans.jpg', 'Fig.1 The GAN papers counts by The GAN Zoo')
show_img('../input/introductiongan/introduction-gan/began_face.jpg', 'Fig.2 Generated facial images by BEGAN', (8,4))
show_img('../input/introductiongan/introduction-gan/cyclegan.jpg', 'Image-to-Image translation by CycleGAN')
show_img('../input/introductiongan/introduction-gan/stackgan.jpg', 'Text-to-Image translation by StackGAN')
show_img('../input/introductiongan/introduction-gan/srgan.png', 'Super resolution by SRGAN')
show_img('../input/introductiongan/introduction-gan/inpainting.png', 'Photo inpainting with GAN')
show_img('../input/introductiongan/introduction-gan/d_simple.png', 'Discriminator with softmax/sigmoid output')
import matplotlib.pyplot as plt
import numpy as np

# a = sigmoid(wp+b)
# error = t - a
error = np.linspace(0, 1, 100)
# loss_mse = e^2
# dL/dw = 2 * e * de/dw = 2 * e * -1 * da/dw = -2 * e * (1 - a) * a * dn/dw
#  = -2 * e * (1 - a) * a * p = -2 * e * e * (1 - e) * p  ... when t = 1
gradient_mse = - error ** 2 * (1 - error)

# loss_ce = - t * log a
# dL/dw = - t * 1/a * da/dw - log a = - t * 1/a * a * (1 - a) dn/dw
#  = - t * (1 - a) * p = - e * p   ... when t = 1
gradient_ce = - error

# loss_mae = t - a = e   ... when t = 1
# dL/dw = - 1 * da/dw = -1 * a * (1 - a) * dn/dw
#  = -1 * a * (1 - a) * p = - (1 - e) * e * p
gradient_mae = - error * (1 - error)

plt.plot(error, -gradient_ce, label='CE')
plt.plot(error, -gradient_mse, label='MSE')
plt.plot(error, -gradient_mae, label='MAE')
plt.xlabel("absolute error")
plt.ylabel("norm of gradient")
plt.title("Fig3. Compare Cross-entropy with MAE, MSE")
plt.legend()
plt.show()
show_img('../input/introductiongan/introduction-gan/g_simple.png', 'The training process for Generator', (8,4))
x = np.linspace(0,1,50)[1:-1]
y_ns = -np.log(x)
y_s = np.log(1-x)
plt.plot(x, y_ns, label='Non-saturating')
plt.plot(x, y_s, label='Saturating')
plt.xlabel('$D(G(z))$')
plt.title('Fig.4 Saturating vs Non-saturating')
plt.legend()
plt.show()
show_img('../input/introductiongan/algorithm.png', 'Pseudo code of GAN')
show_img('../input/introductiongan/introduction-gan/d_train.png', 'Training Discriminator', (8,4))
show_img('../input/introductiongan/introduction-gan/g_train.png', 'Training Generator', (8,4))
show_img('../input/introductiongan/introduction-gan/d_arch.png', 'An example architecture of the Discriminator in DCGAN')
show_img('../input/introductiongan/introduction-gan/conv.png', 'Convolution operation. (Modified from indoml.com)')
show_img('../input/introductiongan/introduction-gan/g_arch.png', 'An example architecture of the Generator in DCGAN')
show_img('../input/introductiongan/introduction-gan/deconv.png', 'Transposed convolution operation. \n (View more visualizations at \n https://github.com/vdumoulin/conv_arithmetic \n by Vincent Dumoulin, Francesco Visin.)')
show_img('../input/introductiongan/introduction-gan/fid_is.png', 'FID vs IS')
# %% --------------------------------------- Load Packages -------------------------------------------------------------
import os
import random
import tensorflow as tf  # tf.__version__ >= 2.2.0
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Input, Reshape, Dense, Dropout, \
    LeakyReLU, Conv2D, Conv2DTranspose, Embedding, \
    Concatenate, multiply, Flatten, BatchNormalization
from tensorflow.keras.initializers import glorot_normal
from tensorflow.keras.optimizers import Adam
# %% --------------------------------------- Fix Seeds -----------------------------------------------------------------
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
weight_init = glorot_normal(seed=SEED)
# # Load MNIST Fashion
from tensorflow.keras.datasets.fashion_mnist import load_data
# %% ---------------------------------- Data Preparation ---------------------------------------------------------------
# change as channel last (n, dim, dim, channel)
def change_image_shape(images):
    shape_tuple = images.shape
    if len(shape_tuple) == 3:
        images = images.reshape(-1, shape_tuple[-1], shape_tuple[-1], 1)
    elif shape_tuple == 4 and shape_tuple[-1] > 3:
        images = images.reshape(-1, shape_tuple[-1], shape_tuple[-1], shape_tuple[1])
    return images

# # Load training set
(x_train, y_train), (x_test, y_test) = load_data()
x_train, x_test = change_image_shape(x_train), change_image_shape(x_test)
y_train, y_test = y_train.reshape(-1), y_test.reshape(-1)

######################## Preprocessing ##########################
# Set channel
channel = x_train.shape[-1]

# It is suggested to use [-1, 1] input for GAN training
x_train = (x_train.astype('float32') - 127.5) / 127.5
x_test = (x_test.astype('float32') - 127.5) / 127.5

# Get image size
img_size = x_train[0].shape

# Get number of classes
n_classes = len(np.unique(y_train))
# %% ---------------------------------- Hyperparameters ----------------------------------------------------------------

# optimizer = Adam(lr=0.0002, beta_1=0.5, beta_2=0.9)
latent_dim = 32
## trainRatio === times(Train D) / times(Train G)
# trainRatio = 5
# %% ---------------------------------- Models Setup -------------------------------------------------------------------
# Build Generator with convolution layer
def generator_conv():
    noise = Input(shape=(latent_dim,))
    x = Dense(3 * 3 * 128)(noise)
    x = LeakyReLU(alpha=0.2)(x)

    ## Out size: 3 x 3 x 128
    x = Reshape((3, 3, 128))(x)

    ## Size: 7 x 7 x 128
    # remove padding='same' to scale 6x6 up to 7x7
    x = Conv2DTranspose(filters=128,
                        kernel_size=(3, 3),
                        strides=(2, 2),
                        # padding='same',
                        kernel_initializer=weight_init)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    ## Size: 14 x 14 x 64
    x = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', kernel_initializer=weight_init)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    ## Size: 28 x 28 x channel
    out = Conv2DTranspose(channel, (3, 3), activation='tanh', strides=(2, 2), padding='same',
                          kernel_initializer=weight_init)(x)

    model = Model(inputs=noise, outputs=out)
    return model
# Build Discriminator with convolution layer
def discriminator_conv():
    # 28 x 28 x channel
    img = Input(img_size)

    # 14 x 14 x 32
    x = Conv2D(32, kernel_size=(3, 3), strides=(2, 2), padding='same', kernel_initializer=weight_init)(img)
    x = LeakyReLU(0.2)(x)

    # 7 x 7 x 64
    x = Conv2D(64, (3, 3), strides=(2, 2), padding='same', kernel_initializer=weight_init)(x)
    x = LeakyReLU(0.2)(x)

    # 3 x 3 x 128
    x = Conv2D(128, (3, 3), strides=(2, 2), kernel_initializer=weight_init)(x)
    x = LeakyReLU(0.2)(x)

    x = Flatten()(x)
    x = Dropout(0.4)(x)
    out = Dense(1)(x)

    model = Model(inputs=img, outputs=out)
    return model
# %% ----------------------------------- GAN Part ----------------------------------------------------------------------
# Build our GAN
class DCGAN(Model):
    def __init__(
        self,
        discriminator,
        generator,
        latent_dim,
        train_ratio=1,
    ):
        super(DCGAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.train_ratio = train_ratio

    def compile(self, d_optimizer, g_optimizer, d_loss_fn, g_loss_fn):
        super(DCGAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss_fn = d_loss_fn
        self.g_loss_fn = g_loss_fn

    def train_step(self, data):
        if isinstance(data, tuple):
            real_images = data[0]
        else:
            real_images = data
        # Get the batch size
        batch_size = tf.shape(real_images)[0]

        ########################### Train the Discriminator ###########################
        # training train_ratio times on D while training once on G
        for i in range(self.train_ratio):
            # Get the latent vector
            random_latent_vectors = tf.random.normal(
                shape=(batch_size, self.latent_dim)
            )
            with tf.GradientTape() as tape:
                # Generate fake images from the latent vector
                fake_images = self.generator(random_latent_vectors, training=True)
                # Get the logits for the fake images
                fake_logits = self.discriminator(fake_images, training=True)
                # Get the logits for real images
                real_logits = self.discriminator(real_images, training=True)

                # Calculate loss of D
                d_loss = self.d_loss_fn(real_logits, fake_logits)


            # Get the gradients w.r.t the discriminator loss
            d_gradient = tape.gradient(d_loss, self.discriminator.trainable_variables)
            # Update the weights of the discriminator using the discriminator optimizer
            self.d_optimizer.apply_gradients(
                zip(d_gradient, self.discriminator.trainable_variables)
            )

        ########################### Train the Generator ###########################
        # Get the latent vector
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        with tf.GradientTape() as tape:
            # Generate fake images using the generator
            generated_images = self.generator(random_latent_vectors, training=True)
            # Get the discriminator logits for fake images
            gen_img_logits = self.discriminator(generated_images, training=True)
            # Calculate the generator loss
            g_loss = self.g_loss_fn(gen_img_logits)

        # Get the gradients w.r.t the generator loss
        gen_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
        # Update the weights of the generator using the generator optimizer
        self.g_optimizer.apply_gradients(
            zip(gen_gradient, self.generator.trainable_variables)
        )
        return {"d_loss": d_loss, "g_loss": g_loss}
# %% ----------------------------------- Compile Models ----------------------------------------------------------------
# Optimizer for both the networks
# learning_rate=0.0002, beta_1=0.5, beta_2=0.9 are recommended
generator_optimizer = Adam(
    learning_rate=0.0002, beta_1=0.5, beta_2=0.9
)
discriminator_optimizer = Adam(
    learning_rate=0.0002, beta_1=0.5, beta_2=0.9
)

# Define the loss functions to be used for discrimiator
def discriminator_loss(real_logits, fake_logits):
    real_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=real_logits, labels=tf.ones_like(real_logits)))
    fake_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_logits, labels=tf.zeros_like(fake_logits)))

    return fake_loss + real_loss

# Define the loss functions to be used for generator
def generator_loss(fake_logits):
    fake_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_logits, labels=tf.ones_like(fake_logits)))
    return fake_loss

d_model = discriminator_conv()
g_model = generator_conv()

dcgan = DCGAN(generator=g_model,
              discriminator=d_model,
              latent_dim=latent_dim,
              train_ratio=1)

dcgan.compile(
    d_optimizer=discriminator_optimizer,
    g_optimizer=generator_optimizer,
    g_loss_fn=generator_loss,
    d_loss_fn=discriminator_loss,
)
# %% ----------------------------------- Start Training ----------------------------------------------------------------
# Plot/save generated images through training
def plt_img(generator):
    np.random.seed(42)
    n = n_classes

    noise = np.random.normal(size=(n * n, latent_dim))
    decoded_imgs = generator.predict(noise)

    decoded_imgs = decoded_imgs * 0.5 + 0.5
    x_real = x_test * 0.5 + 0.5

    plt.figure(figsize=(n, n + 1))
    for i in range(n):
        # display original
        ax = plt.subplot(n + 1, n, i + 1)
        if channel == 3:
            plt.imshow(x_real[y_test == i][0].reshape(img_size[0], img_size[1], img_size[2]))
        else:
            plt.imshow(x_real[y_test == i][0].reshape(img_size[0], img_size[1]))
            plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        for j in range(n):
            # display generation
            ax = plt.subplot(n + 1, n, (i + 1) * n + j + 1)
            if channel == 3:
                plt.imshow(decoded_imgs[i * n + j].reshape(img_size[0], img_size[1], img_size[2]))
            else:
                plt.imshow(decoded_imgs[i * n + j].reshape(img_size[0], img_size[1]))
                plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
    plt.show()
    return

############################# Start training #############################
LEARNING_STEPS = 6
for learning_step in range(LEARNING_STEPS):
    print('LEARNING STEP # ', learning_step + 1, '-' * 50)
    dcgan.fit(x_train, epochs=1, batch_size=128)
    if (learning_step+1)%2 == 0:
        plt_img(dcgan.generator)
