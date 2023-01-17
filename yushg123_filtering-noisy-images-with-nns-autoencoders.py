#Importing necessary libraries. I will explain the different layers later.

import matplotlib.pyplot as plt

import numpy as np; np.random.seed(1337) #Setting random seed for reproducible results

import tensorflow as tf

from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Dense, Activation, Flatten, Reshape, Input

from tensorflow.keras import Model

import tensorflow.keras.backend as K

print("Tensorflow Version " + tf.__version__)
import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
#Loading data - we don't want the training labels

from keras.datasets import mnist

(x_train, _), (x_test, _) = mnist.load_data()
#Normalizing the data between 0 and 1

image_size = x_train.shape[1]

x_train = np.reshape(x_train, [-1, image_size, image_size, 1])

x_test = np.reshape(x_test, [-1, image_size, image_size, 1])

x_train = x_train.astype('float32') / 255

x_test = x_test.astype('float32') / 255



# Generate corrupted MNIST images by adding noise with normal distribution centered at 0.5 and std=0.5

noise = np.random.normal(loc=0.5, scale=0.5, size=x_train.shape)

x_train_noisy = x_train + noise

noise = np.random.normal(loc=0.5, scale=0.5, size=x_test.shape)

x_test_noisy = x_test + noise



#Even after noise, we don't want values below 0 or more than 1, hence we will 

x_train_noisy = np.clip(x_train_noisy, 0., 1.)

x_test_noisy = np.clip(x_test_noisy, 0., 1.)

#Defining how small the middle compression layer should be

latent_dim = 16
#Building the Encoder - Same structure as standard CNN



enc_input = Input(shape=(28, 28, 1))



enc = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='Same', activation='relu')(enc_input)

enc = Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='Same', activation='relu')(enc)

enc = Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='Same', activation='relu')(enc)



enc_shape = K.int_shape(enc)



enc = Flatten()(enc)



#Note that here we want activation to be linear. This is because the denoiser will take in a linear input, and bring it back to image data.

#Using sigmoid or relu will lose more information making it harder for the model to learn.

enc = Dense(latent_dim)(enc)



encoder = Model(inputs=enc_input, outputs=enc)

encoder.summary()
#Building the Decoder - We will use Conv2D Transpose instead. Otherwise, Model will be identical.



dec_input = Input(shape=(latent_dim,))



dec = Dense(enc_shape[1] * enc_shape[2] * enc_shape[3])(dec_input)

dec = Reshape((enc_shape[1], enc_shape[2], enc_shape[3]))(dec)



#Note the descending number of filters

dec = Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='Same', activation='relu')(dec)

dec = Conv2DTranspose(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='Same', activation='relu')(dec)

dec = Conv2DTranspose(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='Same', activation='relu')(dec)



#Final conv2dtranspose layer to reconstruct image. We are using sigmoid activation since image is normalized between 0 and 1.

#If image was between -1 and 1, we would use tanh activation

dec = Conv2DTranspose(filters=1, kernel_size=(3, 3), padding='Same', activation='sigmoid')(dec)



decoder = Model(inputs=dec_input, outputs=dec)

decoder.summary()
#Building Full Denoising AutoEncoder

#Output of encoder is input to decoder.

denoiser = Model(inputs=enc_input, outputs=decoder(encoder(enc_input)))



#Using Adam Optimizer with default values (learning_rate = 0.001)

from tensorflow.keras.optimizers import Adam

opt = Adam()



#Loss should be mean squared error, since we want the generated output to be as close to original image.

#We also want to penalize large deviations more than small deviations.

denoiser.compile(loss='mse', optimizer=opt)



denoiser.summary()
#Training Final AutoEncoder



denoiser.fit(x_train_noisy, x_train, validation_data=(x_test_noisy, x_test), epochs=30, batch_size=500)
#Retreving Model's predictions for the test data.

recovered = denoiser.predict(x_test_noisy)
c = 1

fig=plt.figure(figsize=(8, 8))



for i in range(10):  #This only plots the first 10 images. You can change it to a larget number for more results

    for j in range(3):

        if j == 0:

            img = x_test[i, :, :].reshape(28, 28)

        elif j == 1:

            img = x_test_noisy[i, :, :].reshape(28, 28)

        elif j == 2:

            img = recovered[i, :, :].reshape(28, 28)

        else:

            pass



        fig.add_subplot(10, 3, c)

        plt.imshow(img, cmap='gray')

        c += 1

print("  Real              Noisy              Denoised")

plt.show()
#Saving Weights

denoiser.save("denoiser.h5")



#If you don't want to train the model again, you can just use the weights of this model (in the outputs section). Uncomment below line to load in these weights for fresh model.

#denoiser.load_weights(filepath='./denoiser.h5')