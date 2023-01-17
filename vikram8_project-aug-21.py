import numpy as np

import pandas as pd

from keras.layers import Dense

from keras.models import Sequential

import time

# import matplotlib.pyplot as plt



input_df = pd.read_csv("/kaggle/input/A_T.csv")

Y_train = input_df['Outs']

X_train = input_df.drop(columns='Outs', axis=1, inplace=False)

Y_train = Y_train.values

Y_train = Y_train.reshape(4096, 1, 1)

X_train = X_train.values

X_train = X_train.reshape(-1, 1, 5)

print("X_train shape : ", X_train.shape)

print("Y_train shape : ", Y_train.shape)

X_log = np.log(X_train)

Y_arr = []



for i in range(X_train.shape[0]):

    max_val = max(X_train[i][0])

    int_val = max(1 - max_val, 0)

    Y_arr.append(int_val)



Y_arr = np.asarray(Y_arr)

Y_arr = Y_arr.reshape(-1, 1, 1)



# Adding a Sequential Model



model = Sequential()



model.add(Dense(512, activation='relu', kernel_initializer='random_uniform'))



model.add(Dense(1, activation='linear', kernel_initializer='normal'))



model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])



start = time.time()

result = model.fit(X_log, Y_arr, validation_split=0.4, epochs=100, batch_size=32, verbose=1)

end = time.time()

print(end-start)

batch_size = print(result.params['batch_size'])

epochs = print(result.params['epochs'])

print(batch_size)

print(epochs) #Relu
import tensorflow as tf
Ylog_predictions = log_model.predict(X_log)
plt.plot(result_log.history['mean_squared_error'])

plt.title('MSE')

plt.ylabel('Mean Squared Error')

plt.xlabel('Epochs')

plt.legend(['train'], loc='upper right')
############################################

#

#            estparamz5.py

#

############################################
def estparamz5(X1,X2,X3,X4,X5,sigma,rho,dt,r,w):

    r = r*np.array([1,1,1,1,1]) #array r

    rho2 = rho



    Sx = np.zeros([5,X1.shape[0]])

    Sx[0,:] = w[0]*X1.T

    Sx[1,:] = w[1]*X2.T

    Sx[2,:] = w[2]*X3.T

    Sx[3,:] = w[3]*X4.T

    Sx[4,:] = w[4]*X5.T

    

    LZ = 0

    for i in range(0,5):

        LZ = np.exp(r[i]*dt)*Sx[i,:] + LZ

    r = r.reshape(5, 1)

    mu = (r-0.5*(sigma**2))*dt;

    LZ2 = 0

    for i1 in range(0,5):

        for i2 in range(0,5):

            rho2x = rho2[i1,i2]

            siga  = (sigma[i1]**2+sigma[i1]**2)+2*rho2x*(sigma[i1]*sigma[i2])

            LZ2   = np.exp((mu[i1]+mu[i2])+0.5*dt*siga)

            LZ2 = LZ2*(Sx[i1,:]*Sx[i2,:])

            LZ2+=LZ2

    LZ3 = 0

    for i1 in range(0,5):

        for i2 in range(0,5):

            for i3 in range(0,5):

                siga = (sigma[i1]**2+sigma[i2]**2+sigma[i3]**2)+2*(sigma[i1]*sigma[i2]*rho2[i1,i2]+sigma[i2]*sigma[i3]*rho2[i2,i3]+sigma[i3]*sigma[i1]*rho2[i1,i3])

                LZ3 = np.exp((mu[i1]+mu[i2]+mu[i3])+0.5*dt*siga)*(Sx[i1,:]*Sx[i2,:]*Sx[i3,:]) + LZ3

    

    return LZ, LZ2, LZ3
v1 = 0.2*np.ones([5,1])
rho = np.array([[1.0, 0.79, 0.82, 0.91, 0.84],

               [0.79, 1.0, 0.73, 0.80, 0.76],

               [0.82, 0.73, 1.0, 0.77, 0.72],

               [0.91, 0.80, 0.77,1.0, 0.90],

               [0.84, 0.76, 0.72, 0.90, 1.0]])
def BSbmPut5DBasket_NN(K,r,T,sigma,rho,NSteps,Nexercise,w):

    dt = T/NSteps

    NPaths = 4096

    SPaths1 = np.random.rand(NPaths, NSteps+1) 

    SPaths2 = np.random.rand(NPaths, NSteps+1)

    SPaths3 = np.random.rand(NPaths, NSteps+1)

    SPaths4 = np.random.rand(NPaths, NSteps+1)

    SPaths5 = np.random.rand(NPaths, NSteps+1)

    local_result = any

    modexer = np.floor(NSteps/Nexercise)

    discount = np.exp(-r*dt)

    NRepl = NPaths

    ContinuationValue = np.zeros([NRepl,1])

    

    for step in range(NSteps-1,-1,-1):

        for_step = 1

        ReducedPaths = w[0]*SPaths1[:,step+1]+w[1]*SPaths2[:,step+1]+w[2]*SPaths3[:,step+1]+w[3]*SPaths4[:,step+1]+w[4]*SPaths5[:,step+1]

        IntrinsicValue = np.max(K-ReducedPaths,0)

        ZData1 = ReducedPaths.reshape(NPaths,1)



        X1 = SPaths1[:,step+1].reshape(NPaths,1)

        X2 = SPaths2[:,step+1].reshape(NPaths,1)

        X3 = SPaths3[:,step+1].reshape(NPaths,1)

        X4 = SPaths4[:,step+1].reshape(NPaths,1)

        X5 = SPaths5[:,step+1].reshape(NPaths,1)

        append_list = [ZData1,ZData1**2,ZData1**3,X1,X2,X3,X4,X5]

        RegrMat1 = np.ones([NPaths,1])

        for item_to_be_appended in append_list:

            RegrMat1 = np.append(RegrMat1,item_to_be_appended, axis=1)

        

        #if (np.remainder(step,modexer) == 0):

        OptionValue = np.maximum(IntrinsicValue,ContinuationValue)

        #else:

        #   OptionValue = ContinuationValue

        #print(OptionValue.reshape(NPaths, 1, 1))

        local_result = model.fit(X_log, OptionValue.reshape(NPaths, 1, 1), epochs=100, batch_size=512)

        X1 = SPaths1[:,step].reshape(NPaths,1)

        X2 = SPaths2[:,step].reshape(NPaths,1)

        X3 = SPaths3[:,step].reshape(NPaths,1)

        X4 = SPaths4[:,step].reshape(NPaths,1)

        X5 = SPaths5[:,step].reshape(NPaths,1)



        m1,m2,m3 = estparamz5(X1,X2,X3,X4,X5,sigma,rho,dt,r,w)

        

        m1 = m1.reshape(NPaths,1)

        m2 = m2.reshape(NPaths,1)

        m3 = m3.reshape(NPaths,1)

        #a1 = model.

        #a1 = a1.reshape(len(append_list)+1,1)

        #ContinuationValue = ((a1[0]*np.ones([len(ZData1),1]))+(a1[1]*m1)+(a1[2]*(m2))+(a1[3]*(m3))+a1[4]*X1*np.exp(r*dt)+a1[5]*X2*np.exp(r*dt)+a1[6]*X3*np.exp(r*dt)+a1[7]*X4*np.exp(r*dt)+a1[8]*X5*np.exp(r*dt))*discount

    #price = ContinuationValue[0]

    #return price
price = BSbmPut5DBasket_NN(1,0.06,1,v1,rho,9,9,0.2*np.ones([5,1]))
from keras.optimizers import Adam

from keras.layers import Input, BatchNormalization, Reshape, Flatten

from keras.layers import LeakyReLU

from keras import initializers
gan_data = []
def get_optimizer():

    return Adam(lr=0.0002, beta_1=0.5)
def get_generator(optimizer):

    generator = Sequential()

    generator.add(CuDNNLSTM(64, kernel_initializer=initializers.RandomNormal(stddev=0.02)))

    generator.add(Dense(5, activation='tanh'))

    generator.compile(loss='binary_crossentropy', optimizer=optimizer)

    

    return generator
def get_discriminator(optimizer):

    discriminator = Sequential()

    discriminator.add(Dense(64, kernel_initializer=initializers.RandomNormal(stddev=0.02)))

    discriminator.add(Dense(1, activation='sigmoid'))

    discriminator.compile(loss='binary_crossentropy', optimizer=optimizer)

    

    return discriminator
def get_gan_network(discriminator, random_dim, generator, optimizer):

    discriminator.trainable = False

    gan_input = Input(shape=(X_train.shape[0], 100,))

    x = generator(gan_input)

    gan_output = discriminator(x)

    gan = Model(inputs=gan_input, outputs=gan_output)

    gan.compile(loss='binary_crossentropy', optimizer=optimizer)

    return gan
def plot_generated_data(epoch, generator, examples=100):

    noise = np.random.normal(0, 1, size=[examples, 100])

    generated_data = generator.predict(noise)

    print(generated_data.shape)

    gan_data.append(generated_data)
def train(epochs=1, batch_size=128):

    batch_count = X_train.shape[0] // batch_size



    # Build our GAN netowrk

    adam = get_optimizer()

    generator = get_generator(adam)

    discriminator = get_discriminator(adam)

    gan = get_gan_network(discriminator, 100, generator, adam)



    for e in range(1, epochs+1):

        print('-'*15, 'Epoch %d' % e, '-'*15)

        for _ in range(1, batch_count+1):

            # Get a random set of input noise and images

            noise = np.random.normal(0, 1, size=[batch_size, 100])

            image_batch = X_train[np.random.randint(0, X_train.shape[0], size=batch_size)]



            # Generate fake data

            print(image_batch.shape)

            generated_images = generator.predict(noise.reshape(4096, 1, 100))

            print(generated_images.shape)

            X = np.concatenate([image_batch.reshape(batch_size, 5), generated_images])



            # Labels for generated and real data

            y_dis = np.zeros(2*batch_size)

            # One-sided label smoothing

            y_dis[:batch_size] = 0.9



            # Train discriminator

            discriminator.trainable = True

            discriminator.train_on_batch(X, y_dis)



            # Train generator

            noise = np.random.normal(0, 1, size=[batch_size, 100])

            y_gen = np.ones(batch_size)

            discriminator.trainable = False

            gan.train_on_batch(noise, y_gen)

            

            if e == 1 or e%20 == 0:

                plot_generated_data(e, generator)
if __name__ == '__main__':

    train(400, 4096)
import numpy as np

import pandas as pd

from keras.layers import Dense

from keras.models import Sequential

import time

import csv



# import matplotlib.pyplot as plt



act_bool = False

input_df = pd.read_csv("C:\\Users\\user3\\Downloads\\A_T _Updated.csv")

Y_train = input_df['Outs']

X_train = input_df.drop(columns='Outs', axis=1, inplace=False)

Y_train = Y_train.values

Y_train = Y_train.reshape(4096, 1, 1)

X_train = X_train.values

X_train = X_train.reshape(-1, 1, 5)

print("X_train shape : ", X_train.shape)

print("Y_train shape : ", Y_train.shape)

X_log = np.log(X_train)

Y_arr = []



for i in range(X_train.shape[0]):

    max_val = max(X_train[i][0])

    int_val = max(1 - max_val, 0)

    Y_arr.append(int_val)



Y_arr = np.asarray(Y_arr)

Y_arr = Y_arr.reshape(-1, 1, 1)



# Adding a Sequential Model



model = Sequential()



model.add(Dense(512, activation='relu', kernel_initializer='random_uniform'))



model.add(Dense(1, activation='linear', kernel_initializer='normal'))



model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])



start = time.time()

result = model.fit(X_log, Y_arr, validation_split=0.4, epochs=100, batch_size=32, verbose=1)

end = time.time()

print(end-start)

batch_size = (result.params['batch_size'])

epochs = (result.params['epochs'])

if "relu" in str(result.model.layers[0].activation):

    act_bool = True

# pd.DataFrame(result.history).to_csv("C:\\Users\\user3\\Desktop\\Vikram\\GAN_OUT.csv")

with open('C:\\Users\\user3\\Desktop\\Vikram\\GAN_OUT.csv', 'a') as fd:

    wr = csv.writer(fd, dialect='excel')

    if act_bool:

        wr.writerow("ReLU")

    else:

        wr.writerow("RMSProp")

    wr.writerow(str(batch_size))

    wr.writerow(str(epochs))

    wr.writerow(result.history['mean_squared_error'])