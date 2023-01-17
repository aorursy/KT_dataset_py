# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import pandas as pd
import numpy as np
import pathlib
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


print(tf.__version__)
print(keras.__version__)

    
data_train_file = "../input/thermoelectric9/Book1.csv"

df_train = pd.read_csv(data_train_file)
df_train.head()
#df_train.dtypes
input_train = df_train.iloc[0:499,0:39].values.astype('int32')
input_train_single = df_train.iloc[[0],0:39].values.astype('int32')
output_train = df_train.iloc[0:499,41].values.astype('int32')
output_train_single = df_train.iloc[[0],[40,41]].values.astype('int32')

input_train
#output_train

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import InputLayer

input_shape = [39]
model = Sequential()
model.add(InputLayer(input_shape))

#Train_X = 'All possible input values of mankind'

#add model layers
model.add(Dense(1000, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(1000, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(1000, activation='relu'))
#model.add(Dense(2000, activation='sigmoid'))
#model.add(Dense(20000, activation='sigmoid'))
#model.add(Dense(2000, activation='sigmoid'))
#model.add(Dense(100, activation='relu'))
model.add(Dense(1))

#compile model using mse as a measure of model performance
model.compile(optimizer='adam', loss='mean_squared_error')
#train model
model.fit(input_train, output_train, epochs=30)

#test_loss, test_acc = model.evaluate(input_train, output_train)

#print("Tested acc", test_acc)

# evaluate the model

prediction = model.predict(input_train)

      
print(prediction)

#print(output_train)
#print(prediction)
output_train1 = output_train.tolist()
prediction1 = prediction.tolist()
a = 0
b = 0
c = 0
d = 0
e = 0

for i in range(0,len(prediction1)):
#     print(abs(prediction[i][0]))
#     print(abs(output_train[i]))
    a += (abs(prediction1[i][0])) * (abs(output_train1[i])) #X*Y
    
    b += abs(prediction1[i][0]) #x
    c += abs(output_train1[i]) #y
    
    d += ((abs(prediction1[i][0]))**2) #X^2
    e += ((abs(output_train1[i]))**2) #Y^2
    
R = ((((len(prediction1))*a)-(b*c))/((((len(prediction1))*d)-(b**2))*(((len(prediction1))*e)-(c**2)))**(1/2))**2

print(R)
import matplotlib
import matplotlib.pyplot as plt

data = prediction
plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
plt.plot(data) # plotting by columns
plt.plot(output_train)
plt.show()
Tracking_R = []
model.summary(print_fn=lambda x: Tracking_R.append(x))
Tracking_R.append(R)
print(Tracking_R)
# Rsquared 0.4734
#
#
#
# model.add(Dense(1000, activation='relu'))
# #model.add(Dense(2000, activation='sigmoid'))
# model.add(Dense(200000, activation='sigmoid'))
# #model.add(Dense(2000, activation='sigmoid'))
# model.add(Dense(1000, activation='relu'))
# model.add(Dense(1))
print(Tracking_R)
plt.plot(data)
data_train_file = "../input/element-datate/Element_DataTE.csv"


df_element = pd.read_csv(data_train_file)
df_element.head()
import random
xxx = 1
trak= []
array_alpha = []
element_composition_alpha = []
for length in range(1000000):
    element_composition = []
    import random
    a = random.randint(1,2)
    random = random.randint(0,47)
    element_composition.append(a)
    element_composition.append(random)
    b = df_element.iloc[random,2]
    c = df_element.iloc[random,4]
    d = df_element.iloc[random,6]
    e = df_element.iloc[random,8]
    f = df_element.iloc[random,10]
    g = df_element.iloc[random,12]
    h = df_element.iloc[random,14]
    

    import random
    j = random.randint(1,3)
    random = random.randint(0,47)
    element_composition.append(j)
    element_composition.append(random)
    k = df_element.iloc[random,2]
    l = df_element.iloc[random,4] 
    m = df_element.iloc[random,6]
    n = df_element.iloc[random,8]
    o = df_element.iloc[random,10]
    p = df_element.iloc[random,12]
    q = df_element.iloc[random,14]
    
    import random
    r = random.randint(1,3)
    random = random.randint(0,47)
    element_composition.append(r)
    element_composition.append(random)
    s = df_element.iloc[random,2]
    t = df_element.iloc[random,4] 
    u = df_element.iloc[random,6]
    v = df_element.iloc[random,8]
    x = df_element.iloc[random,10]
    y = df_element.iloc[random,12]
    z = df_element.iloc[random,14]

    import random
    a1 = random.randint(0,2)
    if a1 == 1:
        aa = random.randint(1,5) 
        random = random.randint(0,47)
        element_composition.append(aa)
        element_composition.append(random)
        bb = df_element.iloc[random,2]
        cc = df_element.iloc[random,4]
        dd = df_element.iloc[random,6]
        ee = df_element.iloc[random,8]
        ff = df_element.iloc[random,10]
        gg= df_element.iloc[random,12]
        hh = df_element.iloc[random,14]
    else:
        aa = 0
        bb = 0
        cc = 0
        dd = 0
        ee = 0
        ff = 0
        gg = 0
        hh = 0
    
    import random
    i1 = random.randint(0,1)
    if i1 == 1 and a1 == 1:
        ii = random.randint(0,5)
        random = random.randint(0,47)
        element_composition.append(ii)
        element_composition.append(random)
        jj = df_element.iloc[random,2]
        kk = df_element.iloc[random,4]
        ll = df_element.iloc[random,6]
        mm = df_element.iloc[random,8]
        nn = df_element.iloc[random,10]
        oo = df_element.iloc[random,12]
        pp = df_element.iloc[random,14]
    else:
        ii = 0
        jj = 0
        kk = 0
        ll = 0
        mm = 0
        nn = 0
        oo = 0
        pp = 0
    try:
        qq = aa/(bb+cc+dd+ee+ff+gg+hh)
    except:
        qq = 0
    try:
        rr = jj/(kk+ll+mm+nn+oo)
    except:
        rr = 0
        
    zz = ((b/(c+d+e+f+g+h))+(k/(l+m+n+o+p+q))+(s/(t+u+v+x+y+z))+qq+rr)
    #print(zz)
    
    arraya = []
    if  (l+m+n+o+p+q) == (c+d+e+f+g+h):
        #print("fail1")
        pass
    elif  (c+d+e+f+g+h) == (t+u+v+x+y+z):
        #print("fail2")
        pass
    elif  (l+m+n+o+p+q) == (t+u+v+x+y+z):
        #print("fail3")
        pass
    elif (a*b + j*k + r*s + aa*bb + ii*jj)%8 != 0:
        #print((a*b + j*k + r*s + aa*bb + ii*jj)%8)
        #print("not equal to 0")
        pass
    #elif zz >= 0.6:
        #pass
        #print("zz to big")
        #print(zz)
    else:
        #print("zz small enough")
        #print("pass")
        trak.append(zz)
        arraya.extend((a,b,c,d,e,f,g,h,j,k,l,m,n,o,p,q,r,s,t,u,v,x,y,z,aa,bb,cc,dd,ee,ff,gg,hh,ii,jj,kk,ll,mm,nn,oo))
        xxx=xxx+1
        element_composition_alpha.append([element_composition])
        array_alpha.append(arraya)

array_alpha = np.array(array_alpha).astype('int32')
array_alpha
print('inputs used for training Machine Learning Algorithm')
print((input_train[0]).dtype)
print(input_train[0])

print('Procedurally generated inputs for discovering new materials')
print((array_alpha[0]).dtype)
print((array_alpha[0]))
print('')
print(array_alpha)



# test = []


# test.extend((1,2,2,8,8,2,0,0,1,2,2,8,13,2,0,0,3,6,2,6,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0))
# print(len(test))
# test1 = []
# test1.append(test)

# test1 = np.array(test1).astype('int32')

# #print(test1.dtype)
# #print(array_alpha[0].dtype)
# print(test1)
# print(type(test1))

# # print(input_train[0])
# # print(array_alpha.dtype)
# # print(input_train.dtype)
# import random
# array_alpha = []
# for length in range(1000):
#     a = random.randint(0,4)
#     b = random.randint(0, 6)
#     c = random.randint(0, 2)
#     d = random.randint(0, 8) 
#     e = random.randint(0, 18) 
#     f = random.randint(0, 32)
#     g = random.randint(0, 18)
#     h = random.randint(0, 8)


#     j = random.randint(0, 4)
#     k = random.randint(0, 6)
#     l = random.randint(0, 2) 
#     m = random.randint(0, 8) 
#     n = random.randint(0, 18)
#     o = random.randint(0, 32)
#     p = random.randint(0, 18)
#     q = random.randint(0, 8)

#     r = random.randint(0, 4)
#     s = random.randint(0, 6)
#     t = random.randint(0, 2) 
#     u = random.randint(0, 8) 
#     v = random.randint(0, 18)
#     x = random.randint(0, 32)
#     y = random.randint(0, 18)
#     z = random.randint(0, 8)

#     aa = random.randint(0, 4)
#     bb = random.randint(0, 6)
#     cc = random.randint(0, 2) 
#     dd = random.randint(0, 8) 
#     ee = random.randint(0, 18)
#     ff = random.randint(0, 32)
#     gg= random.randint(0, 18)
#     hh = random.randint(0, 8)

#     ii = random.randint(0, 4)
#     jj = random.randint(0, 6)
#     kk = random.randint(0, 2) 
#     ll = random.randint(0, 8) 
#     mm = random.randint(0, 18)
#     nn = random.randint(0, 32)
#     oo = random.randint(0, 18)
#     pp = random.randint(0, 8)
#     array = []
#     array.extend((a,b,c,d,e,f,g,h,j,k,l,m,n,o,p,q,r,s,t,u,v,x,y,z,aa,bb,cc,dd,ee,ff,gg,hh,ii,jj,kk,ll,mm,nn,oo))
    
#     array_alpha.append(array)
    
# array_alpha = np.array(array_alpha)
# array_alpha
data_train_file = "../input/thermoelectric9/Book1.csv"

df_train1 = pd.read_csv(data_train_file)
df_train1.head()
input_train1 = df_train1.iloc[0:500,0:39].values.astype('int32')
output_train1 = df_train1.iloc[0:500,41].values.astype('int32')
#prediction = model.predict(input_train1)
prediction = model.predict(array_alpha)
data = prediction

from matplotlib.pyplot import figure
figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
#plt.plot(data) # plotting by columns
plt.plot(data)
plt.plot(output_train1)
plt.show()

element_find = []
temp1 = []
from numpy import *
where_are_NaNs = isnan(data)
data[where_are_NaNs] = 0
data = data.tolist()
data = [[int(float(j)) for j in i] for i in data]
for i in range(0,len(data)):
    temp = data[i]
    temp = int(temp[0])
    if 500 < temp or temp < -500:
        #prnt(temp)
        element_find.append(i)
        temp1.append(temp)
#print(data[2])
# type(data)
#data = ''.join(str(e) for e in data)
# data = list(map(int, data))

len(element_find)
for i in range(0,len(element_find)):
    x = element_composition_alpha[element_find[i]]
    print(x)
    b = x[0]
#     c = b[0]
#     print(c)
    print(temp1[i])
    for i in range(1,len(b),2):
        d = b[i]
        z = df_element.iloc[d,1]
        print(z)
valence_sums = "../input/valence/sum.csv"
valence_sum = pd.read_csv(valence_sums)
# check if sum of element valence/sum of shell less than 0.6
t = 0
for i in range(0,len(element_find)):
    x = element_composition_alpha[element_find[i]]
    print(x)
    b = x[0]
#     c = b[0]
#     print(c)
    o = (temp1[i])
    print(o)
    y = 0
    u = []
    for i in range(1,len(b),2):
        d = b[i]
        z = df_element.iloc[d,1]
        #print(x)
        #print(o)
        print(z)
        
# valence_sum iloc doesn't match elements loc now...
#         #y += valence_sum.iloc[d,16]
#         #u.append(z)
#         #if i == len(b)-1:
#             #if y < 0.6:
#                 t +=1
#                 print(x)
#                 print(u)
#                 print(o)
#                 print(y)
print(len(element_find))
print(t)
print(u)
print(y)

#print(((b/(c+d+e+f+g+h))+(k/(l+m+n+o+p+q))+(s/(t+u+v+x+y+z))+qq+rr))



!pip install -q imageio

import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers
import time

from IPython import display
#(train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
train_images = input_train
train_images = train_images.reshape(train_images.shape[0], 39, 1, 1).astype('float32')
train_images = (train_images - 214) / 214 # Normalize the images to [-1, 1]

BUFFER_SIZE = 60000
BATCH_SIZE = 256
# Batch and shuffle the data
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
print(train_images)
def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
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

    return model
generator = make_generator_model()

noise = tf.random.normal([1, 100])
generated_image = generator(noise, training=False)

plt.imshow(generated_image[0, :, :, 0], cmap='gray')
def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                     input_shape=[28, 28, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model
discriminator = make_discriminator_model()
decision = discriminator(generated_image)
print (decision)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss
def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

EPOCHS = 50
noise_dim = 100
num_examples_to_generate = 16

# We will reuse this seed overtime (so it's easier)
# to visualize progress in the animated GIF)
seed = tf.random.normal([num_examples_to_generate, noise_dim])

# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator(noise, training=True)

      real_output = discriminator(images, training=True)
      fake_output = discriminator(generated_images, training=True)

      gen_loss = generator_loss(fake_output)
      disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
def train(dataset, epochs):
  for epoch in range(epochs):
    start = time.time()

    for image_batch in dataset:
      train_step(image_batch)

    # Produce images for the GIF as we go
    display.clear_output(wait=True)
    generate_and_save_images(generator,
                             epoch + 1,
                             seed)

    # Save the model every 15 epochs
    if (epoch + 1) % 15 == 0:
      checkpoint.save(file_prefix = checkpoint_prefix)

    print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

  # Generate after the final epoch
  display.clear_output(wait=True)
  generate_and_save_images(generator,
                           epochs,
                           seed)

def generate_and_save_images(model, epoch, test_input):
  # Notice `training` is set to False.
  # This is so all layers run in inference mode (batchnorm).
  predictions = model(test_input, training=False)

  fig = plt.figure(figsize=(4,4))

  for i in range(predictions.shape[0]):
      plt.subplot(4, 4, i+1)
      plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
      plt.axis('off')

  plt.savefig('image_at_epoch.png'.format(epoch))
  plt.show()

train(train_dataset, EPOCHS)
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

# Display a single image using the epoch number
def display_image(epoch_no):
  return PIL.Image.open('image_at_epoch_{:04d}.png'.format(epoch_no))

display_image(EPOCHS)

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

