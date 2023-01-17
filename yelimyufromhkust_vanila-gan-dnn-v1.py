# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import seaborn as sns

from tqdm import tqdm_notebook as tqdm

from keras.models import Model

from keras.layers import Input, Reshape

from keras.layers.core import Dense, Activation, Dropout, Flatten

from keras.layers.normalization import BatchNormalization

from keras.layers.convolutional import UpSampling1D, Conv1D

from keras.layers.advanced_activations import LeakyReLU

from keras.optimizers import Adam, SGD

from keras.callbacks import TensorBoard

import tensorflow as tf

import matplotlib.pyplot as plt

%matplotlib inline
try: 

    data=pd.read_csv("../input/creditcard.csv")

except Exception as e:

    data=pd.read_csv("creditcard.csv")

data.head()
class1_data=data[data.Class==1]

class1_data=class1_data.drop(['Time','Amount','Class'],axis=1)

class1_data.values
real_data = class1_data.values
def get_generative(G_in, dense_dim=200, out_dim=28, lr=1e-3): #out_dim = 28인 이유는 한 array 의 []당 28개씩 real 데이터가 들어가니까.. 

    x = Dense(dense_dim)(G_in)

    x = Activation('tanh')(x)

    G_out = Dense(out_dim, activation='tanh')(x)

    G = Model(G_in, G_out)

    opt = SGD(lr=lr)

    G.compile(loss='binary_crossentropy', optimizer=opt)

    return G, G_out



G_in = Input(shape=[10]) # <tf.Tensor 'input_1:0' shape=(?, 10) dtype=float32>

G, G_out = get_generative(G_in) #G_out= <tf.Tensor 'dense_2/Tanh:0' shape=(?, 50) dtype=float32>

G.summary()
def get_discriminative(D_in, lr=1e-3, drate=.25, n_channels=28, conv_sz=5, leak=.2):

    x = Reshape((-1, 1))(D_in)

    x = Conv1D(n_channels, conv_sz, activation='relu')(x)

    x = Dropout(drate)(x)

    x = Flatten()(x)

    x = Dense(n_channels)(x)

    D_out = Dense(2, activation='sigmoid')(x)

    D = Model(D_in, D_out)

    dopt = Adam(lr=lr)

    D.compile(loss='binary_crossentropy', optimizer=dopt)

    return D, D_out



D_in = Input(shape=[28])

D, D_out = get_discriminative(D_in)

D.summary()
def set_trainability(model, trainable=False):

    model.trainable = trainable

    for layer in model.layers:

        layer.trainable = trainable

        

def make_gan(GAN_in, G, D):

    set_trainability(D, False)

    x = G(GAN_in)

    GAN_out = D(x)

    GAN = Model(GAN_in, GAN_out)

    GAN.compile(loss='binary_crossentropy', optimizer=G.optimizer)

    return GAN, GAN_out



GAN_in = Input([10])

GAN, GAN_out = make_gan(GAN_in, G, D)

GAN.summary()
#n_samples=10000 -> real_data 속 [] 의 갯수는 총 492개..! 그래서 n_samples 자리에 492가 들어감. 



def sample_data_and_gen(G, noise_dim=10 ):

    XT = real_data #예시임.. 

    XN_noise = np.random.uniform(0,1,size=[492,noise_dim ])

    XN = G.predict(XN_noise)

    X = np.concatenate((XT,XN))

    y = np.zeros((2*492, 2))

    y[:492, 1] = 1

    y[492:, 0] = 1

    return X, y
def pretrain(G, D, noise_dim=10,batch_size=32):

    X, y = sample_data_and_gen(G, noise_dim=noise_dim)

    set_trainability(D, True)

    D.fit(X, y, epochs=1, batch_size=batch_size)



pretrain(G, D)
#여기서 492는 array[[],[],[],...[]] 일떄 안쪽 [] 의 갯수가 492개라는 얘기..!



def sample_noise(G, noise_dim=10, ):

    X = np.random.uniform(0, 1, size=[492, noise_dim])

    y = np.zeros((492, 2))

    y[:, 1] = 1

    return X, y
def train(GAN, G, D, epochs=500, noise_dim=10, batch_size=32, verbose=False, v_freq=50):

    d_loss = []

    g_loss = []

    e_range = range(epochs)

    if verbose:

        e_range = tqdm(e_range)

    for epoch in e_range:

        X, y = sample_data_and_gen(G, noise_dim=noise_dim)

        set_trainability(D, True)

        d_loss.append(D.train_on_batch(X, y))

        

        X, y = sample_noise(G, noise_dim=noise_dim)

        set_trainability(D, False)

        g_loss.append(GAN.train_on_batch(X, y))

        if verbose and (epoch + 1) % v_freq == 0:

            print("Epoch #{}: Generative Loss: {}, Discriminative Loss: {}".format(epoch + 1, g_loss[-1], d_loss[-1]))

    return d_loss, g_loss
d_loss, g_loss = train(GAN, G, D, verbose=True)
ax = pd.DataFrame(

    {

        'Generative Loss': g_loss,

        'Discriminative Loss': d_loss,

    }

).plot(title='Training loss', logy=True)

ax.set_xlabel("Epochs")

ax.set_ylabel("Loss")
#원래 데이터 492개를 을 2배로 만들엇음..! -> 총 981개의 샘플 생성..! (뒤에서 492개만 보면댐.. )

N_VIEWED_SAMPLES = 2

data_and_gen, _ = sample_data_and_gen(G)

pd.DataFrame(np.transpose(data_and_gen[N_VIEWED_SAMPLES:])).plot()
all_class1_df = [pd.DataFrame(data_and_gen)]

for i in range (577):

    new_data,_=sample_data_and_gen(G)

    all_class1_df.append(pd.DataFrame(new_data)[492:])

class1_data_balanced = pd.concat(all_class1_df)    
#컬럼명 추가하기

class1_data_balanced.columns=["V1","V2","V3","V4","V5","V6","V7","V8","V9","V10","V11","V12","V13","V14","V15","V16","V17","V18","V19","V20","V21","V22","V23","V24","V25","V26","V27","V28"]
label_class1=[]

for j in range(284868):

    label_class1.append(1)

label_class1 = pd.DataFrame(label_class1)

label_class1.columns=["Class"]

label_class1.tail()
#CLASS 1 리얼 마지막! -> index가 이상하긴 하지만 일단 row갯수는 284868개 임!!





class1_data_balanced=class1_data_balanced.reset_index()

class1_data_balanced = pd.concat([class1_data_balanced,label_class1],axis=1)

class1_data_balanced.tail()
class0_data=data[data.Class==0].drop(["Time","Amount"],axis=1)

class0_data.head()

len(class0_data)
class1_data_balanced=class1_data_balanced.drop(["index"],axis=1)

class1_data_balanced.tail()
frame=[class0_data,class1_data_balanced]

final_df=pd.concat(frame)

final_df=final_df.reset_index()

final_df=final_df.drop(["index"],axis=1)

final_df.tail()
from sklearn.model_selection import train_test_split

X = final_df.drop(['Class'], axis = 1)

y = final_df['Class']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
x_val = X_train[:100000]

partial_x_train = X_train[100000:]



y_val = y_train[:100000]

partial_y_train = y_train[100000:]
x_val = np.asarray(x_val).astype('float32')

partial_x_train = np.asarray(partial_x_train).astype('float32')

y_val = np.asarray(y_val).astype('float32')

partial_y_train = np.asarray(partial_y_train).astype('float32')
from keras import models

from keras import layers



model = models.Sequential()



model.add(layers.Dense(28, activation='relu', input_dim=28))

model.add(Dropout(0.5))

model.add(layers.Dense(27, activation='relu'))

model.add(Dropout(0.5))

model.add(layers.Dense(26, activation='relu'))

model.add(Dropout(0.5))

model.add(layers.Dense(25, activation='relu'))

model.add(Dropout(0.5))

model.add(layers.Dense(24, activation='relu'))

model.add(Dropout(0.5)) 

model.add(layers.Dense(12, activation='relu'))

model.add(Dropout(0.5))  

model.add(layers.Dense(6, activation='relu'))

model.add(Dropout(0.5))  

model.add(layers.Dense(3, activation='relu'))

model.add(Dropout(0.5))  





model.add(layers.Dense(2, activation='softmax'))
from keras import optimizers



model.compile(optimizer=optimizers.RMSprop(lr=0.001),

              loss='binary_crossentropy',

              metrics=['accuracy'])
from keras.utils import to_categorical

y_val = to_categorical(y_val)

partial_y_train = to_categorical(partial_y_train)
history = model.fit(partial_x_train,

                    partial_y_train,

                    epochs=50,

                    batch_size=500,

                    validation_data=(x_val, y_val))
history_dict = history.history

history_dict.keys()


acc = history.history['acc']

val_acc = history.history['val_acc']

loss = history.history['loss']

val_loss = history.history['val_loss']



epochs = range(1, len(acc) + 1)



# ‘bo’는 파란색 점을 의미합니다

plt.plot(epochs, loss, 'bo', label='Training loss')

# ‘b’는 파란색 실선을 의미합니다

plt.plot(epochs, val_loss, 'b', label='Validation loss')

plt.title('Training and validation loss')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()



plt.show()
plt.clf()   # 그래프를 초기화

acc = history_dict['acc']

val_acc = history_dict['val_acc']



plt.plot(epochs, acc, 'bo', label='Training acc')

plt.plot(epochs, val_acc, 'b', label='Validation acc')

plt.title('Training and validation accuracy')

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.legend()



plt.show()
X_test = np.asarray(X_test).astype('float32')

y_test = np.asarray(y_test).astype('float32')

y_test = to_categorical(y_test)
results = model.evaluate(X_test, y_test)
results #99% 정확도!