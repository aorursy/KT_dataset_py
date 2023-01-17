## Using Google's WaveNet to create a

## user-intent-to-speech call-center engine



import pandas as pd 

import numpy as np



import os

import numpy as np

import matplotlib.pyplot as plt

from keras.utils.np_utils import to_categorical

from keras.preprocessing.sequence import pad_sequences

from sklearn.preprocessing import LabelEncoder

import pandas as pd

from keras.metrics import top_k_categorical_accuracy

def top_3_accuracy(x,y): return top_k_categorical_accuracy(x,y, 3)

from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau

from glob import glob

import gc

gc.enable()

def get_available_gpus():

    from tensorflow.python.client import device_lib

    local_device_protos = device_lib.list_local_devices()

    return [x.name for x in local_device_protos if x.device_type == 'GPU']



## IMPT!: spectogram and speech PRE-PROCESSED prior

#TRAIN_PATH = '../input/train-npy/'

#TEST_PATH = '../input/test-npy/'

#TRAIN_PATH = '../input/train-5000-npy/'

#TEST_PATH = '../input/test-5000-npy/'

TRAIN_PATH = '../input/train-npy-10000/'

TEST_PATH = '../input/test-npy-10000/'



# male

#train_X_male = np.load(TRAIN_PATH + 'train_X_male.npy')

#train_Y_male = np.load(TRAIN_PATH + 'train_Y_male.npy')

#test_X_male = np.load(TEST_PATH + 'test_X_male.npy')

#test_Y_male = np.load(TEST_PATH + 'test_Y_male.npy')



#train_X_male = np.load(TRAIN_PATH + 'train_X_male_5000.npy')

#train_Y_male = np.load(TRAIN_PATH + 'train_Y_male_5000.npy')

#test_X_male = np.load(TEST_PATH + 'test_X_male_5000.npy')

#test_Y_male = np.load(TEST_PATH + 'test_Y_male_5000.npy')



train_X_male = np.load(TRAIN_PATH + 'train_X_male_10000.npy')

train_Y_male = np.load(TRAIN_PATH + 'train_Y_male_10000.npy')

test_X_male = np.load(TEST_PATH + 'test_X_male_10000.npy')

test_Y_male = np.load(TEST_PATH + 'test_Y_male_10000.npy')





# female

#train_X_female = np.load(TRAIN_PATH + 'train_X_female.npy')

#train_Y_female = np.load(TRAIN_PATH + 'train_Y_female.npy')

#test_X_female = np.load(TEST_PATH + 'test_X_female.npy')

#test_Y_female = np.load(TEST_PATH + 'test_Y_female.npy')



#train_X_female = np.load(TRAIN_PATH + 'train_X_female_5000.npy')

#train_Y_female = np.load(TRAIN_PATH + 'train_Y_female_5000.npy')

#test_X_female = np.load(TEST_PATH + 'test_X_female_5000.npy')

#test_Y_female = np.load(TEST_PATH + 'test_Y_female_5000.npy')



train_X_female = np.load(TRAIN_PATH + 'train_X_female_10000.npy')

train_Y_female = np.load(TRAIN_PATH + 'train_Y_female_10000.npy')

test_X_female = np.load(TEST_PATH + 'test_X_female_10000.npy')

test_Y_female = np.load(TEST_PATH + 'test_Y_female_10000.npy')

# IMPT: make sure train_X and train_Y dimensions match

print(train_X_male)

print(train_Y_male)

print(train_X_male.shape)

print(train_Y_male.shape)

print(train_X_female.shape)

print(train_Y_female.shape)

# IMPT: check duration match for train X and Y

assert(train_X_male.shape[1] == train_Y_male.shape[1])

assert(train_X_female.shape[1] == train_Y_female.shape[1])
train_X = train_X_female

train_y = train_Y_female
## IMPT!: WaveNet Mu-Law Quantize 



## Quantize and Categorize audio output y in order to use Softmax according to paper

## Softmax final activation according to paper performs better than linear 



# IMPT: need to keep MIN_VALUE and MAX_VALUE constants for inverse later! 

MIN_Y = np.min(train_y)

MAX_Y = np.max(train_y) 



# mu-law transformation according to paper 

train_y = 2 * (train_y - MIN_Y) / (MAX_Y - MIN_Y) - 1 # normalize to -1 and 1

assert(np.min(train_y) == -1 and np.max(train_y) == 1) # check

mu = 255

train_y = np.sign(train_y) * np.log(1 + mu * np.abs(train_y)) / (np.log(1 + mu)) # mu-law transformation



# quantize from 0 to 256 categorical labels acording to paper 

NO_BINS = 256

_, bin_labels = np.histogram(train_y, NO_BINS-1, range=(MIN_Y, MAX_Y))

quant_y = np.empty((train_y.shape[0], train_y.shape[1])) 

for i in range(train_y.shape[0]):

    # IMPT: return bin labels of range(256)

    quant_y[i,:] = pd.cut(train_y[i,:], NO_BINS, right=False, labels=range(256))    

assert(len(np.unique(quant_y)) == 256) # check

train_y = quant_y # assign 



# one hot encoding to *2-dimensional* 256 categorical for softmax

# IMPT!: ouput only kept at 2-dimesional: (num_data * audio_duration x 256 one hot categories)

train_y = to_categorical(train_y) # assign 

assert(np.sum(train_y[0]) != 0)
print(train_y)

print(train_y.shape)
from keras.layers import Conv1D, Input, Activation, AveragePooling1D, Add, Multiply, GlobalAveragePooling1D

from keras.layers import AveragePooling2D, GlobalAveragePooling2D

from keras.models import Model



# parameters

n_filters = 64

dilation_depth = 8

# softmax performs better according to paper

# but need to perform mu-law quantization to 256 categoricals according to paper 

final_activation = 'softmax' 

#final_activation = 'linear'

scale_ratio = 1

kernel_size = 2

pool_size_1 = 4

pool_size_2 = 8

STROKE_COUNT = 196

TRAIN_SAMPLES = 750

VALID_SAMPLES = 75

TEST_SAMPLES = 50



input_shape = train_X.shape[1:]

output_shape = train_y.shape[1:]



def residual_block(x, i):

    tanh_out = Conv1D(n_filters, 

                      kernel_size, 

                      dilation_rate = kernel_size**i, 

                      padding='causal', 

                      name='dilated_conv_%d_tanh' % (kernel_size ** i), 

                      activation='tanh'

                      )(x)

    sigm_out = Conv1D(n_filters, 

                      kernel_size, 

                      dilation_rate = kernel_size**i, 

                      padding='causal', 

                      name='dilated_conv_%d_sigm' % (kernel_size ** i), 

                      activation='sigmoid'

                      )(x)

    z = Multiply(name='gated_activation_%d' % (i))([tanh_out, sigm_out])

    skip = Conv1D(n_filters, 1, name='skip_%d'%(i))(z)

    res = Add(name='residual_block_%d' % (i))([skip, x])

    return res, skip

x = Input(shape=input_shape, name='original_input')

skip_connections = []

out = Conv1D(n_filters, 2, dilation_rate=1, padding='causal', name='dilated_conv_1')(x)

for i in range(1, dilation_depth + 1):

    out, skip = residual_block(out,i)

    skip_connections.append(skip)

out = Add(name='skip_connections')(skip_connections)

out = Activation('relu')(out)



out = Conv1D(n_filters, pool_size_1, strides = 1, padding='same', name='conv_5ms', activation = 'relu')(out)

#out = AveragePooling1D(pool_size_1, padding='same', name='downsample_to_200Hz')(out)

#out = AveragePooling1D(padding='same', name='downsample_to_200Hz')(out)



out = Conv1D(n_filters, pool_size_2, padding='same', activation='relu', name='conv_500ms')(out)

#out = Conv1D(output_shape[0], pool_size_2, padding='same', activation='relu', name='conv_500ms_target_shape')(out)

out = Conv1D(output_shape[1], pool_size_2, padding='same', activation='relu', name='conv_500ms_target_shape')(out)

#out = AveragePooling1D(pool_size_2, padding='same',name = 'downsample_to_2Hz')(out)



#out = Conv1D(output_shape[0], (int) (input_shape[0] / (pool_size_1*pool_size_2)), padding='same', name='final_conv')(out)

out = Conv1D(output_shape[1], (int) (input_shape[0] / (pool_size_1*pool_size_2)), padding='same', name='final_conv')(out)

#out = GlobalAveragePooling1D(name='final_pooling')(out)

out = Activation(final_activation, name='final_activation')(out)



wavenet_model = Model(x, out)  

wavenet_model.compile(optimizer='adam', 

      loss='categorical_crossentropy',

      metrics=['accuracy'])

wavenet_model.summary()
epochs = 1000 # approximately acc of 0.7131, 2 hours run time

#epochs = 50

wavenet_model.fit(train_X, train_y, epochs = epochs)
## IMPT: Save model 



acc = wavenet_model.evaluate(train_X, train_y)[1]

print("Accuracy: {}".format(acc))

model_json = wavenet_model.to_json()

with open("wavenet_model_{}_epochs_{}_acc.json".format(epochs, acc), "w") as json_file:

    json_file.write(model_json)



wavenet_model.save_weights("wavenet_model_{}_epochs_{}_acc.h5".format(epochs, acc))



# check

assert(os.path.exists("wavenet_model_{}_epochs_{}_acc.json".format(epochs, acc)) == True)

assert(os.path.exists("wavenet_model_{}_epochs_{}_acc.h5".format(epochs, acc)) == True)
predicted = wavenet_model.predict(train_X)

predicted = np.argmax(predicted, axis=2) # take argmax axis=1

print(predicted[0]) # check

print(predicted.shape)
## Inverse predicted back to wav data using bin_labels, MAX_Y and MIN_Y from before



# reverse: reverse quantize -> reverse mu-law -> reverse normalize -1 1



# reverse quantization from bin_labels  

predicted_wav = np.zeros((predicted.shape[0], predicted.shape[1])) 

for i in range(predicted.shape[0]):

    predicted_wav[i,:] = bin_labels[predicted[i]] 

assert(np.sum(predicted_wav) != 0)

print(predicted_wav[0]) # check



# inverse mu-law function 

mu = 255 

predicted_wav = (np.exp(predicted_wav * np.log(1 + mu)) - 1) / mu 

assert(np.sum(predicted_wav) != 0)

print(predicted_wav[0])



# re-normalize back to wav from -1 1

# IMPT: need MAX_Y and MIN_Y constants before not new np.min

predicted_wav = (predicted_wav + 1 / 2) * (MAX_Y - MIN_Y) + MIN_Y  

print(predicted_wav[0]) # check
import scipy.io.wavfile

predicted_0 = predicted_wav[0]

original_0 = train_Y_female[0]

sr = 4410



print("Predicted")

print('Data=wave amplitude L/R stereo channels:', predicted_0)

print('Sampling rate=timesteps/seconds:', sr)

print('Audio length=data.shape[0]/sr:', predicted_0.shape[0]/sr, 'seconds')

print('Lowest amplitude L channel:', min(predicted_0))

print('Highest amplitude L channel:', max(predicted_0))

print('Data file written as predicted_0.wav and original_0.wav')

print("\n")



print("Original")

print('Data=wave amplitude L/R stereo channels:', original_0)

print('Sampling rate=timesteps/seconds:', sr)

print('Audio length=data.shape[0]/sr:', original_0.shape[0]/sr, 'seconds')

print('Lowest amplitude L channel:', min(original_0))

print('Highest amplitude L channel:', max(original_0))

print('Data file written as predicted_0.wav and original_0.wav')



# IMPT: need to cast as np.int16 if not audio will not work

scipy.io.wavfile.write('predicted_0.wav', 44100, np.int16(predicted_0))

scipy.io.wavfile.write('original_0.wav', 44100, np.int16(original_0))
import IPython

IPython.display.Audio('predicted_0.wav')
import IPython

IPython.display.Audio('original_0.wav')
# plot waves

import matplotlib.pyplot as plt

import scipy.io.wavfile

pred_sr, pred_data = scipy.io.wavfile.read('predicted_0.wav')

ori_sr, ori_data = scipy.io.wavfile.read('original_0.wav')

plt.title("Predicted (blue) and Original (orange) data")

plt.plot(pred_data)

plt.plot(ori_data)

plt.show()