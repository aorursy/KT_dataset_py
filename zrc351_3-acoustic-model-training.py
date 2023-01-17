import os

print(os.listdir("./"))

print(os.listdir("../input/pfb-recognition-train/single_world_train_1"))
!pip install python_speech_features
import tensorflow as tf

from python_speech_features import *

import wave

import numpy as np

import pylab as plt

import random
start_letter = 0

end_letter = 10

start_letter_asccii = start_letter

end_letter_asccii = end_letter + 1

letter_count = end_letter_asccii - start_letter_asccii
nchannels, sampwidth, framerate, nframes = None,None,None,None

def decodeWavByPath(wavPath):

    global nchannels, sampwidth, framerate, nframes

    wf = wave.open(wavPath, "rb")

    nchannels, sampwidth, framerate, nframes = wf.getparams()[:4]

    data = wf.readframes(nframes)

    wf.close()

    soundBytes = np.fromstring(data, dtype=np.int16)

    soundBytes.shape = (-1, nchannels)

    graph = soundBytes[:, 0]

    return graph
def preprocessing(soundBytes):

    soundBytes = (soundBytes - soundBytes.mean())/soundBytes.std()

    mfcc_feat0 =  mfcc(soundBytes)

    mfcc_feat1 = delta(mfcc_feat0, 1)

    mfcc_feat2 = delta(mfcc_feat0, 2)    

    feature = np.hstack((mfcc_feat0, mfcc_feat1, mfcc_feat2))

    return feature
wavsPath = [] 

indexOfWavs = []

decodeWavs = []

labelOfWavs = []



wavs_folder = r'../input/pfb-recognition-train/single_world_train_1'

file_count = 0



for chrASCII in range(start_letter_asccii, end_letter_asccii):    

    for root, dirs, files in os.walk(wavs_folder + "/" + str(chrASCII)):    

        print("letter {0} has files count: {1}".format( str(chrASCII), len(files)))

        file_count += len(files)

print("find {0} wave's files".format(file_count))



index = 0

for chrASCII in range(start_letter_asccii, end_letter_asccii):    

    for root, dirs, files in os.walk(wavs_folder + "/" + str(chrASCII)):

        for file in files:

            if os.path.splitext(file)[1].lower() == '.wav':

                wavPath = os.path.join(root, file)

                print("\r" + "Feature extraction:{0}/{1}".format(index, file_count), end="", flush=True)

                label = chrASCII - start_letter_asccii   

                decodeWavs.append(decodeWavByPath(wavPath))

                wavsPath.append(wavPath)   

                labelOfWavs.append(label)

                indexOfWavs.append(index)

                index += 1

print("\r" + "Feature extraction:{0}/{1}".format(index, file_count), end="", flush=True)

print("\nFeature extraction finished")
seq_length = 1024

divWavs = []

divLabel = []



def divWav1024(soundBytes, label):

    n = (len(soundBytes) // seq_length)

    cd_soundBytes = soundBytes[0: n * seq_length]

    cd_soundBytes = np.reshape(cd_soundBytes, (n,seq_length ))

    cd_label = [label] * n  

    return cd_soundBytes, cd_label

    

for ix in range(len(decodeWavs)):    

    print("\r" + "Div wav to seq length:{0}/{1}".format(ix, file_count), end="", flush=True)

    cd_soundBytes, cd_label = divWav1024(decodeWavs[ix], labelOfWavs[ix])

    divWavs.extend(cd_soundBytes)  

    divLabel.extend(cd_label)   



proprocessed_wavs = [preprocessing(x) for x in divWavs]

print("\ndivWavs shape:", np.shape(divWavs))

print("proprocessed_wavs shape:", np.shape(proprocessed_wavs))

print("divLabel shape:", np.shape(divLabel))
set_x = np.expand_dims(proprocessed_wavs, axis=-1)

set_y = np.expand_dims(divLabel, axis=-1)

print("set_x shape:{} ,set_y shape:{}" .format(set_x.shape, set_y.shape)) 
samples_count = set_y.shape[0]

split_boundary = int(0.8 * samples_count)

train_x = set_x[:split_boundary]

train_y = set_y[:split_boundary]

test_x = set_x[split_boundary:]

test_y = set_y[split_boundary:]



train_count = np.shape(train_x)[0]

test_count = np.shape(test_x)[0]



tf_train_x = tf.data.Dataset.from_tensor_slices(train_x)

tf_train_y = tf.data.Dataset.from_tensor_slices(train_y)

tf_test_x = tf.data.Dataset.from_tensor_slices(test_x)

tf_test_y = tf.data.Dataset.from_tensor_slices(test_y)

tf_train_set = tf.data.Dataset.zip((tf_train_x, tf_train_y))

tf_test_set = tf.data.Dataset.zip((tf_test_x, tf_test_y))

(tf_train_set, tf_test_set)
batch_size = 50

tf_train_set = tf_train_set.shuffle(samples_count).repeat().batch(batch_size)

tf_test_set = tf_train_set.batch(batch_size)
model = tf.keras.Sequential()

model.add(tf.keras.layers.Conv2D(256, (3,3), input_shape=(5,39,1), activation='relu'))

model.add(tf.keras.layers.Conv2D(256, (3,3), activation='relu'))

model.add(tf.keras.layers.GlobalAveragePooling2D())

model.add(tf.keras.layers.Dense(512, activation='relu'))

model.add(tf.keras.layers.Dense(32, activation='relu'))

model.add(tf.keras.layers.Dense(letter_count, activation='softmax'))

print(model.summary())
model.compile(optimizer=tf.keras.optimizers.Adam(), loss='sparse_categorical_crossentropy', metrics=['acc'])

steps_per_epoch = train_count//batch_size

validation_steps = test_count//batch_size
history = model.fit(tf_train_set, epochs=50, steps_per_epoch=steps_per_epoch, validation_data=tf_train_set, validation_steps=validation_steps)
MODEL_SAVE_PATH = r"./PFB_Recognition_1024.h5"

model.save(MODEL_SAVE_PATH)
plt.plot(range(len(history.history["acc"])), history.history["acc"])

plt.plot(range(len(history.history["val_acc"])), history.history["val_acc"])

plt.show()