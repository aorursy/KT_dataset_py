import numpy as np 
import pandas as pd
import IPython.display as ipd
import math
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import soundfile as sf
import keras
import glob
import scipy
from scipy.signal import decimate
from scipy.io import wavfile
import seaborn as sns
import matplotlib.pyplot as plt

train_folder = "../input/train/Train"
train_df = pd.read_csv('../input/train/train.csv')
train_df['file'] = train_df['ID'].apply(lambda x: train_folder+'/'+str(x)+'.wav')
test_folder = "../input/test/Test"
test_df = pd.read_csv('../input/test/test.csv')
test_df['file'] = test_df['ID'].apply(lambda x: test_folder+'/'+str(x)+'.wav')
labelEncoder=LabelEncoder()
train_df['Class_id'] = labelEncoder.fit_transform(train_df['Class'])
train_df['Class'].describe()
samples_channels = [sf.read(f, dtype='float32')[0].shape for f in test_df['file']]
framerates = [sf.read(f, dtype='float32')[1] for f in test_df['file']]
channels = [1 if len(x)==1 else x[1] for x in samples_channels]
samples = [x[0] for x in samples_channels]
lengths = np.array(samples) / np.array(framerates)

pd.DataFrame({'framerate':framerates, 'channel':channels, 'sample':samples, 'length':lengths}).describe()
N_CLASSES=10
RATE = 8000
CHANNELS = 1
LENGTH = 4
SAMPLES = RATE*LENGTH
def proc_sound(data, rate):
    data = decimate(data, rate // RATE, axis=0)
    if 2==len(data.shape):
        data = np.sum(data, axis=1)
    pad = SAMPLES - len(data)
    if pad > 0:
        data = np.pad(data, ((0, pad)), mode = 'wrap')
    else:
        data = data[:SAMPLES]
    return data.reshape((-1, 1))
def fit_generator(files, labels, augments, per_batch):
    while True:
        for i in range(0, len(files), per_batch):
            signals = []
            _labels = []
            for j in range(i, min(len(files), i+per_batch)):
                file = files[j]
                label = labels[j]
                data, rate = sf.read(file, dtype='float32')
                data = proc_sound(data, rate)
                for _ in range(augments+1):
                    signals.append(np.roll(data, np.random.randint(0, SAMPLES)))
                    _labels.append(label)
            yield np.array(signals), np.array(_labels)
            
def test_generator(files, labels, per_batch):
    while True:
        signals = []
        _labels = []
        for i in range(0, per_batch):
            j = np.random.randint(0, len(files))
            file = files[j]
            label = labels[j]
            data, rate = sf.read(file, dtype='float32')
            data = proc_sound(data, rate)
            signals.append(np.roll(data, np.random.randint(0, SAMPLES)))
            _labels.append(label)
        yield np.array(signals), np.array(_labels)
        
def predict_generator(files, per_batch):
    while True:
        for i in range(0, len(files), per_batch):
            signals = []
            for j in range(i, min(len(files), i+per_batch)):
                file = files[j]
                data, rate = sf.read(file, dtype='float32')
                data = proc_sound(data, rate)
                signals.append(data)
            yield np.array(signals)
        
def steps_per_epoch(total, batch):
    return int(math.ceil(total / batch))
model = keras.models.Sequential()
model.add(keras.layers.InputLayer((SAMPLES, CHANNELS,)))
for n, k, s in ((30, 25, 5),(50, 19, 5), (100, 19, 5), (100, 19, 4), (100, 19, 4), (100, 15, 4), (100, 7, 4)):
    model.add(keras.layers.Conv1D(n, kernel_size=k, strides=s, padding='same'))
    model.add(keras.layers.LeakyReLU())
    model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(N_CLASSES, activation='softmax'))

model.summary()
per_batch = 100
epochs = 25
augments = 1

model.compile(loss='sparse_categorical_crossentropy',
              optimizer=keras.optimizers.Adam(0.01),
             metrics=['accuracy'])
model.fit_generator(fit_generator(train_df['file'], train_df['Class_id'], augments, per_batch),
                   epochs=epochs,
                   steps_per_epoch=steps_per_epoch(len(train_df), per_batch),
                   verbose=2)
predicted_probs = model.predict_generator(predict_generator(train_df['file'],per_batch),
                   steps=steps_per_epoch(len(train_df), per_batch))
predicted = np.argmax(predicted_probs, axis=1)
print(classification_report(train_df['Class_id'], predicted))
sns.heatmap(confusion_matrix(train_df['Class_id'], predicted));
predict_probs = model.predict_generator(predict_generator(test_df['file'], per_batch),
                   steps=steps_per_epoch(len(test_df), per_batch))
predicts = np.argmax(predict_probs, axis=1)
out_df = test_df[['ID']]
out_df['Class'] = labelEncoder.inverse_transform(predicts)
out_df.to_csv('submission.csv')
out_df.head(10)
for data in next(predict_generator(test_df['file'], 10)):
    ipd.display(ipd.Audio(data.flatten(), rate=RATE))