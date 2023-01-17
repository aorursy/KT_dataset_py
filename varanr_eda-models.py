# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

from scipy.io import wavfile as wav

print(os.listdir("../input"))

import matplotlib.pyplot as plt

from IPython.display import Audio

from tqdm import tqdm

from sklearn.preprocessing import MultiLabelBinarizer

from scipy import signal

import numpy as np

import librosa

from keras.preprocessing.sequence import pad_sequences

import sklearn.metrics

import glob

import json

import pickle

%matplotlib inline

# Any results you write to the current directory are saved as output.
MAX_LEN = 750
df_c = pd.read_csv('../input/freesound-audio-tagging-2019/train_curated.csv')

df_c_n = pd.read_csv('../input/freesound-audio-tagging-2019/train_noisy.csv')
df_c_n.info()
df_c.head()
df_c['labels'].describe()
#visualization

rate, data = wav.read('../input/freesound-audio-tagging-2019/train_noisy/00097e21.wav')

plt.plot(data)

plt.show()
#Listening to a sample

Audio(filename='../input/freesound-audio-tagging-2019/train_curated/0026c7cb.wav')
# feature extraction 

def extract_feature(path):

    X, sample_rate = librosa.load(path)

    stft = np.abs(librosa.stft(X))

    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)

    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)

    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)

    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)

    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)

    return mfccs,chroma,mel,contrast,tonnetz



def parse_audio_files(files):

    # n: number of classes

    features = np.empty((0,193))

    for i in tqdm(range(len(files))):

        f = files[i]

        mfccs, chroma, mel, contrast,tonnetz = extract_feature(f)

        ext_features = np.hstack([mfccs,chroma,mel,contrast,tonnetz])

        features = np.vstack([features,ext_features])

    return np.array(features)

#for creating dataset with only single label entries

def create_single_labels(df):

    df_temp = df

    idx = len(df_temp)

    rem_idx = []

    for index, row in df_temp.iterrows():

        labels = row['labels'].split(',')

        if len(labels) == 1:

            continue

        for label in labels:

            df_temp.loc[idx] = [row['fname']] + [label]

            idx += 1

        rem_idx.append(index)    

        #print(labels)

        #print(row['labels'], row['fname'], index)

    df_temp.drop(rem_idx, inplace=True)

    df_temp.reset_index(drop=True, inplace=True)

    return df_temp
#generate log spectrogram for audio files

def log_specgram(audio, sample_rate, window_size=10,

                 step_size=10, eps=1e-10):

    nperseg = int(round(window_size * sample_rate / 1e3))

    noverlap = int(round(step_size * sample_rate / 1e3))

    _, _, spec = signal.spectrogram(audio,

                                    fs=sample_rate,

                                    window='hann',

                                    nperseg=nperseg,

                                    noverlap=noverlap,

                                    detrend=False)

    return np.log(spec.T.astype(np.float32) + eps)
def audio_to_data(path):

    # we take a single path and convert it into data

    sample_rate, audio = wav.read(path)

    spectrogram = log_specgram(audio, sample_rate, 10, 0)

    return spectrogram.T
spec = audio_to_data('../input/freesound-audio-tagging-2019/train_noisy/00097e21.wav')

plt.figure(figsize = (15,4))

plt.imshow(spec, aspect='auto', origin='lower')
#one hot encoding

y = []

y_n = []

for index, row in df_c.iterrows():

    labels = row['labels'].split(',')

    y.append(labels)

    

for index, row in df_c_n.iterrows():

    labels_n = row['labels'].split(',')

    y_n.append(labels_n)

# Create MultiLabelBinarizer object

one_hot = MultiLabelBinarizer()

one_hot_n = MultiLabelBinarizer()



# One-hot encode label data

y = one_hot.fit_transform(y)

y_n = one_hot_n.fit_transform(y_n)
def load_data(df, loc):

#     y = []

    data = []

    for idx, row in df.iterrows():

#         labels = row['labels'].split(',')

#         y.append(labels)

        data.append(audio_to_data(loc + row['fname']))

    temp = [pad_sequences(x, maxlen=MAX_LEN, padding='post', truncating='post') for x in data]

    return np.array(temp)

        
def fit_feature_model(model, y, path='../input/audio-feature-extraction/d.pkl', epochs=2, batch_size=128, save_w = True):

    data = np.load(path, allow_pickle=True)

    data = np.reshape(data , (data.shape[0],data.shape[1], 1))

    model.fit(data, y, batch_size=batch_size, epochs=epochs, validation_split=0.2)

    if save_w:

        model.save_weights("weights.h5")

        model_json = model.to_json()

        with open("model.json", "w") as json_file:

            json_file.write(model_json)

    return model
def learn_by_parts(df, y, model, part_size=497, loc='../input/freesound-audio-tagging-2019/train_curated/', epochs=2, batch_size=128, wrn=False, save_w = True):

    length = len(df) // part_size

    for i in range(length):

        print("PART " + str(i+1)  + ' of ' + str(length) + '\n')

        x = load_data(df[i*part_size:(i+1)*part_size], loc)

        if wrn:

            x = np.reshape(x, (part_size,221,MAX_LEN,1))

        model.fit(x, y[i*part_size:(i+1)*part_size], batch_size=batch_size, epochs=epochs, validation_split=0.2)

        print()

    if save_w:

        model.save_weights("weights.h5")

        model_json = model.to_json()

        with open("model.json", "w") as json_file:

            json_file.write(model_json)

    return model
from keras.models import Sequential, Model

from keras.layers import LSTM, Dense, Dropout, BatchNormalization, Activation, Input, Add, CuDNNLSTM, Bidirectional, Conv2D, MaxPooling2D, AveragePooling2D, Flatten

from keras.regularizers import l2

from sklearn.model_selection import train_test_split

from keras import metrics

import keras.backend as K
def multilabel_loss(y_true, y_pred):

    # Avoid divide by 0

    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())

    # Multi-task loss

    return K.mean(K.sum(- y_true * K.log(y_pred) - (1 - y_true) * K.log(1 - y_pred), axis=1))
model_f = Sequential()

model_f.add(LSTM(units=128, dropout=0.05, recurrent_dropout=0.35, return_sequences=True, input_shape=(193,1)))

model_f.add(LSTM(units=32,  dropout=0.05, recurrent_dropout=0.35, return_sequences=False))

model_f.add(Dense(units=80, activation="sigmoid"))

model_f.compile(optimizer = 'Adam', loss = 'binary_crossentropy', metrics = ['accuracy',multilabel_loss])
#model 1

model = Sequential()

model.add(LSTM(256, input_shape = (221, MAX_LEN)))

model.add(Dense(1028))

#model.add(Dropout(0.2))

model.add(Dense(128))

#model.add(Dropout(0.2))

model.add(Dense(80, activation = 'sigmoid'))

model.compile(optimizer = 'Adam', loss = 'binary_crossentropy', metrics = [multilabel_loss])
def main_block(x, filters, n, strides, dropout):

	# Normal part

	x_res = Conv2D(filters, (3,3), strides=strides, padding="same")(x)# , kernel_regularizer=l2(5e-4)

	x_res = BatchNormalization()(x_res)

	x_res = Activation('relu')(x_res)

	x_res = Conv2D(filters, (3,3), padding="same")(x_res)

	# Alternative branch

	x = Conv2D(filters, (1,1), strides=strides)(x)

	# Merge Branches

	x = Add()([x_res, x])



	for i in range(n-1):

		# Residual conection

		x_res = BatchNormalization()(x)

		x_res = Activation('relu')(x_res)

		x_res = Conv2D(filters, (3,3), padding="same")(x_res)

		# Apply dropout if given

		if dropout: x_res = Dropout(dropout)(x)

		# Second part

		x_res = BatchNormalization()(x_res)

		x_res = Activation('relu')(x_res)

		x_res = Conv2D(filters, (3,3), padding="same")(x_res)

		# Merge branches

		x = Add()([x, x_res])



	# Inter block part

	x = BatchNormalization()(x)

	x = Activation('relu')(x)

	return x



def build_model(input_dims, output_dim, n, k, act= "relu", dropout=None):

	""" Builds the model. Params:

			- n: number of layers. WRNs are of the form WRN-N-K

				 It must satisfy that (N-4)%6 = 0

			- k: Widening factor. WRNs are of the form WRN-N-K

				 It must satisfy that K%2 = 0

			- input_dims: input dimensions for the model

			- output_dim: output dimensions for the model

			- dropout: dropout rate - default=0 (not recomended >0.3)

			- act: activation function - default=relu. Build your custom

				   one with keras.backend (ex: swish, e-swish)

	"""

	# Ensure n & k are correct

	assert (n-4)%6 == 0

	assert k%2 == 0

	n = (n-4)//6 

	# This returns a tensor input to the model

	inputs = Input(shape=(input_dims))



	# Head of the model

	x = Conv2D(16, (3,3), padding="same")(inputs)

	x = BatchNormalization()(x)

	x = Activation('relu')(x)



	# 3 Blocks (normal-residual)

	x = main_block(x, 16*k, n, (1,1), dropout) # 0

	x = main_block(x, 32*k, n, (2,2), dropout) # 1

	x = main_block(x, 64*k, n, (2,2), dropout) # 2

			

	# Final part of the model

	x = AveragePooling2D((8,8))(x)

	x = Flatten()(x)

	outputs = Dense(output_dim, activation="softmax")(x)



	model = Model(inputs=inputs, outputs=outputs)

	return model
model = build_model((221,MAX_LEN,1), 80,16 , 4)

model.compile("adam","binary_crossentropy", [multilabel_loss])
def load_test(paths):

    data = []

    for path in tqdm(paths):

        data.append(audio_to_data(path))

    temp = [pad_sequences(x, maxlen=MAX_LEN, padding='post', truncating='post') for x in data]

    return np.array(temp), paths
def output(model, wrn=False, feature=False):

    if feature:

        test = parse_audio_files(glob.glob('../input/freesound-audio-tagging-2019/test/*.wav'))

        test = np.reshape(test, (test.shape[0], test.shape[1], 1))

    else:

        test, fname = load_test(glob.glob('../input/freesound-audio-tagging-2019/test/*.wav'))

        if wrn:

            test = np.reshape(test,(1120, 221, MAX_LEN, 1))

    pred = model.predict(test)

    df = pd.DataFrame(columns=(['fname'] + list(one_hot.classes_)))

    for i in tqdm(range(len(fname))):

        df.loc[i] = [fname[i][14:]] + list(pred[i])

    df.sort_values(by='fname', inplace=True)

    df.to_csv('submission.csv', index=False)

    return pred
model = learn_by_parts(model=model,y=y_n,df=df_c_n,loc='../input/freesound-audio-tagging-2019/train_noisy/', epochs=5, batch_size=8,part_size=15, wrn=True)

model = learn_by_parts(model=model,y=y,df=df_c, epochs=5, batch_size=8,part_size=142, wrn=True)

#model_f = fit_feature_model(model_f, y, epochs=200)

preds = output(model,wrn=True)
from sklearn.metrics import matthews_corrcoef

from sklearn.metrics import hamming_loss
size = 500

loc = '../input/train_curated/'

data = load_data(df_c.iloc[:size,:], loc)

data = np.array(data)

y_test = y[:size]
preds = model.predict(data)
threshold = np.arange(0.1,0.9,0.1)



acc = []

accuracies = []

best_threshold = np.zeros(preds.shape[1])

for i in range(preds.shape[1]):

    y_prob = np.array(preds[:,i])

    for j in threshold:

        y_pred = [1 if prob>=j else 0 for prob in y_prob]

        acc.append( matthews_corrcoef(y_test[:,i],y_pred))

    acc   = np.array(acc)

    index = np.where(acc==acc.max()) 

    accuracies.append(acc.max()) 

    best_threshold[i] = threshold[index[0][0]]

    acc = []
best_threshold
y_pred = np.array([[1 if preds[i,j]>=best_threshold[j] else 0 for j in range(y_test.shape[1])] for i in range(len(y_test))])
hamming_loss(y_test,y_pred)


total_correctly_predicted = len([i for i in range(len(y_test)) if (y_test[i]==y_pred[i]).sum() == 5])
total_correctly_predicted