# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

import math

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

#         print(os.path.join(dirname, filename))

        pass



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import cv2

import pathlib

import librosa

import librosa.display

import skimage

import skimage.io

from IPython.display import Audio

from matplotlib import pyplot as plt

import seaborn as sns

import warnings

import tensorflow as tf



from scipy.ndimage.measurements import center_of_mass



from keras import Sequential

from keras import layers

from keras.models import Sequential

from keras.layers import Dense, LSTM, Dropout, GRU, Bidirectional, Conv2D, MaxPooling2D,  Activation, Flatten, experimental, BatchNormalization, MaxPool2D

from keras.optimizers import SGD

from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.metrics import mean_squared_error, f1_score

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sklearn.model_selection import train_test_split

from sklearn.utils import shuffle



warnings.filterwarnings('ignore')
## Convert to mono types and Sampling Rate: 44100 (Hz) 

config = {

    "sample_rate": 44100 ## 

}
def spectrogram_image(y, sr,):       

    

    """

    y: audio samples: numpy array (2, n)

    sr: sample rate: number

    """

    

    HOP_SIZE = 1024       

    N_MELS = 128              

    WINDOW_TYPE = 'hann' 

    FEATURE = 'mel'      

    FMIN = 1400 

    

    y_chunks= librosa.effects.split(y) 

    

    mfccs_final = []

    

    for chunk in y_chunks:

        

        mels = librosa.feature.melspectrogram(y=y,sr=sr,

                                        hop_length=HOP_SIZE, 

                                        n_mels=N_MELS, 

                                        htk=True, 

                                        fmin=FMIN, 

                                        fmax=sr/2) 



        mels = librosa.power_to_db(mels**2,ref=np.max)

        mfccs = librosa.feature.mfcc(S=mels, n_mfcc=40) 



        mfcss_img = np.reshape(mfccs, (*mfccs.shape, 1))

        

        ## resize and rescale image

        resize_and_rescale = tf.keras.Sequential([

            layers.experimental.preprocessing.Resizing(40, 40),

            layers.experimental.preprocessing.Rescaling(1./255)

        ])

        

        mfcss_img = resize_and_rescale(mfcss_img)



        mfcss_image = np.reshape(mfcss_img, (mfcss_img.shape[0], mfcss_img.shape[1]))

        mfccs_final.append(mfcss_image)

    

    return np.array(mfccs_final)
def gen_label_encoder():

    return LabelEncoder()



def save_image(y, out):

    skimage.io.imsave(out, y)
raw_datasets = pd.read_csv("/kaggle/input/birdsong-recognition/train.csv")



datasets = raw_datasets.loc[:, 

            ['location', 'rating', 'ebird_code', 'duration', 'filename', 'time', 'primary_label', 'sampling_rate',

             'length', 'channels', 'pitch', 'bird_seen', 'background', 'bitrate_of_mp3', 'volume', 'file_type']]



datasets = datasets[datasets.rating >= 4.]
## Loading data

label_encoder = gen_label_encoder()



datasets['duration'] = datasets.duration.astype(float)

datasets['label'] = label_encoder.fit_transform(datasets.ebird_code.to_numpy())
def data_generator(datasets):

    while True:

        for index, row in datasets.iterrows():

            audio_p = f'/kaggle/input/birdsong-recognition/train_audio/{row.ebird_code}/{row.filename}'

            if os.path.isfile(audio_p):   

                try:

                    audio_numpy, _ = librosa.load(audio_p, mono=True, sr=None)

                    audio_numpy, _ = librosa.effects.trim(audio_numpy, top_db=20)

                    audio_name = row.filename

                    

                    audio_mfccs = spectrogram_image(

                        audio_numpy, 

                        config['sample_rate']

                    )

                    

                    yield (

                        audio_mfccs, 

                        tf.keras.utils.to_categorical(

                            row.label, 

                            num_classes=len(datasets.ebird_code.unique()), 

                        ),

                        row.ebird_code, 

                        row.filename)

                    

                except Exception as e:

                    print(f"ignore error data {audio_name}")

                    raise e

                    pass

        else:

            break

for mfccs, encoded_y, ebird_code, filename in data_generator(datasets):

           

    HOP_SIZE = 1024       

    N_MELS = 128            

    

    path = pathlib.Path(f'/kaggle/working/{ebird_code}')

    

    if not path.exists():

        path.mkdir(parents=True, exist_ok=True)

          

    index = 0

    [file_path, _] = os.path.splitext(os.path.join(*path.parts, filename))

    for mfcc in mfccs:  

        save_image(mfcc, out=f"{file_path}.{index}.png")

        index += 1