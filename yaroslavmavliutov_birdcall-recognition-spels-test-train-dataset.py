import os
import os.path as op
import pandas as pd
import numpy as np
import gc
from librosa.feature import melspectrogram
import librosa.display
import librosa
from tqdm.notebook import tqdm
from joblib import Parallel, delayed
import cv2
import time
from keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
INPUT_ROOT = "../input"
RAW_DATA = op.join(INPUT_ROOT, "birdsong-recognition")
TRAIN_AUDIO_DIR = op.join(RAW_DATA, "train_audio")
TEST_AUDIO_DIR = op.join(INPUT_ROOT, "birdcall-check", "test_audio")
train_csv = pd.read_csv(op.join(RAW_DATA, "train.csv"))
test_csv = pd.read_csv(op.join(INPUT_ROOT, "birdcall-check", "test.csv"))
print('Train .csv dataset has %d rows and %d columns' % train_csv.shape, end="")
print('There are %d unique bird species in the dataset' % train_csv['ebird_code'].nunique(), end="")
test_csv.head()
train_csv.head()
class Params():
    sr = 44100
    n_mels = 128
    fmin = 20
    fmax = 16000
    chunk_duration = 5
    chunk_size = chunk_duration*sr
    img_size = None

classes = train_csv['ebird_code'].unique()
num_classes = train_csv['ebird_code'].nunique()
def mono_to_color(X: np.ndarray, mean=None, std=None, norm_max=None, norm_min=None, eps=1e-6):
    """
    Code from https://www.kaggle.com/daisukelab/creating-fat2019-preprocessed-data
    """
    # Stack X as [X,X,X]
    X = np.stack([X, X, X], axis=-1)

    # Standardize
    mean = mean or X.mean()
    X = X - mean
    std = std or X.std()
    Xstd = X / (std + eps)
    _min, _max = Xstd.min(), Xstd.max()
    norm_max = norm_max or _max
    norm_min = norm_min or _min
    if (_max - _min) > eps:
        # Normalize to [0, 255]
        V = Xstd
        V[V < norm_min] = norm_min
        V[V > norm_max] = norm_max
        V = 255 * (V - norm_min) / (norm_max - norm_min)
        V = V.astype(np.uint8)
    else:
        # Just zero
        V = np.zeros_like(Xstd, dtype=np.uint8)
    return V
def save_train(df, path, parameters):
    Images = []
    for index in range(len(df)):
        file_name = df.filename[index]
        ebird_code = df.ebird_code[index]

        y, _ = librosa.load(path + ebird_code + '/' + file_name, sr=parameters.sr)
        
        length = y.shape[0]
        if length>0: 
            y, sr = librosa.effects.trim(y)

        if length >= CHUNK_SIZE:
            y = y[0:+CHUNK_SIZE]
        else:
            #y = pad_sequences(y.T.flatten().reshape(1, -1), maxlen=CHUNK_SIZE, dtype="float32").reshape(-1)
            y = np.pad(y, (CHUNK_SIZE - length, 0), 'constant')
    
        spectrogram = librosa.feature.melspectrogram(y)
        spectrogram = librosa.power_to_db(spectrogram).astype(np.float32)
        
        image = mono_to_color(spectrogram)
        height, width, _ = image.shape
        image = cv2.resize(image, (int(width * self.img_size / height), self.img_size))
        image = image/255.0
        image = np.moveaxis(image, 2, 0).astype(np.float32)
        Images.append(image)
        
    with open("preprocessed/mels_train.pkl", 'wb') as f:
        pickle.dump(Images, f, pickle.HIGHEST_PROTOCOL)
parameters = Params()
save_train(train_csv, TRAIN_AUDIO_DIR, parameters)
# def load_pkl(filename):
#     with open(filename, 'rb') as f:
#         return pickle.load(f)
class TestDataset(data.Dataset):
    def __init__(self, df, clip, parameters):
        self.clip = clip.astype(np.float32)
        self.parameters = parameters
        
        if df['site'].values[0] == "site_3":
            n_samples = len(clip) // parameters.chunk_size
            self.df = pd.DataFrame(data={'site': ['site_3'] * n_samples,
                                         'row_id': [f'site_3_{audio_id[0]}_{int(s)}' for s in seconds],
                                         'seconds': [i * parameters.chunk_duration for i in range(1, n_samples + 1)],
                                         'audio_id': [df['audio_id'].values[0]] * n_samples
                                        })
        else:
            self.df = df
            
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        
        end_seconds = int(self.df['seconds'][idx])
        start_seconds = int(end_seconds - parameters.chunk_duration)

        start_index = self.parameters.sr * start_seconds
        end_index = self.parameters.sr * end_seconds

        y = self.clip[start_index:end_index].astype(np.float32)

        melspec = librosa.feature.melspectrogram(y, sr=SR, **self.melspectrogram_parameters)
        melspec = librosa.power_to_db(melspec).astype(np.float32)
        
        spectrogram = librosa.feature.melspectrogram(
            y,
            sr=self.parameters.sr,
            n_mels=self.parameters.n_mels,
            fmin=self.parameters.fmin,
            fmax=self.parameters.fmax
        )
        spectrogram = librosa.power_to_db(spectrogram).astype(np.float32)
        
        image = mono_to_color(spectrogram)
        height, width, _ = image.shape
        image = cv2.resize(image, (int(width * self.img_size / height), self.img_size))
        image = image/255.0
        image = np.moveaxis(image, 2, 0).astype(np.float32)
        
        return image