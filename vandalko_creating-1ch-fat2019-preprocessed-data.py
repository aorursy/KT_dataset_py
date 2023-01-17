import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pathlib import Path

import matplotlib.pyplot as plt

from tqdm import tqdm_notebook

import IPython

import IPython.display

import PIL

import pickle



import torch

import torch.nn as nn

import torch.nn.functional as F



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
DATA = Path('../input')

PREPROCESSED = Path('work/fat2019_prep_mels1')

WORK = Path('work')

Path(PREPROCESSED).mkdir(exist_ok=True, parents=True)

Path(WORK).mkdir(exist_ok=True, parents=True)



CSV_TRN_CURATED = DATA/'train_curated.csv'

CSV_TRN_NOISY = DATA/'train_noisy.csv'

CSV_SUBMISSION = DATA/'sample_submission.csv'



TRN_CURATED = DATA/'train_curated'

TRN_NOISY = DATA/'train_noisy'

TEST = DATA/'test'



CSV_TRN_CURATED_TRIMMED = PREPROCESSED/'trn_curated_trimmed.csv'



MELS_TRN_CURATED = PREPROCESSED/'mels_train_curated.pkl'

MELS_TRN_NOISY = PREPROCESSED/'mels_train_noisy.pkl'

MELS_TEST = PREPROCESSED/'mels_test.pkl'



CSV_TRN_NOISY_BEST50S = PREPROCESSED/'trn_noisy_best50s.csv'

MELS_TRN_NOISY_BEST50S = PREPROCESSED/'mels_trn_noisy_best50s.pkl'



CSV_TRN_NOISY_POOR = PREPROCESSED/'trn_noisy_poor.csv'

MELS_TRN_NOISY_POOR = PREPROCESSED/'mels_trn_noisy_poor.pkl'



trn_curated_df = pd.read_csv(CSV_TRN_CURATED)

trn_noisy_df = pd.read_csv(CSV_TRN_NOISY)

test_df = pd.read_csv(CSV_SUBMISSION)
import librosa

import librosa.display

import random



from fastai import *

from fastai.callbacks import *

from fastai.vision import *

from fastai.vision.data import *





def read_audio(conf, pathname, trim_long_data):

    y, sr = librosa.load(pathname, sr=conf.sampling_rate)

    # trim silence

    if 0 < len(y): # workaround: 0 length causes error

        y, _ = librosa.effects.trim(y) # trim, top_db=default(60)

    # make it unified length to conf.samples

    if len(y) > conf.samples: # long enough

        if trim_long_data:

            y = y[0:0+conf.samples]

    else: # pad blank

        padding = conf.samples - len(y)    # add padding at both ends

        offset = padding // 2

        y = np.pad(y, (offset, conf.samples - len(y) - offset), conf.padmode)

    return y





def audio_to_melspectrogram(conf, audio):

    spectrogram = librosa.feature.melspectrogram(audio, 

                                                 sr=conf.sampling_rate,

                                                 n_mels=conf.n_mels,

                                                 hop_length=conf.hop_length,

                                                 n_fft=conf.n_fft,

                                                 fmin=conf.fmin,

                                                 fmax=conf.fmax)

    spectrogram = librosa.power_to_db(spectrogram)

    spectrogram = spectrogram.astype(np.float32)

    return spectrogram





def show_melspectrogram(conf, mels, title='Log-frequency power spectrogram'):

    librosa.display.specshow(mels, x_axis='time', y_axis='mel', 

                             sr=conf.sampling_rate, hop_length=conf.hop_length,

                            fmin=conf.fmin, fmax=conf.fmax)

    plt.colorbar(format='%+2.0f dB')

    plt.title(title)

    plt.show()





def read_as_melspectrogram(conf, pathname, trim_long_data, debug_display=False):

    x = read_audio(conf, pathname, trim_long_data)

    mels = audio_to_melspectrogram(conf, x)

    if debug_display:

        IPython.display.display(IPython.display.Audio(x, rate=conf.sampling_rate))

        show_melspectrogram(conf, mels)

    return mels





class conf:

    sampling_rate = 44100

    duration = 2 # sec

    hop_length = 347*duration # to make time steps 128

    fmin = 20

    fmax = sampling_rate // 2

    n_mels = 128

    n_fft = n_mels * 20

    padmode = 'constant'

    samples = sampling_rate * duration





def get_default_conf():

    return conf



    

def set_fastai_random_seed(seed=666):

    # https://docs.fast.ai/dev/test.html#getting-reproducible-results



    # python RNG

    random.seed(seed)



    # pytorch RNGs

    import torch

    torch.manual_seed(seed)

    torch.backends.cudnn.deterministic = True

    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)



    # numpy RNG

    import numpy as np

    np.random.seed(seed)
def convert_wav_to_image(df, source, trim_long_data):

    X = []

    for i, row in tqdm_notebook(df.iterrows()):

        x = read_as_melspectrogram(conf, source/str(row.fname), trim_long_data=trim_long_data)

        X.append(x)

    return X





def save_as_pkl_binary(obj, filename):

    """Save object as pickle binary file.

    Thanks to https://stackoverflow.com/questions/19201290/how-to-save-a-dictionary-to-a-file/32216025

    """

    with open(filename, 'wb') as f:

        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)





def load_pkl(filename):

    """Load pickle object from file."""

    with open(filename, 'rb') as f:

        return pickle.load(f)
df = trn_curated_df.copy()

df = df.drop([534, 2068, 2304, 2316, 3893, 4798])

df.to_csv(CSV_TRN_CURATED_TRIMMED, index=False)
conf = get_default_conf()



def convert_dataset(df, source_folder, filename, trim_long_data=False):

    X = convert_wav_to_image(df, source=source_folder, trim_long_data=trim_long_data)

    save_as_pkl_binary(X, filename)

    print(f'Created {filename}')

    return X





convert_dataset(df, TRN_CURATED, MELS_TRN_CURATED);

convert_dataset(test_df, TEST, MELS_TEST);
poor_labels = np.array(['Accelerating_and_revving_and_vroom',

               'Bathtub_(filling_or_washing)',

               'Bus',

               'Buzz',

               'Cutlery_and_silverware',

               'Chink_and_clink',

               'Female_singing',

               'Fill_(with_liquid)',

               'Frying_(food)',

               'Mechanical_fan',

               'Motorcycle',

               'Walk_and_footsteps',

               'Water_tap_and_faucet',

              ])
df = trn_noisy_df[trn_noisy_df.labels.isin(poor_labels)]

df.to_csv(CSV_TRN_NOISY_POOR, index=False)



conf.samples = conf.samples * 2

convert_dataset(df, TRN_NOISY, MELS_TRN_NOISY_POOR, trim_long_data=True);
df = trn_noisy_df.copy()

df['singled'] = ~df.labels.str.contains(',')



singles_df = df[df.singled]

labels = singles_df.labels.unique()

idxes_best50s = np.array([random.choices(singles_df[(singles_df.labels == l)].index, k=50)

                          for l in labels]).ravel()

best50s_df = singles_df.loc[idxes_best50s]

best50s_df.to_csv(CSV_TRN_NOISY_BEST50S, index=False)



convert_dataset(best50s_df, TRN_NOISY, MELS_TRN_NOISY_BEST50S, trim_long_data=True);