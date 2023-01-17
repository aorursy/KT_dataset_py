import numpy as np

import pandas as pd

import os

from tqdm.auto import tqdm

import scipy as sp

import sklearn

import pickle

import math

import matplotlib.pyplot as plt



import librosa

import librosa.display



from PIL import Image



from sklearn.model_selection import train_test_split

from sklearn.metrics import f1_score



import tensorflow as tf

import tensorflow_addons as tfa

from tensorflow import keras

from tensorflow.keras import layers, models, initializers, optimizers



from IPython.core.display import display, HTML

import IPython.display as ipd

from IPython.display import Audio

display(HTML("<style>.container { width:100% !important; }</style>"))



np.set_printoptions(threshold=100000)
tf.__version__
os.getcwd()
TRAIN_FOLDER = "../input/birdsong-recognition/train_audio/"

train_info = pd.read_csv("../input/birdsong-recognition/train.csv").drop_duplicates()

print(train_info.shape)

print(len(train_info.ebird_code.unique()))

train_info.head()
pd.DataFrame(train_info.isna().sum(axis=0))[train_info.isna().sum(axis=0)>0]
# check ebird_codes

print(train_info.ebird_code.value_counts())



# number of birds to make prediction for, adding 'nocall'

num_birds = len(train_info.ebird_code.unique())+1

print(num_birds)



# categorize ebird_code

train_info['ebird_code_cat'] = train_info.ebird_code.astype('category').cat.codes



# sampling rate

train_info['sampling_rate'] = train_info.sampling_rate.apply(lambda sr: int(sr.split(' (Hz)')[0]))
# create lookup for abbreviated name

nocall_label='nocall'

name_lookup = dict(zip(train_info.ebird_code.astype('category').cat.codes, 

                       train_info.ebird_code.astype('category')))

name_lookup[np.max(train_info.ebird_code.astype('category').cat.codes.unique())+1]=nocall_label



# create reverse lookup for code (from abbreviated name)

code_lookup={v:k for k,v in name_lookup.items()}



# create lookup for sampling rate

sr_lookup = dict(zip(train_info.filename, train_info.sampling_rate))
# sampling rate

train_info.sampling_rate.hist(bins=50)
print(train_info.rating.value_counts())

train_info.rating.hist(bins=10)
bad_quality = train_info.query("rating<3")

print("%d out of %d has ratings below 3"%(len(bad_quality),len(train_info.filename.unique())))
# confirm the rest still covers all birds

print("%d birds exist in recordings with rating below 3" % len(train_info.query("rating>=3").ebird_code.unique()))



# update

train_info = train_info.query("rating>=3").reset_index(drop=True)
# seconds

train_info.duration.describe()
example = train_info.iloc[69,:]

print(example)
# filename

filename = example.filename



# ebird

bird = example.ebird_code



# sampling rate

sr = example.sampling_rate



# duration of clip

duration = example.duration



# initial frequency, final frequency

fmin, fmax = 20, sr/2



print("#ebird code: {}\n".format(bird))

print("#label: {}\n".format(example.primary_label))

print("#secondary labels: {}\n".format(example.secondary_labels))

print("#description:\n {}\n".format(example.description))

print("#type: {}\n".format(example.type))

print("#saw bird: {}\n".format(example.bird_seen))

print("#sampling rate: {} Hz\n".format(sr))

print("#initial frequency: {} Hz\n".format(fmin))

print("#final frequency: {} Hz\n".format(fmax))

print("#recording length: {} seconds\n".format(duration))
sound_clip, sr= librosa.load(TRAIN_FOLDER + bird + '/' + filename, sr=sr)
librosa.get_duration(sound_clip, sr=sr)
ipd.Audio(TRAIN_FOLDER + bird + '/' + filename)
plt.figure(figsize=(15,5))

librosa.display.waveplot(sound_clip, sr=sr)
X = sp.fft(sound_clip[:len(sound_clip)]) # fourier transform

X_mag = np.absolute(X)        # spectral magnitude

f = np.linspace(0, sr, len(sound_clip))  # frequency variable



plt.figure(figsize=(15,5))

plt.plot(f, X_mag) # magnitude spectrum

plt.xlabel('Frequency (Hz)')
melspectrogram = librosa.feature.melspectrogram(sound_clip, sr=sr, fmin=fmin, fmax=fmax)

print("In this case, melspectrogram computed {} mel-frequency spectrogram coefficients over {} frames.".format(melspectrogram.shape[0], melspectrogram.shape[1]))

melspectrogram = librosa.power_to_db(melspectrogram).astype(np.float32)

plt.figure(figsize=(8,6))

librosa.display.specshow(melspectrogram, sr=sr, x_axis='time', fmin=fmin, fmax=fmax)

plt.colorbar(format='%+2.0f dB')
mfcc = librosa.feature.mfcc(sound_clip, sr=sr)

print("In this case, mfcc computed {} MFCCs over {} frames.".format(mfcc.shape[0], mfcc.shape[1]))

mfcc = librosa.power_to_db(mfcc).astype(np.float32)

plt.figure(figsize=(8,6))

librosa.display.specshow(mfcc, sr=sr, x_axis='time')

plt.colorbar(format='%+2.0f dB')
chromagram = librosa.feature.chroma_stft(sound_clip, sr=sr)

print("In this case, librosa computed {} pitches over {} frames.".format(chromagram.shape[0], chromagram.shape[1]))



plt.figure(figsize=(8,6))

librosa.display.specshow(chromagram, x_axis='time', y_axis='chroma', sr=sr)

plt.colorbar(format='%+2.0f dB')
ipd.Audio(TRAIN_FOLDER + bird + '/' + filename)
def preprocess_audio(sound_clip, sr):

    melspectrogram = librosa.feature.melspectrogram(sound_clip, sr=sr, fmin=fmin, fmax=fmax)

    melspectrogram = librosa.power_to_db(melspectrogram).astype(np.float32)

    return melspectrogram
img = preprocess_audio(sound_clip, sr)
# source: https://www.kaggle.com/daisukelab/cnn-2d-basic-solution-powered-by-fast-ai

def mono_to_color(X, mean=None, std=None, norm_max=None, norm_min=None, eps=1e-6):

    X = standardize(X)

    X = stack(X)

    return X



def standardize(X, mean=None, std=None, norm_max=None, norm_min=None, eps=1e-6):

    mean = mean or X.mean()

    std = std or X.std()

    Xstd = (X - mean) / (std + eps)

    _min, _max = Xstd.min(), Xstd.max()

    norm_max = norm_max or _max

    norm_min = norm_min or _min

    if (_max - _min) > eps:

        V = Xstd

        V[V < norm_min] = norm_min

        V[V > norm_max] = norm_max

        V = 255 * (V - norm_min) / (norm_max - norm_min)

        V = V.astype(np.uint8)

    else:

        # Just zero

        V = np.zeros_like(Xstd, dtype=np.uint8)

    

    return V



def stack(X):

    return np.stack([X,X,X],axis=-1)
image = mono_to_color(img)

plt.figure(figsize=(10,10))

print(image.shape)

plt.imshow(image)
def split_silence(clip, sr=None, num_std=1):

    db = librosa.core.amplitude_to_db(clip)



    x_split = librosa.effects.split(y=sound_clip, top_db = np.abs(db).mean()-num_std*db.std())

    silence_removed=[]

    silence=[]



    if len(x_split)>0: # if some clips are considered non-silence

        for i in x_split:

            silence_removed.extend(sound_clip[i[0]:i[1]])

        silence_removed=np.array(silence_removed)



        x_split = x_split.flatten()

        last_sample = int(librosa.get_duration(clip,sr)*sr)



        if x_split[0]!=0: # if non-silent clip does not start at 0

            x_split = np.insert(x_split, 0, 0, axis=0)

        else:

            x_split = x_split[1:]



        if x_split[-1]!= last_sample: # if non-silent clip does not end with last sample

            x_split = np.append(x_split,[last_sample],axis=0)

        else:

            x_split = x_split[:-1]



        if len(x_split)>0: # if not entire clip is non-silence

            x_split = np.split(x_split, len(x_split)/2)

            for i in x_split:

                start = int(i[0])

                end = int(i[1])

                silence.extend(sound_clip[start:end])

            silence=np.array(silence)

            return silence_removed, silence

        

        else: # if entire clip is non-silence

            return sound_clip, silence

        

    else: # if no clip is non-silence

        return silence_removed, sound_clip
sound_clip_silence_removed, silence = split_silence(sound_clip, sr)
Audio(sound_clip_silence_removed,rate=sr)
Audio(silence,rate=sr)
# original version

img = preprocess_audio(sound_clip, sr)

img = mono_to_color(img)

plt.figure(figsize=(25,10))

print(img.shape)

height, width, channels = img.shape

plt.imshow(img)
# silence removed

img = preprocess_audio(sound_clip_silence_removed, sr)

img = mono_to_color(img)

plt.figure(figsize=(25,10))

print(img.shape)

height, width, channels = img.shape

plt.imshow(img)
# silence only

img = preprocess_audio(silence, sr)

img = mono_to_color(img)

plt.figure(figsize=(10,10))

print(img.shape)

height, width, channels = img.shape

plt.imshow(img)
def crop_image(img, sr, random=True, num_seconds=5, hop_length=512):

    height, width = img.shape

    duration = width*hop_length/sr

    if duration>num_seconds: 

        if random: #randomly crop 5 seconds

            end_second = np.random.uniform(low=num_seconds,high=duration, size=1)[0]

            start_second = end_second-num_seconds

        else: #crop first 5 seconds

            end_second = num_seconds

            start_second = end_second-num_seconds

    else:

        end_second = duration

        start_second = 0

    

    start_frame = int(np.floor(start_second*sr/hop_length))

    end_frame = int(np.floor(end_second*sr/hop_length))

    

    return img[:, start_frame:end_frame].astype(np.float32)
img = preprocess_audio(sound_clip_silence_removed, sr)

cropped_img = crop_image(img,sr)

cropped_img = mono_to_color(cropped_img)

print(cropped_img.shape)

plt.figure(figsize=(10,5))

plt.imshow(cropped_img)
def resize_and_rescale(img, img_size=128):

    img = Image.fromarray(img)

    img = img.resize((img_size, img_size),Image.ANTIALIAS)

    img = np.array(img)

    img = np.divide(img,255)

    return img



resized_img = resize_and_rescale(cropped_img)

plt.figure(figsize=(10,5))

plt.imshow(resized_img)
np.random.seed(2629)

X_Train, X_test, y_Train, y_test = train_test_split(train_info.drop('ebird_code_cat', axis=1), 

                                                     train_info.ebird_code_cat, 

                                                     stratify=train_info.ebird_code_cat, 

                                                     train_size=0.8)



X_train, X_val, y_train, y_val = train_test_split(X_Train, 

                                                  y_Train, 

                                                  stratify=y_Train, 

                                                  train_size=0.8)
epochs = 100

batch_size= 100

img_size=128

lr=.001

early_stopping=10
resnet50 = tf.keras.applications.ResNet50(classes=num_birds,

                                          weights=None,

                                          include_top=True,

                                          input_shape=(img_size,img_size,3))
model = resnet50
# Adam

optimizer = tf.keras.optimizers.Adam(learning_rate=lr)



# cross entropy loss function

loss_fn = tf.keras.losses.CategoricalCrossentropy()



# metrics

f1_train = tfa.metrics.F1Score(num_birds, 'micro')

f1_val = tfa.metrics.F1Score(num_birds, 'micro')



# compile

model.compile(optimizer=optimizer,loss=loss_fn)



# summary

model.summary()
model = keras.models.load_model("../input/birdcall-models/"+model.name+"/model")

metric_log = pickle.load(open("../input/birdcall-models/"+model.name+"/metric_log.pkl", "rb"))
plt.plot(metric_log[1], label='val_f1')

plt.plot(metric_log[2], label='f1')

plt.xlabel('Epoch')

plt.ylabel('F1 Score')

plt.legend(loc='lower right')
plt.plot(metric_log[3], label='val_loss')

plt.plot(metric_log[4], label='loss')

plt.xlabel('Epoch')

plt.ylabel('Loss')

plt.legend(loc='lower right')
BASE_TEST_DIR = '../input/birdsong-recognition' if os.path.exists('../input/birdsong-recognition/test_audio') else '../input/birdcall-check'

TEST_FOLDER = f'{BASE_TEST_DIR}/test_audio/'

TEST_FOLDER
df_test = pd.read_csv(f'{BASE_TEST_DIR}/test.csv')

df_test.head()
# edit crop_image function for use for site_3

def crop_image(img, sr, start_second, end_second, hop_length=512):

    start_frame = int(np.floor(start_second*sr/hop_length))

    end_frame = int(np.floor(end_second*sr/hop_length))

    return img[:, start_frame:end_frame].astype(np.float32)



def load_test_clip(path, start_time, duration=5): # discussion: https://www.kaggle.com/c/birdsong-recognition/discussion/179592

    clip, sr_native = librosa.core.audio.__audioread_load(path, offset=start_time, duration=duration, dtype=np.float32)

    clip = librosa.to_mono(clip)

    sr = 32000

    if sr_native > 0:

        clip = librosa.resample(clip, sr_native, sr, res_type='kaiser_fast')

    return clip, sr



def make_prediction(clip, site):

    audio,sr=clip

    melspec = preprocess_audio(audio, sr=sr)

    if (site=='site_1' or site=='site_2'):

        x = mono_to_color(melspec).astype(np.uint8)

        x = resize_and_rescale(x)

        x = tf.expand_dims(x, axis=0)

        y = np.argmax(model(x),1)[0]

        return name_lookup[y]

    else:

        duration = librosa.get_duration(audio,sr)

        if duration<5:

            x = crop_image(melspec,sr,0,duration)

            x = mono_to_color(x).astype(np.uint8)

            x = resize_and_rescale(x)

            x = tf.expand_dims(x, axis=0)

            y = np.argmax(model([x]),1)[0]

            return name_lookup[y]

        else:

            num_five = duration/5

            start_second = 0

            end_second = 5

            y = []

            

            # predict for each 5 seconds

            while end_second<=duration:

                x = crop_image(melspec,sr,start_second,end_second)

                x = mono_to_color(x).astype(np.uint8)

                x = resize_and_rescale(x)

                x = tf.expand_dims(x, axis=0)

                clip_pred = np.argmax(model([x]),1)[0]

                clip_pred = name_lookup[clip_pred]

                y.append(clip_pred)

                start_second += 5

                end_second += 5

                

            # predict for remaining time: at least 1 second

            if duration-end_second>=1:

                x = crop_image(melspec,sr,end_second,duration)

                x = mono_to_color(x).astype(np.uint8)

                x = resize_and_rescale(x)

                x = tf.expand_dims(x, axis=0)

                clip_pred = np.argmax(model([x]),1)[0]

                clip_pred = name_lookup[clip_pred]

                y.append(clip_pred)

            return y



preds = []

for index, row in tqdm(df_test.iterrows()):

    site = row['site']

    start_time = row['seconds']-5

    row_id = row['row_id']

    audio_id = row['audio_id']



    if site=='site_1' or site=='site_2':

        # if site 1 or site 2, increment at 5 seconds interval from start time

        sound_clip = load_test_clip(TEST_FOLDER + audio_id + '.mp3', start_time) #example_test_audio: BLKFR-10-CPL_20190611_093000.pt540.mp3

    else:

        # if site 3, entire clip

        sound_clip = load_test_clip(TEST_FOLDER + audio_id + '.mp3', start_time=0, duration=None) #example_test_audio: ORANGE-7-CAP_20190606_093000.pt623.mp3



    pred = make_prediction(sound_clip, site)

    pred = ' '.join(np.unique(pred))



    preds.append([row_id, pred])



preds = pd.DataFrame(preds, columns=['row_id', 'birds'])
preds
preds.to_csv('submission.csv', index=False)