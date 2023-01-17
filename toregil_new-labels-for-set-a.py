import numpy as np

from scipy.io import wavfile

import matplotlib.pyplot as plt

%matplotlib inline

import pandas as pd
INPUT_LIB = '../input/'

SAMPLE_RATE = 44100
def clean_filename(fname, string):   

    file_name = fname.split('/')[1]

    if file_name[:2] == '__':        

        file_name = string + file_name

    return file_name



def load_wav_file(name, path):

    _, b = wavfile.read(path + name)

    assert _ == SAMPLE_RATE

    return b
file_info = pd.read_csv(INPUT_LIB + 'set_a.csv')

new_info = pd.DataFrame({'file_name' : file_info['fname'].apply(clean_filename, 

                                                                string='Aunlabelledtest'),

                         'target' : file_info['label'].fillna('unclassified')})   

new_info['time_series'] = new_info['file_name'].apply(load_wav_file, 

                                                      path=INPUT_LIB + 'set_a/')    

new_info['len_series'] = new_info['time_series'].apply(len)  
MAX_LEN = max(new_info['len_series'])
new_info['target'].value_counts()
new_labels = np.zeros((176,), dtype="int")
print("artifacts:")

fig, ax = plt.subplots(10, 4, figsize = (12, 16))

for i in range(40):

    ax[i//4, i%4].plot(new_info['time_series'][i])

    ax[i//4, i%4].set_title(new_info['file_name'][i][:-4])

    ax[i//4, i%4].get_xaxis().set_ticks([])
new_labels[:40] = 0

new_labels[18] = 0

new_labels[23] = 0
print("extrahls:")

fig, ax = plt.subplots(5, 4, figsize = (12, 16))

for i in range(19):

    ax[i//4, i%4].plot(new_info['time_series'][i+40])

    ax[i//4, i%4].set_title(new_info['file_name'][i+40][:-4])

    ax[i//4, i%4].get_xaxis().set_ticks([])
new_labels[40:59] = 1

for x in [40, 55]:

    new_labels[x] = 2
print("murmur")

fig, ax = plt.subplots(9, 4, figsize = (12, 16))

for i in range(34):

    ax[i//4, i%4].plot(new_info['time_series'][i+59])

    ax[i//4, i%4].set_title(new_info['file_name'][i+59][-4])

    ax[i//4, i%4].get_xaxis().set_ticks([])
new_labels[59:93] = 2

for x in [62,63,65,68]:

    new_labels[x] = 1
print("normal")

fig, ax = plt.subplots(8, 4, figsize = (12, 16))

for i in range(31):

    ax[i//4, i%4].plot(new_info['time_series'][i+93])

    ax[i//4, i%4].set_title(new_info['file_name'][i+93][:-4])

    ax[i//4, i%4].get_xaxis().set_ticks([])
new_labels[93:124] = 1

for x in [101, 107, 115, 116, 122]:

    new_labels[x] = 2
print("unclassified")

fig, ax = plt.subplots(13, 4, figsize = (12, 16))

for i in range(52):

    ax[i//4, i%4].plot(new_info['time_series'][i+124])

    ax[i//4, i%4].set_title(new_info['file_name'][i+124][17:-4])

    ax[i//4, i%4].get_xaxis().set_ticks([])
new_labels[124:]= [0,2,2,1,

                   1,1,1,1,

                   0,1,0,1,

                   1,1,2,1,

                   0,1,1,1,

                   1,1,2,0,

                   0,0,0,0,

                   0,0,1,0,

                   0,0,0,0,

                   0,1,0,2,

                   1,2,2,2,

                   2,2,2,2,

                   2,2,2,2]
print("[" + ", ".join([str(x) for x in new_labels]) + "]")