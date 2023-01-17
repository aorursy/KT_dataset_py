import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt
train = pd.read_json('../input/train.json')
# zero out pixels less than the mean + 2 standard deviations

def process_band(band):

    band = np.array(band).reshape((75, 75))

    pband = band - band.mean()

    pband[pband < pband.std() * 2] = 0

    pband = pband / pband.max()

    return band, pband



# get the average of horizontal and vertical slope magnitudes at every point

def get_band_slopes(band):

    band = np.array(band).reshape((75, 75))

    pband = band - np.mean(band)

    pband[pband < band.std() * 2] = 0

    pband = np.mean(np.abs(np.gradient(pband)),axis=0)

    return pband
icebergs = train[train['is_iceberg'] == True].sample(5)

ships = train[train['is_iceberg'] == False].sample(5)



plt.figure(figsize=(15,10))

plt.suptitle('Icebergs', fontsize=16)

for i in range(5):

    band, pband = process_band(icebergs.iloc[i]['band_1'])

    plt.subplot(3, 5, i+1)

    plt.imshow(band)

    plt.subplot(3, 5, i+6)

    plt.imshow(pband)

    plt.subplot(3, 5, i+11)

    plt.imshow(get_band_slopes(band))



plt.figure(figsize=(15,10))

plt.suptitle('Ships', fontsize=16)

for i in range(5):

    band, pband = process_band(ships.iloc[i]['band_1'])

    plt.subplot(3, 5, i+1)

    plt.imshow(band)

    plt.subplot(3, 5, i+6)

    plt.imshow(pband)

    plt.subplot(3, 5, i+11)

    plt.imshow(get_band_slopes(band))