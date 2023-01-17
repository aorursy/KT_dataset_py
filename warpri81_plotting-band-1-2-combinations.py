# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt
train = pd.read_json("../input/train.json")
train_clean = train[train['inc_angle'] != 'na']

train_clean['inc_angle'] = train_clean['inc_angle'].astype(np.float32)
def get_band_coords(band_1, band_2, inc_angle, is_iceberg):

    band_1 = np.array(band_1)

    band_2 = np.array(band_2)

    inc_angle = np.array([inc_angle])

    is_iceberg = np.array([is_iceberg])

    return np.stack(np.broadcast(band_1, band_2, inc_angle, is_iceberg))



band_labels = np.concatenate([

    get_band_coords(row['band_1'], row['band_2'], row['inc_angle'], row['is_iceberg'])

    for _, row in train_clean.iterrows()

])



band_1 = band_labels[:, 0]

band_2 = band_labels[:, 1]

inc_angles = band_labels[:, 2]

is_iceberg = band_labels[:, 3]
band_1_clipped = band_1.clip(band_1.mean() - band_1.std() * 2, band_1.mean() + band_1.std())

band_2_clipped = band_2.clip(band_2.mean() - band_2.std() * 2, band_2.mean() + band_2.std())
band_1_rounded = np.round(band_1_clipped, 1)

band_2_rounded = np.round(band_2_clipped, 1)

band_1_buckets = np.unique(band_1_rounded)

band_2_buckets = np.unique(band_2_rounded)

band_1_mapped = (band_1_rounded - band_1_buckets.min()) * 10

band_1_mapped = band_1_mapped.astype(np.int)

band_2_mapped = (band_2_rounded - band_2_buckets.min()) * 10

band_2_mapped = band_2_mapped.astype(np.int)
INC_ANGLE_BUCKET_COUNT = 10



def map_inc_angle(inc_angle):

    if inc_angle < 35: return 0

    elif inc_angle < 36: return 1

    elif inc_angle < 37: return 2

    elif inc_angle < 38: return 3

    elif inc_angle < 39: return 4

    elif inc_angle < 40: return 5

    elif inc_angle < 41: return 6

    elif inc_angle < 42: return 7

    elif inc_angle < 43: return 8

    else: return 9

    

inc_angles_mapped = np.array(

    [ map_inc_angle(inc_angle) for inc_angle in inc_angles]

)
mapped_icebergs = np.stack([band_1_mapped, band_2_mapped, inc_angles_mapped, is_iceberg.astype(np.int)], axis=1)

bucket_counter = np.zeros((len(band_1_buckets), len(band_2_buckets), INC_ANGLE_BUCKET_COUNT, 2))

for i in range(mapped_icebergs.shape[0]):

    x, y, ang, ib = mapped_icebergs[i, :]

    bucket_counter[x, y, ang, ib] += 1
ship_counters = bucket_counter[:, :, :, 0]

iceberg_counters = bucket_counter[:, :, :, 1]

total_counts = ship_counters + iceberg_counters

material_grid = iceberg_counters / (ship_counters + iceberg_counters)

material_grid[np.isnan(material_grid)] = 0.5

material_grid[total_counts < 10] = 0.5
plt.figure(figsize=(15,15))

for x in range(INC_ANGLE_BUCKET_COUNT):

    plt.subplot(INC_ANGLE_BUCKET_COUNT / 3 + 1, 3, x+1)

    materials = material_grid[:, :, x]

    plt.imshow(materials)
from scipy.ndimage.filters import gaussian_filter

plt.figure(figsize=(15,15))

for x in range(INC_ANGLE_BUCKET_COUNT):

    plt.subplot(INC_ANGLE_BUCKET_COUNT / 3 + 1, 3, x+1)

    materials = material_grid[:, :, x]

    materials = gaussian_filter(materials, 3, mode='nearest')

    plt.imshow(materials)