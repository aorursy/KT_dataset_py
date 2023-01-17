from pathlib import Path

from datetime import datetime



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



from PIL import Image

from PIL.ExifTags import TAGS



from tqdm import tqdm
def get_exif(file):



    img = Image.open(file)



    # check if img have exif or not

    try:

        exif = img._getexif()

    except AttributeError:

        return {}



    if exif is None:

        return {}



    # prepare dict

    exif_table = {}



    # convert keys

    for tag_id, value in exif.items():

        exif_table[TAGS.get(tag_id, tag_id)] = value



    return exif_table
def get_datetiems(dir_images, target_images):



    datetimes = list()



    for img in tqdm(target_images):



        exif = get_exif(Path(dir_images) / img)



        if 'DateTime' in exif.keys():

            try:

                dt = datetime.strptime(exif['DateTime'], '%Y:%m:%d %H:%M:%S')

            except ValueError:

                dt = ''

            datetimes.append(dt)

        else:

            datetimes.append('')



    return datetimes
df = pd.read_csv('../input/train.csv')

print(df.head())

df['DateTime'] = get_datetiems('../input/train', df.Image.tolist())

print(df.head())

df['is_new_whale'] = df['Id'] == 'new_whale'

df = df.dropna(axis=0)

print('number of images which have datetime: {}'.format(len(df)))



df['year'] = df['DateTime'].dt.year



new_whales = df.query('is_new_whale==True').groupby('year').agg(len).Image

print('number of new_whales')

print(new_whales)



non_new_whales = df.query('is_new_whale==False').groupby('year').agg(len).Image

print('number of non new_whales')

print(non_new_whales)

xticks = list(range(2003, 2019, 1))

ind = np.arange(len(xticks))

width = 0.35



plt.figure(figsize=(12,8))



p1 = plt.bar(ind, new_whales.tolist(), width)

p2 = plt.bar(ind, non_new_whales.tolist(), width, bottom=new_whales.tolist())



plt.xticks(ind, xticks)

plt.xlabel('year')

plt.ylabel('counts')

plt.legend((p1[0], p2[0]), ('new_whale', 'non new_whale'))





plt.show()
