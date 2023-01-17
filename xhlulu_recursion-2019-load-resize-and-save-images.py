import os

import sys

import zipfile



import numpy as np

import pandas as pd

import cv2

import matplotlib.pyplot as plt

from tqdm import tqdm

from PIL import Image
!git clone https://github.com/recursionpharma/rxrx1-utils

sys.path.append('rxrx1-utils')

import rxrx.io as rio
for folder in ['train', 'test']:

    os.makedirs(folder)



!ls
train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')

print(train_df.shape)

print(test_df.shape)

train_df.head()
train_df.tail()
def convert_to_rgb(df, split, resize=True, new_size=400, extension='jpeg'):

    N = df.shape[0]



    for i in tqdm(range(N)):

        code = df['id_code'][i]

        experiment = df['experiment'][i]

        plate = df['plate'][i]

        well = df['well'][i]



        for site in [1, 2]:

            save_path = f'{split}/{code}_s{site}.{extension}'



            im = rio.load_site_as_rgb(

                split, experiment, plate, well, site, 

                base_path='../input/'

            )

            im = im.astype(np.uint8)

            im = Image.fromarray(im)

            

            if resize:

                im = im.resize((new_size, new_size), resample=Image.BILINEAR)

            

            im.save(save_path)
convert_to_rgb(train_df, 'train')

convert_to_rgb(test_df, 'test')
def zip_and_remove(path):

    ziph = zipfile.ZipFile(f'{path}.zip', 'w', zipfile.ZIP_DEFLATED)

    

    for root, dirs, files in os.walk(path):

        for file in tqdm(files):

            file_path = os.path.join(root, file)

            ziph.write(file_path)

            os.remove(file_path)

    

    ziph.close()
zip_and_remove('train')

zip_and_remove('test')
def build_new_df(df, extension='jpeg'):

    new_df = pd.concat([df, df])

    new_df['filename'] = pd.concat([

        df['id_code'].apply(lambda string: string + f'_s1.{extension}'),

        df['id_code'].apply(lambda string: string + f'_s2.{extension}')

    ])

    

    return new_df





new_train = build_new_df(train_df)

new_test = build_new_df(test_df)



new_train.to_csv('new_train.csv', index=False)

new_test.to_csv('new_test.csv', index=False)
!rm -r rxrx1-utils