import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import pydicom

from tqdm import tqdm



import os
BASE_PATH = '/kaggle/input/rsna-intracranial-hemorrhage-detection/'

TRAIN_DIR = 'stage_1_train_images/'

TEST_DIR = 'stage_1_test_images/'
def generate_df(base, files):

    train_di = {}



    for filename in tqdm(files):

        path = base + filename

        dcm = pydicom.dcmread(path)

        all_keywords = dcm.dir()

        ignored = ['Rows', 'Columns', 'PixelData']



        for name in all_keywords:

            if name in ignored:

                continue



            if name not in train_di:

                train_di[name] = []



            train_di[name].append(dcm[name].value)



    df = pd.DataFrame(train_di)

    

    return df
train_files = os.listdir(BASE_PATH + TRAIN_DIR)

train_df = generate_df(BASE_PATH + TRAIN_DIR, train_files)



test_files = os.listdir(BASE_PATH + TEST_DIR)

test_df = generate_df(BASE_PATH + TEST_DIR, test_files)
train_df.to_csv('train_metadata.csv')

test_df.to_csv('test_metadata.csv')