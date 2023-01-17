# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from PIL import Image

import torchvision.transforms as transforms



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

import random



# Any results you write to the current directory are saved as output.

df = pd.read_csv('../input/train.csv')

exps = df['experiment'].unique()

exps = [exp.split('-')[0] for exp in exps]

exp_series = pd.Series(exps)

cell_lines = exp_series.unique()

print('four cell lines are: ', cell_lines)
df = pd.read_csv('../input/train.csv')

df['cell_line'], _ = df['experiment'].str.split('-').str

types_select = [1,2,3,4,5]

fig, axes = plt.subplots(figsize=(25, 25), nrows=len(types_select), ncols=5)

for i, sirna in enumerate(types_select):

    sub_df = df[df['cell_line'] == 'HEPG2']

    sub_df = sub_df[df['sirna'] == sirna]

    sub_df_records = sub_df.to_records()

    np.random.shuffle(sub_df_records)

    axes[i][0].set_ylabel('Type ' + str(sirna))

    for j in range(5):

        exp = sub_df_records[j]['experiment']

        plate = sub_df_records[j]['plate']

        well = sub_df_records[j]['well']

        path = os.path.join('../input/train', exp, 'Plate' + str(plate), well + '_' + 's2' + '_' + 'w3' + '.png')

        img = Image.open(path)

        img = transforms.Resize(224)(img)

        axes[i][j].imshow(img)

        axes[i][j].set_title(sub_df_records[j]['id_code'])
df = pd.read_csv('../input/train.csv')

incomplete_list = []

df['cell_line'], _ = df['experiment'].str.split('-').str

cell_types = ['HEPG2', 'HUVEC', 'RPE', 'U2OS']

for i in range(1, max(df['sirna']) + 1):

    sub_df = df[df['sirna'] == i]

    if (len(df['cell_line'].unique()) < 4):

        incomplete_list.append(i)

print('the incomplete list is: ', incomplete_list)
import matplotlib.pyplot as plt



df = pd.read_csv('../input/train.csv')

incomplete_list = []

df['cell_line'], _ = df['experiment'].str.split('-').str

cell_types = ['HEPG2', 'HUVEC', 'RPE', 'U2OS']

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))

for i, cell_type in enumerate(cell_types):

    sub_df = df[df['cell_line'] == cell_type]

    axes[i // 2, i % 2].hist(sub_df['sirna'].tolist(), bins=1108)

    axes[i // 2, i % 2].set_title(cell_type)