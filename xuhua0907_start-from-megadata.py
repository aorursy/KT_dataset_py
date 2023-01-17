# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

json_files = []

for dirname, _, filenames in os.walk('/kaggle/input/samplemegadata/'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        json_files.append(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import json as js

import pathlib

from tqdm import tqdm

json_files
fake_num = 0

real_num = 0

file_num = 0

video_num = 0

fake_real = 0

train_num = 0

test_num = 0

fake_files = []

real_files = []

no_origin = []

for path in tqdm(json_files):    

    with open(path, 'r') as f:

        file_num += 1

        j = js.loads(f.read())

        video_num += len(j.keys())

        for key in j.keys():

            video = j[key]

            if video['split'] == 'train':

                train_num += 1

            else:

                test_num += 1



            if video['label'] == 'FAKE':

                fake_num += 1

                fake_files.append(key)

                if 'original' in video.keys():

                    fake_real += 1

                else:

                    no_origin.append(video)

            else:

                real_num += 1

                real_files.append(key)

        

print('fake_num is: ', fake_num)

print('real_num is: ', real_num)

print('file_num is: ', file_num)

print('video_num is: ', video_num)

print('fake_real is: ', fake_real)

print('train_num is: ', train_num)

print('test_num is: ', test_num)

# print('fake_files is: ', fake_files)

# print('real_files is: ', real_files)

print('no_origin is: ', no_origin)