# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os 

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames[:5]:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from pathlib import Path

import glob

import seaborn as sns

import matplotlib.pyplot as plt

from scipy.io import loadmat

from PIL import Image

import os
#define paths to use later

car_devkit=Path('../input/stanford-cars-dataset/car_devkit')

car_test=Path('../input/stanford-cars-dataset/cars_test/cars_test')

car_train=Path('../input/stanford-cars-dataset/cars_train/cars_train')





cars_metadata=loadmat('../input/stanford-cars-dataset/car_devkit/devkit/cars_meta.mat')

cars_train_annos=loadmat('../input/stanford-cars-dataset/car_devkit/devkit/cars_train_annos.mat')

cars_test_annos=loadmat('../input/stanford-cars-dataset/car_devkit/devkit/cars_test_annos.mat')
#cars_metadata
cars_meta=[]

for x in cars_metadata['class_names'][0]:

    cars_meta.append(x)

    

cars_classes=pd.DataFrame(cars_meta,columns=['cars_classes_exist_in_data'])

cars_classes
#cars_train_annos['annotations']
fname=[[x.flatten()[0] for x in i]  for i in cars_train_annos['annotations'][0]]

column_list=['bbox_x1','bbox_y1','bbox_x2','bbox_y2','class','fname']

train_df=pd.DataFrame(fname,columns=column_list)

train_df['class']=train_df['class'] - 1 ### all values start from zero because above class_classes df index for classes started from zero hence.

train_df['fname']=[car_train/i for i in train_df['fname']]

train_df.head()
train_df=train_df.merge(cars_classes,left_on='class',right_index=True)

train_df=train_df.sort_index()

train_df
import os 

import zipfile



zf=zipfile.ZipFile('Stanford Cars Dataset simplified.zip',mode='w')



try:

    for i in train_df.index:

        try:

            name=train_df['cars_classes_exist_in_data'][i]

            #print(name)

            file_path=train_df['fname'][i]

            file_name=os.path.basename(train_df['fname'][i])

            short_name=name.split(" ")[0]

            zf.write(file_path,os.path.join(short_name,file_name),zipfile.ZIP_DEFLATED)

        except Exception as exc:

            print(str(exc))

            pass

finally:

    print('closing')

    zf.close()
!unzip './Stanford Cars Dataset simplified.zip'