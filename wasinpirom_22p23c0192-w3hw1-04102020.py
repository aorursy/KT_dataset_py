# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt
import PIL
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
train_data = pd.read_csv('../input/super-ai-image-classification/train/train/train.csv')
print(train_data.info())

test_data = pd.read_csv('../input/super-ai-image-classification/val/val/val.csv')
print(test_data.info())
print(train_data.columns)
print(test_data.columns)
print(len(os.listdir('../input/super-ai-image-classification/train/train/images')))
print(len(os.listdir('../input/super-ai-image-classification/val/val/images')))
train_data['category'].value_counts()
train_data.head()
trainimage_dir="../input/super-ai-image-classification/train/train/images"
testimage_dir="../input/super-ai-image-classification/val/val/images"
filenames = [trainimage_dir +"/"+ fname for fname in train_data['id'].tolist()]
filenames[1724]
PIL.Image.open(str(filenames[1724]))