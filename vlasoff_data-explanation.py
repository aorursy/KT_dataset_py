# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
train_x = pd.read_csv('../input/train_x.csv', index_col=0, header=None)
train_y = pd.read_csv('../input/train_y.csv', index_col=0)
test_x = pd.read_csv('../input/test_x.csv', index_col=0, header=None)
#3 слоя размером 32х32 они "вытянуты" в вектор 
train_x.shape, test_x.shape, 
#Функция для визуализации изображений
def viz_img(df, i):
    plt.imshow(df.values.reshape(df.shape[0], 32, 32, 3)[i])
    plt.show()
train_y.head()
plt.title(train_y.iloc[7])
viz_img(train_x, 7)
plt.title(train_y.iloc[5211])
viz_img(train_x, 5211)