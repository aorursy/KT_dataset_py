import pandas as pd

import numpy as np



import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap
#Функция для визуализации изображений

def viz_img(df, i):

    plt.imshow(df.values.reshape(df.shape[0], 32, 32, 3)[i])

    plt.show()
train_x = pd.read_csv('../input/train_x.csv', index_col=0, header=None)

train_y = pd.read_csv('../input/train_y.csv', index_col=0)

test_x = pd.read_csv('../input/test_x.csv', index_col=0, header=None)
#3 слоя размером 32х32, которые "вытянуты" в вектор 

train_x.shape, test_x.shape, 
train_y.head()
plt.title(train_y.loc[7].values[0])

viz_img(train_x, 7)

plt.title(train_y.loc[5211].values[0])

viz_img(train_x, 5211)