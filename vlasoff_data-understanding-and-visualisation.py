import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap

import os



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
#Функция для визуализации изображений

def viz_img(df, i):

    plt.imshow(df.values.reshape(df.shape[0], 32, 32, 3)[i])

    plt.show()
train_x = pd.read_csv('/kaggle/input/bird-or-aircraft-dafe-open/train_x.csv', index_col=0, header=None)

train_y = pd.read_csv('/kaggle/input/bird-or-aircraft-dafe-open/train_y.csv', index_col=0)

test_x = pd.read_csv('/kaggle/input/bird-or-aircraft-dafe-open/test_x.csv', index_col=0, header=None)
#3 слоя размером 32х32 они "вытянуты" в вектор 

train_x.shape, test_x.shape, 
# Класс 1 соответствует Самолету, 0 – Птице

train_y.head()
plt.title("Класс {}".format(train_y.iloc[2][0]))

viz_img(train_x, 2)
plt.title("Класс {}".format(train_y.iloc[5214][0]))

viz_img(train_x, 5214)