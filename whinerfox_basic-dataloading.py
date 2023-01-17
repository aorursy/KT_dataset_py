# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

import skimage.exposure

from matplotlib import pyplot as plt
CLASSES = [

    'AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial', 'Pasture', 'PermanentCrop', 'Residential',

    'River', 'SeaLake'

]
train_dataset = np.load("/kaggle/input/isae-ssl-hackathon-2020/eurosat_train.npz")
x_train, y_train = train_dataset['x'], train_dataset['y']
x_train.shape
def plot_imgs(x, y=None, grid_size=4, title="samples"):

    """

    Plot grid_size*grid_size images 

    """

    fig, ax = plt.subplots(grid_size, grid_size, figsize=(20, 20))

    fig.tight_layout()

    idxs = np.random.randint(len(x), size=16)



    for i in range(grid_size ** 2):

        k = idxs[i]

        if y is not None:

            img, lbl = x[k], CLASSES[y[k]]

        else:

            img, lbl = x[k], "unlabelled"

        img = skimage.exposure.adjust_gamma(img, gamma=0.7)

        ax[i % 4][i // 4].imshow(img)

        ax[i % 4][i // 4].set_title(lbl)

        ax[i % 4][i // 4].axis('off')

    fig.suptitle(title, fontsize=14)

    plt.show()
plot_imgs(x_train, y=y_train, title="training set")