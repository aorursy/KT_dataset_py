# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input/cash-recognition-system/Cash/'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
%pylab inline

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

img=mpimg.imread('/kaggle/input/cash-recognition-system/rs1000/rs1000_0.jpg')

imgplot = plt.imshow(img)

plt.show()
%pylab inline

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

img=mpimg.imread('/kaggle/input/cash-recognition-system/Cash/Cash/500/IMG_20180717_114414.jpg')

imgplot = plt.imshow(img)

plt.show()