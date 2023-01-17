# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

#    for filename in filenames:

#        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
from PIL import Image

import numpy



myimages = []

for dirname, _, filenames in os.walk('/kaggle/input'):    

    for filename in filenames:

            img=Image.open(os.path.join(dirname, filename))

            imgarray=numpy.array(img)

            myimages.append(imgarray)
len(myimages)

print("there are {} images avaliable.".format(len(myimages)))
from matplotlib import pyplot as plt

img1=Image.open("/kaggle/input/make-up-vs-no-make-up/data/data/no_makeup/no_makeup25.jpeg")

attemp=numpy.array(img1)

plt.imshow(attemp, interpolation='nearest')

plt.show()
fig = plt.figure(figsize=(10,20))

fig.subplots_adjust(hspace=0.3, wspace=0.6)

for i in range(1, 11):

    ax = fig.add_subplot(5, 2, i)

    plt.imshow(myimages[i-1], interpolation='nearest')

    plt.xticks(rotation=45)