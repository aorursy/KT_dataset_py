# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt #plotting



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
TEST_DIR = "../input/test.csv"

TRAIN_DIR = "../input/train.csv"



data = pd.read_csv(TRAIN_DIR)
images = data.iloc[:,1:].values

images = images.astype(np.float)

images = np.multiply(images, 1.0/255.0)



image_size = images.shape[1]

image_height = image_width = np.ceil(np.sqrt(image_size)).astype(np.uint8)

image_size, image_height, image_width
def display(img):

    image = img.reshape(image_width, image_height)

    plt.axis('off')

    plt.imshow(image, cmap=plt.cm.binary)
display(images[20])