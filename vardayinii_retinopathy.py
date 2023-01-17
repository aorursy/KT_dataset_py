# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
files=os.listdir("../input")

# Any results you write to the current directory are saved as output.
labelled = pd.read_csv("../input/trainLabels.csv")
labelled.head()
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
import cv2


i=0
width = 200
height = 200

for image in files:
    if i == 5:    
        break
    if image == "trainLabels.csv":
        continue
    image = os.path.join("../input/",image)
    img=Image.open(image)
    pix = np.array(img)

    pix_resized = cv2.resize(pix, dsize=(200, 200), interpolation=cv2.INTER_CUBIC)
    
    plt.imshow(pix_resized)
    plt.show()
    print(pix_resized,pix_resized.shape)
    i = i+1

