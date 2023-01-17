import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

from matplotlib import patches

import matplotlib.pyplot as plt

 

data = pd.DataFrame()

train = pd.read_csv("../input/bccd-csv/train.csv")

test = pd.read_csv("../input/bccd-csv/test.csv")

train.head()
# print a picture to show something

image = plt.imread('../input/bccd-image/BloodImage_00000.jpg')

plt.imshow(image)
# now to make ractangles to point out WBC and RBC

fig = plt.figure()

ax = fig.add_axes([0,0,1,1])

imageName = 'BloodImage_00000'

image = plt.imread("../input/bccd-image/"+imageName+".jpg")

plt.imshow(image)

for _,row in train[train.image_names==imageName].iterrows():

    xmin = row.xmin

    xmax = row.xmax

    ymin = row.ymin

    ymax = row.ymax

    

    width = xmax-xmin

    height = ymax-ymin

    if(row.cell_type=='RBC'):

        edgecolor = 'r'

        ax.annotate('RBC',xy=(xmax-40,ymin+20))

    elif row.cell_type=='WBC':

        edgecolor='b'

        ax.annotate('WBC',xy=(xmax-40,ymin+20))

    elif row.cell_type == 'Platelets':

        edgecolor='g'

        ax.annotate('Platelets',xy=(xmax-40,ymin+20))

        

    rect = patches.Rectangle((xmin,ymin),width,height,edgecolor=edgecolor,facecolor='none')

    ax.add_patch(rect)