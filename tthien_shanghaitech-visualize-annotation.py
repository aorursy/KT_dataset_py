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
from matplotlib import cm as CM

import PIL.Image as Image

import os

import h5py

import scipy

from scipy.io import loadmat

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# print(os.listdir("../input/shanghaitech_with_people_density_map/ShanghaiTech/part_A/train_data/ground-truth/"))


image_path = "../input/shanghaitech_with_people_density_map/ShanghaiTech/part_A/test_data/images/IMG_176.jpg"

density_map_path = "../input/shanghaitech_with_people_density_map/ShanghaiTech/part_A/test_data/ground-truth-h5/IMG_176.h5"

mat_path = "../input/shanghaitech_with_people_density_map/ShanghaiTech/part_A/test_data/ground-truth/GT_IMG_176.mat"
from matplotlib import pyplot as plt



#now see a sample from ShanghaiA

plt.figure(dpi=600)

plt.axis('off')

plt.margins(0,0)

plt.imshow(Image.open(image_path))
img = Image.open(image_path)

img_matrix = np.array(img)

img_matrix.shape
# gt_file = h5py.File(density_map_path,'r')

# groundtruth = np.asarray(gt_file['density'])

# plt.figure(dpi=600)

# plt.axis('off')

# plt.imshow(groundtruth,cmap=CM.jet)
# groundtruth.shape
def show_density(density, name):

    plt.figure(dpi=600)

    plt.axis('off')

    plt.margins(0, 0)

    

    plt.imshow(density, cmap=CM.jet)

    plt.savefig(name, dpi=600, bbox_inches='tight', pad_inches=0)
gt_file = h5py.File(density_map_path,'r')

groundtruth = np.asarray(gt_file['density'])

show_density(groundtruth, "density.png")
import cv2

groundtruth_resize = cv2.resize(groundtruth,(int(groundtruth.shape[1]/8), int(groundtruth.shape[0]/8)),interpolation = cv2.INTER_CUBIC)*64

show_density(groundtruth_resize, "density_resize.png")
# gt_file = h5py.File(density_map_path,'r')

# groundtruth = np.asarray(gt_file['density'])

# plt.figure(dpi=600)

# plt.axis('off')

# plt.margins(0,0)

# plt.imshow(groundtruth*100, cmap= CM.jet)

# plt.savefig('img262_density.png', dpi=600, bbox_inches='tight',pad_inches=0)# 
type(groundtruth)
# mumber of user

groundtruth.sum()
# load img matrix

img = Image.open(image_path)

img_matrix = np.array(img)

img_matrix.shape


# load point gt from mat

mat = scipy.io.loadmat(mat_path)

img_matrix_annotated = np.copy(img_matrix)

k = np.zeros((img_matrix.shape[0], img_matrix.shape[1]))

gt = mat["image_info"][0, 0][0, 0][0]
gt.shape
for i in range(0, len(gt)):

    if int(gt[i][1]) < img_matrix_annotated.shape[0] and int(gt[i][0]) < img_matrix_annotated.shape[1]:

        img_matrix_annotated[int(gt[i][1]), int(gt[i][0]), 0] = 255 # annotated point

        # make the point on figure bigger for visual 

        img_matrix_annotated[int(gt[i][1]+1), int(gt[i][0]), 0] = 255 # 

        img_matrix_annotated[int(gt[i][1]), int(gt[i][0]+1), 0] = 255 # 

        img_matrix_annotated[int(gt[i][1]+1), int(gt[i][0]+1), 0] = 255 #         
# #now see a sample from ShanghaiA

# plt.figure(dpi=600, frameon = False)

# plt.axis('off')

# plt.imshow(img_matrix_annotated)
#now see a sample from ShanghaiA

import matplotlib.cm as cm

plt.figure(dpi=600, frameon = False)

plt.axis('off')

plt.margins(0,0)

plt.imshow(img_matrix_annotated)

plt.savefig('img262_annotate.png', dpi=600, bbox_inches='tight',pad_inches=0)