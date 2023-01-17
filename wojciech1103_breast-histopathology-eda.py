import imageio



import matplotlib as mpl

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

from PIL import Image, ImageOps

import scipy.ndimage as ndi
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
dirname_input = '/kaggle/input/breast-histopathology-images/IDC_regular_ps50_idx5'

dir_input_list = os.listdir(dirname_input)

print(dir_input_list)
#Function for creating path 

def path_img(path, folder='0'): #folder can assume string values '0' or '1'

    return os.path.join(path, folder)
#creating dictionary  for every folder, key - folder name, values - subfolders 0 and 1

dir_img_folders = []

dir_dict = {}

for folder in dir_input_list:

    dir_dict[folder] = os.path.join(path_img(dirname_input, folder), "0"), os.path.join(path_img(dirname_input, folder), "1")
#example

print(dir_dict['13666'])

print(dir_dict['13666'][0])

print(dir_dict['13666'][1])
def plot_imgs(item_dir, top=25):

  all_item_dirs = os.listdir(item_dir)

  item_files = [os.path.join(item_dir, file) for file in all_item_dirs][:25]

  

  plt.figure(figsize=(10, 10))

  for idx, img_path in enumerate(item_files):

    plt.subplot(5, 5, idx+1)

    

    img = plt.imread(img_path)

    plt.imshow(img)



  plt.tight_layout()
plot_imgs(dir_dict[dir_input_list[0]][0])
plot_imgs(dir_dict[dir_input_list[0]][1])