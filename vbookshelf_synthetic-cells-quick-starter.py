import pandas as pd

import numpy as np

import os



import cv2

import matplotlib.pyplot as plt

%matplotlib inline



# Don't Show Warning Messages

import warnings

warnings.filterwarnings('ignore')
len(os.listdir('../input/bbbc005_v1_images/BBBC005_v1_images'))
len(os.listdir('../input/bbbc005_v1_ground_truth/BBBC005_v1_ground_truth'))
# get a list of files in each folder



img_list = os.listdir('../input/bbbc005_v1_images/BBBC005_v1_images')

mask_list = os.listdir('../input/bbbc005_v1_ground_truth/BBBC005_v1_ground_truth')
# check folder BBBC005_v1_images



for item in img_list:

    name = item.split('.')

    extension = name[1]

    

    if extension != 'TIF':

        print(item)
# check folder BBBC005_v1_ground_truth



for item in mask_list:

    name = item.split('.')

    extension = name[1]

    

    if extension != 'TIF':

        print(item)


# create a dataframe

df_images = pd.DataFrame(img_list, columns=['image_id'])



# filter out the non image file that's called .htaccess

df_images = df_images[df_images['image_id'] != '.htaccess']







# Example file name: SIMCEPImages_A13_C53_F1_s23_w2.TIF





# ======================================================

# Add a column showing how many cells are on each image

# ======================================================



def get_num_cells(x):

    # split on the _

    a = x.split('_')

    # choose the third item

    b = a[2] # e.g. C53

    # choose second item onwards and convert to int

    num_cells = int(b[1:])

    

    return num_cells



# create a new column called 'num_cells'

df_images['num_cells'] = df_images['image_id'].apply(get_num_cells)







# ================================================

# Add a column indicating if an image has a mask.

# ================================================



# Keep in mind images and masks have the same file names.



def check_for_mask(x):

    if x in mask_list:

        return 'yes'

    else:

        return 'no'

    

# create a new column called 'has_mask'

df_images['has_mask'] = df_images['image_id'].apply(check_for_mask)







# ===========================================================

# Add a column showing how much blur was added to each image

# ===========================================================



def get_blur_amt(x):

    # split on the _

    a = x.split('_')

    # choose the third item

    b = a[3] # e.g. F1

    # choose second item onwards and convert to int

    blur_amt = int(b[1:])

    

    return blur_amt



# create a new column called 'blur_amt'

df_images['blur_amt'] = df_images['image_id'].apply(get_blur_amt)

df_images.head(10)
path_img = '../input/bbbc005_v1_images/BBBC005_v1_images/'

path_mask = '../input/bbbc005_v1_ground_truth/BBBC005_v1_ground_truth/'





# set up the canvas for the subplots

plt.figure(figsize=(30,10))



# Our subplot will contain 2 rows and 4 columns

# plt.subplot(nrows, ncols, plot_number)

plt.subplot(2,4,1)



# plt.imread reads an image from a path and converts it into an array



# starting from 1 makes the code easier to write

for i in range(1,9):

    

    plt.subplot(2,4,i)

    

    # select the image

    image = img_list[i]

    

    # display the image

    plt.imshow(plt.imread(path_img + image))

    plt.axis('off')

path_img = '../input/bbbc005_v1_images/BBBC005_v1_images/'

path_mask = '../input/bbbc005_v1_ground_truth/BBBC005_v1_ground_truth/'



image_1 = 'SIMCEPImages_A13_C53_F1_s23_w2.TIF'

image_2 = 'SIMCEPImages_A04_C14_F1_s20_w2.TIF'



# set up the canvas for the subplots

plt.figure(figsize=(30,20))

plt.axis('Off')



# Our subplot will contain 2 rows and 4 columns

# plt.subplot(nrows, ncols, plot_number)

plt.subplot(2,2,1)

plt.imshow(plt.imread(path_img + image_1))

plt.axis('off')



plt.subplot(2,2,2)

plt.imshow(plt.imread(path_mask + image_1), cmap='gray')

plt.axis('off')



plt.subplot(2,2,3)

plt.imshow(plt.imread(path_mask + image_2))

plt.axis('off')



plt.subplot(2,2,4)

plt.imshow(plt.imread(path_mask + image_2), cmap='gray')

plt.axis('off')



plt.show()
