import numpy as np

import pandas as pd

import os



import cv2



import matplotlib.pyplot as plt

%matplotlib inline

# Load the train data

df_train = pd.read_csv('../input/train.csv')



df_train.head()
# Enter train image_id (without file extension)

# ..........................



image_id = 'brsk004-029'



# ..........................



# Note:

# If the image does not have any Kuzushiji characters then there will be an error message immediately after the next cell.
# Set the file name

fname = image_id + '.jpg'





from matplotlib.pyplot import figure

figure(figsize=(20,20)) 



path = '../input/train_images/' + fname

image = plt.imread(path)

plt.imshow(image)

plt.axis('off')



plt.show()
# Get the labels for the image

df = df_train[df_train['image_id'] == image_id]

df = df.reset_index(drop=True)

label = df.loc[0, 'labels']





# Set plot_type = 'mask' to see a mask instead of a bounding box.

#plot_type = 'mask'

plot_type = 'bbox'



# blank mask

mask = np.zeros((image.shape[0], image.shape[1], 1))



target_list = label.split(' ')



for i, item in enumerate(target_list):

    if item[0] == 'U':

        

        # coordinates of the bbox

        x = int(target_list[i+1])

        y = int(target_list[i+2])

        w = int(target_list[i+3])

        h = int(target_list[i+4])

        

        

        # =========================================

        # Imagine that this is a crop inside a crop.

        # i.e. two boxes, one within another.

        # =========================================

        

        # set the line thickness

        stroke = 4



        # larger mask

        # Set plot_type = 'mask' to see a mask instead of a bounding box.

        mask[y:y+h, x:x+w] = 1

        

        if plot_type == 'bbox':

            

            # smaller mask inside larger mask

            y1=y + stroke

            x1=x + stroke

            h1 = h - (stroke*2)

            w1 = w - (stroke*2)

            mask[y1:y1+h1, x1:x1+w1] = 0

            

            

            

            # view the center point (top left corner is the center)

            center_x = int(x + (w/2))

            center_y = int(y + (h/2))

            w_center = 5

            h_center = 5

            mask[center_y:center_y+h_center, center_x:center_x+w_center] = 1

            

          

        



# Plot the bounding boxes

figure(figsize=(20,20))

mask = mask[:,:,0]



plt.imshow(mask)

plt.axis('off')



plt.show()
figure(figsize=(20,20))



plt.imshow(image, cmap='Greys')

plt.imshow(mask, cmap='Reds', alpha=0.3)

plt.axis('off')



plt.show()



# Set plot_type = 'mask' to see a mask instead of a bounding box.

plot_type = 'mask'

#plot_type = 'bbox'



# blank mask

mask = np.zeros((image.shape[0], image.shape[1], 1))



target_list = label.split(' ')



for i, item in enumerate(target_list):

    if item[0] == 'U':

        

        # coordinates of the bbox

        x = int(target_list[i+1])

        y = int(target_list[i+2])

        w = int(target_list[i+3])

        h = int(target_list[i+4])

        

        # =========================================

        # Imagine that this is a crop inside a crop.

        # i.e. two boxes, one within another.

        # =========================================

        

        # set the line thickness

        stroke = 4



        # larger mask

        # Set plot_type = 'mask' to see a mask instead of a bounding box.

        mask[y:y+h, x:x+w] = 1

        

        if plot_type == 'bbox':

            # smaller mask inside larger mask

            y1=y + stroke

            x1=x + stroke

            h1 = h - (stroke*2)

            w1 = w - (stroke*2)

            mask[y1:y1+h1, x1:x1+w1] = 0

                



# Plot the bounding boxes

figure(figsize=(20,20))

mask = mask[:,:,0]



plt.imshow(mask)

plt.axis('off')



plt.show()
figure(figsize=(20,20))



plt.imshow(image, cmap='Greys')

plt.imshow(mask, cmap='Reds', alpha=0.3)

plt.axis('off')



plt.show()