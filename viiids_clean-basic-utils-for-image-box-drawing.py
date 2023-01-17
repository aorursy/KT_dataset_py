import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import cv2

import matplotlib.pyplot as plt

import matplotlib.patches as patches

import random

import math

%matplotlib inline
input_data = pd.read_csv('/kaggle/input/global-wheat-detection/train.csv')

input_data['image_id'] = input_data['image_id'].apply(lambda row : row + '.jpg')

input_data.drop(columns=['source'], inplace=True)



# Basic sanity checking

def validate_data(data):

    # Making sure all images have dimension 1024x1024

    height_non_1024 = data[input_data['height'] != 1024]

    width_non_1024 = data[input_data['width'] != 1024]

    assert height_non_1024.shape[0] == 0

    assert width_non_1024.shape[0] == 0



    # Confirming that training images have only one bounding box in training

    multiple_bb = data[input_data['bbox'].apply(lambda x: len(x.split(","))) > 5]

    assert multiple_bb.shape[0] == 0



def create_numeric_list(bbox):

    """

    Takes a string bounding box as present in input data ie. '[xmin, ymin, width, height]' and returns

    a float array of the numbers

    """

    bbox = bbox.replace(" ", "")

    bbox = bbox.replace("[", "")

    bbox = bbox.replace("]", "")

    bbox = bbox.split(",")

    return [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]



validate_data(input_data)

input_data.head()
# Global variables

# This and any other notebook using the utility functions is going to need this.

INPUT_ROOT = '/kaggle/input/global-wheat-detection/'

TRAIN_IMAGES_PATH = INPUT_ROOT + 'train/'

TOTAL_IMAGES_COUNT = len(input_data)
def plot_images(dataframe, image_count=6):

    """

    Plots image_count random images in a subplot containing 3 columns. Note that before using this function generally,

    please set the TRAIN_IMAGES_PATH global variable to the location where images are stored.

    

    Parameters

    ----------

    dataframe: DataFrame containing just one column which contains image names found under TRAIN_IMAGES_PATH

    image_count: Minimum Number of images to display. Will round of to nearest multiple of 3 since images are 

                displayed in 3 columns and show those images.

    """

    assert dataframe.shape[1] == 1

    size = len(dataframe)

    while image_count % 3 != 0:

        image_count = image_count + 1 



    image_column = dataframe.columns[0]

    row_count = (int) (image_count/3)

    fig, ax = plt.subplots(row_count, 3, figsize=(15,15))

    for i in range(row_count):

        for j in range(3):

            index = random.randint(0, size-1)

            tuple_index = (i, j) if row_count > 1 else j

            ax[tuple_index].imshow(cv2.imread(

                TRAIN_IMAGES_PATH + dataframe.iloc[index][image_column]))

            ax[tuple_index].set_title('Image: {}'.format(index))

    fig.show()

    

# Sample usage

plot_images(input_data[['image_id']], 9)
def plot_images_with_bounding_boxes(dataframe, image_indices, image_label='image_id', bbox_label='bbox'):

    """

    Displayes images with bounding boxes. Note that before using this function generally, please set the 

    TRAIN_IMAGES_PATH global variable to the location where images are stored. The images are displayed in

    three columns so the last image may be repeated for filling up the columns.

    

    Parameters

    ----------

    dataframe: A Dataframe containing the following columns:

                image_id: name of the column containing the image file name including the extension. If the name is 

                          different than 'image_id', pass the name as image_label parameter.

                bbox: name of the column containing the bounding box as a string in the format 

                      '[xmin, ymin, width, height]'. If the name is different than bbox, pass the name as bbox_label 

                      parameter.

    image_indices: A list containing the images to display

    image_label: Column name containing the images

    bbox_label: Column name containing the bounding box

    """

    images_count = len(image_indices)

    while images_count % 3 != 0:

        image_indices.append(image_indices[-1] + 1)

        images_count = len(image_indices)

        

    row_count = (int) (images_count / 3)

    fig, ax = plt.subplots(row_count, 3, figsize=(20,20))

    

    for i in range(images_count):

        x = (int)(i/3)

        y = i%3    

        tuple_index = (x,y) if row_count > 1 else y

        ax[tuple_index].imshow(cv2.imread(TRAIN_IMAGES_PATH + dataframe.iloc[image_indices[i]][image_label]))

        ax[tuple_index].set_title('Image {}'.format(image_indices[i]))

        input_row = input_data.iloc[image_indices[i]]

        all_rows_for_image_id = dataframe[dataframe.image_id == input_row[image_label]]

        for index, row in all_rows_for_image_id.iterrows():    

            bbox = create_numeric_list(row[bbox_label])

            rect = patches.Rectangle((bbox[0],bbox[1]),bbox[2],bbox[3],linewidth=1,edgecolor='r',facecolor='none')

            ax[tuple_index].add_patch(rect)



# Example usage

plot_images_with_bounding_boxes(input_data, [3, 376, 267, 33, 9984, 37, 844, 8947, 489])