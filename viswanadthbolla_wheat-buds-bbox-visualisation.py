import os
import cv2
import csv
import glob
import pandas as pd
import numpy as np
import random
import itertools
from collections import Counter
from math import ceil
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
%matplotlib inline

from matplotlib.patches import Rectangle
from PIL import Image

os.listdir('../input/global-wheat-detection')


train = pd.read_csv("/kaggle/input/global-wheat-detection/train.csv")  
image_folder_path = "/kaggle/input/global-wheat-detection/train/"


train
train.nunique()


train['bbox'] = train['bbox'].apply(lambda x: x[1:-1].split(","))
train['x_min'] = train['bbox'].apply(lambda x: x[0]).astype('float32')
train['y_min'] = train['bbox'].apply(lambda x: x[1]).astype('float32')
train['width'] = train['bbox'].apply(lambda x: x[2]).astype('float32')
train['height'] = train['bbox'].apply(lambda x: x[3]).astype('float32')
train = train[['image_id','x_min', 'y_min', 'width', 'height']]
train["x_max"] = train.apply(lambda col: col.x_min + col.width, axis=1)
train["y_max"] = train.apply(lambda col: col.y_min + col.height, axis = 1)
train.head()




train[train["x_max"] > 1024]



train[train["y_max"] > 1024]

train[train["x_min"] < 0]

train[train["y_min"] < 0]
x_max = np.array(train["x_max"].values.tolist())
y_max = np.array(train["y_max"].values.tolist())
train["x_max"] = np.where(x_max > 1024, 1024, x_max).tolist()
train["y_max"] = np.where(y_max > 1024, 1024, y_max).tolist()
train["class"] = "1"
def check_file_type(image_folder_path):
    extension_type = []
    file_list = os.listdir(image_folder_path)
    for file in file_list:
        extension_type.append(file.rsplit(".", 1)[1].lower())
    print(Counter(extension_type).keys())
    print(Counter(extension_type).values())
    
    
check_file_type(image_folder_path)

train["image_id"] = train["image_id"].apply(lambda x: str(x) + ".jpg")
train.head()
train["image_id"] = train["image_id"].astype("str")
train.to_csv("wheat.csv", index=False)


def check_image_size(image_folder_path):
    total_img_list = glob.glob(os.path.join(image_folder_path,"*"))
    counter = 0
    for image in tqdm(total_img_list, desc = "Checking in progress"):
        try:
            img = cv2.imread(image)
            height, width = img.shape[1], img.shape[0]
            if not (height == 1024 and width == 1024):
                counter = counter + 1
        except:
            print("This {} is problematic.".format(image))
    return counter 
        
        




check_image_size(image_folder_path)


wheat = pd.read_csv("wheat.csv") 
image_folder_path = "/kaggle/input/global-wheat-detection/train/"
image_annotation_file = "wheat.csv"
wheat.head()


def sanity_tally(image_folder_path, image_annotation_file):
    img_annotation_list=[]
    with open(image_annotation_file, "r") as file:
        next(file)
        for row in file:
            try:
                image_name, x_min, y_min,width,height ,x_max, y_max, class_idx = row.split(",")
                if image_name not in  img_annotation_list:
                     img_annotation_list.append(image_name)
                
            except ValueError:
                print("Could not convert float to string, likely that your data has empty values.")
        
    
    total_img_list = os.listdir(image_folder_path)
    if set(img_annotation_list) == set(total_img_list):
        print("Sanity Check Status: True")
    else:
        print("Sanity Check Status: Failed. \nThe elements in wheat/train.csv but not in the train image folder is {}. \nThe elements in train image folder but not in wheat/train.csv is {}".format(
                set(img_annotation_list) - set(total_img_list), set(total_img_list) - set(img_annotation_list)))
        return list(set(img_annotation_list) - set(total_img_list)), list(set(total_img_list) - set(img_annotation_list))


set_diff1, set_diff2 = sanity_tally(image_folder_path, image_annotation_file = image_annotation_file)

print("\n\nThere are {} images without annotations in the train/wheat.csv".format(len(set_diff2)))
print('\n\nThere are {} images that are in train_csv but not in train images'.format(len(set_diff1)))
def plot_multiple_img(list1):
    f, axarr = plt.subplots(4,3,figsize=(16,12))

    k=0
    for i in range(0,4):
        for j in range(0,3):
            axarr[i,j].imshow(list1[k])
            k+=1



def plot_random_images(image_folder_path, image_annotation_file, num = 12):
    img_annotation_list=[]
    with open(image_annotation_file, "r") as file:
        next(file)
        for row in file:
            try:
                image_name, x_min, y_min,width,height ,x_max, y_max, class_idx = row.split(",")
                if image_name not in  img_annotation_list:
                     img_annotation_list.append(image_name)
                
            except ValueError:
                print("Could not convert float to string, likely that your data has empty values.")
                
    # randomly choose 12 images to plot
    img_files_list = np.random.choice(img_annotation_list, num)
    print("The images' names are {}".format(img_files_list))
    img_matrix_list = []
    
    for img_file in img_files_list:
        image_file_path = os.path.join(image_folder_path, img_file)
        img = cv2.imread(image_file_path)[:,:,::-1]  
        img_matrix_list.append(img)

    
    plot_multiple_img(img_matrix_list)



plot_random_images(image_folder_path, image_annotation_file, num = 12)



from matplotlib.patches import Rectangle
from PIL import Image

im = Image.open('../input/global-wheat-detection/train/b6ab77fd7.jpg')

# Display the image
plt.imshow(im)

# Get the current reference
ax = plt.gca()

# Create a Rectangle patch
rect = Rectangle((226.0 ,548.0), 130.0 ,58.0 ,linewidth=1,edgecolor='r',facecolor='none')

# Add the patch to the Axes
ax.add_patch(rect)
image_id=wheat['image_id']

def plot_boxes(img,list1):
    fig, axa = plt.subplots(figsize=(20,10))
    axa.imshow(img)

    ax = plt.gca()

    for i in range(0,len(list1)):
        rect = Rectangle((list1[i][0] ,list1[i][1]), list1[i][2] ,list1[i][3] ,linewidth=1,edgecolor='r',facecolor='none')
        ax.add_patch(rect)
def print_random_bbox():
    img_file = np.random.choice(image_id, 1)
    image_file_path = os.path.join(image_folder_path, img_file[0])
    img = cv2.imread(image_file_path)[:,:,::-1]
    box_data=wheat[wheat['image_id']==img_file[0]]
    box_data=box_data[['x_min','y_min','width','height']]
    list1=[]
    for i in list(box_data.index):
        rowData = box_data.loc[ i , : ]
        list1.append(list(rowData))
        
 
    plot_boxes(img,list1)


print_random_bbox()
listofzeros = [0] * len(set_diff2)
wheat1=pd.DataFrame({'image_id':[*set_diff2],'x_min':[*listofzeros], 'y_min':[*listofzeros], 'width':[*listofzeros], 'height':[*listofzeros], 'x_max':[*listofzeros], 'y_max':[*listofzeros],'class':[*listofzeros]})
    
wheat1.head()
wheat1.shape
wheat.shape
wheat=wheat.append(wheat1)
wheat.shape
wheat.tail()
