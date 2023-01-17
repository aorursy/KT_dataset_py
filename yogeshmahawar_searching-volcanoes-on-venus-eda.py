from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd
import cv2
import seaborn as sns
#Paths for training and test
input_dir  = '../input'
train_dir = input_dir + '/volcanoes_train'
test_dir = input_dir + '/volcanoes_test'
# Removing index column and header = none
train_images = pd.read_csv(train_dir + '/train_images.csv',index_col=False,header=None)
train_labels = pd.read_csv(train_dir + '/train_labels.csv')
# Count plot on volcano presence in train image
sns.countplot(x = 'Volcano?',data=train_labels)
#count plot on no. of volcanos if there is a volcanoes in a image
sns.countplot(x = 'Number Volcanoes',data=train_labels)
image_sample = np.reshape(train_images.iloc[9].values,(110,110))
plt.imshow(image_sample,cmap='gray')
plt.title(train_labels.iloc[0])
image_ids_with_volcanos = train_labels[train_labels['Volcano?']==1].iloc[0:5].index
image_ids_without_volcanos = train_labels[train_labels['Volcano?']==0].iloc[0:5].index
image_ids_with_volcanos
def display_images(ids,gaussin_blur=False,median_blur=False):
    columns = 5
    rows = 1
    fig=plt.figure(figsize=(12, 12))
    indx = 0
    for i in range(1, columns*rows +1):
        img = np.uint8(np.reshape(train_images.iloc[ids[indx]].values,(110,110)))
        if gaussin_blur:
            img = cv2.GaussianBlur(img,(5,5),0)
        if median_blur:
            img = cv2.medianBlur(img,9)
        fig.add_subplot(rows, columns, i)
        plt.title('type : ' + str(train_labels.iloc[ids[indx]].Type))
        plt.imshow(img,cmap='gray')        
        indx = indx + 1
    plt.show() 
display_images(image_ids_with_volcanos)
display_images(image_ids_with_volcanos, gaussin_blur=True)
display_images(image_ids_with_volcanos, median_blur=True)
display_images(image_ids_without_volcanos)
display_images(image_ids_without_volcanos,gaussin_blur=True)
display_images(image_ids_without_volcanos,median_blur=True)
