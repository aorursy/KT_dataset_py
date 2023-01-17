import os
print(os.listdir("../input"))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import PIL
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator
image="../input/siim-isic-melanoma-classification/jpeg/train/ISIC_0015719.jpg"
PIL.Image.open(image)
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# Defining data path
IMAGE_PATH = "../input/siim-isic-melanoma-classification/"

train_df = pd.read_csv('../input/siim-isic-melanoma-classification/train.csv')
test_df = pd.read_csv('../input/siim-isic-melanoma-classification/test.csv')


#Training data
print('Training data shape: ', train_df.shape)
train_df.head(5)
train_df.groupby(['benign_malignant']).count()['sex'].to_frame()
# Null values and Data types
print('Train Set')
print(train_df.info())
print('-------------')
print('Test Set')
print(test_df.info())
# Total number of images in the dataset(train+test)
print("Total images in Train set: ",train_df['image_name'].count())
print("Total images in Test set: ",test_df['image_name'].count())
columns = train_df.keys()
columns = list(columns)
print(columns)
train_df['target'].value_counts()
train_df['sex'].value_counts(normalize=True)
images = train_df['image_name'].values

# Extract 9 random images from it
random_images = [np.random.choice(images+'.jpg') for i in range(9)]

# Location of the image dir
img_dir = IMAGE_PATH+'/jpeg/train'

print('Display Random Images')

# Adjust the size of your images
plt.figure(figsize=(10,8))

# Iterate and plot random images
for i in range(9):
    plt.subplot(3, 3, i + 1)
    img = plt.imread(os.path.join(img_dir, random_images[i]))
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    
# Adjust subplot parameters to give specified padding
plt.tight_layout()   
benign = train_df[train_df['benign_malignant']=='benign']
malignant = train_df[train_df['benign_malignant']=='malignant']
images = malignant['image_name'].values

# Extract 9 random images from it
random_images = [np.random.choice(images+'.jpg') for i in range(9)]

# Location of the image dir
img_dir = IMAGE_PATH+'/jpeg/train'

print('Display malignant Images')

# Adjust the size of your images
plt.figure(figsize=(10,8))

# Iterate and plot random images
for i in range(9):
    plt.subplot(3, 3, i + 1)
    img = plt.imread(os.path.join(img_dir, random_images[i]))
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    
# Adjust subplot parameters to give specified padding
plt.tight_layout()   
images = benign['image_name'].values

# Extract 9 random images from it
random_images = [np.random.choice(images+'.jpg') for i in range(9)]

# Location of the image dir
img_dir = IMAGE_PATH+'/jpeg/train'

print('Display benign Images')

# Adjust the size of your images
plt.figure(figsize=(10,8))

# Iterate and plot random images
for i in range(9):
    plt.subplot(3, 3, i + 1)
    img = plt.imread(os.path.join(img_dir, random_images[i]))
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    
# Adjust subplot parameters to give specified padding
plt.tight_layout()   
df =pd.read_csv('../input/siim-isic-melanoma-classification/train.csv')
df
df = df.drop(columns=['patient_id','sex','age_approx','anatom_site_general_challenge','diagnosis','benign_malignant'])
df
df.columns = ['name','label']
df
data = pd.DataFrame(df[df['label']==1])
data
data2 = pd.DataFrame(df[df['label']==0])
data2 = data2[:584]
data = data.append(data2)
data