# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

from keras.preprocessing.image import load_img,img_to_array

from keras import models, Model

from keras.applications.vgg16 import VGG16

from keras.applications.xception import Xception

from keras.preprocessing import image

from keras.applications.vgg16 import preprocess_input

import matplotlib.pyplot as plt

import PIL.Image as Image

from scipy import spatial

from sklearn.cluster import KMeans
base_path = "/kaggle/input/avantari-technologies-task/dataset/train/"
def load_images_from_folder(folder):

    images = []

    for filename in os.listdir(folder):

        img = load_img(os.path.join(folder,filename),  target_size =(224, 224)) 

        img = img_to_array(img)

        img = img.reshape((1,) + img.shape)

        if img is not None:

            images.append(img)

    return images
file_names = os.listdir("/kaggle/input/avantari-technologies-task/dataset/train/")

file_names.sort()

print('The number of  images: ', len(file_names))
def get_all_images():

    images1 = load_images_from_folder(base_path)

    all_imgs_arr = np.array(images1)

    return all_imgs_arr
all_imgs_arr = get_all_images()

preds_all = np.zeros((len(all_imgs_arr),4096))
vgg = VGG16(include_top=True)

model2 = Model(vgg.input, vgg.layers[-2].output)

# model2.save('vgg_4096.h5') # saving the model just in case
all_imgs_arr.shape
file = "op.csv"

output = open(file, "w")

for j in range(all_imgs_arr.shape[0]):

    featues = model2.predict(all_imgs_arr[j])

    features = [str(f) for f in featues[0]]

    output.write("%s,%s\n" % (file_names[j], ",".join(features)))
op = pd.read_csv("op.csv")

op.head()
def img2array(im):

    if im.mode != 'RGB':

        im = im.convert(mode='RGB')

    return np.fromstring(im.tobytes(), dtype='uint8').reshape((im.size[1], im.size[0], 3))

query_image = "/kaggle/input/avantari-technologies-task/dataset/train/795.jpg"
qu_image = Image.open(query_image)

query_img_arr = img2array(qu_image)

plt.figure()

plt.imshow(query_img_arr)
img_names = query_image.split("/")

img_name = img_names[-1]
def load_images_from_file(file_path):

    images = []

    img = image.load_img(file_path,  target_size=(224, 224))

    img = image.img_to_array(img)

    img = img.reshape((1,) + img.shape)

    if img is not None:

            images.append(img)

    return images
new_img_features = load_images_from_file(query_image)
new_img_pred = model2.predict(new_img_features)
def calculate_similarity(vector1, vector2):

 return (1 - spatial.distance.cosine(vector1, vector2))
op.head()
values = op["0.jpg"]

op = op.drop(["0.jpg"],axis=1)
similar_index = {}

for fea in range(0,len(op)):

   sim_val = calculate_similarity(op.iloc[fea],new_img_pred)

   similar_index.update({values[fea]:sim_val})

sorted_similarity = {}

for key, value in sorted(similar_index.items(), key=lambda kv: kv[1], reverse=True):

    sorted_similarity.update({key:value})
out = dict(list(sorted_similarity.items())[0: 10])

similar_images = list(out.keys())
all_img_path = []

for img in similar_images:

    new_path = base_path+img

    all_img_path.append(new_path)

print(all_img_path)
images = [Image.open(f) for f in all_img_path ]

np_images = [ img2array(im) for im in images ]
for img in np_images:

    plt.figure()

    plt.imshow(img)