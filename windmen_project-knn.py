from keras.applications import inception_v3

from keras.engine import Model

from numpy import zeros

from keras.preprocessing import image

import csv

import os

import numpy as np

import pandas as pd

from sklearn.neighbors import NearestNeighbors

import matplotlib.pyplot as plt
paths = []

k = 0

for dirname, _, filenames in os.walk('/kaggle/input/paintings-0-test'):

    for filename in filenames:

        paths.append(dirname+'/'+filename)

        k+=1
image.load_img('/kaggle/input/paintings-0-test/set_2_test/18988.jpg')
X = zeros((len(paths), 299, 299, 3))
model = inception_v3.InceptionV3(include_top=True, weights='/kaggle/input/model1/my_model(5).h5', classes=21)

model = Model(model.input, model.get_layer(index=-3).output)

for_paths = 0

for i in range(len(paths)):

    X[i] = (image.img_to_array(image.load_img(paths[i])))/255

vec_bath_size = model.predict(X).reshape(k,-1)

'''with open('/kaggle/working/test', "a", newline='') as csv_file:

    writer = csv.writer(csv_file)

    for line in vec_bath_size:

        writer.writerow(line)

data = pd.read_csv('/kaggle/working/test',names = range(131072))

knn = NearestNeighbors(metric='cosine', algorithm='brute')

knn.fit(data)'''
X.shape
k
knn = NearestNeighbors(metric='cosine', algorithm='brute')

knn.fit(vec_bath_size)
path_ = '/kaggle/input/paintings-0-test/set_2_test/1035641.jpg'

test = np.expand_dims(image.img_to_array(image.load_img(path_))/255,axis=0)

test_pred = model.predict(test).reshape(1,-1)

dist, indices = knn.kneighbors(test_pred, n_neighbors=10)

similar_images = [paths[indices[0,i]] for i in range(len(indices[0]))]
fig, axes = plt.subplots(2,5,figsize=(15, 6),subplot_kw={'xticks': (), 'yticks': ()})

for path,ax,dis in zip(similar_images,axes.ravel(),dist[0]):

    ax.imshow(image.load_img(path))

    ax.set_title(dis)

    
'''model = inception_v3.InceptionV3(include_top=True, weights='/kaggle/input/balanse-classes/my_model.h5', classes=7)

model = Model(model.input, model.get_layer(index=-2).output)

for_paths = 0

for j in range(max_index//batch_size):

    for i in range(1,batch_size):

        X[i] = (image.img_to_array(image.load_img(paths[i+for_paths])))/255

    for_paths += batch_size

    vec_bath_size = model.predict(X)

    with open('/kaggle/working/test', "a", newline='') as csv_file:

        writer = csv.writer(csv_file)

        for line in vec_bath_size:

            writer.writerow(line)'''

#Если оперативки не хватает