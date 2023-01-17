import numpy as np

import os

print(os.listdir("../input"))



import requests

from tqdm import tqdm as tqdm

import math



import matplotlib.pyplot as plt

%matplotlib inline

%pylab inline

pylab.rcParams['figure.figsize'] = (10, 6) # adjust the size of figures
#Preparation: Recall HW0003-1: 下載MNIST四個檔案, 並顯示training set的第一張圖

%run '../input/MNIST_utils.py'



filename_list = ['train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz',

                't10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz']

#Download

for filename in filename_list:

    download_file('http://yann.lecun.com/exdb/mnist/' + filename, filename)



image, label = read_mnist(images=filename_list[0], labels=filename_list[1])

print('first label:{}'.format(label[0]))

print('first image:')

print_image(image[0])
plt.imshow(image[0], cmap=plt.cm.gray)
#Pooling layer parameters

n_prev = image[0].shape

kernel_size = (2,2)

stride = kernel_size[0]

padding = 0

n = int((n_prev[0] + 2*padding - kernel_size[0])/stride) + 1

print("feature size after pooling: ",n)
def pooling(img, kernel_size, stride, padding, mode):

    n_prev = img.shape

    n = int((n_prev[0] + 2*padding - kernel_size[0])/stride) + 1

    pooling_img = np.zeros((n,n))

    steps = n

    for i in range(steps):

        for j in range(steps):

            row = i*stride

            col = j*stride

            if mode == 'max':

                pooling_img[i,j] = (img[row:row+kernel_size[0],col:col+kernel_size[1]]).max()

            elif mode == 'avg':

                pooling_img[i,j] = (img[row:row+kernel_size[0],col:col+kernel_size[1]]).mean()

            #print(i,j)

    return pooling_img



pooling_img = pooling(image[0], kernel_size, stride, padding, mode='max')

print(pooling_img.shape)
for i in range(5):

    plt.subplot(1, 5, i + 1)

    plt.imshow(pooling(image[i], kernel_size, stride, padding, mode='max'), cmap=plt.cm.gray_r)

    plt.axis('off')
for i in range(5):

    plt.subplot(1, 5, i + 1)

    plt.imshow(pooling(image[i], kernel_size, stride, padding, mode='avg'), cmap=plt.cm.gray_r)

    plt.axis('off')
for i in range(5):

    plt.subplot(1, 5, i + 1)

    plt.imshow(image[i], cmap=plt.cm.gray_r)

    plt.axis('off')