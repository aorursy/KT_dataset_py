import numpy as np

import requests

import os

from tqdm import tqdm as tqdm

import matplotlib.pyplot as plt

import math

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
### mAiLab_0004：Edge Detection

# HW0004-1: 將作業3的前五個圖，增大至30x30後，

# 分別跑Gx與Gy，共輸出10個圖，以圖示呈現，並請反白處理。



#define Gx, Gy 

Gx = np.array([[-1, 0, 1], 

               [-2, 0, 2], 

               [-1, 0, 1]])

Gy = np.array([[ 1, 2, 1], 

               [ 0, 0, 0], 

               [-1,-2,-1]])



kernel_size = Gx.shape

    

#for kernel_size=3, padding=1, stride=1, this is the 'SAME' padding.

#verify conv image size

stride = 1

n_prev = (30, 30)

padding = 1

n = int((n_prev[0] + 2*padding - kernel_size[0])/stride) + 1



#generate padded image

print('image size after padding: ', (n, n))

plt.imshow(np.pad(image[0], [padding, padding], mode='edge'), cmap=plt.cm.gray)

    
# convolution computation  

def conv(image, conv_kernel, stride, padding):

    #initialize

    n_prev = image.shape[0]

    kernel_size = conv_kernel.shape[0]

    n = int((n_prev + 2*padding - kernel_size)/stride) + 1

    conv_image = np.zeros((n,n))

    padded_image = np.pad(image, [padding, padding], mode='edge')

    

    #origin at left-up, move the kernel

    half_len = initial_center =  math.ceil(kernel_size/2 - 1) #round up

    kernel_center = np.array((initial_center, initial_center))

    steps = n

    for i in range(steps):

        for j in range(steps):

            #define image window dimension

            left = kernel_center[1] - half_len

            right = kernel_center[1] + half_len

            up = kernel_center[0] - half_len

            down = kernel_center[0] + half_len

            

            #define image window

            image_window = padded_image[up:down+1, left:right+1]

            

            #debug

            #print(i,j,image_window.shape)

            

            conv_image[i,j] = np.sum(conv_kernel * image_window)

            kernel_center[1] += stride

        kernel_center[1] = initial_center

        kernel_center[0] += stride

    #conv_image = (conv_image)

    

    

    return conv_image.astype('int32')
L = 5

print('Gx convolution:')

for i in range(L):

    plt.subplot(1, L, i + 1)

    plt.imshow(conv(image=image[i], conv_kernel=Gx, stride=1, padding=2), cmap=plt.cm.gray)

    plt.axis('off')



print('Gy convolution:')

for i in range(L):

    plt.subplot(1, L, i + 1)

    plt.imshow(conv(image=image[i], conv_kernel=Gy, stride=1, padding=2), cmap=plt.cm.gray)

    plt.axis('off')
# HW0004-1: 將作業3的第一個圖，分別增大至 32x32、34x34、36x36，然後跑5x5、7x7、9x9的filter

# ，輸出其值為 28x28 的矩陣

def kernel(kernel_size):

    kernel = np.zeros((kernel_size, kernel_size))

    #center = math.ceil(kernel_size/2)

    

    space_by_row = np.concatenate((np.ones(kernel_size//2), np.zeros(1), -1*np.ones(kernel_size//2)))

    up_row = bottom_row = np.linspace(kernel_size//2, -1*(kernel_size//2), kernel_size)

    

    for i in range(kernel_size//2):

        kernel[i] = up_row + i*space_by_row

        kernel[-1*(i+1)] = bottom_row + i*space_by_row

    kernel[kernel_size//2] = kernel[kernel_size//2 - 1] + space_by_row



    return kernel.astype('int')



kernel(9)
img32x32 = conv(image[0], kernel(5), 1, 2)

img34x34 = conv(image[0], kernel(7), 1, 3)

img36x36 = conv(image[0], kernel(9), 1, 4)
plt.imshow(img32x32, cmap=plt.cm.gray)
plt.imshow(img34x34, cmap=plt.cm.gray)
plt.imshow(img36x36, cmap=plt.cm.gray)
#print out image by a two digits hexadecimal matrix

def print_image(file):

    for i in file:

        for j in i:

            # {:02X} output the pixel numbers by two digits hexadecimal

            # example: 255 -> FF ; 14 -> 1E

            print("{:02X}".format(j.astype("uint8")), end=' ') 

        print()

    print()
print('32x32')

print()

print_image(img32x32)
print('34x34')

print()

print_image(img34x34)
print('36x36')

print()

print_image(img36x36)