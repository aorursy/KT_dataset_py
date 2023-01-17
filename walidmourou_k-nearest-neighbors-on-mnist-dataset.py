# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train_data = np.load("../input/train_data.npy") #import train data

train_label = np.load("../input/train_labels.npy") #import train labels

test_data =np.load("../input/test_data.npy") #import test data

test_label = np.load("../input/test_labels.npy") #import test labesls
print('The shape of the Training data : ', train_data.shape) # print the dimentions of the train data

print('The shape of the Testing data  : ', test_data.shape) # print the size of the training set
def image_show(i, data, label):

    x = data[i] # get the vectorized image

    x = x.reshape((28,28)) # reshape it into 28x28 format

    print('The image label of index %d is %d.' %(i, label[i]))

    plt.imshow(x, cmap='gray') # show the image
# Showing image from trainning set 

image_show(100, train_data, train_label)
# Showing image from testing set

image_show(100, test_data, test_label)
# L2 square distance between two vectorized images x and y

def distance1(x,y):

    return np.sum(np.square(x-y))

# L2 distance between two vectorized images x and y

def distance2(x,y):

    return np.sqrt(np.sum(np.square(x-y)))

# and can be coded as below

def distance3(x,y):

    return np.linalg.norm(x-y)



def kNN(x, k, data, label):

    #create a list of distances between the given image and the images of the training set

#     distances =[np.linalg.norm(x-data[i]) for i in range(len(data))]

    distances =[distance1(x,data[i]) for i in range(len(data))]

    #Use "np.argpartition". It does not sort the entire array. 

    #It only guarantees that the kth element is in sorted position 

    # and all smaller elements will be moved before it. 

    # Thus the first k elements will be the k-smallest elements.

    idx = np.argpartition(distances, k)

    clas, freq = np.unique(label[idx[:k]], return_counts=True)

    return clas[np.argmax(freq)]
i=524

print('The predicted value is : ', 

      kNN(test_data[i], 5, train_data, train_label), 

      ' and the true value is ', 

      test_label[i])
def accuracy_set(data, label, train_data, train_label, k):

    cnt = 0

    for x, lab in zip(data,label):

        if kNN(x,k, train_data, train_label) == lab:

            cnt += 1

    return cnt/len(label)
k_acc = [accuracy_set(test_data, test_label, train_data, train_label, k) for k in range(1,10)]

k_acc
X = [k for k in range(1,10)]

plt.figure(figsize = (10,5))

plt.xlabel("k")

plt.ylabel("Accuracy")

plt.ylim(0,1)

plt.plot(X,k_acc)