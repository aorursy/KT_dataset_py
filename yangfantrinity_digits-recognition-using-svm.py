# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import svm, metrics

import csv

import matplotlib.pyplot as plt





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# read data from csv files

trainfilename = open("../input/train.csv")

trainfilereader = csv.reader(trainfilename)

trainds = list(trainfilereader)

train_array = np.array(trainds)

train_array = train_array[1:, :]

train_array = train_array.astype(np.float)

print(train_array.shape)

print(train_array[0,0])
# conver the feature values to be in the range of 0,1 

train_data = train_array[:,1:]

train_data = train_data/255.0

train_data[0, 134]
# build the classifier

classifier = svm.SVC(gamma=0.001)

classifier.fit(train_data, train_array[:, 0])
# import the test data

testfilename = open("../input/test.csv")

testfilereader = csv.reader(testfilename)

testds = list(testfilereader)

test_array=np.array(testds)

test_array = test_array[1:, ]

test_array = test_array.astype(np.float)

print(test_array.shape)

print(test_array[0, 187])
test_array = test_array/255.0

print(test_array[0,187])
# made prediction using test data

predict_result = classifier.predict(test_array)
print(len(predict_result))

result_list = predict_result.tolist()

# plot the first four digits

test_img=test_array*255.0

test_img.shape

plt.subplot(321)

plt.imshow(test_img[0].reshape(28,28), cmap=plt.get_cmap('gray'))

plt.subplot(322)

plt.imshow(test_img[1].reshape(28,28), cmap=plt.get_cmap('gray'))

plt.subplot(323)

plt.imshow(test_img[2].reshape(28,28), cmap=plt.get_cmap('gray'))

plt.subplot(324)

plt.imshow(test_img[3].reshape(28,28), cmap=plt.get_cmap('gray'))

plt.subplot(325)

plt.imshow(test_img[4].reshape(28,28), cmap=plt.get_cmap('gray'))

eachrow = np.zeros([len(result_list),2])

for i in range(len(predict_result)):

    eachrow[i][0]=i

    eachrow[i][1]=predict_result[i]
eachrow[0:5, ]