# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix

filename = '../input/MNIST.csv'
data = pd.read_csv(filename)
#dataset
data
train_data = data.sample(frac=0.1, random_state=100)
test_data = data.drop(train_data.index)


test_y = test_data['label'].values
test_data = test_data.drop(['label'], axis=1)
test_x = test_data.as_matrix()/255


train_y = train_data['label'].values
train_data = train_data.drop(['label'], axis=1)
train_x = train_data.as_matrix()/255


data
len(train_x)
len(test_y)
cols = list(data.columns)
#cols
l = []
for col in cols:
    #print('--------------------')
    #print (dataset[col][1])
    l.append(data[col][1])
#print (l)
k=0
td = []
for i in range(28):
    r = []
    for j in range(28):
        r.append(l[k])
        k = k+1
    td.append(r)
print(td)


import matplotlib.pyplot as plt
from scipy.misc import imread,imresize
plt.imshow(test_x[11].reshape(28,28), cmap = 'gray')

#plt.imshow(tb, cmap = 'gray')
#Actually displaying the plot if you are not in interactive mode
#plt.show()



