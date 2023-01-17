# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# for graphic 
import matplotlib.pyplot as plt 

# for model 
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv ('../input/column_2C_weka.csv')
data.head()
# set claass  as number
data['class'].unique()
data['class'] = [1 if each == 'Abnormal' else 0 for each in data['class']]
data.head()
# train test split

y = data['class']
X = data.drop(['class'], axis=1)
X_train,  X_test , y_train, y_test = train_test_split(X, y, test_size = 0.3,random_state=1)
print('X_train shape : ', X_train.shape)
print('y_train shape : ', y_train.shape)
print('X_test shape : ', X_test.shape)
print('y_test shape : ', y_test.shape)

# find best n value for knn
neig= range(1,25) 
train_accuracy_list =[]
test_accuracy_list =[]

for each in neig:
    knn = KNeighborsClassifier(n_neighbors =each)
    knn.fit(X_train,  y_train)
    train_accuracy_list.append( knn.score(X_train, y_train))    
    test_accuracy_list.append( knn.score(X_test, y_test))    
    
        
print( 'best k for Knn : {} , best accuracy : {}'.format(test_accuracy_list.index(np.max(test_accuracy_list))+1, np.max(test_accuracy_list)))
plt.figure(figsize=[13,8])
plt.plot(neig, train_accuracy_list,label = 'Train Accuracy')
plt.plot(neig, test_accuracy_list,label = 'Test Accuracy')
plt.title('Neighbors vs accuracy ')
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()
plt.xticks(neig)
plt.show()