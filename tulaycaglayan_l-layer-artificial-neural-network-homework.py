# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm
from skimage import io, transform

# import keras library 
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential # initialize neural network library
from keras.layers import Dense # build our layers library

# import sklearn
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score

# For graph 
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

import warnings
# filter warnings
warnings.filterwarnings('ignore')

# Any results you write to the current directory are saved as output.
# initialize 
category_dir = "../input/animals/raw-img"
img_size = 100

print(os.listdir(category_dir))
# Read and load data 
def get_data(folder_path):
    imgs = []
    indices = []
    labels = []
    for idx, folder_name in enumerate(os.listdir(folder_path)[:10]):
        if folder_name in ( "gatto","elefante") : # cat , elephant 
            labels.append(folder_name)
            for file_name in tqdm(os.listdir(folder_path + '/' + folder_name)):
                if file_name.endswith('jpeg'):
                    img_file = io.imread(folder_path + '/' +  folder_name + '/' + file_name)
                    if img_file is not None:
                       img_file = transform.resize(img_file, (img_size, img_size))
                       imgs.append(np.asarray(img_file))
                       indices.append(idx)
    imgs = np.asarray(imgs)
    indices = np.asarray(indices).reshape(-1,1)
    labels = np.asarray(labels)
    return imgs, indices, labels
    
X, y, labels = get_data(category_dir)    

print('X shape:', X.shape)
print('y shape:', y.shape)
print("labels:",labels)
# show data 
plt.subplot(2, 3, 1)
plt.imshow(X[1].reshape(img_size, img_size,3))
plt.axis('off')
plt.subplot(2, 3, 2)
plt.imshow(X[2].reshape(img_size, img_size,3))
plt.axis('off')
plt.subplot(2, 3, 3)
plt.imshow(X[3].reshape(img_size, img_size,3))
plt.axis('off')
plt.subplot(2, 3, 4)
plt.imshow(X[1942].reshape(img_size, img_size,3))
plt.axis('off')
plt.subplot(2, 3, 5)
plt.imshow(X[1943].reshape(img_size, img_size,3))
plt.axis('off')
plt.subplot(2, 3, 6)
plt.imshow(X[1944].reshape(img_size, img_size,3))
plt.axis('off')
plt.show()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
number_of_train = X_train.shape[0]
number_of_test = X_test.shape[0]

x_train = X_train.reshape(number_of_train,  X_train.shape[1]*X_train.shape[2]*X_train.shape[3])
x_test = X_test .reshape(number_of_test,  X_test.shape[1]*X_test.shape[2]* X_test.shape[3])
 
print("x train: ",x_train.shape)
print("x test: ",x_test.shape)
print("y train: ",y_train.shape)
print("y test: ",y_test.shape)


# Evaluating the ANN
def build_classifier():  # layer sayim 3 . 1.layer 8 , 2.layer 4. 3.layer 1 node iceriyor 
    classifier = Sequential() # initialize neural network
    classifier.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'relu', input_dim = x_train.shape[1]))
    classifier.add(Dense(units = 4, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    ## adam  learning rate i adaptive sekilde ogrenmesini ve degismesini sagliyor 
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy']) 
    return classifier

#epochs number of iterations 
classifier = KerasClassifier(build_fn = build_classifier, epochs = 100  )
accuracies = cross_val_score(estimator = classifier, X = x_train, y = y_train, cv = 2)  # Execute  KerasClassifier 2 times
mean = accuracies.mean()
variance = accuracies.std()
# Evaluate test data predictions 
predictions = cross_val_predict(estimator = classifier,  X = x_test, y = y_test ,  cv=2,verbose=0)
accuracy_pred = accuracy_score(predictions.astype(int), y_test.astype(int))
x_range = np.array(range (1,y_test.shape[0]+1))

plt.figure(figsize=(30,5))
plt.scatter (x_range , y_test.reshape( y_test.shape[0] ), color='green', alpha=0.3, label='Test label')
plt.scatter (x_range , predictions.reshape( y_test.shape[0] ), color='red', alpha=0.3, label='Predicted label')
plt.xlabel("Test Data Index")
plt.ylabel("Test Data Label")
plt.legend()
plt.show()
print("Accuracy mean: "+ str(mean))
print("Accuracy variance: "+ str(variance))
print("Predictions Accuracy: " ,accuracy_pred)
