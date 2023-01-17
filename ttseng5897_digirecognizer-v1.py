dataAugment = False
import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

%matplotlib inline

if(dataAugment): from keras.preprocessing.image import ImageDataGenerator
np.random.seed(2)
# Load the data

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
Y_train = train["label"]

X_train = train.drop(labels = ["label"],axis = 1) 

Y_train.value_counts()

# free some space

del train 
X_train.describe()
X_train.isnull().any().describe()

test.isnull().any().describe()
ndTrain = X_train.values.reshape(-1,28,28,1)



ndTrain.shape

ndTrain = ndTrain.astype('float32') / 255.0



plt.imshow(ndTrain[2])

#plt.imshow(arr[0], cmap='gray')