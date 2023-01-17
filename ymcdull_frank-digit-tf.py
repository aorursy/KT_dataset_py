# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
### Import models
%matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.cm as cm
#from sklearn.linear_model import LogisticRegression
#from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split

import tensorflow as tf
# settings
LEARNING_RATE = 1e-4
# set to 20000 on local environment to get 0.99 accuracy
TRAINING_ITERATIONS = 2500        
    
DROPOUT = 0.5
BATCH_SIZE = 50

# set to 0 to train on all available data
VALIDATION_SIZE = 2000

# image number to output
IMAGE_TO_DISPLAY = 10
### Read in data
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
  ### Split train set to X and Y
train.shape
X = train.iloc[:, 1:]
Y = train["label"]
print(X.shape)
print(Y.shape)
### Exploration 
#images_size = X.shape[1]
#image_width = image_height = np.ceil(np.sqrt(images_size)).astype(np.uint8)
'''
def display(img):
    one_image = img.reshape(image_width, image_height)
    plt.axis('off')
    plt.imshow(one_image, cmap=cm.binary)
display(X.iloc[1, :])
'''
#print(check_output(["head", "../input/test.csv"]).decode("utf8"))
### RF
#rf = RandomForestClassifier(n_estimators = 100).fit(X, Y)
#result = rf.predict(test)
### CNN
'''
net1 = NeuralNet(
        layers=[('input', layers.InputLayer),
                ('hidden', layers.DenseLayer),
                ('output', layers.DenseLayer),
                ],
        # layer parameters:
        input_shape=(None,1,28,28),
        hidden_num_units=1000, # number of units in 'hidden' layer
        output_nonlinearity=lasagne.nonlinearities.softmax,
        output_num_units=10,  # 10 target values for the digits 0, 1, 2, ..., 9

        # optimization method:
        update=nesterov_momentum,
        update_learning_rate=0.0001,
        update_momentum=0.9,

        max_epochs=15,
        verbose=1,
        )
'''
### TF
labels = pd.get_dummies(Y)
labels = labels.astype(np.uint8)

X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.1)
print(X_train.shape)
print(y_test.shape)
### Use np to save to output file
np.savetxt('sub.csv', 
           np.c_[range(1,len(test)+1),result], 
           delimiter=',', 
           header = 'ImageId,Label', 
           comments = '', 
           fmt='%d')
#print(check_output(["head", "sub.csv"]).decode("utf8"))