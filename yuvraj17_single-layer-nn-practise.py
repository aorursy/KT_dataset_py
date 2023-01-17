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
from sklearn.metrics import classification_report,accuracy_score
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')
df_features = df_train.iloc[:, 1:785]
df_label = df_train.iloc[:, 0]

X_test = df_test.iloc[:, 0:784]

X_test = np.reshape(X_test.values,(28000,28,28))
X_train = np.reshape(df_features.values,(42000,28,28))
classes_train = df_label.values
#training procedure
def fit(images,classes):
    #initialize random output layer weight matrix
    output_layers = [np.random.randint(low=0,high=255, size=np.shape(images[0])) for _ in range(10)]
    # for each image in training set
#     print(len(images))
    for index,image in enumerate(images):
        # stack the input layer 10 times to create (10,8,8) ndarray similar to shape of output layer weight matrix.
        # this becomes input (xi)
        q = np.dstack([image.T]*10).transpose()
        # multiply the output and input layer and sum on 1st axis twice to get sum for all nodes in output layer
        # linear neurons
        # find max to get the predicted node
        predicted_class = np.argmax(np.sum(np.sum(output_layers*q,axis=1),axis=1))
        actual_class = classes[index]
        # update weights
        output_layers[predicted_class] = output_layers[predicted_class] - image
        output_layers[actual_class]  = output_layers[actual_class] + image
    return output_layers
def predict(images,weights):
    predictions = []
    for index,image in enumerate(images):
        # stack the input layer 10 times to create (10,8,8) ndarray similar to shape of output layer weight matrix.
        # this becomes input (xi)
        q = np.dstack([image.T]*10).transpose()
        # multiply the output and input layer and sum on 1st axis twice to get sum for all nodes in output layer
        # linear neurons
        # find max to get the predicted node
        predicted_class = np.argmax(np.sum(np.sum(weights*q,axis=1),axis=1))
        predictions.append([index,predicted_class])
    return (predictions)
final_op_layer = fit(X_train,classes_train)
pred = predict(X_test, final_op_layer)
test_pred2 = pd.DataFrame(pred)
df = pd.DataFrame(test_pred2.values[:,1:])
df.index.name = 'ImageId'
df = df.rename(columns = {0: 'Label'}).reset_index()
df['ImageId'] = df['ImageId'] + 1
df.head()
df.to_csv('mnist_submission.csv', index = False)

