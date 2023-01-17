# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

# print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import math

import random

import matplotlib.pyplot as plt



from sklearn.metrics import accuracy_score

from sklearn.metrics import make_scorer

from sklearn.model_selection import GridSearchCV



from sklearn.neural_network import MLPClassifier



from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Conv1D, MaxPooling1D, Conv2D, MaxPooling2D, Flatten
train_data = pd.read_csv('../input/fashion-mnist_train.csv')

# train_data = pd.read_csv('fashion-mnist_train.csv')

print(train_data.shape)

im_sz= int(math.sqrt(train_data.shape[1] - 1))

print(im_sz)

train_data.head(5)
test_data = pd.read_csv('../input/fashion-mnist_test.csv')

# test_data = pd.read_csv('fashion-mnist_test.csv')

print(test_data.shape)

test_data.head(5)
label_dict = {

 0: 'T-shirt/top',

 1: 'Trouser',

 2: 'Pullover',

 3: 'Dress',

 4: 'Coat',

 5: 'Sandal',

 6: 'Shirt',

 7: 'Sneaker',

 8: 'Bag',

 9: 'Ankle boot'

}
tst_idx = random.randint(0, train_data.shape[1])

image_row = train_data.iloc[tst_idx, 1:]

plt.imshow(image_row.values.reshape(im_sz, im_sz), cmap="Greys") # c-olor map
X = train_data.iloc[:, 1:]

y = train_data.iloc[:, 0]

print(X.shape, y.shape)
scorer = make_scorer(accuracy_score)

parameters = {'hidden_layer_sizes': ((1000), (1500), (2000), (1500, 500), (2000, 500), (1500, 1000)), 'alpha':(0.001, 0.01, 0.1)} 

mlpc = MLPClassifier(verbose=False, max_iter=8, solver='adam', activation='relu', random_state=42)

clf = GridSearchCV(mlpc, parameters, scoring=scorer, cv=3, n_jobs=-1, verbose=2)

clf.fit(X, y)

print(clf.cv_results_['mean_test_score'])

print(clf.cv_results_['params'])
mlp_last_classifier = MLPClassifier(verbose=True, max_iter=200, solver='adam', alpha=0.01, 

                                    activation='relu', hidden_layer_sizes = (2000, 500))

mlp_last_classifier = mlp_last_classifier.fit(X, y)

mlp_last_predictions = mlp_last_classifier.predict(test_data.iloc[:, 1:])

print("Accuracy: {}%".format(accuracy_score(test_data.iloc[:, 0], mlp_last_predictions) * 100))
tst_idx = random.randint(0, test_data.shape[1])

image_row = test_data.iloc[tst_idx, 1:]

print('На картинке {}'.format(label_dict[test_data.iloc[tst_idx, 0]]))

plt.imshow(image_row.values.reshape(im_sz, im_sz), cmap="Greys") # c-olor map
print('Предсказано {}'.format(label_dict[ mlp_last_predictions[tst_idx]]))