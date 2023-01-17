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


from patsy import dmatrices

from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn import metrics 

import matplotlib.pyplot as plt
data = pd.read_csv('../input/train.csv')
columns = data.columns[1:-1]
X = data[columns]

y = np.ravel(data['target'])
distribution = data.groupby('target').size()/data.shape[0] * 100.0

distribution.plot(kind = 'bar')

plt.show()
model = MLPClassifier(hidden_layer_sizes=(30,10), random_state = 1, verbose = True)

model.fit(X,y)
model.intercepts_
print(model.coefs_[0].shape)

print(model.coefs_[1].shape)

print(model.coefs_[2].shape)
pred = model.predict(X)

pred
model.score(X,y)
test_data = pd.read_csv('../input/test.csv')

Xtest = test_data[test_data.columns[1:]]

Xtest
test_prob = model.predict_proba(Xtest)
solution = pd.DataFrame(test_prob, columns=['Class_1','Class_2','Class_3','Class_4','Class_5','Class_6','Class_7','Class_8','Class_9'])
solution['id'] = test_data['id']

cols = solution.columns.tolist()

cols = cols[-1:] + cols[:-1]

solution = solution[cols]

solution
solution.to_csv('prediction.csv', index = False)