import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
#print(os.listdir("../input"))

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
sample_submission = pd.read_csv('../input/sample_submission.csv')
train.head()

# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt
first_digit =train.iloc[3].drop('label').values.reshape(28,28)
plt.imshow(first_digit)
train1, validation = train_test_split(train,
                               test_size = 0.3,
                               random_state=100)
print(train1.shape)
print(validation.shape)

# Segregating input and output
train1_y = train1['label']
validation_y = validation['label']

train1_x = train1.drop('label', axis=1)
validation_x = validation.drop('label', axis=1)

train1_x.shape
# Creating/Fitting a model
model = RandomForestClassifier (random_state=100, n_estimators=300)
model.fit(train1_x, train1_y)

# predicting on test data
test_pred = model.predict(validation_x)

test_pred

df_pred = pd.DataFrame({'actual': validation_y,
                         'predicted': test_pred})
df_pred.head()

df_pred['pred_status'] = df_pred['actual'] == df_pred['predicted']
df_pred.head()
df_pred['pred_status'].value_counts() / df_pred.shape[0]* 100
test_pred = model.predict(test)
df_test_pred = pd.DataFrame(test_pred, columns=['Label'])
df_test_pred['ImageId'] = test.index + 1
df_test_pred[['ImageId', 'Label']].to_csv('submission.csv', index=False)

pd.read_csv('submission.csv').head()
sample_submission.head()
