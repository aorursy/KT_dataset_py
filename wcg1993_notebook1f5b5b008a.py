# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import pandas as pd

import numpy as np





#데이터

submit = pd.read_csv('/kaggle/input/mlregression-cabbage-price/sample_submit.csv')

test = pd.read_csv('/kaggle/input/mlregression-cabbage-price/test_cabbage_price.csv')

train = pd.read_csv('/kaggle/input/mlregression-cabbage-price/train_cabbage_price.csv')
test
train
np_train=np.array(train)

np_test=np.array(test)
x_train=np_train[:,:-1]

y_train=np_train[:,-1]
x_test=np_test[:,:]
x_train.shape
x_test.shape
from sklearn.neighbors import KNeighborsRegressor

knn=KNeighborsRegressor(n_neighbors=5,weights='distance')

knn.fit(x_train,y_train)
y_pred=knn.predict(x_test)
y_pred.shape
for i in range(731):

    submit['Expected'][i]=(y_pred[i])
submit.dtypes
submit.to_csv('submit.csv', mode='w', header= True, index= False)