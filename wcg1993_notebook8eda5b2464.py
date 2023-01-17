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

train = pd.read_csv('/kaggle/input/logistic-classification-diabetes-knn/train.csv')

submit = pd.read_csv('/kaggle/input/logistic-classification-diabetes-knn/submission_form.csv')

test = pd.read_csv('/kaggle/input/logistic-classification-diabetes-knn/test_data.csv')
np_train=np.array(train)
np_test=np.array(test)
train
test
x_train=np_train[:,1:-1]
y_train=np_train[:,-1]
x_test=np_test[:,1:-1]
x_test
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score as ac
knn=KNeighborsClassifier(n_neighbors=8,p=2)

knn.fit(x_train,y_train)
y_train.shape
y_train_pred=knn.predict(x_train)
y_train_pred.shape
ac(y_train,y_train_pred)
y_test_pred=knn.predict(x_test)
y_test_pred.shape
y_test_pred[2]
for i in range(50):

    submit['Label'][i]=int(y_test_pred[i])
submit2=submit.astype(np.int64)
submit2.to_csv('submit.csv', mode='w', header= True, index= False)
submit.dtypes
submit2.dtypes