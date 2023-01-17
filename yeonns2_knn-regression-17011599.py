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
import numpy as np

import pandas as pd

import seaborn as sns



#데이터

train = pd.read_csv('/kaggle/input/mlregression-cabbage-price/train_cabbage_price.csv')

test = pd.read_csv('/kaggle/input/mlregression-cabbage-price/test_cabbage_price.csv')

submit = pd.read_csv('/kaggle/input/mlregression-cabbage-price/sample_submit.csv')
x = train.drop(['year','avgPrice'],axis=1)

y = train['avgPrice']

print(x.shape)

x.head()
from sklearn.preprocessing import LabelEncoder

classle=LabelEncoder()

y=classle.fit_transform(train['avgPrice'].values)

print(np.unique(y))

yo=classle.inverse_transform(y) 

print(np.unique(yo))
#데이터 분할

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,yo,test_size=0.2,random_state=0)

print(x_train.shape)

print(x_test.shape)

print(y_train.shape)

print(y_test.shape)
#데이터 표준화

from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

sc.fit(x_train)

x_train_std = sc.transform(x_train)

x_test_std = sc.transform(x_test)



#표준화된 data의 확인

print(x_train.head())

x_train_std[1:5,]
#knn의 적용

from sklearn.neighbors import KNeighborsRegressor

regressor = KNeighborsRegressor(n_neighbors=101,weights="distance")

regressor.fit(x_train_std,y_train)
test = test.drop('year',axis=1)

predict = regressor.predict(test)

print(predict)
for i in range(len(predict)):

  submit['Expected'][i]=predict[i]
submit=submit.astype(np.int32)

submit.to_csv('submit.csv', mode='w', header= True, index= False)