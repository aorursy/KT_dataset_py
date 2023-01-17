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
train = pd.DataFrame(pd.read_csv('/kaggle/input/mlregression-cabbage-price/train_cabbage_price.csv'))

test = pd.DataFrame(pd.read_csv('/kaggle/input/mlregression-cabbage-price/test_cabbage_price.csv'))

submit = pd.DataFrame(pd.read_csv('/kaggle/input/mlregression-cabbage-price/sample_submit.csv'))



train=train.drop('year', axis=1)

test=test.drop('year', axis=1)



print(train)

print(test)

print(submit)
tain_x = train.iloc[:, :-1] # year  avgTemp  minTemp  maxTemp  rainFall

tain_y = train.iloc[:, -1] # avgPrice
from sklearn.neighbors import KNeighborsRegressor

rnd = np.random.RandomState(42)



knn = KNeighborsRegressor(n_neighbors=5)

knn.fit(tain_x, tain_y)

pred_y = knn.predict(test)

submit["Expected"] = pred_y

print(submit)



submit.to_csv('baseline.csv', index=False)