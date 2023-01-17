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
train_data = pd.read_csv("../input/train.csv")
train_data.head()
train_data.info()
train_data.describe()
test_data = pd.read_csv('../input/test.csv')
test_data.head()
pred_id =test_data['Id'] 
y_label = train_data['winPlacePerc']
train_data.drop(['winPlacePerc'], axis=1, inplace=True)
print("is OK")
train_data.drop(['Id', 'groupId', 'matchId'], axis=1, inplace=True)
test_data.drop(['Id', 'groupId', 'matchId'], axis=1, inplace=True)
print("Finish")
train_data.head()
from sklearn import linear_model
lr = linear_model.RidgeCV(alphas=[0.1,0.2,0.5, 1.0,2.,5.,10.0])
#lr = linear_model.LassoCV(alphas = [0.1,0.2,0.5,1.,2.,5.,10.])
lr.fit(train_data,y_label)
pred = lr.predict(test_data)
print("finish",lr.alpha_)
df_submission=pd.read_csv('../input/sample_submission.csv')
df_submission['Id']=pred_id
df_submission['winPlacePerc']=pred
df_submission.to_csv('Finish.csv', index=False)