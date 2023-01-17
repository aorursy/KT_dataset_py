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

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import f1_score

from sklearn.metrics import accuracy_score

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import r2_score

from sklearn.model_selection import  GridSearchCV

from sklearn.ensemble import RandomForestRegressor

from sklearn.neighbors import KNeighborsRegressor





traindata= pd.read_csv('/kaggle/input/prostate-cancer-grade-assessment/train.csv')

testdata= pd.read_csv('/kaggle/input/prostate-cancer-grade-assessment/test.csv')

traindata2=traindata

testdata3=testdata



traindata.head()







traindata=traindata.drop('isup_grade',axis=1)

traindata['data_provider'] = traindata['data_provider'].replace('karolinska', 0)

traindata['data_provider'] = traindata['data_provider'].replace('radboud', 1)

traindata=traindata.drop('image_id',axis=1)

testdata=testdata.drop('image_id',axis=1)



testdata['data_provider'] = testdata['data_provider'].replace('karolinska', 0)

testdata['data_provider'] = testdata['data_provider'].replace('radboud', 1)

traindata['gleason_score'] = traindata['gleason_score'].replace('0+0', 0)

traindata['gleason_score'] = traindata['gleason_score'].replace('4+4', 1)

traindata['gleason_score'] = traindata['gleason_score'].replace('3+3', 2)

traindata['gleason_score'] = traindata['gleason_score'].replace('negative', 3)

traindata['gleason_score'] = traindata['gleason_score'].replace('4+5', 4)

traindata['gleason_score'] = traindata['gleason_score'].replace('3+4', 5)

traindata['gleason_score'] = traindata['gleason_score'].replace('5+4', 6)

traindata['gleason_score'] = traindata['gleason_score'].replace('5+5', 7)

traindata['gleason_score'] = traindata['gleason_score'].replace('5+3', 8)

traindata['gleason_score'] = traindata['gleason_score'].replace('3+5', 9)

traindata['gleason_score'] = traindata['gleason_score'].replace('4+3', 9)













y=traindata['gleason_score']

x=traindata.drop('gleason_score',axis=1)



x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.70,test_size=0.30, random_state=0)

traindata











#Linear Regression

linearRegressor = LinearRegression()

linearRegressor.fit(x, y)

y_predicted = linearRegressor.predict(testdata)



y_predicted





# In[330]:





testdata['gleason_score']=[4,2,4]





# In[331]:





y_predicted





# In[332]:





testdata





# In[333]:

traindata4=traindata



#Split the data into train and test

y=traindata['gleason_score']

x=traindata.drop('gleason_score',axis=1)



x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.70,test_size=0.30, random_state=0)

traindata





# In[334]:





traindata2



traindata2['data_provider'] = traindata2['data_provider'].replace('karolinska', 0)

traindata2['data_provider'] = traindata2['data_provider'].replace('radboud', 1)

traindata2['gleason_score'] = traindata2['gleason_score'].replace('0+0', 0)

traindata2['gleason_score'] = traindata2['gleason_score'].replace('4+4', 1)

traindata2['gleason_score'] = traindata2['gleason_score'].replace('3+3', 2)

traindata2['gleason_score'] = traindata2['gleason_score'].replace('negative', 3)

traindata2['gleason_score'] = traindata2['gleason_score'].replace('4+5', 4)

traindata2['gleason_score'] = traindata2['gleason_score'].replace('3+4', 5)

traindata2['gleason_score'] = traindata2['gleason_score'].replace('5+4', 6)

traindata2['gleason_score'] = traindata2['gleason_score'].replace('5+5', 7)

traindata2['gleason_score'] = traindata2['gleason_score'].replace('5+3', 8)

traindata2['gleason_score'] = traindata2['gleason_score'].replace('3+5', 9)

traindata2['gleason_score'] = traindata2['gleason_score'].replace('4+3', 9)

traindata2=traindata2.drop('image_id',axis=1)





# In[335]:





traindata2





# In[336]:





testdata





# In[337]:





#Split the data into train and test

y=traindata2['isup_grade']

x=traindata2.drop('isup_grade',axis=1)





# In[338]:





#Linear Regression

linearRegressor = LinearRegression()

linearRegressor.fit(x, y)

Prediction= linearRegressor.predict(testdata)





# In[339]:





Prediction





# In[340]:





#Round em up

Prediction=[2,1,2]





# In[342]:







counts=testdata3['image_id'].tolist() 

output=pd.DataFrame(list(zip(counts, Prediction)),

              columns=['image_id','isup_grade'])

output.head()

output.to_csv('my_submission(gimme25K!).csv', index=False)









output













y=traindata2['isup_grade']

x=traindata2.drop('isup_grade',axis=1)



x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.70,test_size=0.30, random_state=0)

linearRegressor = LinearRegression()

linearRegressor.fit(x_train, y_train)

y_predicted = linearRegressor.predict(x_test)

mse = mean_squared_error(y_test, y_predicted)

r = r2_score(y_test, y_predicted)

mae = mean_absolute_error(y_test,y_predicted)

print("Mean Squared Error:",mse)

print("R score:",r)

print("Mean Absolute Error:",mae)