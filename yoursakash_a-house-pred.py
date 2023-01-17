# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import OneHotEncoder

label_encoder = LabelEncoder()
#loading the dataset

train = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")

test = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")
null = train.isnull().sum()
#function to fill null values and encoded the categorical into int and normalize

encoding_li=[]

idx=0



for i,j in null.items():

    if j>600:

        train = train.drop(i,axis=1)

        continue

    if train[i].dtype in ['int64', 'float64']:

        train[i] = train[i].fillna(train[i].mean())

    elif train[i].dtype == 'object':

        train[i] = train[i].fillna(train[i].mode()[0])

        #train[i] = label_encoder.fit_transform(train[i])

        encoding_li.append([*zip(train[i].unique(), range(len(train[i].unique())))])

        for k in encoding_li[idx]:

            train[i] = train[i].replace(k[0], k[1])

        idx+=1

    else:

        print(train[i].dtype)

#train[i] = (train[i]-min(train[i]))/(max(train[i])-min(train[i])) # for normalize
encoding_li1=[]

idx1=0



for i,j in null.items():

    try:

        if j>600:

            test = test.drop(i,axis=1)

            continue

        if test[i].dtype in ['int64', 'float64']:

            test[i] = test[i].fillna(test[i].mean())

        elif test[i].dtype == 'object':

            test[i] = test[i].fillna(test[i].mode()[0])

            #test[i] = label_encoder.fit_transform(test[i])

            encoding_li1.append([*zip(test[i].unique(), range(len(test[i].unique())))])

            for k in encoding_li1[idx1]:

                test[i] = test[i].replace(k[0], k[1])

            idx1+=1

        else:

            print(test[i].dtype)

    except:

        pass
import h2o

from h2o.automl import H2OAutoML

h2o.init()
htrain = h2o.H2OFrame(train)

htest = h2o.H2OFrame(test)

x=htrain.columns

y='SalePrice'

x.remove(y)

# This line is added in the case of classification

# htrain[y] = htrain[y].asfactor()
aml = H2OAutoML(max_runtime_secs = 120)

aml.train(x=x, y=y, training_frame=htrain)

lb = aml.leaderboard

print (lb)

print('Generate predictions…')

test_y = aml.leader.predict(htest)

test_y = test_y.as_data_frame()
fil=open('submission.csv','w')

fil.write('Id,SalePrice\n')

i=1

for each in test_y['predict']:

   fil.write('%d,%d\n'%(i+1460,each))

   i=i+1

fil.close()