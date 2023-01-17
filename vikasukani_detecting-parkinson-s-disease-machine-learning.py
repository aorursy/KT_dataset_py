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
#  install necessary packages

!pip install numpy pandas sklearn xgboost
# os packages

import os, sys
#  let’s read the data into a DataFrame 



df = pd.read_csv('/kaggle/input/parkinsons.data')

df.tail() # shows the last 5 rows



# head() <= Use for first 5 rows
# descrive the data



df.describe()
#  To know how many rows and cols and NA values



df.info()
#  shape of the dataset 



df.shape
#  get the all features except "status"



features = df.loc[:, df.columns != 'status'].values[:, 1:] # values use for array format







# get status values in array format



labels = df.loc[:, 'status'].values



# to know how many values for 1 and how many for 0 labeled status



df['status'].value_counts()


#  import MinMaxScaler class from sklearn.preprocessing



from sklearn.preprocessing import MinMaxScaler


#  Initialize MinMax Scaler classs for -1 to 1



scaler = MinMaxScaler((-1, 1))



# fit_transform() method fits to the data and

# then transforms it.



X = scaler.fit_transform(features)

y = labels



#  Show X and y  here

# print(X, y)
#  import train_test_split from sklearn. 



from sklearn.model_selection import train_test_split
# split the dataset into training and testing sets with 20% of testings



x_train, x_test, y_train, y_test=train_test_split(X, y, test_size=0.15)

# Load an XGBClassifier and train the model



from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score
# make a instance and fitting the model



model = XGBClassifier()

model.fit(x_train, y_train) # fit with x and y train

#  Finnaly pridict the model



y_prediction = model.predict(x_test)



print("Accuracy Score is", accuracy_score(y_test, y_prediction) * 100)