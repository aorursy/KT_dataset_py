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

        

df_data = pd.read_csv(os.path.join(dirname, filenames[1]))

df_train = pd.DataFrame(df_data)



df_data = pd.read_csv(os.path.join(dirname, filenames[0]))

df_test = pd.DataFrame(df_data)



# Any results you write to the current directory are saved as output.
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report

from sklearn.svm import SVC
print(df_train)
data = df_train.loc[1]["tempo":"mfcc20"]

data

label = df_train.loc[1]["label"]

label

len(df_train)

len(data)



scaler = StandardScaler()

data
vectors = []

label = []



for i in range(len(df_train)):

  vector = df_train.loc[i]["tempo":"mfcc20"]

  vectors.append(vector)

  label.append(df_train.loc[i]["label"])

scaler.fit(vectors) 

vectors = scaler.transform(vectors)

print(vectors, label)
clf = SVC(kernel='linear', class_weight='balanced', random_state=42)



param_grid = {'C':[0.0001, 0.01, 0.1, 1, 10], 'gamma':[0.0001,0.001, 0.01, 0.1, 1 ,10]}

#param_grid = {'C':[0.0001, 0.01, 0.1, 1, 10, 100]}



clf = GridSearchCV(clf, param_grid, cv=3)



clf.fit(vectors, label)



model = clf.best_estimator_





print(clf.best_params_)

vectors_val = []

label_val = []

data_id = []



for i in range(len(df_test)):

  vector = df_test.loc[i]["tempo":"mfcc20"]

  vectors_val.append(vector)

  label_val.append(df_test.loc[i]["label"])

  data_id.append(i)

  

vectors_val = scaler.transform(vectors_val)

print(vectors_val, label_val)
len(vectors_val)

print(df_test)
y_predict = model.predict(vectors_val)



score = classification_report(label_val, y_predict)



print(score)
#df = pd.DataFrame(y_predict, columns=['label'])



name=[]

data1={}



df = df.replace('blues',0)

df = df.replace('classical',1)

df = df.replace('country',2)

df = df.replace('disco',3)

df = df.replace('hiphop',4)

df = df.replace('jazz',5)

df = df.replace('metal',6)

df = df.replace('pop',7)

df = df.replace('reggae',8)

df = df.replace('rock',9)
df.index = np.arange(1,len(df)+1)

df.index.name = 'id'



print(df, score)