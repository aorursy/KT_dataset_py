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
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#from pandas.tools.plotting import scatter_matrix

from pandas.plotting import autocorrelation_plot



import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap

from mpl_toolkits.mplot3d import axes3d, Axes3D

import seaborn as sns



from sklearn.preprocessing import scale

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import StratifiedShuffleSplit



from sklearn.svm import SVC

from sklearn.neighbors import NearestCentroid

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import MultinomialNB

from sklearn.decomposition import PCA



from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn import metrics



from itertools import product



import warnings

warnings.filterwarnings('ignore')
df_data = pd.read_csv('/kaggle/input/2019-pr-midterm-musicclassification/data_train.csv')

df_train = pd.DataFrame(df_data)



df_train.loc(1)



data = df_train.loc[1]["tempo":"mfcc20"]

data

label = df_train.loc[1]["label"]

label

len(df_train)

len(data)



scaler = StandardScaler()

#scaler = MinMaxScaler()



features = []

label = []



for d in range(len(df_train)):

  features.append(df_train.loc[d]["tempo":"mfcc20"])

  label.append(df_train.loc[d]["label"])

scaler.fit(features) 

features = scaler.transform(features)
x_train, x_test, y_train, y_test = train_test_split(features, label, test_size=0.25, random_state=42)
clf = SVC(kernel='rbf', class_weight='balanced', random_state=42)



param_grid = {'C':[0.01, 0.05, 0.1, 0.5, 1, 5, 10], 'gamma':[0.01, 0.05, 0.1,1, 5,10, 50]}



clf = GridSearchCV(clf, param_grid, cv=3)



clf.fit(x_train, y_train)



model = clf.best_estimator_

print(clf.best_params_)



y_predict = model.predict(x_test)



score = classification_report(y_test, y_predict)

print(score)
df_data = pd.read_csv('/kaggle/input/2019-pr-midterm-musicclassification/data_test.csv')

df_test = pd.DataFrame(df_data)



feature_test = []

label_test = []

data_id = []



for i in range(len(df_test)):

  feature_test.append(df_test.loc[i]["tempo":"mfcc20"])

  label_test.append(df_test.loc[i]["label"])

  data_id.append(i)

  

feature_test = scaler.transform(feature_test)
test_predict = model.predict(feature_test)





score = classification_report(label_test, test_predict)



print(test_predict)

#print(score)



#data = pd.DataFrame(test_predict,index=['id'],columns=['label'])

data_list = {''}



#data = pd.DataFrame(test_predict,index=['label'])

data = pd.DataFrame(test_predict)

#data.insert(0,['label',id])

df = data

df.index = df.index+1

#df.index.name = 'id'

#df.rename(columns={0:'label'})



f = ['label']

df2 = pd.DataFrame(f,index=['id'])



print(df2)

df3 = df2.append(df)



print(df3)

#df.loc['id'] = ['label']

df= df3

#print(df)
import pandas as pd



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



print(df)



df.to_csv('results-sm-v5.csv',index=True, header=False)

! kaggle competitions submit -c 2019-pr-midterm-musicclassification -f results-sm-v5.csv -m "smwi-1101"