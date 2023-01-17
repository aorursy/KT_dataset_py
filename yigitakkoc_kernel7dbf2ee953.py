# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data=pd.read_csv('../input/biomechanical-features-of-orthopedic-patients/column_2C_weka.csv')
data.head()
#making class values numeric 0 for normal, 1 for abnormal



data['class']=[1 if each =="Abnormal" else 0 for each in data['class']]

dc=data['class'].values

f_data=data.drop(["class"],axis=1)
#normalization

fn=(f_data-np.min(f_data))/(np.max(f_data)-np.min(f_data))
#train, test split

from sklearn.model_selection import train_test_split

fn_train,fn_test,dc_train,dc_test=train_test_split(fn,dc,test_size=0.3,random_state=1)
#model with KNN

from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors=27)

knn.fit(fn_train,dc_train)

prediction=knn.predict(fn_test)
print(knn.score(fn_test,dc_test))
#specifying best k value, k=27 is the best, accuracy is %80

a_list=[]

for each in range(1,30):

    knn2=KNeighborsClassifier(n_neighbors=each)

    knn2.fit(fn_train,dc_train)

    a_list.append(knn2.score(fn_test,dc_test))

plt.plot(range(1,30),a_list)

plt.xlabel='k values'

plt.ylabel='accuracy'

plt.show()

    