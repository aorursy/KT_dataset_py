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
#helpers

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split
data=pd.read_csv(os.path.join(dirname, filename))
data.head()
#Female 



f_count=len(data[data.gender=='F'])

f_placed=len(data[(data.gender=='F') & (data.status=='Placed') ])

#percentage of female placed

f_placed_percent=f_placed/f_count*100

f_placed_percent
#Male

m_count=len(data[data.gender=='M'])

m_placed=len(data[(data.gender=='M') & (data.status=='Placed') ])

#percentage of male placed

m_placed_percent=m_placed/m_count*100

m_placed_percent
f_count/m_count
data.describe()
y = data['status']

features = list(data.columns)

features.remove('status')

#print(features)

X = data[features].copy()

X=X.drop('sl_no',axis=1)

X
%matplotlib inline

pd.crosstab(data.workex,data.status).plot(kind='bar')

plt.title('experience vs placement')

plt.xlabel('experience')

plt.ylabel('Placement_status')

plt.savefig('placement Vs experience')
%matplotlib inline

pd.crosstab(data.degree_t,data.status).plot(kind='bar')

plt.title('domain vs placement')

plt.xlabel('Domain')

plt.ylabel('Placement_status')

plt.savefig('Domain Vs placement')
_={'Placed':1,'Not Placed':0}

y=y.map(_)
#X=pd.get_dummies(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)#