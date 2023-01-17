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

import matplotlib.pyplot as plt

from catboost import CatBoostRegressor

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

%matplotlib inline
data=pd.read_csv("/kaggle/input/weight-height/weight-height.csv")

data.head()

for i in range(0,10000):

    j=data['Weight'][i]

    k=data['Height'][i]

    data['Weight'][i]=j*0.453592

    data['Height'][i]=k*0.0328084

data.head(10)
data.tail()
data.plot(kind='scatter',x='Height',y='Weight')

plt.show()
data['gendercolor']=data['Gender'].map({'Male':'blue','Female':'red'})

data.plot(kind='scatter',x='Height',y='Weight',c=data['gendercolor'],title='male vs female distribution')
data.isnull().sum()
data.skew()
data.drop(['gendercolor'],axis=1,inplace=True)

data.head()
x=data.drop(['Weight'],axis=1)

y=data['Weight']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=10)

model=CatBoostRegressor(cat_features=['Gender'],verbose=False)

model.fit(x_train,y_train)

y_pred=model.predict(x_test)

error=np.sqrt(mean_squared_error(y_test,y_pred))

print(error)

from sklearn.metrics import r2_score

r2_er=r2_score(y_test,y_pred)

print(r2_er)
print(model.predict(['Female',5.1]))