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
import matplotlib.pyplot as plt

%matplotlib inline

from sklearn.model_selection import train_test_split as tts

from sklearn.metrics import mean_absolute_error as mae

import seaborn as sns
data = pd.read_csv('../input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv') 
data.head()
data.describe()
data.shape
data.isnull().sum()
data.columns
data['gender'] = data['gender'].replace('M', 0)

data['gender'] = data['gender'].replace('F', 1)
data.head()
data['ssc_b'] = data['ssc_b'].replace(["Others"], 0) 

data['ssc_b'] = data['ssc_b'].replace(["Central"], 1)
data.head()
data['hsc_b'] = data['hsc_b'].replace(["Others"], 0) 

data['hsc_b'] = data['hsc_b'].replace(["Central"], 1)
data.head()
data['hsc_s'] = data['hsc_s'].replace(["Commerce"], 0)

data['hsc_s'] = data['hsc_s'].replace(["Science"], 1) 

data['hsc_s'] = data['hsc_s'].replace(["Arts"], 2) 
data.head()
data['degree_t'] = data['degree_t'].replace(["Sci&Tech"], 1) 

data['degree_t'] = data['degree_t'].replace(["Comm&Mgmt"], 0) 

data['degree_t'] = data['degree_t'].replace(["Others"], 2)
data.head()
data['specialisation'] = data['specialisation'].replace(["Mkt&HR"], 1) 

data['specialisation'] = data['specialisation'].replace(["Mkt&Fin"], 0) 
data['workex'] = data['workex'].replace(["Yes"], 1) 

data['workex'] = data['workex'].replace(["No"], 0)
data['status'] = data['status'].replace(["Placed"], 1)

data['status'] = data['status'].replace(["Not Placed"], 0)
data
def impute_salary(cols):

    sal = cols[0]

    status = cols[1]

    

    if pd .isnull(sal):

        

        if status == 0:

            return 0.0

    else:

        return sal
data['salary'] = data[['salary', 'status']].apply(impute_salary, axis=1)
data.isnull().sum()
sns.heatmap(data.isnull(), yticklabels=False, cbar=False, cmap='viridis')
y = data[['status', 'salary']]
feature_list = ['gender', 'ssc_p', 'ssc_b', 'hsc_p', 'hsc_b', 'hsc_s', 'degree_p', 'degree_t', 'workex']
X = data[feature_list]
# Divide Data using validation and training data

train_X, val_X, train_y, val_y = tts(X, y, random_state=1)
from sklearn.tree import DecisionTreeRegressor as dtr
model1 = dtr()
model1.fit(train_X, train_y)
predict1 = model1.predict(val_X)
print(mae(val_y, predict1))
from sklearn.ensemble import RandomForestRegressor as rfr
model2 = rfr()
model2.fit(train_X, train_y)
predict2 = model2.predict(val_X)
print(mae(val_y, predict2))