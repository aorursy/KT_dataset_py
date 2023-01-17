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

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.linear_model import LinearRegression

from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

pd.set_option('display.max_columns',None)

sns.set_style(style='darkgrid')
dataset = pd.read_csv('../input/car-price-prediction/CarPrice_Assignment.csv')

dataset.head()
dataset.info()
dataset.shape
dataset.describe()
dataset.columns
cols =['doornumber','carbody','enginetype','fueltype','drivewheel', 'enginelocation']

for col in cols:

    sns.barplot(x =col,y ='price',data=dataset)

    plt.show()
cols =['wheelbase', 'carlength', 'carwidth',

       'carheight', 'curbweight', 'enginesize', 'boreratio', 'stroke',

       'compressionratio', 'horsepower', 'peakrpm', 'citympg', 'highwaympg']

for col in cols:

    sns.regplot(x=col,y='price',data=dataset)

    plt.show()
df_num = dataset.select_dtypes(exclude=[np.number])
df_num
le = LabelEncoder()
for col in df_num.columns:

    dataset[col]=le.fit_transform(dataset[col])
dataset.sample(5)
dataset.drop('car_ID',axis=1,inplace=True)
plt.figure(figsize = (20,20))

sns.heatmap(dataset.corr(),annot=True, cmap='Blues')
df = dataset.copy()

df.head()
X= df.iloc[:,:-1].values

y= df.iloc[:,-1].values
print(X)
print(y)
x_train,x_test,y_train,y_test= train_test_split(X,y,test_size=0.2,random_state=0)
lr=LinearRegression()

lr.fit(x_train,y_train)
pred = lr.predict(x_test)
sns.distplot(y_test-pred)
from sklearn import metrics
metrics.r2_score(y_test,pred)