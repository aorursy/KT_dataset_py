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
n=pd.read_csv("/kaggle/input/graduate-admissions/Admission_Predict.csv")

n
# checking for null values

n.isnull().sum()
#draw a graph to check the independent variable

import seaborn as sns

sns.pairplot(n)
df=n.drop(["Serial No."], axis=1)

df
#creating dependent and independent variable

X=df.drop(['Chance of Admit '],axis=1)

y=df['Chance of Admit ']
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2)

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVR

from sklearn.model_selection import cross_val_score
#using this we can check which model is best 

cross_val_score(LinearRegression(),X,y).mean()


cross_val_score(RandomForestRegressor(),X,y).mean()
cross_val_score(DecisionTreeRegressor(),X,y).mean()
cross_val_score(SVR(),X,y).mean()
#from above we can see linear rehression is best

s=LinearRegression()

s.fit(X_train,y_train)
s.predict(X_test)