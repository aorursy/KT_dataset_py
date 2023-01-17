# this section is just to import the dataset file , so don't worry about it

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# let's just read our data
df=pd.read_csv('/kaggle/input/regression-116-categorical-16-numeric-features/insur.csv',index_col=0)
df.head()
# let's see how many rows and columns we have in our dataset

df.shape

# we have 117794 rows and 131 columns
# let's look at the data in depth
df.describe()
# let's see if we have any missing values in our dataset

df.isnull().sum()
df.isna().sum()
# As we can see there are no missing values 
# let's create our x and y 
y=df.iloc[:,-1]
x=df.iloc[:,:-1]
x
# let's split our data into train and test sets
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.3,random_state=2)
# let's define our model , we will use CatBoostRegressor because it can handle categorical variable so we need not
# preprocess our data 

from catboost import CatBoostRegressor

model=CatBoostRegressor()
# let's seperate our categorical columns
cat_col=x_train.select_dtypes(include='object').columns
cat_col

# let's fit our model to the training dataset

model.fit(x_train,y_train,cat_features=cat_col)
# let's check training accuracy of our model

model.score(x_train,y_train)
# as we can see our model is still underfitting and has 63%  training accuracy
# let's check our test accuracy

model.score(x_test,y_test)

# model has 58% test accuracy which is much like tossing a coin