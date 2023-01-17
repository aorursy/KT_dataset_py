# this cell is just to access the csv files from Kaggle so don't worry about it

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# reading the data into the dataframe
df=pd.read_csv('../input/mushroom-classification/mushrooms.csv')
# let's take a peek into our data
df.head()
# let's check how many records we have and how many columns / features are there in our data

df.shape

# So we have 8124 records/rows and 23 columns/features
# let's check our target columns/feature
df

# let's look at our target varibale
df.columns

# Here class is our target varibale so let's start by separating it 
# let's move our target into y ( just a convetion for better readbality)
y=df['class']
y
# let's define our x with all other columns except the target columns
x=df.iloc[:,1:]
x
# let's check if there are any missing values in our  dataset

x.isna().sum()
# as we can see there are no missing values so we can start creating our model
from catboost import CatBoostClassifier

model=CatBoostClassifier()

# we have just defined our model with default parameters 
# Cat boost Classifier is a boosting based appraoch and can work with categorical variables which spare us from the 
# gard work of converting our non numeric columns into numeric using ( LabelEncoder , get_dummies etc)
# now let's make our train and test datasets
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=.25,random_state=3)

# we have split our x , y into x_train, x_test and y_train,y_test respectively with 25% as our test size and 75% train size 
# let's convert our target to numeric value 
from sklearn.preprocessing import LabelEncoder
y_train=pd.get_dummies(y_train,drop_first=True)

# let's tell our model which columns are of object type
a=x_train.select_dtypes(include='object')
li=a.columns.to_list()
# now let's fit our model 
model.fit(x_train,y_train,cat_features=li)
# let's check the training accuracy
model.score(x_train,y_train)

# model gives us a 100% accuracy ( looks like the model might be overfitting)
# let's see how our model performs on Test Data

model.score(x_test,y_test)

# model gives us a 100% accuracy 
# let's look at a classification report

from sklearn.metrics import classification_report
classification_report(model.predict(x_test),y_test)