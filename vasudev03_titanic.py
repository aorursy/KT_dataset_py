# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from sklearn.impute import SimpleImputer # Intermediate Machine Learning



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train='/kaggle/input/titanic/train.csv'

gender='/kaggle/input/titanic/gender_submission.csv'

test='/kaggle/input/titanic/test.csv'



train_data = pd.read_csv(train)

gender_sub = pd.read_csv(gender)

test_data = pd.read_csv(test)



print("Data loaded.")
# print("===============================Test Data===============================\n{}\nShape: {}\n".format(test_data.head(),test_data.shape))

# print("===============================Gender Submission===============================\n{}\nShape: {}\n".format(gender_sub.head(),gender_sub.shape))

# print("===============================Train Data===============================\n{}\nShape: {}\n".format(train_data.head(),train_data.shape))
print(train_data.columns)

print(test_data.columns)
features = ['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']

print("===============================Train Stats===============================\n")

for f in features:

    print("-------- {} Stats --------\n{}\n".format(f,train_data[f].describe()))

    

print("===============================Test Stats===============================\n")

for f in features:

    print("-------- {} Stats --------\n{}\n".format(f,test_data[f].describe()))
# # Old approach where I tried to fill missing values with mode 

# #thereby not changing the stats by much. (Hopefully)

# # Finding mode of age

# age_mode = train_data.Age.mode()

# age_mode
help(pd.Series.mode)
help(train_data.fillna)
# age_mode[0]
help(SimpleImputer.transform)
help(SimpleImputer.fit_transform)
# print(age_mode)

# train_data.Age.fillna(value=age_mode[0],inplace=True)

# test_data.Age.fillna(value=age_mode[0],inplace=True)

# train_data.Age.describe(),test_data.Age.describe()
test_data.Sex.replace(['male','female'],[0,1],inplace=True)

train_data.Sex.replace(['male','female'],[0,1],inplace=True)

test_data.Sex.describe(),train_data.Sex.describe()
features = ['Pclass','Sex','SibSp','Parch','Fare','Age']

features
X = train_data[features]

y = train_data["Survived"]



X,y
train_X,val_X,train_y,val_y = train_test_split(X,y,random_state=1)

train_X,val_X,train_y,val_y
my_imputer = SimpleImputer()



# NotFittedError: This SimpleImputer instance is not fitted yet. 

#         Call 'fit' with appropriate arguments before using this estimator.

# Hence using fit_transform()



trainX = pd.DataFrame( my_imputer.fit_transform(train_X) )

validX = pd.DataFrame( my_imputer.transform(val_X) )



trainX,validX
# from sklearn.ensemble import RandomForestRegressor

help(RandomForestRegressor.fit)
rf_model = RandomForestRegressor(n_estimators=300,max_depth=10,random_state=1) #,criterion='mae',random_state=1)

rf_model.fit(trainX,train_y) 

print("Training Complete")
X.describe()
rf_preds = rf_model.predict(validX)

rf_preds
rf_preds = [int(round(v,0)) for v in rf_preds]

rf_preds
mae = mean_absolute_error(val_y,rf_preds)

mae
test_data['Fare'].fillna(test_data.Fare.mode()[0],inplace=True)

test_data.describe()
testD = pd.DataFrame( my_imputer.transform(test_data[features]) )

testD.columns = features

testD.describe()
rf_preds = rf_model.predict(testD)

rf_preds
rf_preds = [int(round(v,0)) for v in rf_preds]

rf_preds
output = pd.DataFrame({"PassengerId":test_data.PassengerId,"Survived":rf_preds})

output=output.set_index("PassengerId")

output.to_csv('/kaggle/working/rf_impute_submission.csv')

print(output)

print("Submission saved")