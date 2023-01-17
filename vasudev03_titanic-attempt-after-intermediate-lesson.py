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
train = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')



# Understanding the size and the type of data

'Train Data',train.shape, train.dtypes.value_counts(),'Test Data', test.shape, test.dtypes.value_counts()
train.describe(), test.describe()
# Finding all object columns (Python considers strings as objects)

object_cols = [col for col in train if train[col].dtype=='object']

object_cols
# Finding missing values in all columns

train_temp=[col for col in train if train[col].isnull().any()]

train_tempc = (train.isnull().sum())





test_temp=[col for col in test if test[col].isnull().any()]

test_tempc = (test.isnull().sum())



train_temp, train_tempc, test_temp,test_tempc

    
# Dropping columns with large missing values i.e. 'Embarked' column 

# and columns that might not add significant value.

train_data = train.drop(columns=['Embarked','Name','Ticket','PassengerId'],axis=1)

test_data = test.drop(columns=['Embarked','Name','Ticket','PassengerId'],axis=1)

object_cols = [col for col in train_data if train_data[col].dtype=='object']



train_data.columns, test_data.columns

# Replacing missing values in object columsn with 'None'

train_data[object_cols] = train_data[object_cols].fillna('None')

test_data[object_cols] = test_data[object_cols].fillna('None')



train_data[object_cols].describe,test_data[object_cols].describe
# Train-Test Split

from sklearn.model_selection import train_test_split

X = train_data.drop(columns=['Survived']).copy()

y = train_data['Survived']



X_train, X_valid, y_train, y_valid = train_test_split(X,y,random_state=0)

X_train.shape, X_valid.shape
# Finding number of unique values in each column

# Get number of unique entries in each column with categorical data

object_nunique = list(map(lambda col: X_train[col].nunique(), object_cols))

d = dict(zip(object_cols, object_nunique))



# Print number of unique entries by column, in ascending order

sorted(d.items(), key=lambda x: x[1])

# One-Hot Encoding Gender column

from sklearn.preprocessing import OneHotEncoder



OH_encoder = OneHotEncoder(handle_unknown='ignore',sparse=False)



OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[['Sex']]))

OH_cols_test = pd.DataFrame(OH_encoder.transform(X_valid[['Sex']]))

OH_test = pd.DataFrame(OH_encoder.transform(test_data[['Sex']]))





OH_cols_train, OH_cols_test,OH_test

# Adding index and column names back to One-hot encoded datasets.

OH_cols_train.index = X_train.index

OH_cols_test.index = X_valid.index

OH_test.index = test_data.index



OH_cols_train.columns = OH_encoder.get_feature_names(['Sex'])

OH_cols_test.columns = OH_encoder.get_feature_names(['Sex'])

OH_test.columns = OH_encoder.get_feature_names(['Sex'])



num_X_train = X_train.drop(['Sex'],axis=1)

num_X_valid = X_valid.drop(['Sex'],axis=1)

num_test = test_data.drop(['Sex'],axis=1)





OH_X_train = pd.concat([num_X_train,OH_cols_train],axis=1)

OH_X_valid = pd.concat([num_X_valid,OH_cols_test],axis=1)

OH_test = pd.concat([num_test,OH_test],axis=1)



OH_X_train.describe(), OH_X_valid.describe(), OH_test
# Finding difference in the unique values in training data and test data for column Cabin

OH_X_train['Cabin'].unique(),OH_X_valid['Cabin'].unique(),'Are the unique values in X_train == X_valid?',set(OH_X_train['Cabin']) == set(OH_X_valid['Cabin'])
# Dropping Cabin column too as the unique values in training dataset is not equal to validation dataset. And 

# I'm not sure how to handle new values that might show up in the validation dataset.

print('X_train: {},\nX_valid: {}\ntest_data: {}'.format(OH_X_train.shape,OH_X_valid.shape,OH_test.shape))

OH_X_train = OH_X_train.drop(columns=['Cabin'],axis=1)

OH_X_valid = OH_X_valid.drop(columns=['Cabin'],axis=1)

OH_test = OH_test.drop(columns=['Cabin'],axis=1)

print('X_train: {},\nX_valid: {}\ntest_data: {}'.format(OH_X_train.shape,OH_X_valid.shape,OH_test.shape))
# Checking to make sure that all object type columns are handled.

OH_X_train.dtypes, OH_X_valid.dtypes,OH_test.dtypes
# Imputing missing data from numerical columns.



from sklearn.impute import SimpleImputer



# Imputation

my_imputer = SimpleImputer(strategy='most_frequent')

imputed_X_train = pd.DataFrame( my_imputer.fit_transform( OH_X_train ) )

imputed_X_valid = pd.DataFrame( my_imputer.transform( OH_X_valid ))

imputed_test = pd.DataFrame( my_imputer.transform (OH_test))



# Imputation removes column names. Need to put them back.

imputed_X_train.columns = OH_X_train.columns

imputed_X_valid.columns = OH_X_valid.columns

imputed_test.columns = OH_test.columns



imputed_X_train.describe(),imputed_X_valid.describe(), imputed_test
# Using random forest regression



from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error



forest_model = RandomForestRegressor(random_state=1,n_estimators=300,max_depth=10)

forest_model.fit(imputed_X_train,y_train)

rf_preds = forest_model.predict(imputed_X_valid)

print(mean_absolute_error(y_valid,rf_preds))

rf_preds = forest_model.predict(imputed_test)

rf_preds
# Rounding values.

rf_preds = [int(round(v,0)) for v in rf_preds]

rf_preds
output = pd.DataFrame({'PassengerId':test.PassengerId,'Survived':rf_preds})

output = output.set_index('PassengerId')

output.to_csv('/kaggle/working/rf_imputed_oh_submission.csv')

print(output)

print('Submission saved')