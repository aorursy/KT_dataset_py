# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Read in both datasets >>>

train = pd.read_csv('../input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')
# Data shape >>>

print(f'Train Size: {train.shape}')

print(f'Test Size: {test.shape}')
# Find NaN's in train

train.isnull().sum()
# Find NaN's in test

test.isnull().sum()
# Examine missing Embarked values: 

train['Embarked'].isnull().sum()
# Drop NaNs from embarked:

train = train.loc[train['Embarked'].notnull(), :].copy()   # Null filter on train df. Make sure to do a .copy() with this sort of filters. 

print("NaN's in Embarked: {0}".format(train['Embarked'].isnull().sum()))
# Examine Fare NaN: Individual missing a fare >>> 'Storey, Mr. Thomas'

nulls_fare = test.loc[test['Fare'].isnull(), :]

nulls_fare.head()
# Impute a fare for Mr. Thomas = 'Pclass 3rd mean fare' >>>

third_class_mean_fare = test.loc[test['Pclass'] == 3, 'Fare'].mean()

test.loc[test['PassengerId'] == 1044, 'Fare'] = third_class_mean_fare

test.loc[test['PassengerId'] == 1044]
# Create outliers for NaN's in Age so the model will consider all NaN's as a one group >>>

train.loc[train['Age'].isnull(), 'Age'] = 999

print(train['Age'].isnull().sum())

test.loc[test['Age'].isnull(), 'Age'] = 999

print(test['Age'].isnull().sum())

train['Age'].describe()
# Cabin is by far the worst data point; I will binarize this as 1 = cabin & 0 = no cabin >>>

train.loc[train['Cabin'].isnull(), 'Cabin'] = 0

train.loc[train['Cabin'].notnull(), 'Cabin'] = 1



test.loc[test['Cabin'].isnull(), 'Cabin'] = 0

test.loc[test['Cabin'].notnull(), 'Cabin'] = 1
# Since we're only dealing with two genders, we need to dummy new columns instead of binarizing to do skew our model further on >>>

# We will also dummy the embarked field. 

train = pd.get_dummies(train, columns=['Sex', 'Embarked'], drop_first=True)

test = pd.get_dummies(test, columns=['Sex', 'Embarked'], drop_first=True)
train.head()
test.head()
# Now lets prepare out training data for our Random Forest model >>>

features = [

    'Pclass',

    'Age',

    'SibSp',

    'Parch',

    'Fare',

    'Cabin',

    'Sex_male',

    'Embarked_Q',

    'Embarked_S'

]



X = train[features]

y = train['Survived']
# Double check training data for nulls >>>

X.isnull().sum()
# Benchmark survival score: 38% survived vs 62% deceased 

y.value_counts(normalize=True)
# Train/Test split >>>

X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=42, stratify=y)
# Instantiate RF >>>

rf = RandomForestClassifier()
# Cross val score >>>

cross_val_score(rf, X_train, y_train, cv=5).mean()
# Run Random Forest Classififer >>>

rf = RandomForestClassifier(random_state=42)

params = {

    'n_estimators': [50, 100],  # 100 trees or 50? (100)

    'max_features': [None, 'auto'],

    'max_depth': [None, 2, 3, 4]

}

gs = GridSearchCV(rf, param_grid=params, cv=5)

gs.fit(X_train, y_train)

print(gs.best_score_)

gs.best_params_
# Generate predictions >>>

pred = gs.predict(test[features])

test['Survived'] = pred

test[['PassengerId', 'Survived']]
# Export submission.csv >>>

test[['PassengerId', 'Survived']].to_csv('submission.csv', index=False)