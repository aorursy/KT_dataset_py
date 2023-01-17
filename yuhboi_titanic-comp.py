# VINCENT CHEN

# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Imports

from sklearn.ensemble import RandomForestClassifier

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import train_test_split

from xgboost import XGBRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.ensemble import GradientBoostingClassifier

import pandas as pd

import math

import statistics 

from statistics import mean 

from sklearn.metrics import accuracy_score

from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from sklearn.preprocessing import StandardScaler

import pandas as pd, catboost

from sklearn.pipeline import make_pipeline

from sklearn.model_selection import GridSearchCV

import seaborn as sns
#Read CSV and misc.

train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

test_data = pd.read_csv("/kaggle/input/titanic/test.csv")



train_data['Embarked'] = train_data['Embarked'].fillna(train_data['Embarked'].mode()[0])

test_data['Embarked'] = test_data['Embarked'].fillna(train_data['Embarked'].mode()[0])

train_data['Cabin'] = train_data['Cabin'].fillna("unknown")

test_data['Cabin'] = test_data['Cabin'].fillna("unknown")

train_data['Fare'] = train_data['Fare'].fillna(train_data['Fare'].mean())

test_data['Fare'] = test_data['Fare'].fillna(train_data['Fare'].mean())

train_data['Surname'], test_data['Surname'] = [df.Name.str.split(',').str[0] for df in [train_data, test_data]]
#go through name column and change all the names to their respective title

i = 0

for i in range(len(train_data)):

    train_data['Name'][i] = train_data['Name'][i].split(',')[1].split('.')[0].strip()

    if (i < 890):

        i = i+1

        

#Change all Titles to their catagories (train)

for j in range(len(train_data)):

    if train_data['Name'][j] == "Capt" or train_data['Name'][j] == "Col" or train_data['Name'][j] == "Major" or train_data['Name'][j] == "Dr" or train_data['Name'][j] == "Rev":

        train_data['Name'][j] = "Officer"

    elif train_data['Name'][j] == "Jonkheer" or train_data['Name'][j] =="Don" or train_data['Name'][j] == "Dona" or train_data['Name'][j] == "the Countess" or train_data['Name'][j] == "Lady" or train_data['Name'][j] == "Sir":

        train_data['Name'][j] = "Royalty"

    elif train_data['Name'][j] == "Mme" or train_data['Name'][j] == "Mrs":

        train_data['Name'][j] = "Mrs"

    elif train_data['Name'][j] == "Mr":

        train_data['Name'][j] = "Mr"

    elif train_data['Name'][j] == "Miss" or train_data['Name'][j] == "Ms" or train_data['Name'][j] == "Mlle":

        train_data['Name'][j] = "Miss"

    elif train_data['Name'][j] == "Master":

        train_data['Name'][j] = "Master"    

        

#Set test_data 'Name'(s) to Titles

r = 0

for r in range(len(test_data)):

    test_data['Name'][r] = test_data['Name'][r].split(',')[1].split('.')[0].strip()

    if (r < 417):

        r = r + 1



#Change all Titles to their catagories (test)

for c in range(len(test_data)):

    if test_data['Name'][c] == "Capt" or test_data['Name'][c] == "Col" or test_data['Name'][c] == "Major" or test_data['Name'][c] == "Dr" or test_data['Name'][c] == "Rev":

        test_data['Name'][c] = "Officer"

    elif train_data['Name'][j] == "Jonkheer" or train_data['Name'][c] =="Don" or train_data['Name'][c] == "Dona" or train_data['Name'][c] == "the Countess" or train_data['Name'][c] == "Lady" or train_data['Name'][c] == "Sir":

        train_data['Name'][c] = "Royalty"

    elif train_data['Name'][c] == "Mme" or train_data['Name'][c] == "Mrs":

        train_data['Name'][c] = "Mrs"

    elif train_data['Name'][c] == "Mr":

        train_data['Name'][c] = "Mr"

    elif test_data['Name'][c] == "Miss" or test_data['Name'][c] == "Ms" or test_data['Name'][c] == "Mlle":

        test_data['Name'][c] = "Miss"

    elif test_data['Name'][c] == "Master":

        test_data['Name'][c] = "Master" 

#Finds average ages (train)

officer_age = []

royalty_age = []

mrs_age = []

mr_age = []

miss_age = []

master_age = []

f_officer_age = []

f_royalty_age = []

d = 0

for name in train_data['Name']:

    if train_data['Name'][d] == "Officer" and not(math.isnan(train_data['Age'][d])):

        if train_data['Sex'][d] == 'female':

            f_officer_age.append(train_data['Age'][d])

        else:

            officer_age.append(train_data['Age'][d])

            

    if train_data['Name'][d] == "Royalty" and not(math.isnan(train_data['Age'][d])):

        if train_data['Sex'][d] == 'female':

            f_royalty_age.append(train_data['Age'][d])

        else:

            royalty_age.append(train_data['Age'][d])            

            

    if train_data['Name'][d] == "Mr" and not(math.isnan(train_data['Age'][d])):

        mr_age.append(train_data['Age'][d])

        

    if train_data['Name'][d] == "Miss" and not(math.isnan(train_data['Age'][d])):

        miss_age.append(train_data['Age'][d])

        

    if train_data['Name'][d] == "Master" and not(math.isnan(train_data['Age'][d])):

        master_age.append(train_data['Age'][d])

        

    if (d < 890):

        d = d + 1

#-------------------------------------------------------------------------------------------#

d = 0

for name in test_data['Name']:

    if test_data['Name'][d] == "Officer" and not(math.isnan(test_data['Age'][d])):

        if test_data['Sex'][d] == 'female':

            f_officer_age.append(test_data['Age'][d])

        else:

            officer_age.append(test_data['Age'][d])

            

    if test_data['Name'][d] == "Royalty" and not(math.isnan(test_data['Age'][d])):

        if test_data['Sex'][d] == 'female':

            f_royalty_age.append(test_data['Age'][d])

        else:

            royalty_age.append(test_data['Age'][d])            

            

    if test_data['Name'][d] == "Mrs" and not(math.isnan(test_data['Age'][d])):

        mrs_age.append(test_data['Age'][d])

            

    if test_data['Name'][d] == "Mr" and not(math.isnan(test_data['Age'][d])):

        mr_age.append(test_data['Age'][d])

        

    if test_data['Name'][d] == "Miss" and not(math.isnan(test_data['Age'][d])):

        miss_age.append(test_data['Age'][d])

        

    if test_data['Name'][d] == "Master" and not(math.isnan(test_data['Age'][d])):

        master_age.append(test_data['Age'][d])

    if (d < 418):

        d = d + 1

#-------------------------------------------------------------------------------------------#

officer_mean = mean(officer_age)

royalty_mean = mean(royalty_age)

f_officer_mean = mean(f_officer_age)

f_royalty_mean = mean(f_royalty_age)

mrs_mean = mean(mrs_age)

mr_mean = mean(mr_age)

miss_mean = mean(miss_age)

master_mean = mean(master_age)
#Age Assignment train

train_data["Age"] = train_data["Age"].fillna(-1)

test_data["Age"] = test_data["Age"].fillna(-1)

for counter in range(len(train_data)):

    if train_data['Name'][counter] == 'Officer' and train_data['Sex'][counter] == 'female' and train_data['Age'][counter] == -1:

        train_data['Age'][counter] = f_officer_mean

    if train_data['Name'][counter] == 'Officer' and train_data['Sex'][counter] == 'male' and train_data['Age'][counter] == -1:

        train_data['Age'][counter] = officer_mean

    if train_data['Name'][counter] == 'Royalty' and train_data['Sex'][counter] == 'female' and train_data['Age'][counter] == -1:

        train_data['Age'][counter] = f_royalty_mean

    if train_data['Name'][counter] == 'Royalty' and train_data['Sex'][counter] == 'male' and train_data['Age'][counter] == -1:

        train_data['Age'][counter] = royalty_mean

    if train_data['Name'][counter] == 'Mrs' and train_data['Age'][counter] == -1:

        train_data['Age'][counter] = mrs_mean

    if train_data['Name'][counter] == 'Mr' and train_data['Age'][counter] == -1:

        train_data['Age'][counter] = mr_mean

    if train_data['Name'][counter] == 'Miss' and train_data['Age'][counter] == -1:

        train_data['Age'][counter] = miss_mean

    if train_data['Name'][counter] == 'Master' and train_data['Age'][counter] == -1:

        train_data['Age'][counter] = master_mean

for counter in range(len(test_data)):

    if test_data['Name'][counter] == 'Officer' and test_data['Sex'][counter] == 'female' and test_data['Age'][counter] == -1:

        test_data['Age'][counter] = f_officer_mean

    if test_data['Name'][counter] == 'Officer' and test_data['Sex'][counter] == 'male' and test_data['Age'][counter] == -1:

        test_data['Age'][counter] = officer_mean

    if test_data['Name'][counter] == 'Royalty' and test_data['Sex'][counter] == 'female' and test_data['Age'][counter] == -1:

        test_data['Age'][counter] = f_royalty_mean

    if test_data['Name'][counter] == 'Royalty' and test_data['Sex'][counter] == 'male' and test_data['Age'][counter] == -1:

        test_data['Age'][counter] = royalty_mean

    if test_data['Name'][counter] == 'Mrs' and test_data['Age'][counter] == -1:

        test_data['Age'][counter] = mrs_mean

    if test_data['Name'][counter] == 'Mr' and test_data['Age'][counter] == -1:

        test_data['Age'][counter] = mr_mean

    if test_data['Name'][counter] == 'Miss' and test_data['Age'][counter] == -1:

        test_data['Age'][counter] = miss_mean

    if test_data['Name'][counter] == 'Master' and test_data['Age'][counter] == -1:

        test_data['Age'][counter] = master_mean
model = catboost.CatBoostClassifier(one_hot_max_size=4, iterations=100, random_seed=0, verbose=False)

model.fit(train_data[['Sex', 'Pclass', 'Embarked', 'Age', 'Name']].fillna(''), train_data['Survived'], cat_features=[0, 2, 4])

pred = model.predict(test_data[['Sex', 'Pclass', 'Embarked', 'Age', 'Name']].fillna('')).astype('int')

pd.concat([test_data['PassengerId'], pd.DataFrame(pred, columns=['Survived'])], axis=1).to_csv('submission.csv', index=False)