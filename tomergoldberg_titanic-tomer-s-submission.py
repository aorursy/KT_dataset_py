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
base_input_train = pd.read_csv('../input/titanic/train.csv')




base_input_train.head()
# Helper methods:

def get_data_splits(dataframe, valid_fraction=0.2):

    valid_size = int(len(dataframe) * valid_fraction)

    train = dataframe[:-valid_size * 2]

    # valid size == test size, last two sections of the data

    valid = dataframe[-valid_size * 2:-valid_size]

    

    return train, valid
from sklearn.impute import SimpleImputer

from sklearn.preprocessing import LabelEncoder



def preprocess_data(input_data):

    '''

    Removes the unsed columns.

    Applies label encoder to categorical data.

    Fill missing values using the 'SimpleImputer'

    '''



    input_copy = input_data.copy()

    

    # Remove unsed columns

    cols_to_remove = ['Ticket', 'Name', 'PassengerId']

    input_copy = input_copy.drop(cols_to_remove, axis = 1)

    

    # Apply label encoder to each column with categorical data

    cols_with_label = ['Sex','Embarked', 'Cabin']

    label_encoder = LabelEncoder()

    for col in cols_with_label:

        label_encoder.fit(input_copy[col].map(str))

        input_copy[col] = label_encoder.transform(input_copy[col].map(str))

        

    #Handle with missing values

    my_imputer = SimpleImputer()

    

    input_copy_imput = pd.DataFrame(my_imputer.fit_transform(input_copy))

    

    #Imputation removed column names; putting them back

    input_copy_imput.columns = input_copy.columns

    

    return input_copy_imput



preprocess_data(base_input_train).head()  
import lightgbm as lgb

from sklearn import metrics

from sklearn.ensemble import RandomForestClassifier



def train_model_lgb(train, validation):

    y_train = train['Survived']

    x_train = preprocess_data(train.drop('Survived', axis =1))

   

    y_valid = validation['Survived']

    x_valid = preprocess_data(validation.drop('Survived', axis =1))





    dtrain = lgb.Dataset(x_train, label= y_train)

    dvalid = lgb.Dataset(x_valid, label= y_valid)

    param = {'num_leaves': 64, 'objective': 'binary','metric': 'auc', 'seed': 7}

   

    print("Training model!")

    bst = lgb.train(param, dtrain, num_boost_round=1000, valid_sets=[dvalid], 

                    early_stopping_rounds=10, verbose_eval=False)



    valid_pred = bst.predict(x_valid)

    valid_score = metrics.roc_auc_score(y_valid, valid_pred)

    print(f"LGB - Validation AUC score: {valid_score:.4f}")

    return bst



def train_model_random_forest_classifier(train, validation):

    y_train = train['Survived']

    x_train = preprocess_data(train.drop('Survived', axis =1))

   

    y_valid = validation['Survived']

    x_valid = preprocess_data(validation.drop('Survived', axis =1))



    clf = RandomForestClassifier(max_depth=5, random_state=1)

    print("Training model!")

    clf.fit(x_train, y_train)

    

    valid_pred = clf.predict(x_valid)

    valid_score = metrics.roc_auc_score(y_valid, valid_pred)

    print(f"RFC - Validation AUC score: {valid_score:.4f}")

    return clf
# Running all:



base_input = pd.read_csv('../input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')



train,valid = get_data_splits(base_input)



lgb_model = train_model_lgb(train, valid)

rfc_model = train_model_random_forest_classifier(train,valid)



x_test = preprocess_data(test)

test_pred = lgb_model.predict(x_test)



output = pd.DataFrame({'PassengerId': test.PassengerId.tolist(), 'Survived': test_pred.tolist()})

output.to_csv('result.csv', index=False)
