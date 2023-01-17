# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # data visualization library 
import matplotlib.pyplot as plt

from scikitplot.metrics import plot_roc_curve

%matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv', index_col=False)
test = pd.read_csv('../input/test.csv', index_col=False)
test_y = pd.read_csv('../input/gender_submission.csv', index_col=False)
train.info()
train.describe()
sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap='viridis')
def impute_age(cols):
    ''' Imputing the age attribute by checking the pclass
        @param: cols list of column values
    '''
    pclass = cols[0]
    age = cols[1]
    
    if pd.isnull(age):        
        if pclass==3:
            age = 22.0
        elif pclass==2:
            age = 35.0
        else:
            age = 40.0
    return age

def cleanse_data(dataset, drop_cols):
    ''' Cleansing the dataframe data by removing the empty values and NaN
        @param: 
            dataset Dataframe
            drop_cols List of column names
    '''
    embark = pd.get_dummies(dataset['Embarked'], drop_first=True)
    sex = pd.get_dummies(dataset['Sex'], drop_first=True)
    dataframe = dataset.copy()
    
    dataframe['Age'] = dataset[['Pclass', 'Age']].apply(impute_age, axis=1)
        
    dataframe = dataframe.drop(drop_cols, axis=1)
    
    dataframe = pd.concat([dataframe, sex, embark], axis=1)
    
    return dataframe


drop_cols = ['Cabin', 'Name', 'Ticket', 'Fare', 'Sex', 'Embarked', 'PassengerId']

train_df = cleanse_data(train, drop_cols)
test_df = cleanse_data(test, drop_cols)
train_df.head()
from sklearn.linear_model import LogisticRegression
X = train_df.loc[:, train_df.columns != 'Survived']
y = train_df['Survived']

log_model = LogisticRegression()

log_model.fit(X, y)
pred = log_model.predict(test_df)
from sklearn.metrics import confusion_matrix, classification_report

print("Classification Report: \n %s" % classification_report(pred, test_y['Survived']))
print("Confusion Matrix : \n %s " % confusion_matrix(pred, test_y['Survived']))

result = test.copy()

result['Survived'] = pd.Series(pred)
result.head()

result[['PassengerId', 'Survived']].to_csv('submission.csv', sep=',')
import os

os.listdir('.')