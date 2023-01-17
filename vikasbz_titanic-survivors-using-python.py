'''Titanic Survivors Prediction using Python
    A classification project by Vikas Zingade.
    Submitted to Kaggle competition.
'''
#Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn

print('All Good!')
#Read and write datasets
#Re-usable utility functions

#import pandas as pd

def read_data(train = 'train', test = 'test'):
    '''Inputs: "Train" CSV filename, "Test" CSV filename
        To read "train" and "test" data from CSV
        Outputs: "Train" and "Test" DataFrames
        
    Don't add ".csv" at the end of the filanmes
    '''
    
    train = pd.read_csv("../input/" + train + ".csv")
    test  = pd.read_csv("../input/" + test + ".csv")
    
    return train, test

#To write a DataFrame to CSV
def write_df(df, filename):
    '''Inputs: DataFRame to be written, target CSV filename
        To write DataFrame to CSV
        Outputs: None
        
    Don't add ".csv" at the end of the filanmes
    '''
    
    df.to_csv(filename + '.csv', index = False)
    print(filename, 'written to csv.')
    
print('All Good!')
#Read data: train and test
train, test = read_data()

print('All Good!')
print('Train data:', train.columns.values)
print('\n')
print('Test data:', test.columns.values)
#Combine train and test datasets for preprocessing: Missing Value Analysis and Outlier Analysis
predictor_cols = test.columns.values

#Combine train and test
data = pd.concat([train[predictor_cols], test])
data.is_copy = False

print('All Good!')
data.head()
data['Travelling_alone'] = data['SibSp'] + data['Parch']
data['Travelling_alone'] = np.where(data['Travelling_alone'] > 0, 0, 1)

data.drop(['SibSp', 'Parch'], axis=1, inplace=True)

print('All Good!')
data.drop(['PassengerId', 'Ticket', 'Name'], axis=1, inplace=True)

print('All Good!')
data.head()
data.info()
#Missing values count
#Re-usable utility function to get the columns with missing values

def na_count(data):
    '''Inputs: "data" DataFrame, "id" primary-key column
        To count the number of NAs in every column of the DataFrame.
        Outputs: A list of count of NAs in every column of teh DataFrame
    '''

    na_cols = []
    for i in data.columns:
        if data[i].isnull().sum() != 0:
            na_cols.append([i, data[i].isnull().sum(), round( data[i].isnull().sum() / len(data[i]) * 100, 4)])
    
    return na_cols

print('All Good!')
na_cols = na_count(data)

print('Missing Values:')
for i in na_cols:
    print(i)
plt.figure()
_ = data["Age"].hist()
data['Age'].fillna(data['Age'].median(skipna=True), inplace=True)

print('All Good!')
data['Fare'].head()
data['Fare'].describe()
plt.figure()
_ = data['Fare'].hist()
data['Fare'].fillna(data['Fare'].median(skipna=True), inplace=True)

print('All Good!')
data.drop('Cabin', axis=1, inplace = True)

print('All Good!')
data['Embarked'].head()
data['Embarked'].describe()
plt.figure()
_ = sbn.countplot(x='Embarked',data=data)
data['Embarked'].fillna('S', inplace = True)

print('All Good!')
na_cols = na_count(data)

print('Missing Values:')
for i in na_cols:
    print(i)
data.info()
plt.figure(figsize=(15,3))
_ = data.boxplot(column='Pclass', vert=False)
plt.figure(figsize=(15,3))
_ = data.boxplot(column='Age', vert=False)
plt.figure(figsize=(15,3))
_ = data.boxplot(column='Fare', vert=False)
data[data['Fare'] == max(data['Fare'])]
#If decided to go ahead with the outlier for 'Fare'
#Fare_max = max(data['Fare'])
#data.loc[data['Fare'] == Fare_max] = -999
#data.loc[data['Fare'] == -999] = data['Fare'].median()
data.info()
data['Pclass'].head()
data['Pclass'].unique()
data = pd.get_dummies(data, columns=['Pclass'])

print('All Good!')
data.drop('Pclass_3', axis=1, inplace=True)

print('All Good!')
data.head()
data['Sex'].head()
data['Sex'].unique()
data = pd.get_dummies(data, columns=['Sex'])

print('All Good!')
data.head()
data.drop('Sex_female', axis=1, inplace=True)

print('All Good!')
data['Embarked'].head()
data['Embarked'].unique()
data = pd.get_dummies(data, columns=['Embarked'])

print('All Good!')
data.drop('Embarked_Q', axis=1, inplace=True)

print('All Good!')
data.head()
data['is_minor'] = np.where(data['Age'] <= 16, 1, 0)
data.head()
train_df = data.iloc[:891]
test_df = data.iloc[891:]

print('All Good!')
test_df.head()
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(train_df, train['Survived'], test_size = 0.33, random_state = 0)

print('All Good!')
from sklearn import tree

DTC3 = tree.DecisionTreeClassifier(max_depth = 3)
DTC3.fit(X_train, Y_train)
Y_pred_DTC3 = DTC3.predict(X_test)

print('All Good!')
from sklearn.metrics import mean_absolute_error

DTC3_mae = mean_absolute_error(Y_test, Y_pred_DTC3)
print('Decision Tree Classifier with max_depth = 3:\nMean Absolute Error: ', DTC3_mae)
import graphviz 

DTC3_view = tree.export_graphviz(DTC3, out_file=None, feature_names = X_train.columns.values, rotate=True) 
DTC3viz = graphviz.Source(DTC3_view)
DTC3viz
DTC3_pred = DTC3.predict(test_df)

print('All Good!')
DTC3_submission = pd.DataFrame({'PassengerId':test['PassengerId'], 'Survived':DTC3_pred})

DTC3_submission.to_csv('DTC3_submission.csv', index=False)

print('All Good!')
from sklearn.ensemble import RandomForestClassifier

RFC = RandomForestClassifier(n_estimators = 100)
RFC.fit(X_train, Y_train)
Y_pred_RFC = RFC.predict(X_test)

print('All Good!')
RFC_mae = mean_absolute_error(Y_test, Y_pred_RFC)

print('Random Forest Classifier with max_depth = 3:\nMean Absolute Error: ', RFC_mae)
RFC_pred = RFC.predict(test_df)

print('All Good!')
RFC_submission = pd.DataFrame({'PassengerId':test['PassengerId'], 'Survived':RFC_pred})

RFC_submission.to_csv('RFC_submission.csv', index=False)

print('All Good!')
from sklearn.linear_model import LogisticRegression

LogReg = LogisticRegression()
LogReg.fit(X_train, Y_train)
Y_pred_LogReg = LogReg.predict(X_test)

print('All Good!')
LogReg_mae = mean_absolute_error(Y_test, Y_pred_LogReg)

print('Random Forest Classifier with max_depth = 3:\nMean Absolute Error: ', LogReg_mae)
LogReg_pred = LogReg.predict(test_df)

print('All Good!')
LogReg_submission = pd.DataFrame({'PassengerId':test['PassengerId'], 'Survived':LogReg_pred})

LogReg_submission.to_csv('LogReg_submission.csv', index=False)

print('All Good!')