import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-whitegrid')
sns.set_style('whitegrid')
titanic_train = pd.read_csv('../input/titanic/train.csv')
titanic_test = pd.read_csv('../input/titanic/test.csv')

len(titanic_train), len(titanic_test)
titanic_train.head()
titanic_train.describe().T
titanic_train.info()
titanic_test.head()
titanic_test.describe().T
titanic_test.info()
titanic_train.isnull().sum()
sns.heatmap(titanic_train.isnull(), yticklabels=False, cbar=False, cmap='YlGnBu')
titanic_test.isnull().sum()
sns.heatmap(titanic_test.isnull(), yticklabels=False, cbar=False, cmap='YlGnBu')
sns.countplot(titanic_train['Survived'])
sns.countplot('Survived', hue='Sex', data=titanic_train)
sns.countplot(x='Survived',hue='Pclass',data=titanic_train)
plt.figure(figsize=(12, 7))
sns.boxplot(x='Pclass',y='Age',data=titanic_train,palette='winter')
titanic_train['Name'].str.split(' ')[24][1]
titanic_train['Name'] = titanic_train['Name'].apply(lambda x: str(x).split(' ')[1])
titanic_train['Name'] = titanic_train['Name'].apply(lambda x: x.replace('.',''))

titanic_test['Name'] = titanic_test['Name'].apply(lambda x: str(x).split(' ')[1])
titanic_test['Name'] = titanic_test['Name'].apply(lambda x: x.replace('.',''))
plt.figure(figsize=(15,5))
sns.countplot(titanic_test['Name'])
titanic_train['Name'].unique()
titanic_test['Name'].unique()
def change_name(cols):
    '''
    Change into simple categories
    '''
    Name = cols[0]
    Sex = cols[1]
    
    if Name == 'Mr' or Name == 'Mrs' or Name == 'Miss':
        return Name

    else:
        if Sex == 1:
            return 'Mr'
        elif Sex == 0:
            return 'Miss'
titanic_train['Name'] = titanic_train[['Name','Sex']].apply(change_name, axis=1)
titanic_test['Name'] = titanic_test[['Name','Sex']].apply(change_name, axis=1)
titanic_test[titanic_test['Pclass'] == 1]['Age'].median()
titanic_train[titanic_train['Pclass'] == 2]['Age'].median()
titanic_test[titanic_test['Pclass'] == 3]['Age'].median()
def impute_age(cols):
    '''
    Function to impute null values in age base on 'pclass'
    '''
    Pclass = cols[0]
    Age = cols[1]
    
    if pd.isnull(Age):

        if Pclass == 1:
            return 37

        elif Pclass == 2:
            return 29

        else:
            return 24

    else:
        return Age
titanic_train['age_isnull'] = titanic_train['Age'].isnull()
titanic_train['Age'] = titanic_train[['Pclass','Age']].apply(impute_age, axis=1)
titanic_train['Cabin'].str[0].value_counts()
titanic_train['cabin_isnull'] = titanic_train['Cabin'].isnull() 
titanic_train['Cabin'] = titanic_train['Cabin'].apply(lambda x: str(x)[0] if x!=0 else 'Z')
titanic_train['Embarked'].fillna('C', inplace=True)
titanic_test['age_isnull'] = titanic_test['Age'].isnull()
titanic_test['Age'] = titanic_test[['Pclass','Age']].apply(impute_age, axis=1)
titanic_test['cabin_isnull'] = titanic_test['Cabin'].isnull() 
titanic_test['Cabin'] = titanic_test['Cabin'].apply(lambda x: str(x)[0] if x!=0 else 'Z')
titanic_test['Fare'].fillna(14.4, inplace=True)
titanic_train.info()
def labelling(data, col): # i don't know why this function error
    temp = pd.get_dummies(data[col], drop_first=True)
    data.drop(col, axis=1, inplace=True)
    data = pd.concat([data,temp], axis=1)
# labelling(titanic_train, 'Sex')
# labelling(titanic_train, 'Cabin')
# labelling(titanic_train, 'Embarked')
temp = pd.get_dummies(titanic_train['Sex'], drop_first=True)
titanic_train['Sex'] = temp

temp = pd.get_dummies(titanic_train['Cabin'], drop_first=True)
titanic_train.drop('Cabin',axis=1, inplace=True)
titanic_train = pd.concat([titanic_train,temp], axis=1)

temp = pd.get_dummies(titanic_train['Embarked'], drop_first=True)
titanic_train.drop('Embarked',axis=1,inplace=True)
titanic_train = pd.concat([titanic_train,temp], axis=1)

temp = pd.get_dummies(titanic_train['Name'], drop_first=True)
titanic_train = pd.concat([titanic_train,temp], axis=1)
titanic_train.drop('Name', axis=1, inplace=True)
titanic_test.info()
temp = pd.get_dummies(titanic_test['Sex'], drop_first=True)
titanic_test['Sex'] = temp

temp1 = pd.get_dummies(titanic_test['Cabin'], drop_first=True)
temp2 = pd.get_dummies(titanic_test['Embarked'], drop_first=True)
temp3 = pd.get_dummies(titanic_test['Name'], drop_first=True)

titanic_test.drop('Cabin',axis=1,inplace=True)
titanic_test.drop('Embarked',axis=1,inplace=True)
titanic_test.drop('Name', axis=1, inplace=True)

titanic_test = pd.concat([titanic_test,temp1,temp2,temp3], axis=1)
titanic_test.head().T
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score
X = titanic_train.drop(['PassengerId', 'Survived', 'Ticket'], axis=1)
y = titanic_train['Survived']

X_test_real = titanic_test.drop(['PassengerId', 'Ticket'], axis=1)
X.drop('T', axis=1, inplace=True)

X.shape, X_test_real.shape
lgbc = LGBMClassifier(n_estimators=68, learning_rate=0.09, min_child_samples=25, colsample_bytree=0.8)
logmc = LogisticRegression(max_iter=500)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42).get_n_splits(X)
score = cross_val_score(lgbc, X, y)
score, round(score.mean() * 100, 4)
# score = cross_val_score(logmc, X, y)
# score, round(score.mean() * 100, 4)
param = dict(boosting_type=['gbdt','rf'],
            n_estimators=[100,200,400,700,1000],
            num_leaves=[31,20,50])
grSCV = GridSearchCV(lgbc, param, cv=skf, verbose=0)
# grSCV.fit(X, y)
# grSCV.best_params_
lgbc.fit(X, y)
pred = lgbc.predict(X_test_real)
output_data = pd.DataFrame()
output_data['PassengerId'] = titanic_test['PassengerId']
output_data['Survived'] = pred
output_data.head()
output_data.to_csv('titanic_pred.csv', index=False)