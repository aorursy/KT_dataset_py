import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
train = pd.read_csv("../input/titanic/train.csv")

test = pd.read_csv("../input/titanic/test.csv")
train.sample(10)
train.info()
print ('Survives percentage -', train.Survived.mean()) 
print(pd.DataFrame(train.groupby('Sex').mean().Survived))

print('------------------------------')

print(pd.DataFrame(train.groupby('Pclass').mean().Survived))
print(train.groupby(['Pclass','Sex']).mean().Survived.unstack()), 

sns.catplot(x='Survived', kind='count', hue='Sex', col = 'Pclass', data=train)
print(train.groupby(['Embarked','Sex']).mean().Survived.unstack()), 

sns.catplot(x='Sex', y='Survived', kind='bar', col='Embarked', data=train)
fig, (ax0,ax1) = plt.subplots (1, 2, figsize=(15, 5))

sns.catplot(x='Survived', y='Age', kind="violin", hue = 'Pclass', data=train, ax = ax0)

plt.close(2)

sns.catplot(x='Survived', y='Fare',  kind="violin", hue = 'Sex', split = True, data=train, ax = ax1)

plt.close(2)
train[train.Fare > 300][['Fare','Survived']]
train.isnull().mean()
print(train.groupby(['Sex','Pclass']).Age.median().unstack()),

sns.boxplot(x="Pclass", y="Age",hue="Sex",data=train)
for data in [train, test]:

    # Delete useless columns

    data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis = 1, inplace = True)

    

    # Fill missing data

    # Use median in groups of Sex and Class for Age

    data.Age = data.groupby(['Sex','Pclass']).Age.transform(lambda x: x.fillna(np.nanmedian(x)))

    # Use mode for Embarked and median for Fare

    data.Embarked.fillna(data.Embarked.mode()[0], inplace = True)

    data.Fare.fillna(np.nanmedian(data.Fare), inplace = True)

    

    # Convert categorical features to numeric

    data['Sex_num'] = data.Sex.map({'female': 0, 'male':1})

    data['Embarked_num'] = data.Embarked.map({'S': 0, 'C':1, 'Q': 2})

    

    # Create a feature Isalone where 1 means a person without family on boart, 0 means a person with family members on Titanic

    data['Family_size'] = data.SibSp + data.Parch + 1

    data['Isalone'] = 0

    data.loc[data.Family_size == 1, ['Isalone']] = 1

    

    # Delete useless columns

    data.drop(['Sex', 'Embarked', 'SibSp', 'Parch'], axis = 1, inplace = True)

    

    # Divide Age and Fare into categorical groups

    data['Age_num'] = pd.cut(data.Age, 5, labels = [0,1,2,3,4]).astype(int)

    data['Fare_num'] = pd.qcut(data.Fare, 5, labels = [0,1,2,3,4]).astype(int)

    data.drop(['Age', 'Fare'], axis = 1, inplace = True)
train.sample(10)
print(train.corr()['Survived'])

sns.heatmap(train.corr())
from sklearn.model_selection import cross_val_score

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

import xgboost as xgb
# Divide train data to independent and target variables

data_y = train.Survived

train.drop (['Survived'], axis = 1, inplace = True)
# Create a model variable

clf = xgb.XGBClassifier ()

# Create a dictionary with model parameters

params = {

    "colsample_bytree": [0.1, 0.2, 0.3, 0.4, 0.5],

    "learning_rate": [0.2, 0.3, 0.4], 

    "max_depth": range (5, 15),

    "n_estimators": [50, 70, 100, 150],

    "subsample": [0.4, 0.5, 0.6, 1],

    'min_child_weight': [1, 5, 10],

    'gamma': [0.5, 1, 1.5, 2, 5]}
# Select the best params from dictionary

search = GridSearchCV(clf, params, n_jobs=5, scoring='accuracy')

search.fit(train, data_y)

search.best_score_
# Get best parameters

search.best_params_
# Apply model with selected parameters

model = xgb.XGBClassifier(colsample_bytree = 0.5, gamma = 5, learning_rate = 0.4, max_depth = 5, n_estimators = 50, subsample = 1, min_child_weight= 1)

model.fit (train, data_y)

result = model.predict(test)
# Save final file

submission = pd.DataFrame({

        'PassengerId': pd.read_csv("../input/titanic/test.csv")['PassengerId'],

        'Survived': result})

submission.to_csv('submission.csv', index = False)