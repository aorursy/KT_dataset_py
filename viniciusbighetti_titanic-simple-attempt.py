import pandas as pd

import numpy as np



train_data = pd.read_csv('../input/titanic/train.csv')

test_data = pd.read_csv('../input/titanic/test.csv')

datas = [train_data, test_data]
train_data.info()
test_data.info()
sex_map = {'male' : 0, 'female' : 1}



train_data.Sex = train_data.Sex.map(sex_map)



test_data.Sex = test_data.Sex.map(sex_map)
# Filling the age missing values based on Sex and Class mean

age_test = train_data.groupby(['Sex', 'Pclass'])['Age']

age_test.mean()
# To improve: Find a better way to automatize the process!



for ds in datas:

    ds.loc[(ds['Sex'] == 0) & (ds['Pclass'] == 1) & ds['Age'].isnull(), 'Age'] = 41.281386

    ds.loc[(ds['Sex'] == 0) & (ds['Pclass'] == 2) & ds['Age'].isnull(), 'Age'] = 30.740707

    ds.loc[(ds['Sex'] == 0) & (ds['Pclass'] == 3) & ds['Age'].isnull(), 'Age'] = 26.507589

    ds.loc[(ds['Sex'] == 1) & (ds['Pclass'] == 1) & ds['Age'].isnull(), 'Age'] = 34.611765

    ds.loc[(ds['Sex'] == 1) & (ds['Pclass'] == 2) & ds['Age'].isnull(), 'Age'] = 28.722973

    ds.loc[(ds['Sex'] == 1) & (ds['Pclass'] == 3) & ds['Age'].isnull(), 'Age'] = 21.750000

    ds['Age'] = ds['Age'].astype(int)
pd.qcut(train_data.Age, 8)
for ds in datas:

    ds.loc[ds['Age'] <= 17, 'Age'] = 0

    ds.loc[(ds['Age'] > 17) & (ds['Age'] <= 21), 'Age'] = 1

    ds.loc[(ds['Age'] > 21) & (ds['Age'] <= 25), 'Age'] = 2

    ds.loc[(ds['Age'] > 25) & (ds['Age'] <= 26), 'Age'] = 3

    ds.loc[(ds['Age'] > 26) & (ds['Age'] <= 30), 'Age'] = 4

    ds.loc[(ds['Age'] > 30) & (ds['Age'] <= 36), 'Age'] = 5

    ds.loc[(ds['Age'] > 36) & (ds['Age'] <= 45), 'Age'] = 6

    ds.loc[ds['Age'] > 45, 'Age'] = 7
train_data['Age'].value_counts()
test_data['Age'].value_counts()
train_data[train_data['Embarked'].isnull()]
embarked_treatment = train_data.groupby('Embarked')

embarked_treatment.Survived.value_counts()
train_data.fillna({'Embarked' : 'S'}, inplace = True)



embarked_map = {'S' : 0, 'C' : 1, 'Q' : 2}



for ds in datas:

    ds.Embarked = ds.Embarked.map(embarked_map)
test_data[test_data.Fare.isnull()]
test_data.fillna({'Fare' : test_data.Fare.mean()}, inplace = True)
pd.qcut(train_data.Fare, 5)
for ds in datas:

    ds['Fare'] = ds['Fare'].astype(int)

    ds.loc[ds['Fare'] <= 7.854, 'Fare'] = 0

    ds.loc[(ds['Fare'] > 7.854) & (ds['Fare'] <= 10.5), 'Fare'] = 1

    ds.loc[(ds['Fare'] > 10.5) & (ds['Fare'] <= 21.679), 'Fare'] = 2

    ds.loc[(ds['Fare'] > 21.679) & (ds['Fare'] <= 39.688), 'Fare'] = 3

    ds.loc[ds['Fare'] > 39.688, 'Fare'] = 4
for ds in datas:

    ds['Title'] = ds.Name.str.extract(' ([A-Za-z]+)\.', expand = False)

    

train_data['Title'].value_counts()
titles = {'Mr' : 1, 'Miss' : 2, 'Mrs' : 3, 'Master' : 4, 'Other' : 5}



for ds in datas:

    ds['Title'].replace(['Dr', 'Rev', 'Major', 'Col', 'Lady', 'Capt', 'Jonkheer', 'Countess', 'Don', 'Sir', 'Dona'], 'Other', inplace = True)

    ds['Title'].replace('Ms', 'Miss', inplace = True)

    ds['Title'].replace('Mlle', 'Miss', inplace = True)

    ds['Title'].replace('Mme', 'Mrs', inplace = True)

    ds['Title'] = ds['Title'].map(titles)

    ds['Title'].fillna(0, inplace = True)
for ds in datas:

    ds['Floor'] = ds.Cabin.str.extract('([A-Za-z]+)', expand = False)

    ds['Floor'].fillna('T', inplace = True)
train_data
train_data['Cabin'].unique()
train_data['Floor'].value_counts()
test_data['Floor'].value_counts()
floors = {'ABC' : 1, 'DE' : 2, 'FG' : 3, 'T' : 4}



for ds in datas:

    ds['Floor'].replace(['A', 'B', 'C'], 'ABC', inplace = True)

    ds['Floor'].replace(['D', 'E'], 'DE', inplace = True)

    ds['Floor'].replace(['F', 'G'], 'FG', inplace = True)

    ds['Floor'] = ds['Floor'].map(floors)
for ds in datas:

    ds['Age_Pclass'] = ds['Age'] * ds['Pclass']

    ds['Relatives'] = ds['SibSp'] + ds['Parch']
for ds in datas:

    ds.drop('Cabin', axis = 1, inplace = True)

    ds.drop('Name', axis = 1, inplace = True)

    ds.drop('Ticket', axis = 1, inplace = True)
train_data
test_data
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score



model = RandomForestClassifier(criterion = "gini", 

                     min_samples_leaf = 1, 

                     min_samples_split = 10,   

                     n_estimators=100, 

                     max_features='auto', 

                     oob_score=True, 

                     random_state=1, 

                     n_jobs=-1)



train_x = train_data.drop(['PassengerId', 'Survived'], axis = 1)

train_y = train_data['Survived']

test_x = test_data.drop('PassengerId', axis = 1)



model.fit(train_x, train_y)



prediction = model.predict(test_x)



model_accuracy = model.score(train_x, train_y)

print(model_accuracy)
parametros = pd.DataFrame({'feature':train_x.columns,'Parametros':np.round(model.feature_importances_,3)})

parametros = parametros.sort_values('Parametros',ascending=False).set_index('feature')
parametros.plot.bar()
submission = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived' : prediction})



submission.to_csv('Submission.csv', index = False)