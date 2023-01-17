import numpy as np

import pandas as pd

from sklearn.ensemble import RandomForestClassifier

# disable Pandas notification 

pd.options.mode.chained_assignment = None

#to write model in file

from sklearn.externals import joblib

# for plotting

%matplotlib inline

from matplotlib import pyplot as plt



#function for check missing data in Dataframes

def missing_data(data):

    total = data.isnull().sum()

    percent = (data.isnull().sum()/data.isnull().count()*100)

    alldata = pd.concat([total, percent], axis=1, keys=['Total', 'in Percent'])

    types = []

    for col in data.columns:

        dtype = str(data[col].dtype)

        types.append(dtype)

    alldata['Types'] = types

    return(np.transpose(alldata)) 
train_df = pd.read_csv("../input/train.csv")

test_df = pd.read_csv('../input/test.csv')
train_df.head()
test_df.head()
missing_data(train_df)
missing_data(test_df)
train_df['Embarked'].value_counts()
# Getting median for Age

median_age = train_df.Age.median()

# Getting mean value for Fare

mean_fare = train_df.Fare.mean()

# Feature Engineering

for line in [train_df, test_df]:    

    # filling 'Age' Nan values and converting to integer type

    line['Age'].fillna(median_age, inplace = True)

    line['Age'] = line['Age'].astype(int)    

    # Filling 'Fare' feature with mean value

    line['Fare'].fillna(mean_fare, inplace = True)    

    # Filling 'Embarked' nan with S

    line['Embarked'].fillna('S', inplace = True)    

    # Creating new feature "Family Size"

    line['FamSize'] = line.SibSp + line.Parch
# Creating dictionarys for mapping values

map_embarked = {'S': 1, 'C': 2, 'Q': 3}

map_sex = {'male': 1, 'female': 2}



for dataset in [train_df, test_df]:

    dataset['Embarked'] = dataset.Embarked.map(map_embarked)

    dataset['Sex'] = dataset.Sex.map(map_sex)
train_df = train_df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
test_df = test_df.drop(['Name', 'Ticket', 'Cabin'], axis=1)
train_df.head()
test_df.head()
train_df.info()
test_df.info()
X_train = train_df.drop('Survived', axis=1)

Y_train = train_df["Survived"]

X_test  = test_df.drop('PassengerId', axis=1).copy()
# Random Forest with oob



random_forest = RandomForestClassifier(n_estimators=100, oob_score = True)

random_forest.fit(X_train, Y_train)



Y_prediction = random_forest.predict(X_test)



random_forest.score(X_train, Y_train)

acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
print('Accuracy:', acc_random_forest, '%')
print("oob score:", round(random_forest.oob_score_, 4)*100, "%")
feature_imp = pd.DataFrame({'Feature':X_train.columns,'Importance':np.round(random_forest.feature_importances_,3)})

feature_imp = feature_imp.sort_values('Importance',ascending=False).set_index('Feature')

feature_imp.head(15)
submission = pd.DataFrame({'PassengerId':test_df['PassengerId'],'Survived':Y_prediction})
submission.head()
# submission.to_csv('submission.csv', index=False)
mean_fare
im = random_forest.predict([[2, 1, 32, 0, 0, 32.2000, 1, 0]])
print ('Survived =', im[0])