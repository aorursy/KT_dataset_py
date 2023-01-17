import numpy as np    # linear algebra

import pandas as pd   # data processing/feature engineering

import matplotlib.pyplot as plt       # Data visualization

import seaborn as sns                 # Enhanced Data Visualization



%matplotlib inline



from sklearn.linear_model import LogisticRegression # Logistic Regression Model
train_df = pd.read_csv('../input/titanic/train.csv')

test_df = pd.read_csv('../input/titanic/test.csv')
train_df.head(2)
train_df.describe()
train_df.info()
test_df.head(2)
test_df.describe()
test_df.info()
sns.heatmap(train_df.isnull(), yticklabels=False, cbar=False, cmap='viridis')
sns.set_style('whitegrid')

sns.countplot(train_df['Survived'], palette='RdBu_r')
survived = train_df[train_df['Survived']==1]['Survived'].sum()

survived
sns.countplot(train_df['Survived'], hue=train_df['Sex'], palette='rainbow')
sns.countplot(train_df['Survived'], hue=train_df['Pclass'], palette='rainbow')
train_df['Age'].hist(color='darkred', bins=30, alpha=0.6)
train_df['Fare'].hist(color='purple', bins=30, figsize=(8,4))
plt.figure(figsize=(12,7))

sns.boxplot(x=train_df['Pclass'], y=train_df['Age'], palette='winter')
sns.boxplot(x=train_df['Pclass'], y=train_df['Fare'])
train_df.isnull().sum()
meanAge = train_df.groupby('Pclass').mean()['Age']

meanAge
# Defining a function for calculating mean age

def imputeAge(cols):

    Age = cols[0]

    Class = cols[1]

    

    if pd.isnull(Age):

        

        if Class == 1:

            return meanAge[1]

        elif Class == 2:

            return meanAge[2]

        else:

            return meanAge[3]

    else:

        return Age        
# Applying above function in Age column

train_df['Age'] = train_df[['Age', 'Pclass']].apply(imputeAge, axis=1)
train_df.drop('Cabin', axis=1, inplace=True)
train_df['Embarked'] = train_df['Embarked'].fillna(train_df['Embarked'].mode()[0])
train_df.isnull().sum()
test_df.isnull().sum()
meanAge_test = test_df.groupby('Pclass').mean()['Age']

meanAge_test
def imputAge_test(cols):

    Age = cols[0]

    Class = cols[1]

    

    if pd.isnull(Age):

        if Class == 1:

            return meanAge_test[1]

        elif Class == 2:

            return meanAge_test[2]

        else:

            return meanAge_test[3]

    else:

        return Age
test_df['Age'] = test_df[['Age', 'Pclass']].apply(imputAge_test, axis=1)
meanFare_test = test_df.groupby('Pclass').mean()['Fare']

meanFare_test
# Check the number of missing values

test_df['Fare'].isnull().sum()
test_df[test_df['Fare'].isnull() == True]['Pclass']
test_df['Fare'] = test_df['Fare'].fillna(meanFare_test[3])
test_df.drop('Cabin',axis=1, inplace=True)
test_df.isnull().sum()
sex = pd.get_dummies(train_df['Sex'], drop_first=True)

embark = pd.get_dummies(train_df['Embarked'], drop_first=True)
train_df.drop(['Sex', 'Embarked', 'Name', 'Ticket'], axis =1, inplace=True)
train_df = pd.concat([train_df, sex, embark], axis=1)
train_df.head(2)
sex = pd.get_dummies(test_df['Sex'], drop_first=True)

embark = pd.get_dummies(test_df['Embarked'], drop_first=True)
test_df.drop(['Sex', 'Embarked', 'Name', 'Ticket'], axis= 1, inplace=True)
test_df.head(2)
test_df = pd.concat([test_df, sex, embark], axis =1)
test_df.head(2)
X_train = train_df.drop(['Survived','PassengerId'], axis=1)

y_train = train_df['Survived']

X_test = test_df.drop('PassengerId', axis=1)



X_train.shape, y_train.shape, X_test.shape
logmodel = LogisticRegression(max_iter=150)
logmodel.fit(X_train, y_train)
predictions = logmodel.predict(X_test)
submission = pd.DataFrame({

    'PassengerId': test_df['PassengerId'],

    'Survived': predictions

})



submission.to_csv('submission.csv', index = False)

submission.head()