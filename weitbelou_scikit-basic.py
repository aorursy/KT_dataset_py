%matplotlib inline
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_style('whitegrid')

train = pd.read_csv('../input/train.csv')

train.sample(10)
train.isna().sum()
train.Embarked.unique()
def normalize_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert all features to numeric
    """
    new_df = df.copy(deep=True)

    new_df['Child'] = new_df.Age.fillna(new_df.Age.median()).map(lambda x: int(x <= 16))
    
    new_df.Fare.fillna(new_df.Age.mean(), inplace=True)

    new_df.Sex = new_df.Sex.map({
        'male': 0,
        'female': 1,
    })

    new_df.drop([
        'Ticket',
        'PassengerId',
        'Cabin',
        'Name',
        'Age',
    ], axis=1, inplace=True)

    new_df.Embarked.fillna('S', inplace=True)
    new_df.Embarked = new_df.Embarked.map({
        'S': 0,
        'C': 1,
        'Q': 2,
    })

    return new_df
normalized_train = normalize_data(train)
normalized_train.sample(10)
sns.catplot(x='Sex', y='Survived', kind='bar', data=normalized_train)
sns.catplot(x='Child', y='Survived', kind='bar', data=normalized_train)
sns.catplot(x='Pclass', y='Survived', kind='point', hue='Sex', data=normalized_train)
sns.catplot(x='SibSp', y='Survived', kind='bar', hue='Sex', data=normalized_train)
sns.catplot(x='Parch', y='Survived', kind='bar', hue='Sex', data=normalized_train)
from sklearn.model_selection import train_test_split

X = normalized_train.drop('Survived', axis=1)
y = normalized_train.Survived

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,random_state=98)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

knn = KNeighborsClassifier( )

grid = GridSearchCV(
    knn,
    dict(n_neighbors=list(range(1,10)), weights=['uniform', 'distance']),
    cv=10, scoring = 'accuracy',
)
grid.fit(X_train, y_train)
print(f'Best score: {grid.best_score_}')
print (f'Best Parameters: {grid.best_params_}')
print (f'Best Estimator: {grid.best_estimator_}')
from sklearn.metrics import accuracy_score

print(f'Accuracy: {accuracy_score(y_test, grid.predict(X_test))}')
test = pd.read_csv('../input/test.csv')
passenger_id = test.PassengerId
normalized_test = normalize_data(test)

normalized_test.sample(10)
survived = grid.predict(normalized_test)
pd.DataFrame(dict(
    PassengerId=passenger_id,
    Survived=survived,
)).to_csv('./result.csv', index=False)
