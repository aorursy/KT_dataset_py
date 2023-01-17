import re

import pandas as pd

import seaborn as sns



from sklearn.ensemble import RandomForestClassifier 

from sklearn.preprocessing import LabelEncoder



df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')

df_both = pd.concat((df_train, df_test))
df_both.isnull().sum()
df_both[df_both['Embarked'].isnull()]
sns.barplot(x='Embarked', y='Fare', data=df_both)
df_both['EmbarkedFill'] = df_both['Embarked'].fillna('C')
df_both[df_both['Sex'].isnull()]
sns.distplot(df_both['Age'].dropna())
median_ages = df_both.groupby(['Sex', 'Pclass']).apply(lambda x: x['Age'].median())



for (sex, pclass), something in median_ages.iteritems():

    fill_age = median_ages[sex, pclass]

    mask = (df_train['Sex'] == sex) & (df_train['Pclass'] == pclass) 

    df_both.loc[mask, 'AgeFill'] = df_both.loc[mask, 'Age'].fillna(fill_age)
sns.distplot(df_both['Fare'].dropna())
df_both['FareFill'] = df_both['Fare'].fillna(df_both['Fare'].median())
df_both['SexInt'] = df_both['Sex'].map({'female': 0, 'male': 1})

df_both['EmbarkedFillInt'] = df_both['EmbarkedFill'].map({'C': 0, 'Q': 1, 'S': 2})
df_both['FamilySize'] = df_both['SibSp'] + df_both['Parch']
def get_title(name):

    return re.match('[^,]+, (.*?)\.', name).groups()[0]



titles = df_both['Name'].map(get_title)
titles.value_counts()
title_counts = titles.value_counts()

rare_titles = title_counts[title_counts < 10].index



for title in rare_titles:

    titles.replace(title, 'Rare', inplace=True)



df_both['Title'] = titles

df_both['TitleInt'] = LabelEncoder().fit_transform(df_both['Title'])
df_train = df_both.iloc[:df_train.shape[0]]

df_test = df_both.iloc[df_train.shape[0]:]
features = ['Parch', 'Pclass', 'SibSp', 'AgeFill', 'FareFill', 

            'SexInt', 'EmbarkedFillInt', 'FamilySize', 'TitleInt']



clf = RandomForestClassifier(n_estimators=1000)

clf.fit(df_train[features], df_train['Survived'])

survived_pred = clf.predict(df_test[features])
sns.barplot(x=clf.feature_importances_, y=features) 
df_output = pd.DataFrame({'PassengerId': df_test['PassengerId'],

                          'Survived': survived_pred.astype(int)})

df_output.to_csv('forest.csv', index=False)