from pprint import pprint

# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
train_df = pd.read_csv('../input/train.csv', index_col='PassengerId')
test_df = pd.read_csv('../input/test.csv', index_col='PassengerId')
df = train_df.append(test_df)
df.head()
df.info()
df.describe()
df.corr()
null_series = df.isnull().any()
null_series = null_series[null_series == True]
null_series.index
def extract_mean_survived_groupby_column(df, row_name, column_name):
    return df[[row_name, column_name]].groupby([row_name], as_index=True).mean() \
        .sort_values(by=column_name, ascending=False)
extract_mean_survived_groupby_column(df, 'Pclass', 'Survived')
extract_mean_survived_groupby_column(df, 'Sex', 'Survived')
extract_mean_survived_groupby_column(df, 'SibSp', 'Survived')
extract_mean_survived_groupby_column(df, 'Parch', 'Survived')
g = sns.FacetGrid(df, col='Survived')
g.map(plt.hist, 'Age', bins=20)
g = sns.FacetGrid(df, col='Survived', row='Pclass', size=2.2, aspect=1.6)
g.map(plt.hist, 'Age', alpha=.5, bins=20)
g.add_legend()
grid = sns.FacetGrid(df, row='Embarked', size=2.2, aspect=1.6)
grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
grid.add_legend()
grid = sns.FacetGrid(df, row='Embarked', col='Survived', size=2.2, aspect=1.6)
grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)
grid.add_legend()
grid = sns.FacetGrid(df, row='Pclass', col='Survived', size=2.2, aspect=1.6)
grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)
grid.add_legend()
grid = sns.FacetGrid(df, row='Pclass', col='Survived', size=2.2, aspect=1.6)
grid.map(sns.barplot, 'Embarked', 'Fare', alpha=.5, ci=None)
grid.add_legend()
df = df.drop(['Ticket', 'Cabin'], axis=1)
df['Title'] = df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
pd.crosstab(df['Title'], df['Sex'])
df.Title.value_counts()
title_series = df.Title.value_counts()
title_series = title_series >= 61
title_majority_list = title_series[title_series == True].index.tolist()
title_minority_list = title_series[title_series == False].index.tolist()
title_majority_df = df[df.Title.isin(title_majority_list)]
title_minority_df = df[df.Title.isin(title_minority_list)]
fig, ax = plt.subplots(figsize=(10, 10))
ax = sns.swarmplot(x='Survived', y="Age", hue='Title', data=title_majority_df, size=10, ax=ax, palette='muted')
fig, ax = plt.subplots(figsize=(10, 10))
ax = sns.swarmplot(x='Survived', y='Age', hue='Title', data=title_minority_df, size=10, ax=ax, palette=sns.color_palette("Set1", n_colors=len(title_minority_list), desat=1))
title_dummy = pd.get_dummies(title_majority_df.Title)
tmp_df = df.drop(['Title'], axis=1)
tmp_df = pd.concat([tmp_df, title_dummy], axis=1)
tmp_df.corr()
'''
Please refer here:
Mr. on wiki: https://en.wikipedia.org/wiki/Mr.
Miss on wiki: https://en.wikipedia.org/wiki/Miss
Ms on wiki: https://en.wikipedia.org/wiki/Ms.
'''
title_map = {
    'Capt': 'Mr',
    'Col': 'Mr',
    'Don': 'Mr',
    'Dona': 'Mrs',
    'Dr': 'Mr',
    'Jonkheer': 'Mr',
    'Lady': 'Mrs',
    'Major': 'Mr',
    'Mlle': 'Miss',
    'Mme': 'Mrs',
    'Ms': 'Miss',
    'Rev': 'Mr',
    'Sir': 'Mr',
    'Countess': 'Mrs'}
df.Title = df.Title.replace(title_map)
df.Title.value_counts()
extract_mean_survived_groupby_column(df, 'Title', 'Survived')
df = df.drop(columns=['Name'], axis=1)
title_dummies = pd.get_dummies(df.Title, prefix='title')
df = pd.concat([df, title_dummies], axis=1)
df = df.drop(columns=['Title'], axis=1)
grid = sns.FacetGrid(df, row='Embarked', col='Sex', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend()
grid = sns.FacetGrid(df, row='Pclass', col='Sex', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend()
pclass_dummies = pd.get_dummies(df.Pclass)
embarked_dummies = pd.get_dummies(df.Embarked)
sex_dummies = pd.get_dummies(df.Sex)
tmp_df = pd.concat([df, pclass_dummies, embarked_dummies, sex_dummies], axis=1)
tmp_df.corr()
grid = sns.FacetGrid(df[df.Embarked == 'S'], row='Pclass', col='Sex', 
                     size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend()
# data of Embarked C is too few
grid = sns.FacetGrid(df[df.Embarked == 'C'], row='Pclass', col='Sex', 
                     size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend()
# data of Embarked Q is too few
grid = sns.FacetGrid(df[df.Embarked == 'Q'], row='Pclass', col='Sex', 
                     size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend()
for embark_index in df.Embarked.value_counts().index:
    the_embark_series = df.Embarked == embark_index
    for sex_index in df.Sex.value_counts().index:
        the_sex_series = df.Sex == sex_index
        median = df[the_embark_series & the_sex_series].Age.median()
        df.loc[the_embark_series & the_sex_series & df.Age.isnull(), 'Age'] = median
df.Age = df.Age.astype(int)
age_intervals = (
    (-1, 3),
    (3, 6),
    (6, 9),
    (9, 12),
    (12, 15),
    (15, 18),
    (18, 21),
    (21, 25),
    (25, 30),
    (30, 35),
    (35, 40),
    (40, 45),
    (45, 50),
    (50, 55),
    (55, 60),
    (60, 65),
    (65, 80)
)
def apply_age_intervals(age):
    for index, age_interval in enumerate(age_intervals):
        left, right = age_interval
        if left < age <= right:
            return index
df['age_stage'] = df.Age.apply(apply_age_intervals).astype(int)
df[['age_stage', 'Survived']].groupby(['age_stage'], as_index=False).mean() \
    .sort_values(by='age_stage', ascending=True)
age_stage_dummies = pd.get_dummies(df.age_stage, prefix='age_stage')
df = pd.concat([df, age_stage_dummies], axis=1)
df = df.drop(columns=['Age', 'age_stage'], axis=1)
for embark_index in df.Embarked.value_counts().index:
    the_embark_series = df.Embarked == embark_index
    for pclass_index in df.Pclass.value_counts().index:
        the_pclass_series = df.Pclass == pclass_index
        median = df[the_embark_series & the_pclass_series].Fare.median()
        df.loc[the_embark_series & the_pclass_series & df.Fare.isnull(), 'Fare'] = median
most_embarked = df.Embarked.value_counts().index[0]
df.Embarked.fillna(most_embarked, inplace=True)
embarked_dummies = pd.get_dummies(df.Embarked, prefix='embarked')
df = pd.concat([df, embarked_dummies], axis=1)
df = df.drop(columns=['Embarked'], axis=1)
is_alone_series = (df.Parch == 0) & (df.SibSp == 0)
df['is_alone'] = is_alone_series.astype(int)
sex_dummies = pd.get_dummies(df.Sex, prefix='sex')
df = pd.concat([df, sex_dummies], axis=1)
df = df.drop(columns=['Sex'], axis=1)
from sklearn.model_selection import cross_val_score
train_df = df[df.Survived.notnull()]
test_df = df[df.Survived.isnull()]
train_df.Survived = train_df.Survived.astype(int)
test_df = test_df.drop(columns=['Survived'], axis=1)
scores = []
the_range = list(range(10, 40, 5))
for n_estimators in the_range:
    clf = RandomForestClassifier(n_estimators=n_estimators)
    #score = clf.score(train_df.drop(columns=['Survived'], axis=1), train_df.Survived)
    cv_scores = cross_val_score(clf,
                                train_df.drop(columns=['Survived'], axis=1), 
                                train_df.Survived,
                                cv=20)
    scores.append(cv_scores.mean())
    
plt.plot(the_range, scores)
plt.xlabel('n_estimators')
plt.ylabel('score')
plt.show()

pprint(dict(zip(the_range, scores)))
scores = []
the_range = list(range(15, 31))
for n_estimators in the_range:
    clf = RandomForestClassifier(n_estimators=n_estimators)
    #score = clf.score(train_df.drop(columns=['Survived'], axis=1), train_df.Survived)
    cv_scores = cross_val_score(clf,
                                train_df.drop(columns=['Survived'], axis=1), 
                                train_df.Survived,
                                cv=10)
    scores.append(cv_scores.mean())
    
plt.plot(the_range, scores)
plt.xlabel('n_estimators')
plt.ylabel('score')
plt.show()

pprint(dict(zip(the_range, scores)))
test_df.info()
clf = RandomForestClassifier()
clf = clf.fit(train_df.drop(columns=['Survived'], axis=1), 
              train_df.Survived)
predict_result = clf.predict(test_df)
test_df = pd.read_csv('../input/test.csv')
test_df['Survived'] = predict_result.astype(int)
test_df[['PassengerId', 'Survived']].to_csv('submission.csv', index=False)
