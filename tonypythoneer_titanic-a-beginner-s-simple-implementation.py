# Import basic packages

from pprint import pprint

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.tools.plotting import scatter_matrix

import seaborn as sns

pd.set_option('display.max_rows', 10)
# Files

train_file = '../input/train.csv'
test_file = '../input/test.csv'
train_and_test_files = (train_file, test_file)
train_df, test_df = [pd.read_csv(f, index_col='PassengerId') for f in (train_file, test_file)]
print('train_df:')
train_df.info()
print('---')
print('test_df:')
test_df.info()
Correlation=train_df.corr()
print(pd.DataFrame(Correlation))

correlation_Y = pd.DataFrame(Correlation["Survived"])
correlation_Y.sort_values(by = "Survived", ascending = False)
print(correlation_Y)
test_df.hist()
unnecessary_columns = ['Ticket', 'Cabin']
train_df = train_df.drop(columns=unnecessary_columns)
test_df = test_df.drop(columns=unnecessary_columns)
merged_df = train_df.append(test_df)
# Sould check what columns are including NaN
columns_series = merged_df.isnull().any()
columns_with_nan_series = columns_series[columns_series == True]
columns_with_nan = columns_with_nan_series.index.values.tolist()
columns_with_nan

for c in columns_with_nan:
    subset_df = merged_df[merged_df[c].isnull()]
    row_count = len(subset_df.index)
    print('{} column has {} of row count of NaN data '.format(c, row_count))
    subset_df.head()
p_class_types = merged_df['Pclass'].unique()
p_class_types.sort()
for p_class_type in p_class_types:
    fare_series = merged_df[merged_df['Pclass'] == p_class_type]['Fare']
    median = fare_series.median()
    print('Median fare of {} Pclass is: {}'.format(p_class_type, median))
for p_class_type in p_class_types:
    fare_series = merged_df[merged_df['Pclass'] == p_class_type]['Fare']
    has_any_null = fare_series.isnull().any()
    if not has_any_null:
        continue

    s = fare_series.isnull()
    filled_series = s[s == True]
    merged_df.loc[filled_series.index, ['Fare']] = fare_series.median()
embarked_count_series = merged_df['Embarked'].value_counts()

median_age = merged_df['Age'].median()
merged_df['Age'] = merged_df['Age'].fillna(median_age)
idxmax = merged_df['Embarked'].value_counts().idxmax()
print("Embarked 最常出現:", idxmax)
merged_df['Embarked'] = merged_df['Embarked'].fillna(idxmax)
last_name_series = merged_df['Name'].str.split(", ", expand=True)[1]
last_name_series.head(5)
title_series = last_name_series.str.split('.', expand=True)[0]
merged_df['title'] = title_series
merged_df = merged_df.drop(columns=['Name'])
merged_df.head(5)
pd.crosstab(merged_df['title'], merged_df['Sex']).T.style.background_gradient(cmap='summer_r')
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
    'the Countess': 'Mrs',
    'Master': 'Mr'}
merged_df['title'] = merged_df['title'].replace(title_map)
dummy = pd.get_dummies(merged_df['title'])
merged_df = pd.concat([merged_df, dummy], axis=1)
merged_df = merged_df.drop(columns=['title'])
merged_df.head()
for column in ['Sex', 'Embarked']:
    dummy = pd.get_dummies(merged_df[column])
    merged_df = pd.concat([merged_df, dummy.astype(bool)], axis=1)
    merged_df = merged_df.drop([column], axis=1)
    
dummy = pd.get_dummies(merged_df['Pclass'], prefix='pclass')
merged_df = pd.concat([merged_df, dummy.astype(bool)], axis=1)
merged_df = merged_df.drop(['Pclass'], axis=1)
merged_df.head()
merged_df['family'] = merged_df['Parch'] + merged_df['SibSp']
merged_df = merged_df.drop(columns=['Parch', 'SibSp'])
merged_df['family'] = merged_df['family'].astype(int)
merged_df.head()
new_train_df = merged_df[pd.notnull(merged_df['Survived'])]
plt.figure(figsize=(14, 14))
sns.heatmap(merged_df.astype(float).corr(), cmap = 'BrBG',
            linewidths=0.1, square=True, linecolor='white',
            annot=True)
new_train_df = merged_df[merged_df['Survived'].notnull()]
new_test_df = merged_df[merged_df['Survived'].isnull()]
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
clf = DecisionTreeClassifier(max_depth=3)
scores = cross_val_score(clf,
                        new_train_df.drop(['Survived'], axis=1), 
                        new_train_df['Survived'],
                        cv=10)
scores
scores.mean()
clf.fit(new_train_df.drop(['Survived'], axis=1), new_train_df['Survived'])
from sklearn.tree import export_graphviz
import graphviz
g = export_graphviz(clf,out_file=None,
                    feature_names=new_train_df.drop(['Survived'], axis=1).columns,
                    class_names=["No", "Yes"],
                    filled=True, 
                    rounded=True,
                    special_characters=True)
graphviz.Source(g)
pprint(dict(zip(new_train_df.drop(['Survived'], axis=1).columns.tolist(), clf.feature_importances_)))
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
x_train = new_train_df.drop(['Survived'], axis=1)
x_test = new_train_df['Survived']
scores = []
for n_estimators in range(10, 110, 5):
    clf = RandomForestClassifier(n_estimators=n_estimators)
    clf = clf.fit(x_train, x_test)
    score = clf.score(x_train, x_test)
    scores.append(score)
plt.plot(range(10, 110, 5), scores)
plt.xlabel('n_estimators')
plt.ylabel('score')
plt.show()
pprint(dict(zip(range(10, 110, 5), scores)))
scores = []
for n_estimators in range(25, 36):
    clf = RandomForestClassifier(n_estimators=n_estimators)
    clf = clf.fit(x_train, x_test)
    score = clf.score(x_train, x_test)
    scores.append(score)
plt.plot(range(25, 36), scores)
plt.xlabel('n_estimators')
plt.ylabel('score')
plt.show()
pprint(dict(zip(range(25, 36), scores)))
scores = []
for max_depth in range(3, 21):
    clf = RandomForestClassifier(n_estimators=26, max_depth=max_depth)
    clf = clf.fit(x_train, x_test)
    score = clf.score(x_train, x_test)
    scores.append(score)
plt.plot(range(3, 21), scores)
plt.xlabel('max_depth')
plt.ylabel('score')
plt.show()
pprint(dict(zip(range(3, 21), scores)))
clf = RandomForestClassifier(n_estimators=26, max_depth=15)
clf = clf.fit(x_train, x_test)
predict_result = clf.predict(new_test_df.drop(['Survived'], axis=1))
new_test_df['Survived'] = predict_result
survived = new_test_df.loc[:,['Survived']]
survived['Survived'] = survived['Survived'].astype(int) 
#survived = new_test_df['Survived']
survived.to_csv('submission.csv')
