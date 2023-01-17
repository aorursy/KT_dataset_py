import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

import os
print(os.listdir("../input"))
train_df = pd.read_csv('../input/train.csv')
train_df.head()
train_df.shape
train_df.isna().any()
print('Number of null ages: {}'.format(len([age for age in train_df.Age.isna() if age])))
print('Number of null cabins: {}'.format(len([cabin for cabin in train_df.Cabin.isna() if cabin])))
print('Number of null embarked: {}'.format(len([embarked for embarked in train_df.Embarked.isna() if embarked])))
train_df['cabin_blocks'] = [cabin[0] if cabin is not np.NaN else cabin for cabin in train_df.Cabin]
train_df.head()
cabin_blocks = [cabin for cabin in train_df.cabin_blocks.unique() if cabin is not np.NaN]
print(cabin_blocks)

def find_average_fare_for_cabin_block(cabin_block: str):
    sum_of_fares = sum([row['Fare'] for _, row in train_df.iterrows() if row['cabin_blocks'] == cabin_block])
    number_of_cabins = len([cabin for cabin in train_df.cabin_blocks if cabin == cabin_block])
    return sum_of_fares / number_of_cabins

cabin_averages = [find_average_fare_for_cabin_block(cabin) for cabin in cabin_blocks]

plt.bar(cabin_blocks, cabin_averages)
plt.title('Average Fare Cost Per Cabin Block')
plt.xlabel('Cabin Block')
plt.ylabel('Fare Cost')
label_encoder = preprocessing.LabelEncoder()
label_encoder.fit(list(train_df['cabin_blocks']))
encoded_cabin_blocks = label_encoder.transform(list(train_df['cabin_blocks']))
print('Pearson Correlation Coefficient of Fare and Cabin Block: {}'.format(round(train_df['Fare'].corr(pd.Series(encoded_cabin_blocks)), 3)))
print('Pearson Correlation Coefficient of Cabin Block and Survival: {}'.format(round(train_df['Survived'].corr(pd.Series(encoded_cabin_blocks)), 3)))
train_df['family_size'] = [row['Parch'] + row['SibSp'] for _, row in train_df.iterrows()]
train_df.head()
print('Pearson Correlation of Family Size and Survived: {}'.format(round(train_df['family_size'].corr(train_df['Survived']), 3)))
train_df['is_alone'] = [1 if size == 0 else 0 for size in train_df['family_size']]
train_df.head()
print('Pearson Correlation of Alone and Survived: {}'.format(round(train_df['Survived'].corr(train_df['is_alone']), 3)))
train_df['cabin_blocks'] = encoded_cabin_blocks
label_encoder.fit(list(train_df['Sex']))
train_df['Sex'] = label_encoder.transform(list(train_df['Sex']))
label_encoder.fit(list(train_df['Embarked']))
train_df['Embarked'] = label_encoder.transform(list(train_df['Embarked']))
train_df.head()
sns.heatmap(train_df.corr(), 
            xticklabels=train_df.corr().columns.values,
            yticklabels=train_df.corr().columns.values)
train_df.drop(columns=['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket', 'Cabin'])
y = train_df['Survived']
X = train_df.drop(columns=['Survived', 'PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp', 'Parch'])
imput = preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0)
imput = imput.fit(X)
X = imput.transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y)
clf = RandomForestClassifier()
clf.fit(X_train, y_train)
clf.score(X_test, y_test)