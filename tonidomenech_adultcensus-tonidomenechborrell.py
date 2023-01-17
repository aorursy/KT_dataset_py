import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
# First of all we charge our DataSet into df:

df = pd.read_csv('/kaggle/input/adult-census-income/adult.csv')
df.head()
# What are the type of our columns and what do we have.

df.info()
# How many null values do we have.

df.isnull().sum()

# We can see the null information comes with a ?. Therefore the .isnull() method doesn't detect correctly the nulls.

# We will work on that later. On the Feature Engineering Section.
df['age'].value_counts()
sns.set_style('darkgrid')

sns.distplot(df['age'], bins=70)
plt.figure(figsize = (14, 6))

sns.violinplot(y='age', x='education.num', hue='sex', palette = 'RdBu', data=df, split=True)

# We will discus it later but the education num it references in a scale of 1-16 how much had each person studied.
sns.countplot(x='sex', hue='income', data=df)
df['workclass'].value_counts()
plt.figure(figsize=(12,6))

sns.countplot(x='workclass', hue='income', data=df)
df[df['workclass']=='Never-worked']
df[df['workclass']=='Without-pay']
df['fnlwgt'].value_counts()
print(df['fnlwgt'].min())

print('\n')

print(df['fnlwgt'].max())
plt.figure(figsize=(12,6))

sns.countplot(x = 'education.num', hue='race', data=df)
df['marital.status'].value_counts()
plt.figure(figsize=(12,6))

sns.countplot(x='marital.status', hue = 'income', data = df)
plt.figure(figsize=(6,6))

df['marital.status'].value_counts().plot.pie()
plt.figure(figsize=(10,6))

sns.scatterplot(x='marital.status', y='relationship', data = df)
df['occupation'].value_counts()
plt.figure(figsize=(16,12))





plt.subplot(2,2,1)

sns.violinplot(x = df['race'], y = df['capital.gain'], hue='sex', split=True, data = df ,palette = 'RdBu')

plt.subplot(2,2,2)

sns.violinplot(x = df['race'], y = df['capital.loss'],  hue='income', split=True,data = df)

plt.subplot(2,2,3)

sns.violinplot(x = df['race'], y = df['hours.per.week'], hue='income', split=True, data = df)

plt.subplot(2,2,4)

sns.violinplot(x = df['race'], y = df['education.num'], hue='income', split=True,data = df)
df['native.country'].value_counts()
df = df[df['native.country']!='?']
native_more = df.loc[df['income'] == '>50K',['native.country']].groupby('native.country').size()

native_less = df.loc[df['income'] == '<=50K',['native.country']].groupby('native.country').size()



index_more = list(native_more.index)

index_less = list(native_less.index)



# Checking if the Countries in both aspects are same or not

print(index_more)

print(len(index_more))

print(index_less)

print(len(index_less))
# Checking which Countries are not in the list of more than 50k.

[country for country in index_less if country not in index_more]
# Making DataFrames of the Data

df_more = pd.DataFrame({'Countries' : index_more, '>50K' : list(native_more) })

df_less = pd.DataFrame({'Countries' : index_less, '<=50K' : list(native_less) })



# Adding the entries of the missing countries

df_more.loc[40] = 'Holand-Netherlands', 0

df_more.loc[41] = 'Outlying-US(Guam-USVI-etc)', 0



df_bycountry = pd.merge(df_more, df_less, on='Countries')
# Removing USA in order to have a more clear and scalatted plot.

df_bycountry = df_bycountry[df_bycountry['Countries'] != 'United-States']
fig, axes = plt.subplots(ncols=2, sharey=True, figsize=(20,20))

axes[0].barh(df_bycountry['Countries'], df_bycountry['>50K'], align='center', color='green', zorder=10)

axes[0].set(title='>50K')

axes[1].barh(df_bycountry['Countries'], df_bycountry['<=50K'], color='red', zorder=10)

axes[1].set(title='<=50K')



axes[0].invert_xaxis()

axes[0].set(yticklabels=df_bycountry['Countries'], )

axes[0].yaxis.tick_right()



for ax in axes.flat:

    ax.margins(0.03)

    ax.grid(True)



fig.tight_layout()

fig.subplots_adjust(wspace=0.09)

plt.show()
sns.countplot(x='income', data=df)
df['income'] = df['income'].map({'<=50K':0, '>50K':1})
df.head()
df_copy = df.copy()

df_copy
df = df.drop('education', axis = 1)
# Deleting all the ? 



categorical_features = list(df.select_dtypes(include=['object']).columns)





for feature in categorical_features:

    df = df[df[feature] != '?']

df.head()
categorical_features = list(df.select_dtypes(include=['object']).columns)

categorical_features



for feature in categorical_features:

    s = pd.get_dummies(df[feature])

    df = df.join(s)

    df = df.drop(feature, axis = 1)

df.head()
df.drop('Female', axis = 1, inplace=True)
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
X = df.drop('income', axis = 1)

y = df['income']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
rfc.fit(X_train,y_train)
pred_rfc = rfc.predict(X_test)
from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(y_test, pred_rfc))

print('\n')

print(classification_report(y_test, pred_rfc))
df[df['income']==1].count()
df[df['income']==0].count()