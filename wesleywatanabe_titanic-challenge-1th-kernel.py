import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
# Import 

df_train = pd.read_csv("../input/train.csv", sep=",")

df_test = pd.read_csv("../input/test.csv", sep=",")



# Check import

df_train.head()
# % survived split by sex

df_train[['Sex','Survived']].groupby(['Sex'],as_index=False).mean().sort_values(by='Survived',ascending=False).round(2)
# Shape of train dataset

df_train.shape
# Types of features in train dataset

df_train.dtypes
# Drop columns of datasets

df_train = df_train.drop(["PassengerId", "Name"], axis=1)

df_test = df_test.drop(["Name"], axis=1)
# Check NaN in column 'Cabin'

df_train['Cabin'].isna().sum()
# Check Cabin columns values unique 

df_train['Cabin'].unique()
# Drop more columns of datasets

df_train = df_train.drop(["Cabin"], axis=1)

df_test = df_test.drop(["Cabin"], axis=1)
# Check Ticket columns values unique 

df_train['Ticket'].unique()
# (TRAIN) - 'Create' category variables

df_train['Ticket'] = df_train['Ticket'].astype("category").cat.codes

df_train['Sex']= df_train['Sex'].astype("category").cat.codes

df_train['Embarked'] = df_train['Embarked'].astype("category").cat.codes

df_train['Age'] = round(df_train['Age'])



# (TRAIN) - Create a feature to see if passeger is alone in ship

is_alone_train=[]

for x in df_train['SibSp']:

  if(x==0):is_alone_train.append(0)

  else: is_alone_train.append(1)

df_train['is_alone'] = is_alone_train



# (TRAIN) - Create a feature to agroup people by age

age_group_train=[]

for x in df_train['Age']:

  if x <=16: age_group_train.append(1)

  elif x >=17 and x <= 32:age_group_train.append(2)

  elif x >= 33 and x <= 42:age_group_train.append(3)

  else:age_group_train.append(4)   

df_train['age_group'] = age_group_train



# ---------------------------------------------------------------------



# (TEST) - 'Create' category variables

df_test['Ticket'] = df_test['Ticket'].astype("category").cat.codes

df_test['Sex']= df_test['Sex'].astype("category").cat.codes

df_test['Embarked'] = df_test['Embarked'].astype("category").cat.codes

df_test['Age'] = round(df_test['Age'])



# (TEST) - Create a feature to see if passeger is alone in ship

is_alone_test=[]

for x in df_test['SibSp']:

  if(x==0): is_alone_test.append(0)

  else: is_alone_test.append(1)

df_test['is_alone'] = is_alone_test



# (TEST) - Create a feature to agroup people by age

age_group_test=[]

for x in df_test['Age']:

  if x <=16: age_group_test.append(1)

  elif x >=17 and x <= 32:  age_group_test.append(2)

  elif x >= 33 and x <= 42: age_group_test.append(3)

  else: age_group_test.append(4)

df_test['age_group'] = age_group_test
# Fill missing values with 'mode' and 0



df_train['Age'].fillna(0,inplace=True)

df_train['Embarked'].fillna(df_train['Embarked'].mode()[0],inplace=True)

df_train['Fare'].fillna(0, inplace=True)



df_test['Age'].fillna(0,inplace=True)

df_test['Embarked'].fillna(df_test['Embarked'].mode()[0],inplace=True)

df_test['Fare'].fillna(0, inplace=True)
# Plotting 'Age Group'

ax = sns.kdeplot(df_train['age_group'],shade=True)

ax2 = sns.kdeplot(df_test['age_group'],shade=True)

sns.set_context("notebook")

plt.legend(["Group train","Group test"])

plt.title("Age group by dataset", size=15, x=0.5,y=1.1)
# (TRAIN) - Correlation table 

df_train.corr().round(2)
# Using seaborn to plot features correlation 



# Appply style

sns.set(style="white")



# Compute the correlation matrix

corr = df_train.corr().round(2)



# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(13, 8))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})
# (TRAIN) - Check dataset after changes

df_train.head()
# Set 'y' as a target

y = df_train['Survived']



# Create a dataframe

export = pd.DataFrame()

export['PassengerId'] = df_test['PassengerId']



df_test = df_test.drop(['PassengerId'], axis=1)



# Removing target from data to train

df_train = df_train.drop('Survived', axis=1 )



# Set X

X = df_train
# Libriries to create model

from sklearn.ensemble import RandomForestClassifier

from sklearn import metrics

from sklearn.model_selection import train_test_split



# Split train dataset 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.18, random_state=0)



# Create a model and parameters

clf = RandomForestClassifier(n_estimators=50, 

                               max_depth=6,

                               n_jobs = -1,

                               random_state=0)

# Training model

clf.fit(X_train, y_train) 





print("Teste: {}%".format(clf.score(X_test, y_test).round(2)))

print("Train: {}%".format(clf.score(X_train, y_train).round(2)))

print('-----------------------------')

print('The mae of prediction is:', metrics.mean_absolute_error(y_train, clf.predict(X_train)).round(2) )

print('Mean Squared Error:', metrics.mean_squared_error(y_train, clf.predict(X_train)).round(2))  

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_train, clf.predict(X_train))).round(2))

# Create X_test to predict (test dataset)

X_test = df_test



# Predic y from X_test

y_pred_test = clf.predict(X_test)



# Fill dataframe with values predict

export['Survived'] = y_pred_test 



# Export file

export_csv = export.to_csv (r'titanic.csv', index = None, header=True)
# Libriries to create model

from sklearn.model_selection import cross_validate

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import KFold



# Scaling variables

sc = StandardScaler()

X_scale = sc.fit_transform(X)



# Create k-fold

cv = KFold(n_splits = 6, shuffle = True)



result = cross_validate(clf,X_scale,y,cv=cv, return_train_score=False)

print("Cross validate median {}%".format(np.median(result['test_score']).round(2)*100))

print("Cross validate average {}%".format(np.average(result['test_score']).round(2)*100))