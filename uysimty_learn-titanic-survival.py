import pandas as pd

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



from sklearn.model_selection import RandomizedSearchCV

from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split



import warnings

warnings.filterwarnings('ignore')
raw_data = pd.read_csv("../input/train.csv")

raw_test = pd.read_csv('../input/test.csv')
def survival_estimator(x):

    return len(x[x==1])/len(x)*100.0
print(raw_data.columns)
raw_data.info()
raw_data.head(10)
print(raw_data.isnull().sum())

print("-"*10)

print(raw_data.isnull().sum()/raw_data.shape[0])
cor_matrix = raw_data.drop(columns=['PassengerId']).corr().round(2)

# Plotting heatmap 

fig = plt.figure(figsize=(12,12));

sns.heatmap(cor_matrix, annot=True, cmap='autumn');
sns.countplot(x='Survived', data=raw_data)
sns.barplot(x="Sex",y="Survived", data=raw_data, estimator=survival_estimator)
sns.barplot(x='Pclass', y='Survived', data=raw_data, estimator=survival_estimator)
plt.figure(figsize=(18, 30))

sns.countplot(y='Age', data=raw_data)
raw_data['Age'] = raw_data['Age'].dropna().astype(int)

sns.FacetGrid(raw_data, hue='Survived', aspect=4).map(sns.kdeplot, 'Age', shade= True).set(xlim=(0 , raw_data['Age'].max())).add_legend()
raw_data['AgeGroup'] = pd.cut(raw_data.Age, bins=np.arange(start=0, stop=90, step=10), include_lowest=True)

plt.figure(figsize=(18, 5))

sns.barplot(x='AgeGroup', y='Survived', data=raw_data, estimator=survival_estimator)
raw_data['IsChildren'] = np.where(raw_data['Age']<10, 1, 0)

sns.barplot(x='IsChildren', y='Survived', data=raw_data, estimator=survival_estimator)
raw_data['IsAgeNull'] = np.where(np.isnan(raw_data['Age']), 1, 0)

raw_data['Age'].fillna((raw_data['Age'].mean()), inplace=True)

raw_data['Age'] = raw_data['Age'].round().astype(int)
raw_data['AgeLabel'] = pd.cut(raw_data['Age'], bins=np.arange(start=0, stop=90, step=10), labels=np.arange(start=0, stop=8, step=1), include_lowest=True)
plt.figure(figsize=(18, 8))

sns.barplot(x='SibSp', y='Survived', data=raw_data, estimator=survival_estimator)
plt.figure(figsize=(18, 8))

sns.barplot(x='Parch', y='Survived', data=raw_data, estimator=survival_estimator)
plt.figure(figsize=(18, 8))

raw_data['FamilySize'] = raw_data.apply (lambda row: row['SibSp']+row['Parch'], axis=1)

sns.barplot(x='FamilySize', y='Survived', data=raw_data, estimator=survival_estimator)
raw_data['NoFamily'] = np.where(raw_data['FamilySize']==0, 1, 0)

raw_data['SmallFamily'] = np.where((raw_data['FamilySize']>0)&(raw_data['FamilySize']<4), 1, 0)

raw_data['MediumFamily'] = np.where((raw_data['FamilySize']>3)&(raw_data['FamilySize']<7), 1, 0)

raw_data['LargeFamily'] = np.where(raw_data['FamilySize']>=7, 1, 0)



fig, axes = plt.subplots(1, 4, figsize=(18, 8))

sns.barplot(x='NoFamily', y='Survived', data=raw_data, estimator=survival_estimator, ax=axes[0])

sns.barplot(x='SmallFamily', y='Survived', data=raw_data, estimator=survival_estimator, ax=axes[1])

sns.barplot(x='MediumFamily', y='Survived', data=raw_data, estimator=survival_estimator, ax=axes[2])

sns.barplot(x='LargeFamily', y='Survived', data=raw_data, estimator=survival_estimator, ax=axes[3])
plt.subplots(1,1,figsize=(18, 8))

sns.distplot(raw_data['Fare'])
raw_data['FareRange'] = pd.cut(raw_data.Fare, bins=np.arange(start=0, stop=600, step=50), precision=0, include_lowest=True)

plt.figure(figsize=(18, 8))

sns.countplot('FareRange', data=raw_data)
plt.figure(figsize=(18, 8))

sns.barplot(x='FareRange', y='Survived', data=raw_data, estimator=survival_estimator)
raw_data['FareLabel'] = pd.cut(raw_data.Fare, bins=np.arange(start=0, stop=600, step=50), precision=0, labels=np.arange(start=0, stop=11, step=1), include_lowest=True)
raw_data['LowFare'] = np.where(raw_data['Fare']<=50, 1, 0)

sns.barplot(x='LowFare', y='Survived', data=raw_data, estimator=survival_estimator)
raw_data['HighFare'] = np.where(raw_data['Fare']>300, 1, 0)

sns.barplot(x='HighFare', y='Survived', data=raw_data, estimator=survival_estimator)
def medium_fare(row):

    if(row['LowFare']==0 & row['HighFare']==0):

        return 1

    else:

        return 0

raw_data['MediumFare'] = raw_data.apply(medium_fare, axis=1)

sns.barplot(x='MediumFare', y='Survived', data=raw_data, estimator=survival_estimator)
sns.barplot(x='Embarked', y='Survived', data=raw_data, estimator=survival_estimator)
plt.figure(figsize=(18, 8))

sns.FacetGrid(raw_data,size=5, col="Sex", row="Embarked", hue = "Survived").map(plt.hist, "Age", edgecolor = 'white').add_legend();
raw_data['Title'] = raw_data.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

plt.figure(figsize=(18, 8))

print(raw_data['Title'].unique())



raw_data['Title'] = raw_data['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

raw_data['Title'] = raw_data['Title'].replace('Mlle', 'Miss')

raw_data['Title'] = raw_data['Title'].replace('Ms', 'Miss')

raw_data['Title'] = raw_data['Title'].replace('Mme', 'Mrs')



sns.barplot(x="Title", y="Survived", data=raw_data, estimator=survival_estimator)
print(raw_data['Cabin'].unique())
raw_data['NoCabin'] = np.where(raw_data['Cabin'].isnull(), 1, 0)

sns.barplot(x="NoCabin", y="Survived", data=raw_data, estimator=survival_estimator)
raw_data = pd.concat([raw_data, pd.get_dummies(raw_data['Sex'])], axis=1)
raw_data = pd.concat([raw_data, pd.get_dummies(raw_data['Title'], prefix='title')], axis=1)
raw_data = pd.concat([raw_data, pd.get_dummies(raw_data['Pclass'], prefix='Pclass')], axis=1)
raw_data.head(10)
need_columns = [

    'female', 'male', 

    'Pclass_1', 'Pclass_2', 'Pclass_3', 

    'title_Master', 'title_Miss', 'title_Mr', 'title_Mrs', 'title_Rare', 

    'AgeLabel', 'IsAgeNull', 'IsChildren', 

    'FareLabel', 'SibSp', 'Parch', 'FamilySize', 

    'NoFamily', 'SmallFamily', 'MediumFamily', 'LargeFamily', 

    'LowFare', 'HighFare', 'MediumFare', 

    'NoCabin'

]

data = raw_data[need_columns]



x = data

y = raw_data.Survived

# Number of trees in random forest

n_estimators = [int(x) for x in np.linspace(start = 100, stop = 2000, num = 20)]

# Number of features to consider at every split

max_features = ['auto', 'sqrt', 'log2', None]

# Maximum number of levels in tree

max_depth = [int(x) for x in np.linspace(10, 200, num = 20)]

max_depth.append(None)

# Minimum number of samples required to split a node

min_samples_split = [2, 5, 10]

# Minimum number of samples required at each leaf node

min_samples_leaf = [1, 2, 4]

# Method of selecting samples for training each tree

bootstrap = [True, False]

max_leaf_nodes = [2, 5, 8, 10, None]

criterion=['gini', 'entropy']



random_grid = {

    'n_estimators': n_estimators,

    'criterion': criterion,

    'max_features': max_features,

    'max_depth': max_depth,

    'min_samples_split': min_samples_split,

    'min_samples_leaf': min_samples_leaf,

    'bootstrap': bootstrap,

    'max_leaf_nodes': max_leaf_nodes

}

estimator = RandomForestClassifier()
rf_random = RandomizedSearchCV(

    estimator=estimator, 

    param_distributions=random_grid, 

    random_state=42, 

    n_jobs=-1

)

rf_random.fit(x, y)
best_params = rf_random.best_params_

print(best_params)
best_score = rf_random.best_score_

print("best score {}".format(best_score))
test_model = RandomForestClassifier(

    n_estimators=best_params['n_estimators'],

    criterion=best_params['criterion'],

    max_features=best_params['max_features'],

    max_depth=best_params['max_depth'],

    min_samples_split=best_params['min_samples_split'],

    min_samples_leaf=best_params['min_samples_leaf'],

    max_leaf_nodes=best_params['max_leaf_nodes'],

    bootstrap=best_params['bootstrap']

)
from sklearn.model_selection import cross_val_score

scores = cross_val_score(test_model, x, y, cv=10, scoring='accuracy')
print(scores)

print("Mean Accuracy: {}".format(scores.mean()))
model = RandomForestClassifier(

    n_estimators=best_params['n_estimators'],

    criterion=best_params['criterion'],

    max_features=best_params['max_features'],

    max_depth=best_params['max_depth'],

    min_samples_split=best_params['min_samples_split'],

    min_samples_leaf=best_params['min_samples_leaf'],

    max_leaf_nodes=best_params['max_leaf_nodes'],

    bootstrap=best_params['bootstrap']

)

model.fit(x, y)
print(raw_test.isnull().sum())

print("-"*10)

print(raw_test.isnull().sum()/raw_test.shape[0])
raw_test['Fare'].fillna(raw_test['Fare'].mean(), inplace=True)

raw_test['NoCabin'] = np.where(raw_test['Cabin'].isnull(), 1, 0)



raw_test['IsChildren'] = np.where(raw_test['Age']<=10, 1, 0)

raw_test['IsAgeNull'] = np.where(np.isnan(raw_test['Age']), 1, 0)

raw_test['Age'].fillna(raw_test['Age'].mean(), inplace=True)

raw_test['Age'] = raw_test['Age'].round().astype(int)

raw_test['AgeLabel'] = pd.cut(raw_test['Age'], bins=np.arange(start=0, stop=90, step=10), labels=np.arange(start=0, stop=8, step=1), include_lowest=True)



raw_test['FamilySize'] = raw_test.apply (lambda row: row['SibSp']+row['Parch'], axis=1)

raw_test['LowFare'] = np.where(raw_test['Fare']<=50, 1, 0)

raw_test['HighFare'] = np.where(raw_test['Fare']>300, 1, 0)

raw_test['MediumFare'] = raw_test.apply(medium_fare, axis=1)

raw_test['FareLabel'] = pd.cut(raw_test.Fare, bins=np.arange(start=0, stop=600, step=50), precision=0, labels=np.arange(start=0, stop=11, step=1), include_lowest=True)



raw_test['NoFamily'] = np.where(raw_test['FamilySize']==0, 1, 0)

raw_test['SmallFamily'] = np.where((raw_test['FamilySize']>0)&(raw_test['FamilySize']<4), 1, 0)

raw_test['MediumFamily'] = np.where((raw_test['FamilySize']>3)&(raw_test['FamilySize']<7), 1, 0)

raw_test['LargeFamily'] = np.where(raw_test['FamilySize']>=7, 1, 0)



raw_test['Title'] = raw_test.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

raw_test['Title'] = raw_test['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

raw_test['Title'] = raw_test['Title'].replace('Mlle', 'Miss')

raw_test['Title'] = raw_test['Title'].replace('Ms', 'Miss')

raw_test['Title'] = raw_test['Title'].replace('Mme', 'Mrs')



raw_test = pd.concat([raw_test, pd.get_dummies(raw_test['Sex'])], axis=1)

raw_test = pd.concat([raw_test, pd.get_dummies(raw_test['Title'], prefix='title')], axis=1)

raw_test = pd.concat([raw_test, pd.get_dummies(raw_test['Pclass'], prefix='Pclass')], axis=1)

data_test = raw_test[need_columns]



ids = raw_test['PassengerId']

predictions = model.predict(data_test)



output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })

output.to_csv('submission.csv', index = False)

output.head(10)