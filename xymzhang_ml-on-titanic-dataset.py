import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
%matplotlib inline

import warnings
warnings.filterwarnings('ignore')
data_train = pd.read_csv('../input/train.csv')
data_test = pd.read_csv('../input/test.csv')
data_train.info()
data_train.head()
def detect_outliers(df,n,features):
    outlier_indices = []
    for col in features:
        Q1 = np.percentile(df[col],25)
        Q3 = np.percentile(df[col],75)
        IQR = Q3 - Q1
        outlier_step = 1.5 * IQR
        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step)].index
        outlier_indices.extend(outlier_list_col)

    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list(k for k, v in outlier_indices.items() if v>n)
    return multiple_outliers
data1 = data_train.copy()
Outliers_to_drop = detect_outliers(data1,2,['Age','Parch','Fare','SibSp'])
data1.iloc[Outliers_to_drop]
data2 = data1.drop(Outliers_to_drop).reset_index(drop=True)
data3 = data2.copy()
data2.info()
data2.head(3)
def categorical_plot(df, feature):
    sns.countplot(data=df, x=feature)
    sns.factorplot(data=df, x=feature,y ='Survived', kind='bar')
categorical_plot(data2, 'Pclass')
categorical_plot(data2, 'Sex')
sns.factorplot(data=data2, x='Pclass',y ='Survived', hue='Sex',kind='bar')
data2['Sex'] = data2['Sex'].apply(lambda x: 1 if x=='male' else 0)
categorical_plot(data2, 'Embarked')
data2['Embarked'] = data2.Embarked.fillna('S')
sns.factorplot(data=data2, x='Embarked', y ='Survived', hue='Sex',kind='bar')
data2['Cabin_Initial'] = data2['Cabin'].apply(lambda x: 'NA' if pd.isna(x) else str(x)[0])
data2.Cabin_Initial.value_counts()
categorical_plot(data2, 'Cabin_Initial')
sns.factorplot(data=data2, x='Cabin_Initial', y ='Survived', hue='Sex',kind='bar')
sns.distplot(data2[-(data2['Age'].isna())].Age)
data2.head(3)
na_index = list(data2[data2['Age'].isna()].index)
age_median = data2[-(data2['Age'].isna())].Age.median()
for i in na_index:
    age_median2 = data2[((data2['Sex']==data2.iloc[i]['Sex'])&(data2['SibSp']==data2.iloc[i]['SibSp'])&(data2['Parch']==data2.iloc[i]['Parch']))]['Age'].median()
    if not np.isnan(age_median2):
        data2['Age'].iloc[i] = age_median2
    else:
        data2['Age'].iloc[i] = age_median
sns.distplot(data2.Age)
data2['Age_bucket'] = pd.cut(data2['Age'], 6, labels=['A','B','C','D','E','F'])
data2.head()
sns.factorplot(data=data2, x='Age_bucket', y='Survived',kind='bar')
def age_gap(x):
    if x < 8:
        return 'A'
    elif x < 12:
        return 'B'
    elif x < 18:
        return 'C'
    elif x < 50:
        return 'D'
    elif x < 60:
        return 'E'
    else:
        return 'F'
data2['Age_bucket2'] = data2['Age'].apply(age_gap)
sns.factorplot(data=data2, x='Age_bucket2', y='Survived',kind='bar')
sns.factorplot(data=data2, x='SibSp', y='Survived',kind='bar')
sns.factorplot(data=data2, x='Parch', y='Survived',kind='bar')
data2['Family'] = data2['Parch']+data2['SibSp']+1
sns.factorplot(data=data2, x='Family', y='Survived',kind='bar')
data2['Title'] = data2['Name'].map(lambda i: i.split(',')[1].split('.')[0].strip())
data2['Title'].value_counts()
data2['Title'] = data2['Title'].replace(['Lady', 'the Countess','Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
data2['Title'] = data2['Title'].map({'Master':0, 'Miss':1, 'Ms' : 1 , 'Mme':1, 'Mlle':1, 'Mrs':1, 'Mr':2, 'Rare':3})
data2['Title'] = data2['Title'].astype(int)

sns.factorplot(data=data2, x='Title', y='Survived',kind='bar')
data2.head(3)
data2['Ticket_Initial'] = data2['Ticket'].apply(lambda x: 'NA' if x.isdigit() else 
                                               x.replace('.','').replace('/','').strip().split(' ')[0])
data2[['Ticket_Initial','Survived']].groupby(by='Ticket_Initial', as_index=True).mean().sort_values(by='Survived', ascending=False)
sns.distplot(data2.Fare)
data2['Fare_log'] = data2['Fare'].apply(lambda x: np.log(x) if x !=0 else 0)
sns.distplot(data2.Fare_log)
data2['Fare_bucket'] = pd.cut(data2['Fare_log'], bins=4, labels=['A','B','C','D'])
sns.factorplot(data=data2, x='Fare_bucket', y='Survived',kind='bar')
data_test.head()
data_all = pd.concat([data3, data_test], axis=0).reset_index(drop=True)
train_len = len(data3)
data_all['Embarked'] = data_all.Embarked.fillna('S')
data_all['Cabin_Initial'] = data_all['Cabin'].apply(lambda x: 'NA' if pd.isna(x) else str(x)[0])

na_index = list(data_all[data_all['Age'].isna()].index)
age_median = data_all[-(data_all['Age'].isna())].Age.median()
for i in na_index:
    age_median2 = data_all[((data_all['Sex']==data_all.iloc[i]['Sex'])&(data_all['SibSp']==data_all.iloc[i]['SibSp'])&(data_all['Parch']==data_all.iloc[i]['Parch']))]['Age'].median()
    if not np.isnan(age_median2):
        data_all['Age'].iloc[i] = age_median2
    else:
        data_all['Age'].iloc[i] = age_median
        
data_all['Age_bucket2'] = data_all['Age'].apply(age_gap)

data_all['Family'] = data_all['Parch']+data_all['SibSp']+1
data_all['Title'] = data_all['Name'].map(lambda i: i.split(',')[1].split('.')[0].strip())
data_all['Title'] = data_all['Title'].replace(['Lady', 'the Countess','Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
data_all['Title'] = data_all['Title'].map({'Master':0, 'Miss':1, 'Ms' : 1 , 'Mme':1, 'Mlle':1, 'Mrs':1, 'Mr':2, 'Rare':3})
data_all['Title'] = data_all['Title'].astype(int)

data_all['Ticket_Initial'] = data_all['Ticket'].apply(lambda x: 'NA' if x.isdigit() else 
                                               x.replace('.','').replace('/','').strip().split(' ')[0])
data_all['Fare_log'] = data_all['Fare'].apply(lambda x: np.log(x) if x !=0 else 0)
data_all['Fare_bucket'] = pd.cut(data_all['Fare_log'], bins=4, labels=['A','B','C','D'])
data_all.tail()
dummies_Age = pd.get_dummies(data_all['Age_bucket2'], prefix='Age_bucket2')
dummies_Cabin = pd.get_dummies(data_all['Cabin_Initial'], prefix='Cabin_Initial')
dummies_Embarked = pd.get_dummies(data_all['Embarked'], prefix='Embarked')
dummies_Fare = pd.get_dummies(data_all['Fare_bucket'], prefix='Fare_bucket')
dummies_Ticket = pd.get_dummies(data_all['Ticket_Initial'], prefix='Ticket_Initial')
dummies_Pclass = pd.get_dummies(data_all['Pclass'], prefix='Pclass')
dummies_Sex = pd.get_dummies(data_all['Sex'], prefix='Sex')
dummies_Family = pd.get_dummies(data_all['Family'], prefix='Family')
dummies_Name = pd.get_dummies(data_all['Title'], prefix='Title')

data_all = pd.concat([data_all, dummies_Age, dummies_Cabin, dummies_Embarked, dummies_Fare,
                     dummies_Ticket,dummies_Pclass,dummies_Sex,dummies_Family,dummies_Name], axis=1)
data_all.head(3)
train_df = data_all[:train_len].filter(regex='Age_bucket2_.*|Fare_bucket_.*|Cabin_Initial_.*|Embarked_.*|Sex_.*|Pclass_.*|Family_.*|Ticket_Initial_.*|Title_.*')
test_df = data_all[train_len:].filter(regex='Age_bucket2_.*|Fare_bucket_.*|Cabin_Initial_.*|Embarked_.*|Sex_.*|Pclass_.*|Family_.*|Ticket_Initial_.*|Title_.*')

train = train_df.as_matrix()
test = test_df.as_matrix()

X = train
y = data_all[:train_len]['Survived'].as_matrix()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=666)
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

lgr = LogisticRegression()
C = [0.001,0.01,0.1,1,10,100]
penalty = ['l1','l2']

param_grid = dict(C=C, penalty=penalty)

grid_search = GridSearchCV(lgr, param_grid, scoring='accuracy', cv=5)
grid_result = grid_search.fit(X_train, y_train)

result_lgr = pd.DataFrame(grid_result.cv_results_)
result_lgr.sort_values(by='mean_test_score', ascending=False)
best_lgr = grid_search.best_estimator_
y_pred = best_lgr.predict(X_test)
print(accuracy_score(y_test,y_pred))
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier()
n_estimators = [i for i in range(50,350,10)]
max_depth = [i for i in range(7,13,1)]

param_grid = dict(max_depth=max_depth, n_estimators=n_estimators)

grid_search = GridSearchCV(rfc, param_grid, scoring='accuracy', cv=5)
grid_result = grid_search.fit(X_train, y_train)

result_rfc = pd.DataFrame(grid_result.cv_results_)
result_rfc.sort_values(by='mean_test_score', ascending=False)
best_rfc = grid_search.best_estimator_
y_pred = best_rfc.predict(X_test)
print(accuracy_score(y_test,y_pred))
from sklearn.ensemble import GradientBoostingClassifier

gbc = GradientBoostingClassifier()
n_estimators = [i for i in range(50,350,10)]
max_depth = [i for i in range(7,13,1)]

param_grid = dict(max_depth=max_depth, n_estimators=n_estimators)

grid_search = GridSearchCV(gbc, param_grid, scoring='accuracy', cv=5)
grid_result = grid_search.fit(X_train, y_train)

result_gbc = pd.DataFrame(grid_result.cv_results_)
result_gbc.sort_values(by='mean_test_score', ascending=False)
best_gbc = grid_search.best_estimator_
y_pred = best_gbc.predict(X_test)
print(accuracy_score(y_test,y_pred))
