import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import (RandomForestClassifier,
                             AdaBoostClassifier,
                             GradientBoostingClassifier)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import (GridSearchCV,
                                     cross_val_score,
                                     StratifiedKFold, 
                                     learning_curve)
train = pd.read_csv('../input/titanic/train.csv')
test = pd.read_csv('../input/titanic/test.csv')
ids = test['PassengerId']
print('Train shape : ',train.shape)
print('Test shape : ',test.shape)
train.head()
test.head()
train.isna().sum().sort_values(ascending=False)
train['Embarked'].value_counts()
train['Embarked'].fillna('S',inplace=True)
train.isna().sum().sort_values(ascending=False)
sns.scatterplot(x='Age',y='SibSp',data=train)
sns.scatterplot(x='Age',y='Parch',data=train)
train[['Age','SibSp']].groupby('SibSp').median()
train[['Age','SibSp']].groupby('SibSp').mean()
train[['Age','Parch']].groupby('Parch').median()
train[['Age','Parch']].groupby('Parch').mean()
train[train['SibSp']==8]
print('Mean of age is : ',train['Age'].mean())
print('Median of age is : ',train['Age'].median())
train['Age'].fillna(train['Age'].median(),inplace=True)
train.isna().sum().sort_values(ascending=False)
sns.heatmap(train.corr(),annot=True)
train['FamilySize'] = train['Parch'] + train['SibSp'] +1
train.head()
sns.heatmap(train.corr(),annot=True)
train['Single'] = train['FamilySize'].map(lambda i: 1 if i==1 else 0)
train['Small'] = train['FamilySize'].map(lambda i: 1 if i==2 else 0)
train['Medium'] = train['FamilySize'].map(lambda i: 1 if 3<=i<=4 else 0)
train['Large'] = train['FamilySize'].map(lambda i: 1 if i>4 else 0)
train.head()
train['Sex'] = train['Sex'].map(lambda i : 1 if i=='male' else 0)
train.head()
train['Embarked_S'] = train['Embarked'].map(lambda i: 1 if i=='S' else 0)
train['Embarked_C'] = train['Embarked'].map(lambda i: 1 if i=='C' else 0)
train['Embarked_Q'] = train['Embarked'].map(lambda i: 1 if i=='Q' else 0)
train.drop(['Embarked'],axis=1,inplace=True)
train.head()
titles = [i.split(',')[1].split('.')[0].strip() for i in train['Name']]
train['Title'] = pd.Series(titles)
train['Title'].head()
train['Title'].value_counts()
rare_surnames = ['Rev','Col','Mlle','Don','Mme','Jonkheer','the Countess']
mapping_other_surnames = {'Mr':1,
                         'Mrs':2,
                         'Miss':2,
                         'Master':1,
                         'Dr':3,
                         'Col':1,
                         'Major':3,
                         'Ms':2,
                         'Lady':2,
                         'Capt':3,
                         'Sir':1,
                         'Rare':4}
train['Title'] = train['Title'].replace(rare_surnames,'Rare')
train['Title'] = train['Title'].map(mapping_other_surnames)
train['Title']=train['Title'].astype(int)

train['Title'].head()
train.drop(['Name'],axis=1,inplace=True)
train.head()
train.drop(['PassengerId'],axis=1,inplace=True)
train.head()
train['Cabin'].describe()
train['Cabin'][1][0]
train['Cabin'] = pd.Series([i[0] if not pd.isnull(i) else 'X' for i in train['Cabin']])
train['Cabin'].head()
sns.countplot(train['Cabin'])
sns.barplot(x='Cabin',y='Survived',data=train)
train['Cabin'].value_counts()
train = pd.get_dummies(train,columns=['Cabin'],prefix='Cabin')
train.head()
train.drop(['Ticket'],axis=1,inplace=True)
train.head()
train.shape
test.head()
test.isna().sum().sort_values(ascending=False)
test[test['Fare'].isna()==True]
test.shape
test.drop(['PassengerId'],axis=1,inplace=True)
titles = [i.split(',')[1].split('.')[0].strip() for i in test['Name']]
test['Title'] = pd.Series(titles)
test['Title'].isna().sum()
rare_surnames = ['Rev','Col','Mlle','Don','Mme','Jonkheer','the Countess']
mapping_other_surnames = {'Mr':1,
                         'Mrs':2,
                         'Miss':2,
                         'Master':1,
                         'Dr':3,
                         'Col':1,
                         'Major':3,
                         'Ms':2,
                         'Lady':2,
                         'Capt':3,
                         'Sir':1,
                         'Rare':4}
test['Title'] = test['Title'].replace(rare_surnames,'Rare')
test['Title'] = test['Title'].map(mapping_other_surnames)
test.head()
test.drop(['Name'],axis=1,inplace=True)
test.shape
test['Sex'] = test['Sex'].map(lambda i : 1 if i=='male' else 0)

test['Embarked_S'] = test['Embarked'].map(lambda i: 1 if i=='S' else 0)
test['Embarked_C'] = test['Embarked'].map(lambda i: 1 if i=='C' else 0)
test['Embarked_Q'] = test['Embarked'].map(lambda i: 1 if i=='Q' else 0)
test.drop(['Embarked'],axis=1,inplace=True)

test.drop(['Ticket'],axis=1,inplace=True)

test['FamilySize'] = test['Parch'] + test['SibSp'] +1
test['Single'] = test['FamilySize'].map(lambda i: 1 if i==1 else 0)
test['Small'] = test['FamilySize'].map(lambda i: 1 if i==2 else 0)
test['Medium'] = test['FamilySize'].map(lambda i: 1 if 3<=i<=4 else 0)
test['Large'] = test['FamilySize'].map(lambda i: 1 if i>4 else 0)

test['Age'].fillna(test['Age'].median(),inplace=True)
test['Fare'].fillna(test['Fare'].mean(),inplace=True)

test['Cabin'] = pd.Series([i[0] if not pd.isnull(i) else 'X' for i in test['Cabin']])
test = pd.get_dummies(test,columns=['Cabin'],prefix='Cabin')
print(test.shape)
test.head()
test.columns
train.columns
test.isna().sum().sort_values(ascending=False)
train.isna().sum().sort_values(ascending=False)
test[test['Title'].isna()==True]
test['Title'].fillna('1',inplace=True)
test.isna().sum().sort_values(ascending=False)
test['Cabin_T']=0
plt.figure(figsize=(20,12))
sns.heatmap(train.corr(),annot=True)
plt.figure(figsize=(20,12))
sns.heatmap(test.corr(),annot=True)
test.head()
len(train)
X_train = train.drop(['Survived'],axis=1)
y_train = train['Survived']
print('Shape of X_train is : ',X_train.shape)
print('Shape of Y_train is :',y_train.shape)
kfold = StratifiedKFold(n_splits=10)
classifiers = [
    SVC(),
    DecisionTreeClassifier(),
    AdaBoostClassifier(DecisionTreeClassifier(),learning_rate=0.1),
    RandomForestClassifier(n_estimators=50),
    GradientBoostingClassifier(),
    KNeighborsClassifier(),
    LogisticRegression(),
    LinearDiscriminantAnalysis(solver='eigen',shrinkage='auto'),
    MLPClassifier(learning_rate='adaptive')
]
import warnings
warnings.filterwarnings('ignore')
results = []
for classifier in classifiers:
  results.append(cross_val_score(estimator=classifier,X=X_train,y=y_train,cv=kfold,scoring='accuracy'))
mean = []
std = []
for result in results:
  mean.append(result.mean())
  std.append(result.std())

result_df = pd.DataFrame({'Cross Validation Mean':mean,'Cross Validation Error':std,'Algorithms':['Suppor vector classifier',
                                                                                                  'Decision Tree classifier',
                                                                                                  'AdaBoosting classifier',
                                                                                                  'Random forest classifier',
                                                                                                  'Gradient boosting',
                                                                                                  'K Neighbours classifier',
                                                                                                  'Logistic Regression classifier',
                                                                                                  'Linear discriminant analysis',
                                                                                                  'Multi layer perceptron classifier']})
result_df
sns.barplot(x='Cross Validation Mean',y='Algorithms',data=result_df)

kfold = StratifiedKFold(n_splits=10)
classifiers = [
    AdaBoostClassifier(DecisionTreeClassifier(),learning_rate=1),
    RandomForestClassifier(n_estimators=50),
    GradientBoostingClassifier(),
    LogisticRegression()
]
ada_grid = {'base_estimator__criterion':['gini','entropy'],
           'base_estimator__splitter':['best','random'],
           'algorithm':['SAMME','SAMME.R'],
           'n_estimators':[1,2,3],
           'learning_rate':[0.01,0.1,0.5,1,1.3]}
random_grid = {'max_depth':[None],
              'max_features':[1,3,5,10,20],
              'min_samples_split':[2,4,6,8],
              'min_samples_leaf':[1,2,3,5],
              'n_estimators':[50,100,200],
              'criterion':['gini']}
gradient_grid = {'loss':['deviance'],  #exponential
                'learning_rate':[1,1.3],  #1,1.3
                'n_estimators':[200,250],  # 100
                'criterion':['mae','friedman_mse'],  #'friedman_mse'
                'min_samples_split':[3], #0.5,4
                'min_samples_leaf':[1,2], 
                'max_depth':[3],
                'max_features':['sqrt']}
logistic_grid = {'penalty':['l2'],
                'C':[0.4,0.5,0.6,0.7],
                'solver':['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
                'max_iter':[100,200,250,300]}

main_grid = [ada_grid,random_grid,gradient_grid,logistic_grid]
grid_results = list()
grid_best_estimator = list()
grid_ = GridSearchCV(estimator=classifiers[0],param_grid=main_grid[0],cv=kfold,scoring='accuracy')
grid_.fit(X_train,y_train)
grid_results.append(grid_.best_score_)
grid_best_estimator.append(grid_.best_estimator_)
grid_results

grid_best_estimator

grid_ = GridSearchCV(estimator=classifiers[1],param_grid=main_grid[1],cv=kfold,scoring='accuracy')
grid_.fit(X_train,y_train)
grid_results.append(grid_.best_score_)
grid_best_estimator.append(grid_.best_estimator_)
grid_results
grid_best_estimator
grid_ = GridSearchCV(estimator=classifiers[3],param_grid=main_grid[3],cv=kfold,scoring='accuracy')
grid_.fit(X_train,y_train)
grid_results.append(grid_.best_score_)
grid_best_estimator.append(grid_.best_estimator_)
grid_results
grid_best_estimator
grid_ = GridSearchCV(estimator=classifiers[2],param_grid=main_grid[2],cv=kfold,scoring='accuracy')
grid_.fit(X_train,y_train)
grid_results.append(grid_.best_score_)
grid_best_estimator.append(grid_.best_estimator_)
grid_results
grid_best_estimator
from sklearn.ensemble import VotingClassifier
voting = VotingClassifier(estimators=[('Random_forest',grid_best_estimator[1]),
                                      ('Ada_boost',grid_best_estimator[0]),
                                      ('Gradient_boost',grid_best_estimator[3]),
                                      ('Logistic',grid_best_estimator[2])],
                         voting='hard')

voting.fit(X_train,y_train)
test.head()
predictions = pd.Series(voting.predict(test),name='Survived')
results = pd.concat([ids,predictions],axis=1)
results.to_csv('Predictions.csv',index=False)
