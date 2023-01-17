import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import scipy.stats

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

from sklearn.neural_network import MLPClassifier

%matplotlib inline
#This is the traing dataset hosted on Kaggle.

ti = pd.read_csv('../input/train.csv')

ti.head()
ti.isnull().sum()
ti.describe()
#check the distributions of age and fare by plotting.

fig1, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharey=False, figsize=(13, 3))



ax1.hist(ti['Age'].dropna(), bins=12, edgecolor='white')

ax1.set_xlabel('Age')

ax2.set_xlabel('Fare')

ax2.hist(ti['Fare'], bins=12, edgecolor='white')

ax2.set_xlabel('Fare')



plt.show()

plt.close()
#take a look at NaN in Embarked.

ti[ti['Embarked'].isnull()]
#plot the counts of the groups in Embarked 

fig,ax = plt.subplots(figsize=(5,3))

sns.countplot(data=ti,

              x='Embarked',

              orient='v',

              hue=None,

              order=None,

              hue_order=None,              

              color=None,

              palette='Set2',

              saturation=0.6,

              edgecolor='black',

              linewidth=1,

              ax=ax)
#Fill in the most common category S

ti['Embarked'].fillna('S', inplace=True)



#check to see the filled values.

ti[(ti['PassengerId'] == 62) | (ti['PassengerId'] == 830)].loc[:,'Embarked']



#Or, drop the two NaN values in gembark

#ti.dropna(axis=0, subset=['Embarked'], inplace=True)
#Check to see if the missing values in Age and Cabin occur randomly in the aspect of the survived outcome.

print('survival percentage for NaN Age passengers: ',ti[ti['Age'].isnull()].loc[:,'Survived'].mean())

print('survival percentage for NaN Cabin passengers: ',ti[ti['Cabin'].isnull()].loc[:,'Survived'].mean())

print('survival percentage for all passengers: ',ti['Survived'].mean())
#impute Age Nan, with mean values of Sex and Pclass groups.

ti['Age_fill'] = ti.groupby(['Sex', 'Pclass'])['Age'].transform(lambda x: x.fillna(x.mean()))



#check if there is any NaN in the new column

print(ti['Age'].isnull().sum())

print(ti['Age_fill'].isnull().sum())
#check the distributions of Age, and Age_fill

fig1, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharey=False, figsize=(13, 3))



ax1.hist(ti['Age'].dropna(), bins=12, edgecolor='white', alpha=0.8)

ax1.set_xlabel('Age')

ax2.set_xlabel('Age_fill')

ax2.hist(ti['Age_fill'], bins=12, edgecolor='white', alpha=0.8)

ax2.set_xlabel('Fare')



plt.show()

plt.close()
#percentage of survival by Pclass, Sex, SibSp, Parch, Embarked

print(ti.groupby(['Pclass'])['Survived'].mean())

print('\n',ti.groupby(['Sex'])['Survived'].mean())

print('\n',ti.groupby(['SibSp'])['Survived'].mean())

print('\n',ti.groupby(['Parch'])['Survived'].mean())

print('\n',ti.groupby(['Embarked'])['Survived'].mean())
#code male as 0 female as 1

ti['gsex'] = ti['Sex'].apply(lambda x: 1 if (x == 'female') else 0)



#code Embarked 0 as C, 1 as Q and 2 as S

ti['gembark'] = ti['Embarked'].apply(lambda x: 0 if (x == 'C') else 1 if (x == 'Q') else 2)



#seperate SibSp into 3 groups of different survival rates

ti['gsibsp'] = ti['SibSp'].apply(lambda x: 0 if (x == 0) else 1 if (x in [3,4,5,8]) else 2)



#seperate Parch into 3 groups of different survival rates

ti['gparch'] = ti['Parch'].apply(lambda x: 0 if (x == 0) else 1 if (x in [4,5,6]) else 2)
#create a family column = SibSp + Parch

ti['family'] = ti['SibSp'] + ti['Parch']



#extract salute and create a new column and make sure I extracted something for every row.

ti['sal'] = ti['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

ti['sal'].isnull().sum() 
#check percentage of survival for the new columns

print(ti.groupby(['family'])['Survived'].mean())

print('\n',ti.groupby(['sal'])['Survived'].mean(), ti.groupby(['sal'])['Survived'].count())
#seperate sal into low, medium, and high survival groups, 0, 1, 2

high = ['Countess','Lady','Mlle','Mme','Ms','Sir','Miss','Mrs']

medium = ['Col','Dr','Major','Master']

ti['gsal'] = ti['sal'].apply(lambda x: 2 if x in high else 1 if x in medium else 0) 

ti['gsal'].iloc[10:20]
ti_plot = ti[['Survived','gsex','Age_fill','Fare','gembark','gparch','gsibsp','Pclass', 'gsal']]
ti_plot.head(2)
#count plots for the categorial variables.

fig, axes = plt.subplots(nrows=2, ncols=3, sharex=False, sharey=False, figsize=(15, 6))

for var, ax in zip(['gsex','gembark','gparch','gsibsp','Pclass', 'gsal'],axes.flat):

    sns.countplot(x=var, hue="Survived", data=ti_plot, edgecolor='black', palette="Set2", alpha=0.8, ax=ax)

#plot Fare and Age, colored by survival.

colors = {0:'firebrick', 1:'blue'}

groups = ti_plot.groupby(by=['Survived'])

for name, g in groups:

    plt.scatter(x=g['Age_fill'], y=g['Fare'], label=name, edgecolor='white', color=colors[name], alpha=0.5)

plt.legend(loc='best', prop={'size':15}, frameon=False)

plt.xlabel('Age')

plt.ylabel('Fare')
#dummy coding

pclass = pd.get_dummies(ti['Pclass'],prefix='pclass').drop('pclass_1', axis=1)

sibsp = pd.get_dummies(ti['gsibsp'],prefix='gsibsp').drop('gsibsp_0', axis=1)

parch = pd.get_dummies(ti['gparch'],prefix='gparch').drop('gparch_0', axis=1)

embark = pd.get_dummies(ti['gembark'],prefix='gembark').drop('gembark_0', axis=1)

sal = pd.get_dummies(ti['gsal'],prefix='gsal').drop('gsal_0', axis=1)
#create df_ti for training

df_ti = ti[['Survived','gsex','Age_fill','Fare']].join(pclass).join(sibsp).join(parch).join(embark).join(sal)



#create X_lr for logistic regression

X_lr = df_ti.drop(['Survived'], axis=1)



#create y_all for all type of classification training and logistic regression.

y_all = df_ti['Survived']
#If there is no testing data, split the whole dataset into a training set and test set

# from sklearn.model_selection import train_test_split

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=8)
#grid search with logistic regression

params1 = {'solver':['liblinear'],

           'penalty':['l1','l2'],

           'C':[0.1,1,10]}

kf1 = KFold(n_splits=10, shuffle=True)

gs1 = GridSearchCV(estimator=LogisticRegression(),param_grid=params1,cv=kf1,scoring='accuracy')

lr_mod1 = gs1.fit(X_lr, y_all)

print('Done!')
#output the statistics of the logistic regression model

print(pd.DataFrame(lr_mod1.cv_results_))

print('\n',lr_mod1.best_estimator_)

print('\n','Best Score: ',lr_mod1.best_score_)
#group Pclass and count the total passengers

total = ti.groupby(['Pclass'])['Survived'].count()

total
#group Pclass for the missing Cabin passengers and count the numbers

ca_null = ti[ti['Cabin'].isnull()].groupby(['Pclass'])['Survived'].count()

ca_null
#percentages of missing Cabin values grouped by Pclass

ca_null/total
#percentages of existing Cabin values grouped by Pclass

ca_exist = ti[ti['Cabin'].notnull()].groupby(['Pclass'])['Survived'].count()

ca_exist/total
#create a cabin column with null values coded 0 and notnull values coded 1.

ti['cabin']=ti['Cabin'].notnull().astype('int')



#check the new column.

print(ti['cabin'].isnull().sum())

ti[['cabin','Cabin']].head()
#Add the new cabin column to the plotting dataframe.

ti_plot = ti_plot.join(ti['cabin'])
ti_plot.head(2)
#Plot the number of cabin information status, known (1) and missing (0), across different Pclasses.

ax = sns.countplot(x='Pclass', hue="cabin", data=ti_plot, edgecolor='white', palette="Set2", alpha=0.8)

ax = sns.countplot(x='Survived', hue="cabin", data=ti_plot, edgecolor='white', palette="Set2", alpha=0.8)

#Add the new cabin column to the training dataset

df_ti2 = df_ti.join(ti['cabin'])



#Extract all the predictors for logistic regression

X_lr2 = df_ti2.drop(['Survived'], axis=1)
params2 = {'solver':['liblinear'],

           'penalty':['l1','l2'],

           'C':[0.1,1,10]}

kf2 = KFold(n_splits=20, shuffle=True)

gs2 = GridSearchCV(estimator=LogisticRegression(),param_grid=params2,cv=kf2,scoring='accuracy')

lr_mod2 = gs2.fit(X_lr2, y_all)

print('Done!')
#Output the results

print(pd.DataFrame(lr_mod2.cv_results_))

print('\n',lr_mod2.best_estimator_)

print('\n','Best Score: ',lr_mod2.best_score_)
#group and count the missing Age passengers

total_age = ti.groupby(['Sex'])['Survived'].count()

age_null = ti[ti['Age'].isnull()].groupby(['Sex'])['Survived'].count()

print(age_null, age_null/total_age)
ti[['Ticket', 'Pclass']].head(10)
#extract digits from Ticket and save it in a new column tick.

ti['tick'] = ti['Ticket'].str.extract('(\d+\Z)', expand=False)

#ti[['tick', 'Pclass']]

ti[ti['tick'].isnull()].loc[:,'Ticket']
#create data for decision tree training.

tree_df = ti[['Survived','Pclass','gsal','gsex','gsibsp','gparch','Fare','cabin','gembark']]

tree_df.info()
#create X and y for decision tree training.

X = tree_df.drop('Survived',axis=1)

y = tree_df['Survived']
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier



params3 = {'criterion':['gini','entropy'],

           'splitter':['best','random'],

           'min_samples_split':[2,5,8,11],

           'min_samples_leaf':[10,15,20,25,30]}

kf3 = KFold(n_splits=10, shuffle=True)

gs3 = GridSearchCV(estimator=DecisionTreeClassifier(),param_grid=params3,cv=kf3,scoring='accuracy')

tree_mod = gs3.fit(X, y)

print('Done!')
#output the results

#print(pd.DataFrame(tree_mod.cv_results_))

print('\n',tree_mod.best_estimator_)

print('\n','Best Score: ',tree_mod.best_score_)
#Prepare the test dataset from kaggle for evaluation of the model.

df_test=pd.read_csv('../input/test.csv')

df_test.info()
#prepare the features. 

df_test['gsex'] = df_test['Sex'].apply(lambda x: 1 if (x == 'female') else 0)

df_test['gembark'] = df_test['Embarked'].apply(lambda x: 0 if (x == 'C') else 1 if (x == 'Q') else 2)

df_test['gsibsp'] = df_test['SibSp'].apply(lambda x: 0 if (x == 0) else 1 if (x in [3,4,5,8]) else 2)

df_test['gparch'] = df_test['Parch'].apply(lambda x: 0 if (x == 0) else 1 if (x in [4,5,6]) else 2)

df_test['sal'] = df_test['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

df_test['gsal'] = df_test['sal'].apply(lambda x: 2 if x in high else 1 if x in medium else 0) 

df_test['cabin'] = df_test['Cabin'].notnull().astype('int')

df_test['Age_fill'] = df_test.groupby(['Sex', 'Pclass'])['Age'].transform(lambda x: x.fillna(x.mean()))
df_test.head()
#check where is the Fare NaN

df_test[df_test['Fare'].isnull()]
#calculate the mean of class 3 passenger fares

df_test.groupby(['Pclass']).get_group((3))['Fare'].mean()
#fill in the mean of class 3 passenger fares

df_test['Fare']=df_test['Fare'].fillna(12.5)

df_test.loc[152,'Fare']
#create X test set for evalution of decision tree models

X_tree_test = df_test[['Pclass','gsal','gsex','gsibsp','gparch','Fare','cabin','gembark']]
#create X test set for evalution of logistic regression

pclass_t = pd.get_dummies(df_test['Pclass'],prefix='pclass').drop('pclass_1', axis=1)

sibsp_t = pd.get_dummies(df_test['gsibsp'],prefix='gsibsp').drop('gsibsp_0', axis=1)

parch_t = pd.get_dummies(df_test['gparch'],prefix='gparch').drop('gparch_0', axis=1)

embark_t = pd.get_dummies(df_test['gembark'],prefix='gembark').drop('gembark_0', axis=1)

sal_t = pd.get_dummies(df_test['gsal'],prefix='gsal').drop('gsal_0', axis=1)

X_lr_test1 = df_test[['gsex','Age_fill','Fare']].join(pclass_t).join(sibsp_t).join(parch_t).join(embark_t).join(sal_t)

X_lr_test2 = X_lr_test1.join(df_test['cabin'])
from sklearn.metrics import roc_curve

from sklearn.metrics import auc

from sklearn.metrics import accuracy_score



#creating a plot_roc_curve funstion that we can call later

def plot_roc_curve(target_test, target_predicted_proba, label):

    fpr, tpr, thresholds = roc_curve(target_test, target_predicted_proba[:,1], drop_intermediate=False)

    roc_auc = auc(fpr, tpr)

    

    # Plot ROC curve

    fig, ax = plt.subplots(figsize=(3.5,3.5))

    ax.plot(fpr, tpr, 'r-', label= label + '\n' + 'ROC Area = %0.3f' % roc_auc)

    ax.plot([0, 1], [0, 1], 'k-.')

    ax.set_xlim(0,1)

    ax.set_ylim(0,1)

    ax.set_xlabel('False Positive Rate (1 - Specifity)', fontsize=12)

    ax.set_ylabel('True Positive Rate (Sensitivity)',fontsize=12)

    plt.legend(loc='lower right',fontsize=12)

    plt.title('Receiver Operating Characteristic',fontsize=12)
#calling roc curve function

plot_roc_curve(y_all_test, lr_mod1.predict_proba(X_lr_test1), 'Survived')

print('Model_1 Accuracy (logistic regression): ',accuracy_score(y_all_test,lr_mod1.predict(X_lr_test1)),'\n')

plot_roc_curve(y_all_test, lr_mod2.predict_proba(X_lr_test2), 'Survived')

print('Model_2 Accuracy (logistic regression with cabin info): ',accuracy_score(y_all_test,lr_mod2.predict(X_lr_test2)),'\n')

plot_roc_curve(y_all_test, tree_mod.predict_proba(X_tree_test), 'Survived')

print('Model_3 Accuracy (decision tree): ',accuracy_score(y_all_test,tree_mod.predict(X_tree_test)),'\n')