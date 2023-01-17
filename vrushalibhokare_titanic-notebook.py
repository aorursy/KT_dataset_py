# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



from collections import Counter



from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier



from sklearn.tree import DecisionTreeClassifier



from sklearn.discriminant_analysis import LinearDiscriminantAnalysis



from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold,learning_curve

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report





sns.set(style='white')
train = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')

IDtest = test['PassengerId']
def detect_outliers(df,n,features):

    outlier_indices = []

    

    # iterate over dataframe

    for col in features:

        # Find 1st Quartile

        Q1 = np.percentile(df[col],25)

        # Find 3rd Quartile

        Q3 = np.percentile(df[col],75)

        # Find Interquartile Range

        IQR = Q3 - Q1

        

        # setting outlier_step to 1.5

        outlier_step = 1.5 * IQR

        

        # Determine a lot of indices of outliers for feature column

        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step) ].index

        

        # appending indices 

        outlier_indices.extend(outlier_list_col)

        

    outlier_indices = Counter(outlier_indices)

    

    multiple_outliers = list( k for k,v in outlier_indices.items() if v > n)

    

    return multiple_outliers
train.columns
outliers_to_drop = detect_outliers(train,2,['Age', 'SibSp', 'Parch', 'Fare'])
outliers_to_drop
train.loc[outliers_to_drop]
train = train.drop(outliers_to_drop,axis=0).reset_index(drop=True)
## Joining the trainign and test dataset for categorical conversion



train_len = len(train)

df = pd.concat(objs=[train,test],axis=0).reset_index(drop=True)
df.fillna(np.nan, inplace=True)



df.isnull().sum()
train.isnull().sum()
train.head()
train.info()
## Correlation between numerical variables



g = sns.heatmap(train[['Survived','Age', 'SibSp', 'Parch', 'Fare','Pclass']].corr(),annot=True,cmap='coolwarm')
df.head()
# SibSp



g = sns.factorplot(x='SibSp',y='Survived',data=train,kind='bar',size=6,palette='muted')



g.despine(left=True)



g = g.set_ylabels('Survival Probability')
## Parch



# SibSp



g = sns.factorplot(x='Parch',y='Survived',data=train,kind='bar',size=6,palette='muted')



g.despine(left=True)



g = g.set_ylabels('Survival Probability')
# Age vs Survived



g = sns.FacetGrid(train,col='Survived')



g = g.map(sns.distplot,'Age')
df['Fare'].isnull().sum()
df['Fare'].fillna(df['Fare'].median(),inplace=True)
df['Fare'].isnull().sum()
g = sns.distplot(df['Fare'],color='m',label='Skewness : %.2f'%(df['Fare'].skew()))

g = g.legend(loc='best')
df['Fare'] = df['Fare'].apply(lambda x: np.log(x) if x > 0 else 0)
g = sns.distplot(df['Fare'],color='m',label='Skewness : %.2f'%(df['Fare'].skew()))

g = g.legend(loc='best')
g = sns.barplot(x='Sex',y='Survived',data=train)
train[['Sex','Survived']].groupby('Sex').mean()
g = sns.factorplot(x='Pclass',y='Survived',data=train,kind='bar',size=6,palette='muted')
g = sns.factorplot(x='Pclass',y='Survived',data=train,kind='bar',size=6,palette='muted', hue = 'Sex')

df['Embarked'].value_counts()
df['Embarked'].fillna('S',inplace=True)
g = sns.factorplot(x='Embarked',y='Survived',data=train,kind='bar',size=6,palette='muted')

df.isnull().sum()
## Correlation between numerical variables



g = sns.heatmap(train[['Sex','Survived','Age', 'SibSp', 'Parch', 'Fare','Pclass']].corr(),annot=True,cmap='coolwarm')
g = sns.factorplot(y='Age',x='Sex',data=df,kind='box')



g = sns.factorplot(y='Age',x='Sex', hue='Pclass',data=df,kind='box')



g = sns.factorplot(y='Age',x='Parch',data=df,kind='box')



g = sns.factorplot(y='Age',x='SibSp',data=df,kind='box')
df['Sex'] = df['Sex'].map({'male':0,'female':1})



g = sns.heatmap(df[['Sex','SibSp','Parch','Age','Pclass']].corr(),annot=True,cmap='coolwarm',square=True)
# Filling missing values of age



# Fill Age with median age of similar rows accoring to Pclass, Parch and SibSp



index_nan_age = list(df['Age'][df['Age'].isnull()].index)



for i in index_nan_age:

    age_median = df['Age'].median()

    age_pred = df['Age'][((df['SibSp'] == df.iloc[i]['SibSp']) & (df['Parch'] == df.iloc[i]['Parch']) & 

                          (df['Pclass'] == df.iloc[i]['Pclass']))].median()

    

    if not np.isnan(age_pred):

        df['Age'].iloc[i] = age_pred

    else:

        df['Age'].iloc[i] = age_median
df.isnull().sum()
## Feature Engineering Name - Handling Rare Labels



df['Name'].head(30)
df_title = [i.split(',')[1].split('.')[0].strip() for i in df['Name'] ]



df['Title'] = pd.Series(df_title)
g = sns.countplot(x='Title',data=df)

g = plt.setp(g.get_xticklabels(),rotation=45)
df['Title'].value_counts()
df['Title'] = df['Title'].replace(['Lady','Jonkheer','the Countess','Major',

                                  'Ms','Mlle','Col','Rev','Dr','Sir','Mme','Dona'

                                 ,'Don','Capt'],'Rare')



df['Title'] = df['Title'].map({'Mr':0,'Miss':1,'Mrs':1,'Master':2,'Rare':3})



df['Title'] = df['Title'].astype(int)
g = sns.countplot(x='Title',data=df)

g = plt.setp(g.set_xticklabels(['Mr','Miss, Mrs','Master','Rare']),rotation=45)
g = sns.factorplot(x='Title',y='Survived',data=df,kind='bar')

g = g.set_xticklabels(['Mr','Miss, Mrs','Master','Rare'])

g = g.set_ylabels('survival probability')
df.drop(['Name'],axis=1,inplace=True)
# Family size

df['Fsize'] = df['SibSp'] + df['Parch'] + 1



g = sns.factorplot(x='Fsize',y='Survived',data=df,kind='bar')

df['Single'] = df['Fsize'].map(lambda x:1 if x==1 else 0)



df['SmallF'] = df['Fsize'].map(lambda x:1 if x==2 else 0)



df['MediumF'] = df['Fsize'].map(lambda x:1 if 3<=x<=4 else 0)



df['LargeF'] = df['Fsize'].map(lambda x:1 if x>4 else 0)
g = sns.factorplot(x='Single',y='Survived',data=df,kind='bar')



g = sns.factorplot(x='SmallF',y='Survived',data=df,kind='bar')



g = sns.factorplot(x='MediumF',y='Survived',data=df,kind='bar')



g = sns.factorplot(x='LargeF',y='Survived',data=df,kind='bar')



## Title and Embarked



df = pd.get_dummies(df,columns=['Title'])

df = pd.get_dummies(df,columns=['Embarked'],prefix='Em')





df.head()
df['Cabin'].describe()
df['Cabin'].isnull().sum()
df['Cabin'].value_counts()
# Replace with cabin type, if null then impute X



df['Cabin'] = pd.Series([i[0] if not pd.isnull(i) else 'X' for i in df['Cabin']])
df['Cabin'].unique()
g = sns.countplot(df['Cabin'],order='A B C D E F G T X'.split())
g = sns.factorplot(x='Cabin',y='Survived',data=df,kind='bar')
df = pd.get_dummies(df,columns=['Cabin'],prefix='Cabin_')
df['Ticket'].head()
Ticket = []



for i in df['Ticket']:

    if not i.isdigit():

        Ticket.append(i.replace('.','').replace('/','').strip().split(' ')[0])

    else:

        Ticket.append('X')
df['Ticket'] = Ticket
df = pd.get_dummies(df,columns=['Ticket'],prefix='T_')
df['Pclass'].unique()
df['Pclass'] = df['Pclass'].astype('category')

df = pd.get_dummies(df,columns=['Pclass'],prefix='pc_')
df.drop(labels=['PassengerId'],axis=1,inplace=True)
#Modeling

train = df[:train_len]

test = df[train_len:]

test.drop(['Survived'],axis=1,inplace=True)
train['Survived']=train['Survived'].astype(int)
y_train = train['Survived']

X_train = train.drop('Survived',axis=1)
kfold = StratifiedKFold(n_splits=10)
random_state = 2

classifiers = []



classifiers.append(SVC(random_state=random_state))



classifiers.append(DecisionTreeClassifier(random_state=random_state))



classifiers.append(AdaBoostClassifier(DecisionTreeClassifier(random_state=random_state),

                                      random_state=random_state,learning_rate=0.1))



classifiers.append(RandomForestClassifier(random_state=random_state))



classifiers.append(ExtraTreesClassifier(random_state=random_state))



classifiers.append(GradientBoostingClassifier(random_state=random_state))



classifiers.append(MLPClassifier(random_state=random_state))



classifiers.append(KNeighborsClassifier())



classifiers.append(LogisticRegression(random_state=random_state))



classifiers.append(LinearDiscriminantAnalysis())
cv_results = []



for classifier in classifiers:

    cv_results.append(cross_val_score(classifier,X_train,y=y_train,scoring='accuracy',cv=kfold,n_jobs=4))

    

cv_means = []

cv_std = []





for cv_result in cv_results:

    cv_means.append(cv_result.mean())

    cv_std.append(cv_result.std())



cv_results = pd.DataFrame({'CrossValMeans':cv_means,'CrossValErrors':cv_std,

                           'Algorithms':['SVC','DecisionTree','AdaBoost','RandomForest','ExtraTrees',

                                         'GradientBoosting','MLP','KNN','Logistic','LDA']})



g = sns.barplot('CrossValMeans','Algorithms',data=cv_results)
DTC = DecisionTreeClassifier()



adaDTC = AdaBoostClassifier(DTC, random_state=7)



ada_param_grid = {'base_estimator__splitter':['best','random'],

                 'n_estimators':[1,2],

                 'learning_rate':[0.0001,0.001,0.01,0.1,0.2,0.3,1.5]}



gs_ada = GridSearchCV(adaDTC,param_grid=ada_param_grid,cv=kfold,scoring='accuracy',n_jobs=1,verbose=1)



gs_ada.fit(X_train,y_train)



gs_ada_best = gs_ada.best_estimator_
svc = SVC(probability=True)



svc_parm_grid={'kernel':['rbf'],'C':[1,10,50,100,1000,200,300,500],'gamma':[0.001,0.01,0.1,1]}



gs_svc = GridSearchCV(svc,param_grid=svc_parm_grid,cv=kfold,scoring='accuracy',n_jobs=4,verbose=4)



gs_svc.fit(X_train,y_train)



gs_svc_best = gs_svc.best_estimator_
%%time



from sklearn.metrics import make_scorer



RFC = RandomForestClassifier()

scoring = {'AUC': 'roc_auc', 'Accuracy': make_scorer(accuracy_score)}

param_grid = {'n_estimators': [5, 10, 20,  50,100,200] }



gs_rfc= GridSearchCV(RFC,param_grid=param_grid,cv=kfold,n_jobs=4,verbose=1, scoring=scoring, return_train_score=True,

                                refit= 'AUC')



gs_rfc.fit(X_train,y_train)



gs_rfc_best = gs_rfc.best_estimator_

gs_rfc_best
etc = ExtraTreesClassifier(n_estimators=100, random_state=0)

etc.fit(X_train,y_train)
gdb = GradientBoostingClassifier()

param_grid = {'n_estimators': [5, 10, 20,  50,100,200],'learning_rate':[0.0001,0.001,0.01,0.1,0.2,0.3,1.5] }



gdb_gs= GridSearchCV(gdb,param_grid=param_grid,cv=kfold,n_jobs=4,verbose=1, scoring='accuracy')



gdb_gs.fit(X_train,y_train)



gdb_gs_best =gdb_gs.best_estimator_

gdb_gs_best
test_Survived_RFC = pd.Series(gs_rfc_best.predict(test), 

name="rfc")

test_Survived_ExtC = pd.Series(etc.predict(test), name="etc")

test_Survived_SVMC = pd.Series(gs_svc_best.predict(test), name="SVC")

test_Survived_AdaC = pd.Series(gs_ada_best.predict(test), name="ada")

test_Survived_GDC = pd.Series(gdb_gs_best.predict(test), name="GDC")





# Concatenate all classifier results

ensemble_results = pd.concat([test_Survived_RFC,test_Survived_ExtC,test_Survived_AdaC,test_Survived_GDC, test_Survived_SVMC],axis=1)





g= sns.heatmap(ensemble_results.corr(),annot=True)
votingC = VotingClassifier(estimators=[('adaboost',gs_ada_best),

                                       ('svc',gs_svc_best),

                                       ('RFC',gs_rfc_best),

                                       ('GDB',gdb_gs_best)],

                           voting='soft',n_jobs=4)



votingC = votingC.fit(X_train,y_train)

test_survived = pd.Series(votingC.predict(test),name='Survived')
results = pd.concat([IDtest,test_survived],axis=1)
results.to_csv('ensemble_voting_classifier.csv',index=False)
df=pd.read_csv('ensemble_voting_classifier.csv')

df