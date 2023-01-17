import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use('seaborn')

sns.set(font_scale=2.5) 

import missingno as msno

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline
df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')
df_train.head()
df_train.describe()
df_train.info()
df_test.describe()
df_test.head()
df_test.info()
df_train.isnull().sum()
msno.matrix(df=df_train.iloc[:, :], figsize=(7, 5), color=(0.5, 0.1, 0.2))
msno.bar(df=df_train.iloc[:, :], figsize=(7, 5), color=(0.2, 0.5, 0.2))
f,ax=plt.subplots(1,2,figsize=(16,6))

df_train['Survived'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=False)

ax[0].set_title('Survived')

ax[0].set_ylabel('')

sns.countplot('Survived',data=df_train,ax=ax[1])

ax[1].set_title('Survived')

plt.show()
df_train.groupby(['Sex','Survived'])['Survived'].count()
f,ax=plt.subplots(1,2,figsize=(14,4))

df_train[['Sex','Survived']].groupby(['Sex']).mean().plot.bar(ax=ax[0])

ax[0].set_title('Survived vs Sex')

sns.countplot('Sex',hue='Survived',data=df_train,ax=ax[1])

ax[1].set_title('Sex:Survived vs Dead')

plt.show()
df_train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=True).count()
df_train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=True).sum()
pd.crosstab(df_train.Pclass,df_train.Survived,margins=True)
df_train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=True).mean().sort_values(by='Survived', ascending=False).plot.bar()
f,ax=plt.subplots(1,2,figsize=(16,8))

df_train['Pclass'].value_counts().plot.bar(color=['black','silver','yellow'],ax=ax[0])

ax[0].set_title('Number Of Passengers By Pclass')

ax[0].set_ylabel('Count')

sns.countplot('Pclass',hue='Survived',data=df_train,ax=ax[1])

ax[1].set_title('Pclass:Survived vs Dead')

plt.show()
pd.crosstab([df_train.Sex,df_train.Survived],df_train.Pclass,margins=True)
sns.factorplot('Pclass','Survived',hue='Sex',data=df_train)

plt.show()
print('Oldest Passenger was of:',df_train['Age'].max(),'Years')

print('Youngest Passenger was of:',df_train['Age'].min(),'Years')

print('Average Age on the ship:',df_train['Age'].mean(),'Years')
fig, ax = plt.subplots(1, 1, figsize=(10, 5))

sns.kdeplot(df_train[df_train['Survived'] == 1]['Age'], ax=ax)

sns.kdeplot(df_train[df_train['Survived'] == 0]['Age'], ax=ax)

plt.legend(['Survived == 1', 'Survived == 0'])

plt.show()
plt.figure(figsize=(8, 6))

df_train['Age'][df_train['Pclass'] == 1].plot(kind='kde')

df_train['Age'][df_train['Pclass'] == 2].plot(kind='kde')

df_train['Age'][df_train['Pclass'] == 3].plot(kind='kde')



plt.xlabel('Age')

plt.title('Age Distribution within classes')

plt.legend(['1st Class', '2nd Class', '3rd Class'])
cummulate_survival_ratio = []

for i in range(1, 80):

    cummulate_survival_ratio.append(df_train[df_train['Age'] < i]['Survived'].sum() / len(df_train[df_train['Age'] < i]['Survived']))

    

plt.figure(figsize=(7, 7))

plt.plot(cummulate_survival_ratio)

plt.title('Survival rate change depending on range of Age', y=1.02)

plt.ylabel('Survival rate')

plt.xlabel('Range of Age(0~x)')

plt.show()
f,ax=plt.subplots(1,2,figsize=(18,8))

sns.violinplot("Pclass","Age", hue="Survived", data=df_train,split=True,ax=ax[0])

ax[0].set_title('Pclass and Age vs Survived')

ax[0].set_yticks(range(0,110,10))

sns.violinplot("Sex","Age", hue="Survived", data=df_train,split=True,ax=ax[1])

ax[1].set_title('Sex and Age vs Survived')

ax[1].set_yticks(range(0,110,10))

plt.show()
df_train['Title'] = 0

for salut in df_train:

    df_train['Title'] = df_train.Name.str.extract('([A-Za-z]+)\.')

    

df_test['Title'] = 0

for salut in df_test:

    df_test['Title'] = df_test.Name.str.extract('([A-Za-z]+)\.')  
pd.crosstab(df_train['Title'], df_train['Sex'])

#Sex와 관련된 이니셜(Initials) 체크
df_train['Title'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don'],['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr'],inplace=True)

df_test['Title'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don'],['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr'],inplace=True)
data_df = df_train.append(df_test)



titles = ['Dr', 'Master', 'Miss', 'Mr', 'Mrs', 'Rev']

for title in titles:

    age_to_impute = data_df.groupby('Title')['Age'].median()[titles.index(title)]

    data_df.loc[(data_df['Age'].isnull()) & (data_df['Title'] == title), 'Age'] = age_to_impute
# TRAIN_DF and TEST_DF에서 Age 값 대체:

df_train['Age'] = data_df['Age'][:891]

df_test['Age'] = data_df['Age'][891:]
df_train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
df_train.Age.isnull().any()
df_test.Age.isnull().any()
f,ax=plt.subplots(1,2,figsize=(20,10))

df_train[df_train['Survived']==0].Age.plot.hist(ax=ax[0],bins=20,edgecolor='black',color='red')

ax[0].set_title('Survived= 0')

x1=list(range(0,85,5))

ax[0].set_xticks(x1)

df_train[df_train['Survived']==1].Age.plot.hist(ax=ax[1],color='green',bins=20,edgecolor='black')

ax[1].set_title('Survived= 1')

x2=list(range(0,85,5))

ax[1].set_xticks(x2)

plt.show()
sns.factorplot('Pclass','Survived',col='Title',data=df_train)

plt.show()
pd.crosstab([df_train.Embarked,df_train.Pclass],[df_train.Sex,df_train.Survived],margins=True)
f, ax = plt.subplots(1, 1, figsize=(7, 7))

df_train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=True).mean().sort_values(by='Survived', ascending=False).plot.bar(ax=ax)
f,ax=plt.subplots(2,2,figsize=(20,12))

sns.countplot('Embarked',data=df_train,ax=ax[0,0])

ax[0,0].set_title('No. Of Passengers Boarded')

sns.countplot('Embarked',hue='Sex',data=df_train,ax=ax[0,1])

ax[0,1].set_title('Male-Female Split for Embarked')

sns.countplot('Embarked',hue='Survived',data=df_train,ax=ax[1,0])

ax[1,0].set_title('Embarked vs Survived')

sns.countplot('Embarked',hue='Pclass',data=df_train,ax=ax[1,1])

ax[1,1].set_title('Embarked vs Pclass')

plt.subplots_adjust(wspace=0.2,hspace=0.5)

plt.show()
df_train['Embarked'].fillna('S',inplace=True)
df_train.Embarked.isnull().any()
pd.crosstab([df_train.SibSp],df_train.Survived)
fig, ax = plt.subplots(figsize=(20, 15))



df_train.groupby(['SibSp', 'Survived']).size().unstack().plot(kind='bar', stacked=False, ax=ax)

ax.set_title('SibSp vs Survived - Count - Side by Side')

plt.show()
f,ax=plt.subplots(1,2,figsize=(20,8))

sns.barplot('SibSp','Survived',data=df_train,ax=ax[0])

ax[0].set_title('SibSp vs Survived')

sns.factorplot('SibSp','Survived',data=df_train,ax=ax[1])

ax[1].set_title('SibSp vs Survived')

plt.close(2)

plt.show()
pd.crosstab(df_train.SibSp,df_train.Pclass)
pd.crosstab(df_train.Parch,df_train.Pclass)
f, ax = plt.subplots(figsize=(18, 10))

df_train.groupby(['Parch', 'Survived']).size().unstack().plot(kind='bar', stacked=False, ax=ax)

ax.set_title('Parch vs Survived - Count - Side by Side')

plt.show()
f,ax=plt.subplots(1,2,figsize=(20,8))

sns.barplot('Parch','Survived',data=df_train,ax=ax[0])

ax[0].set_title('Parch vs Survived')

sns.factorplot('Parch','Survived',data=df_train,ax=ax[1])

ax[1].set_title('Parch vs Survived')

plt.close(2)

plt.show()
df_train['FamilySize'] = df_train['SibSp'] + df_train['Parch'] + 1 # 자신을 포함해야하니 1을 더합니다

df_test['FamilySize'] = df_test['SibSp'] + df_test['Parch'] + 1 # 자신을 포함해야하니 1을 더합니다
print("Maximum size of Family: ", df_train['FamilySize'].max())

print("Minimum size of Family: ", df_train['FamilySize'].min())
f,ax=plt.subplots(1, 3, figsize=(40,10))

sns.countplot('FamilySize', data=df_train, ax=ax[0])

ax[0].set_title('(1) No. Of Passengers Boarded', y=1.02)



sns.countplot('FamilySize', hue='Survived', data=df_train, ax=ax[1])

ax[1].set_title('(2) Survived countplot depending on FamilySize',  y=1.02)



df_train[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=True).mean().sort_values(by='Survived', ascending=False).plot.bar(ax=ax[2])

ax[2].set_title('(3) Survived rate depending on FamilySize',  y=1.02)



plt.subplots_adjust(wspace=0.2, hspace=0.5)

plt.show()
print('Highest Fare was:',df_train['Fare'].max())

print('Lowest Fare was:',df_train['Fare'].min())

print('Average Fare was:',df_train['Fare'].mean())
f,ax=plt.subplots(1,3,figsize=(20,8))

sns.distplot(df_train[df_train['Pclass']==1].Fare,ax=ax[0])

ax[0].set_title('Fares in Pclass 1')

sns.distplot(df_train[df_train['Pclass']==2].Fare,ax=ax[1])

ax[1].set_title('Fares in Pclass 2')

sns.distplot(df_train[df_train['Pclass']==3].Fare,ax=ax[2])

ax[2].set_title('Fares in Pclass 3')

plt.show()
df_train.head()
df_train['Ticket'].value_counts()
sns.heatmap(df_train.corr(),annot=True,cmap='RdYlGn',linewidths=0.2) #df_train.corr()-->correlation matrix

fig=plt.gcf()

fig.set_size_inches(10,8)

plt.show()
df_train['Age_band']=0

df_train.loc[df_train['Age']<=20,'Age_band']=0

df_train.loc[(df_train['Age']>20)&(df_train['Age']<=40),'Age_band']=1

df_train.loc[(df_train['Age']>40)&(df_train['Age']<=60),'Age_band']=2

df_train.loc[df_train['Age']>60,'Age_band']=3

df_train.head(2)
df_test['Age_band']=0

df_test.loc[df_test['Age']<=20,'Age_band']=0

df_test.loc[(df_test['Age']>20)&(df_test['Age']<=40),'Age_band']=1

df_test.loc[(df_test['Age']>40)&(df_test['Age']<=60),'Age_band']=2

df_test.loc[df_test['Age']>60,'Age_band']=3

df_test.head(2)
df_train['Age_band'].value_counts().to_frame()
sns.factorplot('Age_band','Survived',data=df_train,col='Pclass')

plt.show()
df_train['Family_Size']=0

df_train['Family_Size']=df_train['Parch']+df_train['SibSp']#family size

df_train['Alone']=0

df_train.loc[df_train.Family_Size==0,'Alone']=1#Alone



df_test['Family_Size']=0

df_test['Family_Size']=df_test['Parch']+df_test['SibSp']#family size

df_test['Alone']=0

df_test.loc[df_test.Family_Size==0,'Alone']=1#Alone







f,ax=plt.subplots(1,2,figsize=(18,6))

sns.factorplot('Family_Size','Survived',data=df_train,ax=ax[0])

ax[0].set_title('Family_Size vs Survived')

sns.factorplot('Alone','Survived',data=df_train,ax=ax[1])

ax[1].set_title('Alone vs Survived')

plt.close(2)

plt.close(3)

plt.show()
df_train['Fare_Range']=pd.qcut(df_train['Fare'],10)

df_test['Fare_Range']=pd.qcut(df_test['Fare'],10)

df_train.groupby(['Fare_Range'])['Survived'].mean().to_frame()
df_train['Fare_cat']=0

df_train.loc[df_train['Fare']<=7.55,'Fare_cat']=0

df_train.loc[(df_train['Fare']>7.55)&(df_train['Fare']<=7.854),'Fare_cat']=1

df_train.loc[(df_train['Fare']>7.854)&(df_train['Fare']<=8.05),'Fare_cat']=2

df_train.loc[(df_train['Fare']>8.05)&(df_train['Fare']<=10.5),'Fare_cat']=3

df_train.loc[(df_train['Fare']>10.5)&(df_train['Fare']<=14.454),'Fare_cat']=4

df_train.loc[(df_train['Fare']>14.454)&(df_train['Fare']<=21.679),'Fare_cat']=5

df_train.loc[(df_train['Fare']>21.679)&(df_train['Fare']<=27.0),'Fare_cat']=6

df_train.loc[(df_train['Fare']>27.0)&(df_train['Fare']<=39.688),'Fare_cat']=7

df_train.loc[(df_train['Fare']>39.688)&(df_train['Fare']<=77.958),'Fare_cat']=8

df_train.loc[(df_train['Fare']>77.958)&(df_train['Fare']<=513),'Fare_cat']=9
df_test['Fare_cat']=0

df_test.loc[df_test['Fare']<=7.55,'Fare_cat']=0

df_test.loc[(df_test['Fare']>7.55)&(df_test['Fare']<=7.854),'Fare_cat']=1

df_test.loc[(df_test['Fare']>7.854)&(df_test['Fare']<=8.05),'Fare_cat']=2

df_test.loc[(df_test['Fare']>8.05)&(df_test['Fare']<=10.5),'Fare_cat']=3

df_test.loc[(df_test['Fare']>10.5)&(df_test['Fare']<=14.454),'Fare_cat']=4

df_test.loc[(df_test['Fare']>14.454)&(df_test['Fare']<=21.679),'Fare_cat']=5

df_test.loc[(df_test['Fare']>21.679)&(df_test['Fare']<=27.0),'Fare_cat']=6

df_test.loc[(df_test['Fare']>27.0)&(df_test['Fare']<=39.688),'Fare_cat']=7

df_test.loc[(df_test['Fare']>39.688)&(df_test['Fare']<=77.958),'Fare_cat']=8

df_test.loc[(df_test['Fare']>77.958)&(df_test['Fare']<=513),'Fare_cat']=9
sns.factorplot('Fare_cat','Survived',data=df_train,hue='Sex')

plt.show()
df_train['Sex'].replace(['male','female'],[0,1],inplace=True)

df_train['Embarked'].replace(['S','C','Q'],[0,1,2],inplace=True)

df_train['Title'].replace(['Mr','Mrs','Miss','Master','Other'],[0,1,2,3,4],inplace=True)
df_test['Sex'].replace(['male','female'],[0,1],inplace=True)

df_test['Embarked'].replace(['S','C','Q'],[0,1,2],inplace=True)

df_test['Title'].replace(['Mr','Mrs','Miss','Master','Other'],[0,1,2,3,4],inplace=True)
df_train.drop(['Name','Age','Ticket','Fare','Cabin','Fare_Range','PassengerId'],axis=1,inplace=True)
df_test.drop(['Name','Age','Ticket','Fare','Cabin','Fare_Range','PassengerId'],axis=1,inplace=True)
df_train.head()
sns.heatmap(df_train.corr(),annot=True,cmap='RdYlGn',linewidths=0.2,annot_kws={'size':20})

fig=plt.gcf()

fig.set_size_inches(18,15)

plt.xticks(fontsize=14)

plt.yticks(fontsize=14)

plt.show()
from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import StandardScaler
df_train.head()
all_features = ['Pclass', 'Sex', 'FamilySize', 'Age_band', 'Fare_cat']
all_transformer = Pipeline(steps = [

    ('stdscaler', StandardScaler())

])
all_preprocess = ColumnTransformer(

    transformers = [

        ('allfeatures', all_transformer, all_features),

    ]

)
y = df_train['Survived']
X = df_train[df_train.columns[1:]]
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.3,random_state=0,stratify=df_train['Survived'])
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import SGDClassifier



from sklearn.model_selection import cross_val_score
classifiers = [

    LogisticRegression(random_state=42),

    RandomForestClassifier(random_state=42),

    SVC(random_state=42),

    KNeighborsClassifier(),

    SGDClassifier(random_state=42),

    ]
first_round_scores = {}

for classifier in classifiers:

    pipe = Pipeline(steps=[('preprocessor', all_preprocess),

                      ('classifier', classifier)])

    pipe.fit(X_train, y_train)   

    print(classifier)

    score = pipe.score(X_test, y_test)

    first_round_scores[classifier.__class__.__name__[:10]] = score

    print("model score: %.3f" % score)
# Plot the model scores

plt.plot(first_round_scores.keys(), first_round_scores.values(), "ro", markersize=10)

fig=plt.gcf()

fig.set_size_inches(8,5)

plt.title('Model Scores of the Classifiers - with no tuning ')

plt.show()
final_pipe = Pipeline(steps=[('preprocessor', all_preprocess)])
X_final_processed = final_pipe.fit_transform(X)
df_test_final_processed = final_pipe.transform(df_test)
knn_hyperparameters = {

    'n_neighbors': [6, 7, 8, 9, 10, 11, 12, 14, 16, 18, 20, 22],

    'algorithm' : ['auto'],

    'weights': ['uniform', 'distance'],

    'leaf_size': list(range(1,50,5)),

}



gd=GridSearchCV(estimator = KNeighborsClassifier(), param_grid = knn_hyperparameters,  

                cv=10, scoring = "roc_auc")



gd.fit(X_final_processed, y)

print(gd.best_score_)

print(gd.best_estimator_)
gd.best_estimator_.fit(X_final_processed, y)

y_pred = gd.best_estimator_.predict(df_test_final_processed)
knn = KNeighborsClassifier(algorithm='auto', leaf_size=26, metric='minkowski', 

                           metric_params=None, n_jobs=None, n_neighbors=6, p=2, 

                           weights='uniform')

knn.fit(X_final_processed, y)

X_pred = knn.predict(X_final_processed)

y_pred = knn.predict(df_test_final_processed)
submission = pd.DataFrame(pd.read_csv("../input/test.csv")['PassengerId'])

submission['Survived'] = y_pred

submission.to_csv("submission.csv", index = False)