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
df_train['Sex'].value_counts()
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
df_train.head()
df_train['Ticket'].value_counts()
sns.heatmap(df_train.corr(),annot=True,cmap='RdYlGn',linewidths=0.2) #df_train.corr()-->correlation matrix

fig=plt.gcf()

fig.set_size_inches(10,8)

plt.show()
df_train['Embarked'].value_counts()
df_train['Embarked'].fillna('S', inplace=True)
df_train['Embarked'].value_counts()
df_train[['Embarked','Survived']].groupby(['Embarked'], as_index=False).mean()
pd.crosstab(df_train['Embarked'],df_train['Survived'])
sns.catplot('Embarked','Survived', data=df_train, kind='point')

fig=plt.gcf()

fig.set_size_inches(5,3)

plt.show()

pd.crosstab(df_train['Embarked'],df_train['Pclass'])
f, ax = plt.subplots(1,2, figsize=(18, 6))

df_train.groupby(['Embarked', 'Survived']).size().unstack().plot(kind='bar', stacked=False, ax=ax[0])

df_train.groupby(['Embarked', 'Pclass']).size().unstack().plot(kind='bar', stacked=False, ax=ax[1])

ax[0].set_title('Embarked vs Survived')

ax[1].set_title('Embarked vs Pclass')

plt.show()
sns.catplot('Pclass', 'Survived', hue='Sex', col='Embarked', data=df_train, kind='point')

plt.show()
df_train['FamilySize'] = df_train['SibSp'] + df_train['Parch']

df_test['FamilySize'] = df_test['SibSp'] + df_test['Parch']
df_train[['FamilySize','Survived']].groupby(['FamilySize'], as_index=False).mean()
pd.crosstab(df_train['FamilySize'], df_train['Survived'])
f, ax = plt.subplots(1, 2, figsize=(18, 5))

sns.barplot('FamilySize', 'Survived', data=df_train, ax=ax[0])

ax[0].set_title('Family Size vs Survived')

sns.catplot('FamilySize', 'Survived', data=df_train, ax=ax[1], kind='point')

ax[0].set_title('Family Size vs Survived')

plt.close(2)

plt.show()
from sklearn.preprocessing import LabelEncoder
df_test['Fare'].fillna(df_test['Fare'].median(), inplace = True)



df_train['FareCategory'] = pd.qcut(df_train['Fare'], 5)

df_test['FareCategory'] = pd.qcut(df_test['Fare'], 5)
df_train['Fare_Code'] = LabelEncoder().fit_transform(df_train['FareCategory'])
df_test['Fare_Code'] = LabelEncoder().fit_transform(df_test['FareCategory'])
df_train[['Fare_Code', 'Survived']].groupby(['Fare_Code']).mean()
pd.crosstab(df_train['Fare_Code'], df_train['Survived'])
f, ax = plt.subplots(1,2, figsize=(18, 6))

df_train.groupby(['Fare_Code','Survived']).size().unstack().plot(kind='bar', ax=ax[0])

sns.catplot('Fare_Code', 'Survived', data=df_train, kind='bar', ax=ax[1])

ax[0].set_title('Fare_Code vs Survived')

ax[1].set_title('Fare_Code vs Survived')

plt.close(2)

plt.show()


df_train['Title'] = 0

for salut in df_train:

    df_train['Title'] = df_train.Name.str.extract('([A-Za-z]+)\.')

    

df_test['Title'] = 0

for salut in df_test:

    df_test['Title'] = df_test.Name.str.extract('([A-Za-z]+)\.')  
pd.crosstab(df_train['Title'], df_train['Sex'])
mapping = {'Mlle': 'Miss', 

           'Major': 'Mr', 

           'Col': 'Mr', 

           'Sir': 'Mr', 

           'Don': 'Mr',

           'Mme': 'Miss',

           'Jonkheer': 'Mr',

           'Lady': 'Mrs', 

           'Capt': 'Mr', 

           'Countess': 'Mrs', 

           'Ms': 'Miss', 

           'Dona': 'Mrs'

           }

df_train.replace({'Title': mapping}, inplace=True)

df_test.replace({'Title': mapping}, inplace=True)
data_df = df_train.append(df_test)



titles = ['Dr', 'Master', 'Miss', 'Mr', 'Mrs', 'Rev']

for title in titles:

    age_to_impute = data_df.groupby('Title')['Age'].median()[titles.index(title)]

    data_df.loc[(data_df['Age'].isnull()) & (data_df['Title'] == title), 'Age'] = age_to_impute    
# TRAIN_DF and TEST_DF에서 Age값 대체하기:

df_train['Age'] = data_df['Age'][:891]

df_test['Age'] = data_df['Age'][891:]
df_train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
plt.style.use('fivethirtyeight')

sns.catplot('Pclass','Survived', col='Title', data=df_train, kind='point')

plt.show()
plt.style.use('default')
df_train['AgeCategory'] = pd.qcut(df_train['Age'], 4)

df_test['AgeCategory'] = pd.qcut(df_test['Age'], 4)
df_train['Age_Code'] = LabelEncoder().fit_transform(df_train['AgeCategory'])

df_test['Age_Code'] = LabelEncoder().fit_transform(df_test['AgeCategory'])
data_df['Last_Name'] = data_df['Name'].apply(lambda x: str.split(x, ",")[0])



DEFAULT_SURVIVAL_VALUE = 0.5

data_df['Family_Survival'] = DEFAULT_SURVIVAL_VALUE



for grp, grp_df in data_df[['Survived','Name', 'Last_Name', 'Fare', 'Ticket', 'PassengerId',

                           'SibSp', 'Parch', 'Age', 'Cabin']].groupby(['Last_Name', 'Fare']):

    

    if (len(grp_df) != 1):

        # A Family group is found.

        for ind, row in grp_df.iterrows():

            smax = grp_df.drop(ind)['Survived'].max()

            smin = grp_df.drop(ind)['Survived'].min()

            passID = row['PassengerId']

            if (smax == 1.0):

                data_df.loc[data_df['PassengerId'] == passID, 'Family_Survival'] = 1

            elif (smin==0.0):

                data_df.loc[data_df['PassengerId'] == passID, 'Family_Survival'] = 0

                

print("Number of passengers with family survival information:", 

      data_df.loc[data_df['Family_Survival']!=0.5].shape[0])
for _, grp_df in data_df.groupby('Ticket'):

    if (len(grp_df) != 1):

        for ind, row in grp_df.iterrows():

            if (row['Family_Survival'] == 0) | (row['Family_Survival']== 0.5):

                smax = grp_df.drop(ind)['Survived'].max()

                smin = grp_df.drop(ind)['Survived'].min()

                passID = row['PassengerId']

                if (smax == 1.0):

                    data_df.loc[data_df['PassengerId'] == passID, 'Family_Survival'] = 1

                elif (smin==0.0):

                    data_df.loc[data_df['PassengerId'] == passID, 'Family_Survival'] = 0

                        

print("Number of passenger with family/group survival information: " 

      +str(data_df[data_df['Family_Survival']!=0.5].shape[0]))



# # Family_Survival in TRAIN_DF and TEST_DF:

df_train['Family_Survival'] = data_df['Family_Survival'][:891]

df_test['Family_Survival'] = data_df['Family_Survival'][891:]
df_train['Sex'].replace(['male','female'],[0,1],inplace=True)

df_test['Sex'].replace(['male','female'],[0,1],inplace=True)
drop_elements = ['PassengerId', 'Name', 'SibSp', 'Parch','Ticket', 'Cabin', 'FareCategory', 'AgeCategory','Age', 'Fare', 'Title', 'Embarked']
df_train = df_train.drop(drop_elements, axis=1)

df_test = df_test.drop(drop_elements, axis=1)
y = df_train['Survived']
X = df_train[df_train.columns[1:]]
df_train.head()
from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import StandardScaler
all_features = ['Pclass', 'Sex', 'FamilySize', 'Fare_Code', 'Age_Code', 'Family_Survival']
all_transformer = Pipeline(steps = [

    ('stdscaler', StandardScaler())

])
all_preprocess = ColumnTransformer(

    transformers = [

        ('allfeatures', all_transformer, all_features),

    ]

)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=df_train['Survived'])
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
grid_scores = {}

log_clf = Pipeline(steps=[('preprocessor', all_preprocess),

                      ('classifier', LogisticRegression(random_state=42))])



log_param_grid = {

    'classifier__C': [0.01, 0.1, 1.0, 10],

    'classifier__solver' : ['liblinear','lbfgs','sag', 'saga'],

    'classifier__max_iter' : [500],

}



log_grid_search = GridSearchCV(log_clf, log_param_grid, cv=10, iid=True)

log_grid_search.fit(X_train, y_train)



print('Logistic Regression - grid_search.best_params_ and best_scores_', log_grid_search.best_params_, log_grid_search.best_score_)

log_model_score = log_grid_search.score(X_test, y_test)

print("Logistic Regression - model score: ", log_model_score)

grid_scores['log'] = log_model_score

rf_clf = Pipeline(steps=[('preprocessor', all_preprocess),

                      ('classifier', RandomForestClassifier(random_state=42))])



rf_param_grid = {

    'classifier__n_estimators' : [50, 100],

    'classifier__max_features' : [2, 3],

    'classifier__criterion' : ['gini', 'entropy']

}



rf_grid_search = GridSearchCV(rf_clf, rf_param_grid, cv=10, iid=True)

rf_grid_search.fit(X_train, y_train)



print('Random Forest grid_search.best_params_ and best_scores_', rf_grid_search.best_params_, rf_grid_search.best_score_)

rf_model_score = rf_grid_search.score(X_test, y_test)

print("Random Forest - model score: ", rf_model_score)

grid_scores['rf'] = rf_model_score
svm_clf = Pipeline(steps=[('preprocessor', all_preprocess),

                      ('classifier', SVC(random_state=42))])

svm_param_grid = [

    {'classifier__kernel': ['linear'], 'classifier__C': [10., 30., 100., 300.]},

    {'classifier__kernel': ['rbf'], 'classifier__C': [1.0, 3.0, 10., 30., 100., 300.],

     'classifier__gamma': [0.01, 0.03, 0.1, 0.3, 1.0]},

    ] 



svm_grid_search = GridSearchCV(svm_clf, svm_param_grid, cv=10, iid=True)

svm_grid_search.fit(X_train, y_train)



print('SVM grid_search.best_params_ and best_scores_', svm_grid_search.best_params_, svm_grid_search.best_score_)

svm_model_score = svm_grid_search.score(X_test, y_test)

print("SVM - model score: ", svm_model_score)

grid_scores['svm'] = svm_model_score
knn_clf = Pipeline(steps=[('preprocessor', all_preprocess),

                      ('classifier', KNeighborsClassifier())])

knn_param_grid = {

    'classifier__n_neighbors': [5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 18, 20, 22, 24, 26 ],

    'classifier__weights': ['uniform', 'distance' ],

    'classifier__leaf_size': list(range(1,50,5)),

}



knn_grid_search = GridSearchCV(knn_clf, knn_param_grid, cv=10, iid=True, scoring ='roc_auc')

knn_grid_search.fit(X_train, y_train)



print('KNN grid_search.best_params_ and best_scores_', knn_grid_search.best_params_, knn_grid_search.best_score_)

knn_model_score = knn_grid_search.score(X_test, y_test)

print("KNN - model score: ", knn_model_score)

grid_scores['knn'] = knn_model_score
sgd_clf = Pipeline(steps=[('preprocessor', all_preprocess),

                      ('classifier', SGDClassifier(random_state=42))])



sgd_param_grid = {

    'classifier__max_iter': [100, 200],

    'classifier__alpha': [0.0001, 0.001, 0.01, 0.1],

}



sgd_grid_search = GridSearchCV(sgd_clf, sgd_param_grid, cv=10, iid=True)

sgd_grid_search.fit(X_train, y_train)



print('SGD grid_search.best_params_ and best_scores_', sgd_grid_search.best_params_, sgd_grid_search.best_score_)

sgd_model_score = sgd_grid_search.score(X_test, y_test)

print("SGD - model score: ", sgd_model_score)

grid_scores['sgd'] = sgd_model_score
plt.plot(grid_scores.keys(), grid_scores.values(), "ro", markersize=10)

fig=plt.gcf()

fig.set_size_inches(8,5)

plt.title('Model Scores of the Classifiers after hyperparameter tuning')

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

y_pred = knn.predict(df_test_final_processed)
submission = pd.DataFrame(pd.read_csv("../input/test.csv")['PassengerId'])

submission['Survived'] = y_pred

submission.to_csv("submission.csv", index = False)