# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import matplotlib.pyplot as plt

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import tree

from sklearn import ensemble

from sklearn.model_selection import GridSearchCV

from sklearn import linear_model

import re

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session



#Read all the csv's

df_train=pd.read_csv('/kaggle/input/titanic/train.csv')

df_example=pd.read_csv('/kaggle/input/titanic/gender_submission.csv')

df_test=pd.read_csv('/kaggle/input/titanic/test.csv')
df_train.head()
##Check if we got any nulls up in here.

##Cabin and Age have some nulls - we'll need to impute them.

df_train.isnull().sum()
print(df_train['Survived'].value_counts())

print(df_train['Survived'].value_counts()/df_train['Survived'].count())
#Ticket Class

print(df_train['Pclass'].value_counts())

print(df_train[['Pclass','Survived']].value_counts().sort_index())
#Age - Significant bunching around 20-30

df_train['Age'].plot.hist()
# Sex - More Male than female. Male have 81.1% death rate. Female have 25.7% death rate.

print(df_train[['Sex']].value_counts())

print(df_train[['Sex','Survived']].value_counts())

df_train[['Sex','Survived']].value_counts()
# Sibsp - Number of Siblings or Spouses aboard - Large Proportion didn't have any.

print(df_train['SibSp'].value_counts())

df_train['SibSp'].plot.hist()
#Parch - Number of Parents and children onboard. Alot of people didn't have any

print(df_train['Parch'].value_counts())

df_train['Parch'].plot.hist()
#Ticket - Ticket Number. Looks like it's all Unique. However, there seems to be prefixes to some of the Ticket Numbers. 

# We can probably introduce two new features - one if the Ticket Number has a letter and another based on the length of the ticket number.

df_train['Ticket'].unique()

# df_train['Ticket'].str.len()

# df_train['Ticket'].str.isnumeric().astype(int)
#Fare. Fare price passenger paid. Big proportion of those below 20 bucks.

print(df_train['Fare'].value_counts().sort_index())

df_train['Fare'].plot.hist(bins=20)
#Cabin. Cabin Number. This one had heaps of NaN's - i'm assuming those without cabins. Also alot of unique values. 

#There are also multi cabins. There's a bunch of way's we can split this: by alphabet, NaN vs. not NaN, by number, multi-cabins...

df_train['Cabin'].unique()

# df_train['Cabin'].isna().astype(int)
# Embarked - Point of Embarkation for the passenger. Heavily weighted towards Southampth. C = Cherbourg, Q = Queenstown, S = Southampton

df_train['Embarked'].value_counts()
#returns the Title within the name columns

def get_title(name):

    title_search = re.search(' ([A-Za-z]+)\.', name)

    # If the title exists, extract and return it.

    if title_search:

        return title_search.group(1)

    return ""
def splitage(age):

    if age<15:

        result='<15'

    elif age>=15:

        result='>=15'

    else:

        result='NULL'

    

    return result
def leftcabin(s):

    if s!=s:

        return 'NULL'

    return s[:1]

    
#This Function accepts the dataframe in the form of test/train.

def preprocess_X(df,trainortest):

    #Grab all the string categorical values and one hot encode them.

    #Gotta One-hot encode before putting into the Tree Algo.

    from sklearn.preprocessing import OrdinalEncoder

    

    #Ticket Feature Creation

    df['Ticket_str_length']=df['Ticket'].str.len()

    df['Ticket_is_numeric']=df['Ticket'].str.isnumeric().astype(int)

    #Cabin Number Feature Creation

    df['Cabin_isna']=df['Cabin'].isna().astype(int)

    df['Cabin_letter']=df['Cabin'].apply(leftcabin)   

    #Age Feature Creation

    df['agebin']=df['Age'].apply(splitage)



    df['Title']=df['Name'].apply(get_title)

    df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    df['Title'] = df['Title'].replace('Mlle', 'Miss')

    df['Title'] = df['Title'].replace('Ms', 'Miss')

    df['Title'] = df['Title'].replace('Mme', 'Mrs')

    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

    df['IsAlone'] = 0

    df.loc[df['FamilySize'] == 1, 'IsAlone'] = 1

    



    print(df['Title'].unique())

    col_string_list=['Sex', 'Embarked','Title','agebin','Cabin_letter']

    X2=df[col_string_list]

    X2=X2.where(pd.notnull(X2), 'None')

    encoder = OrdinalEncoder()

    encoder.fit(X2)

    X_encoded = encoder.transform(X2)

    df_encoded=pd.DataFrame(X_encoded,columns=col_string_list)

    

    #All the numeric stuff

    col_numeric_list=['Pclass', 'Age', 'SibSp',

        'Parch','FamilySize','IsAlone','Fare','Ticket_str_length','Ticket_is_numeric','Cabin_isna']

    if trainortest=='Train':

        col_numeric_list.append('Survived')      

    df_numeric=df[col_numeric_list]

    #Concatenate in the columns

#     print(df_numeric.head())

#     print(df_encoded.head())

    df_result=pd.concat([df_numeric,df_encoded],axis=1)

    df_result = df_result.fillna(-1)

    



    return df_result
# We should get cleaned data ready to place into our tree models

df_train_post=preprocess_X(df_train,'Train')

df_test_post=preprocess_X(df_test,'Test')

df_train_post
#Let's plot a Correlation Matrix.

def plot_corr(df,size=10):

    import seaborn as sns

    '''Function plots a graphical correlation matrix for each pair of columns in the dataframe.



    Input:

        df: pandas DataFrame

        size: vertical and horizontal size of the plot'''



    colormap = plt.cm.viridis

    plt.figure(figsize=(12,12))

    plt.title('Pearson Correlation of Features', y=1.05, size=15)

    sns.heatmap(df.corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)
plot_corr(df_train_post)
#Let's remove highly correlated metrics and leave one's with high magnitude

#This will help with model training faster and model interpretability.

main_features=['Pclass','IsAlone','Fare','Cabin_isna','Sex','Embarked','agebin','Title']

X=df_train_post.drop(['Survived'], axis=1) #[main_features]

Xtest=df_test_post #[main_features]

Y=df_train_post['Survived']
#Cross Validating our Decision Tree

from sklearn.model_selection import cross_val_score

depth = []

for i in range(1,20):

    clf = tree.DecisionTreeClassifier(max_depth=i)

    # Perform 7-fold cross validation 

    scores = cross_val_score(estimator=clf, X=X, y=Y, cv=10, n_jobs=4)

    depth.append((i,scores.mean()))

print(depth)
#Let's train our Decision Tree

clf = tree.DecisionTreeClassifier(max_depth=4)

#Seems to work with a dataframe

clf=clf.fit(X,Y)

plt.figure(figsize=(25,15))

tree.plot_tree(clf, feature_names=X.columns,class_names=True,filled=True,fontsize=12)
#Let's Predict 

submission_arr=clf.predict(df_test_post)

df_survived_output=pd.DataFrame(submission_arr, columns=['Survived'])

df_submission=pd.concat([df_test['PassengerId'],df_survived_output['Survived']], axis=1)

df_submission.to_csv('submissionDecisionTree.csv',index=False)

df_submission
from sklearn.pipeline import Pipeline

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.ensemble import BaggingClassifier



#Using Grid Search and Cross Validation in our Random Forest to get best hyperparameters



#The step in the pipeline doesn't matter because we'll replace them in the param_grid

pipe = Pipeline([ ('classifier', ensemble.ExtraTreesClassifier())])





param_grid = [

    {'classifier': [ensemble.RandomForestClassifier()],"classifier__criterion" : ["gini"], "classifier__min_samples_leaf" : [1, 2, 3], 

               "classifier__min_samples_split" : [2,3], 

               "classifier__n_estimators": [30,50,60,80], 'classifier__max_depth': [1,2,3,4,5,6]},

    {'classifier':[ensemble.BaggingClassifier()],"classifier__n_estimators": [20,50,100],

               'classifier__bootstrap' : [True, False],'classifier__bootstrap_features' : [True, False]},

    {'classifier': [linear_model.LogisticRegression()], 'classifier__penalty' : ['l1', 'l2'], 'classifier__C' : np.logspace(-4, 4, 20), 'classifier__solver' : ['liblinear']}

              ]



gs = GridSearchCV(estimator=pipe, param_grid=param_grid, cv=7)



gs = gs.fit(X, Y)





print(gs.best_score_)

print(gs.best_estimator_.get_params()['classifier'])



#Let's train our random forest using the best performing hyperparameters function

clf = gs.best_estimator_.get_params()['classifier']

#Seems to work with a dataframe

clf=clf.fit(X,Y)



#Let's Predict using our random forest

submission_arr=clf.predict(Xtest)

df_survived_output=pd.DataFrame(submission_arr, columns=['Survived'])

df_submission=pd.concat([df_test['PassengerId'],df_survived_output['Survived']], axis=1)

df_submission.to_csv('submissionOutput.csv',index=False)

df_submission