# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

plt.style.use('seaborn-whitegrid')

import seaborn as sns



from collections import Counter



import warnings

warnings.filterwarnings('ignore')



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
plt.style.available
train_df = pd.read_csv('/kaggle/input/titanic/train.csv')

test_df = pd.read_csv('/kaggle/input/titanic/test.csv')

test_PassengerId = test_df['PassengerId']
train_df.columns
train_df.info()
train_df.describe()
train_df.head()
def bar_plot(variable):

    """

    input: variable ex: 'Sex'

    output: bar plot & value count

    """

    # get feature

    var = train_df[variable]

    # count number of categorical variable(value/sample)

    varValue = var.value_counts()

    

    # visualize

    plt.figure(figsize = (9,3))

    plt.bar(varValue.index, varValue)

    plt.xticks(varValue.index,varValue.index.values)

    plt.ylabel('Frequency')

    plt.title(variable)

    plt.show()

    print('{}: \n{}'.format(variable,varValue))

category1 = ['Survived','Sex','Pclass','Embarked','SibSp','Parch']

for c in category1:

    bar_plot(c)
category2 = ['Cabin','Name','Ticket']

for c in category2:

    print('{} \n'.format(train_df[c].value_counts()))
def plot_hist(variable):

    plt.figure(figsize=(9,3))

    plt.hist(train_df[variable],bins=50)

    plt.xlabel(variable)

    plt.ylabel('Frequency')

    plt.title('{} distribution with histogram'.format(variable))

    plt.show()

numericVar=['Fare','Age','PassengerId']

for n in numericVar:

    plot_hist(n)
train_df[["Pclass","Survived"]]
train_df[["Pclass","Survived"]].groupby(['Pclass'], as_index = False).mean()
# Pclass - Survived



train_df[["Pclass","Survived"]].groupby(['Pclass'], as_index = False).mean().sort_values(by='Survived',ascending = False)
# Sex - Survived



train_df[["Sex","Survived"]].groupby(['Sex'], as_index = False).mean().sort_values(by='Survived',ascending = False)
# SibSp - Survived



train_df[["SibSp","Survived"]].groupby(['SibSp'], as_index = False).mean().sort_values(by='Survived',ascending = False)
# Parch - Survived



train_df[["Parch","Survived"]].groupby(['Parch'], as_index = False).mean().sort_values(by='Survived',ascending = False)
train_df.corr()
# SibSp+Parch - Survived



train_df2 = train_df.copy()

train_df2['Relatives'] = train_df['SibSp'] + train_df['Parch']

train_df2[["Relatives","Survived"]].groupby(['Relatives'], as_index = False).mean().sort_values(by='Survived',ascending = False)
def outlier_detect(variable):

    Q1 = train_df.describe()[[variable]].loc['25%'].values[0]

    Q3 = train_df.describe()[[variable]].loc['75%'].values[0]

    IQR = Q3 - Q1

    lower_threshold = Q1 - 1.5*(Q3-Q1)

    upper_threshold = Q3 + 1.5*(Q3-Q1)

    filter1 = train_df[variable] < lower_threshold

    filter2 = train_df[variable] > upper_threshold

    return train_df[[variable]][filter1 | filter2]
outlier_detect('Fare')
def detect_outliers(df,features):

    outlier_indices = []

    

    for c in features:

        Q1 = np.percentile(df[c],25)

        Q3 = np.percentile(df[c],75)

        IQR = Q3-Q1

        outlier_step = IQR * 1.5

        outlier_list_col = df[(df[c] < (Q1 - outlier_step)) | (df[c] > (Q3 + outlier_step))].index

        outlier_indices.extend(outlier_list_col)

    

    outlier_indices = Counter(outlier_indices)

    multiple_outliers = list(i for i, v in outlier_indices.items() if v > 2)

    return multiple_outliers
train_df.loc[detect_outliers(train_df,['Age','SibSp','Parch','Fare'])]
def detect_outliers2(df,features):

    """

        More suitable for dataset includes Nan values

        The function above cannot find an outlier for Age column because Age column returns nan and nan for Q1 and Q3, respectively.

    """

    outlier_indices = []

    for c in features:

        outlier_list_col = outlier_detect(c).index

        outlier_indices.extend(outlier_list_col)

    

    outlier_indices = Counter(outlier_indices)

    multiple_outliers = list(i for i, v in outlier_indices.items() if v > 2)

    return multiple_outliers
train_df.loc[detect_outliers2(train_df,['Age','SibSp','Parch','Fare'])]
# dropping outliers



train_df = train_df.drop(detect_outliers2(train_df,['Age','SibSp','Parch','Fare']),axis=0).reset_index(drop = True)
train_df.loc[detect_outliers2(train_df,['Age','SibSp','Parch','Fare'])]#checking
train_df_len = len(train_df)

train_df = pd.concat([train_df,test_df],axis=0).reset_index(drop = True)
train_df.columns[train_df.isnull().any()]
train_df.isnull().sum()
# Embarked has 2 missing value



train_df[train_df['Embarked'].isnull()]
train_df.boxplot(column = 'Fare',by = 'Embarked')

plt.show()
train_df['Embarked'] = train_df['Embarked'].fillna('C')

train_df[train_df['Embarked'].isnull()]
test_df['Embarked'] = test_df['Embarked'].fillna('C')
train_df[train_df['Fare'].isnull()]
np.mean(train_df[train_df['Pclass'] == 3]['Fare'])
train_df['Fare'] = train_df['Fare'].fillna(12.741219971469327)

train_df[train_df['Fare'].isnull()]
list1 = ['SibSp','Parch','Age','Fare','Survived']

sns.heatmap(train_df[list1].corr(),annot = True,fmt = '.2f')
g = sns.factorplot(x = 'SibSp', y = 'Survived', data = train_df, kind = 'bar', size = 7)

g.set_ylabels('Survived Probability')

plt.show()
g = sns.factorplot(x = 'Parch', y = 'Survived', data = train_df, kind = 'bar', size = 7)# line means standard deviation

g.set_ylabels('Survived Probability')

plt.show()
g = sns.factorplot(x = 'Pclass', y = 'Survived', data = train_df, kind = 'bar', size = 7)# line means standard deviation

g.set_ylabels('Survived Probability')

plt.show()
g = sns.FacetGrid(train_df, col = 'Survived')

g.map(sns.distplot, 'Age', bins = 25)

plt.show()
g = sns.FacetGrid(train_df, col = 'Survived', row = 'Pclass')

g.map(plt.hist,'Age',bins = 25)

g.add_legend()

plt.show()
g = sns.FacetGrid(train_df, row = 'Embarked')

g.map(sns.pointplot,'Pclass','Survived','Sex')

g.add_legend()

plt.show()
g = sns.FacetGrid(train_df, row = 'Embarked', col = 'Survived',size = 2.5)

g.map(sns.barplot,'Sex','Fare')

g.add_legend()

plt.show()
train_df.isnull().sum()
train_df[train_df['Age'].isnull()]
sns.factorplot(x='Sex',y='Age',data=train_df,kind = 'box')

# Sex is not informative for age prediction
sns.factorplot(x='Sex',y='Age',hue = 'Pclass',data=train_df,kind = 'box')

# Pclass is a good feature to predict Age.
sns.factorplot(x='Parch',y='Age',hue = 'Pclass',data=train_df,kind = 'box')

sns.factorplot(x='SibSp',y='Age',hue = 'Pclass',data=train_df,kind = 'box')

train_df['New_sex'] = [1 if i == 'male' else 0 for i in train_df['Sex']]



test_df['New_sex'] = [1 if i == 'male' else 0 for i in test_df['Sex']]
sns.heatmap(train_df[['Age','New_sex','SibSp','Parch','Pclass']].corr(),annot = True, fmt = '.3f')

plt.show()

# Age is not correlated with Sex
index_nan_age = list(train_df.Age[train_df.Age.isnull()].index)

print(index_nan_age)
for i in index_nan_age:

    age_pred = train_df['Age'][(train_df['SibSp'] == train_df.iloc[i]['SibSp']) & (train_df['Parch'] == train_df.iloc[i]['Parch']) & (train_df['Pclass'] == train_df.iloc[i]['Pclass'])].median
age_pred
for i in index_nan_age:

    age_pred = train_df['Age'][(train_df['SibSp'] == train_df.iloc[i]['SibSp']) & (train_df['Parch'] == train_df.iloc[i]['Parch']) & (train_df['Pclass'] == train_df.iloc[i]['Pclass'])].median()

    age_med = train_df['Age'].median()

    if not np.isnan(age_pred):

        train_df['Age'].iloc[i] = age_pred

    else:

        train_df['Age'].iloc[i] = age_med
train_df.Age[index_nan_age]
train_df['Name'].head(5)
name = train_df['Name']

train_df['Title'] = [i.split('.')[0].split(',')[1].strip() for i in name]
test_df['Title'] = [i.split('.')[0].split(',')[1].strip() for i in test_df['Name']]
train_df['Title'].head(5)
sns.countplot(x='Title',data=train_df)

plt.xticks(rotation=45)

plt.show()
# covert to categorical



train_df['Title'] = train_df['Title'].replace(['Lady','the Countess','Capt','Don','Dr','Major','Rev','Sir','Jonkheer','Dona'],'other')
test_df['Title'] = test_df['Title'].replace(['Lady','the Countess','Capt','Don','Dr','Major','Rev','Sir','Jonkheer','Dona'],'other')
sns.countplot(x='Title',data=train_df)

plt.xticks(rotation=45)

plt.show()
train_df["Title"] = [0 if i == "Master" else 1 if i == "Miss" or i == "Ms" or i == "Mlle" or i == "Mrs" else 2 if i == "Mr" else 3 for i in train_df["Title"]]
test_df["Title"] = [0 if i == "Master" else 1 if i == "Miss" or i == "Ms" or i == "Mlle" or i == "Mrs" else 2 if i == "Mr" else 3 for i in test_df["Title"]]
sns.countplot(x='Title',data=train_df)

plt.xticks(rotation=45)

plt.show()
train_df["Title"].head(20)
g = sns.factorplot(x = 'Title', y = 'Survived', data = train_df, kind = 'bar')

g.set_xticklabels(['Master','Mrs','Mr','Other'])# In order

g.set_ylabels('Survival Probability')
train_df.drop(labels = ['Name'],axis=1,inplace=True)
test_df.drop(labels = ['Name'],axis=1,inplace=True)
train_df.head(6)
train_df = pd.get_dummies(train_df,columns=['Title'])#dummy variable

train_df.head()
test_df = pd.get_dummies(test_df,columns=['Title'])#dummy variable
train_df['Fsize'] = train_df['SibSp'] + train_df['Parch'] + 1#Family size including passenger
test_df['Fsize'] = test_df['SibSp'] + test_df['Parch'] + 1
train_df.head(5)
g = sns.factorplot(x='Fsize',y='Survived',data=train_df,kind='bar')

g.set_ylabels('Survival')

plt.show()
train_df['family_size_category'] = ['alone' if i == 1 else 'small' if (i == 2 or i == 3 or i == 4) else 'big' if i > 4 else 'unknown' for i in train_df['Fsize']]# Alone, small family, big family
test_df['family_size_category'] = ['alone' if i == 1 else 'small' if (i == 2 or i == 3 or i == 4) else 'big' if i > 4 else 'unknown' for i in test_df['Fsize']]
train_df.head(10)
g = sns.factorplot(x='family_size_category',y='Survived',data=train_df,kind='bar')

g.set_ylabels('Survival')

plt.show()
sns.countplot(x='family_size_category',data=train_df)

plt.show()
train_df = pd.get_dummies(train_df,columns=['family_size_category'],prefix = 'FS')
test_df = pd.get_dummies(test_df,columns=['family_size_category'],prefix = 'FS')
train_df = pd.get_dummies(train_df, columns=['Embarked'])
test_df = pd.get_dummies(test_df, columns=['Embarked'])
train_df.head(10)
train_df.corr().iloc[:,0:3]
train_df['Ticket'].head(20)
ticket = []

for i in list(train_df['Ticket']):

    if not i.isdigit():

        ticket.append(i.replace('.','').replace('/','').strip().split(' ')[0])

    else:

        ticket.append('x')

train_df['Ticket'] = ticket
ticket_t = []

for i in list(test_df['Ticket']):

    if not i.isdigit():

        ticket_t.append(i.replace('.','').replace('/','').strip().split(' ')[0])

    else:

        ticket_t.append('x')

test_df['Ticket'] = ticket_t
train_df.head(20)
train_df['Ticket'].unique()
train_df = pd.get_dummies(train_df,columns=['Ticket'], prefix = 'T')# use t instead of ticket word
test_df = pd.get_dummies(test_df,columns=['Ticket'], prefix = 'T')
train_df.head(20)
sns.countplot(x='Pclass',data=train_df)

plt.show()
train_df['Pclass'] = train_df['Pclass'].astype('category')

train_df = pd.get_dummies(train_df,columns=['Pclass'])

train_df.head(20)
test_df['Pclass'] = test_df['Pclass'].astype('category')

test_df = pd.get_dummies(test_df,columns=['Pclass'])
train_df.columns
train_df.drop(axis=1,labels=['Sex','PassengerId','Cabin','Fsize'],inplace=True)
test_df.drop(axis=1,labels=['Sex','PassengerId','Cabin','Fsize'],inplace=True)
train_df.columns
train_df.info()
test_df.columns
for i in train_df.columns:

    if i in test_df.columns:

        pass

    else:

        test_df[i] = 0
test_df.info()
test_df.columns
train_df = train_df.reindex(sorted(train_df.columns), axis=1)
test_df = test_df.reindex(sorted(test_df.columns), axis=1)
test_df.columns
train_df.columns
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier, VotingClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score
train_df.info()
test = train_df[train_df_len:]

test.drop(labels = ["Survived"],axis = 1, inplace = True)
train = train_df[:train_df_len]

x_train = train.drop(labels = "Survived", axis = 1)

y_train = train["Survived"]

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size = 0.33, random_state = 42)
log_reg = LogisticRegression()

log_reg.fit(x_train,y_train)

acc_log_train = log_reg.score(x_train,y_train)

acc_log_test = log_reg.score(x_test,y_test)

print('Train suitability : ',acc_log_train)

print('Test acc : ',acc_log_test)
random_state = 42

classifier = [DecisionTreeClassifier(random_state=random_state),

             SVC(random_state=random_state),

             RandomForestClassifier(random_state=random_state),

             LogisticRegression(random_state=random_state),

             KNeighborsClassifier()]



dt_param_grid = {'min_samples_split':range(10,500,20),

                'max_depth':range(1,20,2)}



svc_param_grid = {'kernel':['rbf'],

                 'gamma':[0.001,0.01,0.1,1],

                 'C':[1,10,50,100,200,300,1000]}



rf_param_grid = {'max_features':[1,3,10],

                'min_samples_split':[2,3,10],

                'min_samples_leaf':[1,3,10],

                'bootstrap':[False],

                'n_estimators':[100,300],

                'criterion':['gini']}



logreg_param_grid = {'C':np.logspace(-3,3,7),

                    'penalty':['l1','l2']}



knn_param_grid = {'n_neighbors': np.linspace(1,19,10, dtype = int).tolist(),

                 'weights':['uniform','distance'],

                 'metric':['euclidean','manhattan']}



classifier_param = [dt_param_grid,

                   svc_param_grid,

                   rf_param_grid,

                   logreg_param_grid,

                   knn_param_grid]



cv_results = []

best_estimators = []

for i in range(len(classifier)):

    clf = GridSearchCV(classifier[i],param_grid=classifier_param[i],cv = StratifiedKFold(n_splits = 10),scoring = 'accuracy', n_jobs = -1, verbose = 1)

    clf.fit(x_train,y_train)

    cv_results.append(clf.best_score_)

    best_estimators.append(clf.best_estimator_)

    print(cv_results[i])
cv_result = pd.DataFrame({'Cross Val Means': cv_results, 'ML models': ['DecisionTreeClassifier','SVM','RandomForestClassifier','LogisticRegression','KNeighborsClassifier']})



g = sns.barplot('Cross Val Means','ML models',data = cv_result)

g.set_xlabel('Mean Acc')

g.set_title('Cross Val Scores')

plt.show()
votingC = VotingClassifier(estimators = [('dt',best_estimators[0]),('rf',best_estimators[2]),('logreg',best_estimators[3])],voting = 'soft', n_jobs = -1)# soft and hard/ posibily, direct

# Using models at the same time

votingC = votingC.fit(x_train,y_train)

print(accuracy_score(votingC.predict(x_test),y_test))
test_survived = pd.Series(votingC.predict(test), name = "Survived").astype(int)

results = pd.concat([test_PassengerId, test_survived],axis = 1)

results.to_csv("titanic.csv", index = False)