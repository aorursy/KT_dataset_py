import pandas as pd

import numpy as np



from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process

from xgboost import XGBClassifier

from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from sklearn import feature_selection

from sklearn import model_selection

from sklearn import metrics

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.linear_model import LogisticRegression, Ridge, LinearRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve

from sklearn.model_selection import train_test_split

from sklearn.svm import SVR

from sklearn.kernel_ridge import KernelRidge

from sklearn.feature_selection import SelectFromModel

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer



import matplotlib as mpl

import matplotlib.pyplot as plt

import matplotlib.pylab as pylab

import seaborn as sns

from pandas.tools.plotting import scatter_matrix

import string

import math

import sys



# disable warnings:

if not sys.warnoptions:

    import warnings

    warnings.simplefilter("ignore")

#axtract data into Dataframe

data_train = pd.read_csv("../input/train.csv")

data_test= pd.read_csv("../input/test.csv")
print("Training Data shape:", data_train.shape)

print("Test Data shape:", data_test.shape)
data_train.head()
data_test.head()
# extract target variable from train set

label = data_train['Survived']
data_train.dtypes
# check for unique values of labels:

label.unique()
#check for missing values:

if label.isnull().sum()==0:

    print("No missing values")

else:

    print(label.isnull().sum(), 'missing values found in dataset')
# Historgam

label.value_counts().plot.pie(autopct='%1.2f%%')
# Check info for train and test dataset

data_train.info()

print("----------------------------------")

data_test.info()
# check how many unique values each feature has:

for column in data_train.columns:

    print(column, len(data_train[column].unique()))
# check variable Age for missing values:

print('Amount of missing data in Fare for train:', data_train.Fare.isnull().sum())

print('Amount of missing data in Fare for test:',data_test.Fare.isnull().sum())

print("--------------------------------------------------")

# check variable Age for missing values:

print('Amount of missing data in Embarked for train:',data_train.Embarked.isnull().sum())

print('Amount of missing data in Embarked for test:', data_test.Embarked.isnull().sum())
data_train['Embarked'] = data_train['Embarked'].fillna("S") 

data_test['Fare'] = data_test['Fare'].fillna(data_train['Fare'].median())
data_train.Age.describe()
# check variable Age for missing values:

print(data_train.Age.isnull().sum())

print(data_test.Age.isnull().sum())
# 1. create feature to show rows with missing values of age:

data_train['Age_NA'] =np.where(data_train.Age.isnull(), 1, 0)

data_test['Age_NA'] =np.where(data_test.Age.isnull(), 1, 0)
# # visualize Age_NA vs survival rate

print(data_train["Age_NA"].value_counts())

sns.factorplot('Age_NA','Survived', data=data_train)
# 2. plot distribution of available Age vs survival rate

a = sns.FacetGrid(data_train, hue = 'Survived', aspect=4 )

a.map(sns.kdeplot, 'Age', shade= True )

a.set(xlim=(0 , data_train['Age'].max()))

a.add_legend()



print('Skew for train data:',data_train.Age.skew())
# create feature where missing age is imputed with mean of age values that are not missing

data_train['Age_mean'] =np.where(data_train.Age.isnull(), data_train['Age'].mean(), data_train['Age'])

data_test['Age_mean'] =np.where(data_test.Age.isnull(), data_test['Age'].mean(), data_test['Age'])

# plot distribution of available Age_mean vs survival rate

a = sns.FacetGrid(data_train, hue = 'Survived', aspect=4 )

a.map(sns.kdeplot, 'Age_mean', shade= True )

a.set(xlim=(0 , data_train['Age_mean'].max()))

a.add_legend()



print('Skew for train data:',data_train.Age.skew())
# check how many unique values each feature has:

for column in data_train.columns:

    print(column, len(data_train[column].unique()))
data_train = data_train.drop(['PassengerId'], axis=1)

data_test = data_test.drop(['PassengerId'], axis=1)
data_train.head()
data_train.Ticket[:10]
#create function that takes ticket feature and returns list of ticket_types

def ticket_sep(data_ticket):

    ticket_type = []



    for i in range(len(data_ticket)):



            ticket =data_ticket.iloc[i]



            for c in string.punctuation:

                ticket = ticket.replace(c,"")

                splited_ticket = ticket.split(" ")   

            if len(splited_ticket) == 1:

                ticket_type.append('NO')

            else: 

                ticket_type.append(splited_ticket[0])

    return ticket_type 
# for train data create new column with ticket_type:

data_train["ticket_type"] = ticket_sep(data_train.Ticket)



data_train.head()
# for test data create new column with ticket_type:

data_test["ticket_type"]= ticket_sep(data_test.Ticket)



data_test.head()
# check how many samples are there for each ticket type and visualize:

print(data_train["ticket_type"].value_counts())

sns.factorplot('ticket_type','Survived', data=data_train,size=4,aspect=3)

# for those types that have less than 15 samples in training set, assign type to 'OTHER':



for t in data_train['ticket_type'].unique():

    if len(data_train[data_train['ticket_type']==t]) < 15:

        data_train.loc[data_train.ticket_type ==t, 'ticket_type'] = 'OTHER_T'

       

    

for t in data_test['ticket_type'].unique():

    if t not in data_train['ticket_type'].unique():

        data_test.loc[data_test.ticket_type ==t, 'ticket_type'] = 'OTHER_T'

        

print(data_train['ticket_type'].unique())

print(data_test['ticket_type'].unique())
# visualize ticket_type vs survival rate

print(data_train["ticket_type"].value_counts()/len(data_train))

sns.barplot(x = 'ticket_type', y = 'Survived', data = data_train)
# where ticket_type is 'SOTONOQ' convert it to 'A5'

data_train["ticket_type"] = np.where(data_train["ticket_type"]=='SOTONOQ', 'A5', data_train["ticket_type"])

data_test["ticket_type"] = np.where(data_test["ticket_type"]=='SOTONOQ', 'A5', data_test["ticket_type"])

# visualize ticket_type vs survival rate

print(data_train["ticket_type"].value_counts()/len(data_train))

sns.barplot(x = 'ticket_type', y = 'Survived', data = data_train)
# drop Ticket from dataset:



data_train = data_train.drop(['Ticket'], axis=1)

data_test = data_test.drop(['Ticket'], axis=1)
print('Missing values in Train set:', data_train.Cabin.isnull().sum())

print('Missing values in Test set:', data_test.Cabin.isnull().sum())
data_train.Cabin[:10]
#create function that takes cabin from dataset and extracts cabin type for each cabin that is not missing.

# If cabin is missing, leaves missing value:



def cabin_sep(data_cabin):

    cabin_type = []



    for i in range(len(data_cabin)):



            if data_cabin.isnull()[i] == True: 

                cabin_type.append('NaN') 

            else:    

                cabin = data_cabin[i]

                cabin_type.append(cabin[:1]) 

            

    return cabin_type
# apply cabin sep on test and train set:

data_train['cabin_type'] = cabin_sep(data_train.Cabin)

data_test['cabin_type'] = cabin_sep(data_test.Cabin)





data_train.head()
# visualize cabin_type vs survival rate:

print(data_train["cabin_type"].value_counts())

sns.factorplot('cabin_type','Survived', data=data_train,size=4,aspect=3)

# for those types that have less than 15 samples in training set, assign type to 'OTHER_C':



for t in data_train['cabin_type'].unique():

    if len(data_train[data_train['cabin_type']==t]) <= 15:

        data_train.loc[data_train.cabin_type ==t, 'cabin_type'] = 'OTHER_C'

       

    

for t in data_test['cabin_type'].unique():

    if t not in data_train['cabin_type'].unique():

        data_test.loc[data_test.cabin_type ==t, 'cabin_type'] = 'OTHER_C'

        

print(data_train['cabin_type'].unique())

print(data_test['cabin_type'].unique())
# visualize cabin_type vs survival rate

print(data_train["cabin_type"].value_counts()/len(data_train))

sns.barplot(x = 'cabin_type', y = 'Survived', data = data_train)
# drop cabin from dataset:



data_train = data_train.drop(['Cabin'], axis=1)

data_test = data_test.drop(['Cabin'], axis=1)
data_train.Name[:10]
# Create function that take name and separates it into title, family name and deletes all puntuation from name column:

def name_sep(data):

    families=[]

    titles = []

    new_name = []

    #for each row in dataset:

    for i in range(len(data)):

        name = data.iloc[i]

        # extract name inside brakets into name_bracket:

        if '(' in name:

            name_no_bracket = name.split('(')[0] 

        else:

            name_no_bracket = name

            

        family = name_no_bracket.split(",")[0]

        title = name_no_bracket.split(",")[1].strip().split(" ")[0]

        

        #remove punctuations accept brackets:

        for c in string.punctuation:

            name = name.replace(c,"").strip()

            family = family.replace(c,"").strip()

            title = title.replace(c,"").strip()

            

        families.append(family)

        titles.append(title)

        new_name.append(name)

            

    return families, titles, new_name    
# apply name_sep on train and test set:

data_train['family'], data_train['title'], data_train['Name']  = name_sep(data_train.Name)

data_test['family'], data_test['title'], data_test['Name'] = name_sep(data_test.Name)



data_train.head()
# check how many samples we have for each title and visualize vs survival rate:

print(data_train["title"].value_counts())

sns.factorplot('title','Survived', data=data_train,size=4,aspect=3)
# for those types that have less than 15 samples in training set, assign type to 'OTHER':



for t in data_train['title'].unique():

    if len(data_train[data_train['title']==t]) <= 15:

        data_train.loc[data_train.title ==t, 'title'] = 'OTHER'

       

    

for t in data_test['title'].unique():

    if t not in data_train['title'].unique():

        data_test.loc[data_test.title ==t, 'title'] = 'OTHER'

        

print(data_train['title'].unique())

print(data_test['title'].unique())
# visualize title vs survival rate:

sns.barplot(x = 'title', y = 'Survived', data = data_train)

# amount of overlapping family names in train and test set:

len([x for x in data_train.family.unique() if x in data_test.family.unique()])
# amount of non overlapping with test set unique family names in train set:

len([x for x in data_train.family.unique() if x not in data_test.family.unique()])
# amount of non overlapping with train set unique family names in test set:

len([x for x in data_test.family.unique() if x not in data_train.family.unique()])
#create a list with all overlapping families

overlap = [x for x in data_train.family.unique() if x in data_test.family.unique()]
data_train.head()
# introduce new column to data called family_size:

data_train['family_size'] = data_train.SibSp + data_train.Parch +1

data_test['family_size'] = data_test.SibSp + data_test.Parch +1



# calculate survival rate for each family in train_set:

rate_family = data_train.groupby('family')['Survived', 'family','family_size'].median()

rate_family.head()
# if family size is more than 1 and family name is in overlap list 

overlap_family ={}

for i in range(len(rate_family)):

    if rate_family.index[i] in overlap and  rate_family.iloc[i,1] > 1:

        overlap_family[rate_family.index[i]] = rate_family.iloc[i,0]
mean_survival_rate = np.mean(data_train.Survived)

family_survival_rate = []

family_survival_rate_NA = []



for i in range(len(data_train)):

    if data_train.family[i] in overlap_family:

        family_survival_rate.append(overlap_family[data_train.family[i]])

        family_survival_rate_NA.append(1)

    else:

        family_survival_rate.append(mean_survival_rate)

        family_survival_rate_NA.append(0)

        

data_train['family_survival_rate']= family_survival_rate

data_train['family_survival_rate_NA']= family_survival_rate_NA
# repeat the same for test set:

mean_survival_rate = np.mean(data_train.Survived)

family_survival_rate = []

family_survival_rate_NA = []



for i in range(len(data_test)):

    if data_test.family[i] in overlap_family:

        family_survival_rate.append(overlap_family[data_test.family[i]])

        family_survival_rate_NA.append(1)

    else:

        family_survival_rate.append(mean_survival_rate)

        family_survival_rate_NA.append(0)

data_test['family_survival_rate']= family_survival_rate

data_test['family_survival_rate_NA']= family_survival_rate_NA
# drop name and family from dataset:

data_train = data_train.drop(['Name', 'family'], axis=1)

data_test = data_test.drop(['Name', 'family'], axis=1)



data_train.head()
sns.boxplot(data_train.Age)
sns.boxplot(data_train.Age_mean)
sns.boxplot(data_train.Fare)
print('Skew for Fare:',data_train.Fare.skew())

print('Skew for Age_mean:',data_train.Fare.skew())
# calculate upper bound for Fair

IQR = data_train.Fare.quantile(0.75) - data_train.Fare.quantile(0.25)

upper_bound = data_train.Fare.quantile(0.75) + 3*IQR

# for train and test sets convert all values in column Fair where age is more than upper_bound to upper_bound:

data_train.loc[data_train.Fare >upper_bound, 'Fare'] = upper_bound 

data_test.loc[data_test.Fare >upper_bound, 'Fare'] = upper_bound



max(data_train.Fare)
# calculate upper bound for Age_mean

IQR = data_train.Age_mean.quantile(0.75) - data_train.Age_mean.quantile(0.25)

upper_bound = data_train.Age_mean.quantile(0.75) + 3*IQR

# for train and test sets convert all values in column Fair where age is more than upper_bound to upper_bound:

data_train.loc[data_train.Age_mean >upper_bound, 'Age_mean'] = upper_bound 

data_test.loc[data_test.Age_mean >upper_bound, 'Age_mean'] = upper_bound



max(data_train.Age_mean)
# calculate upper bound for Age:

upper_bound = data_train.Age.mean() + 3* data_train.Age.std()

# for train and test sets convert all values in column Fair where age is more than upper_bound to upper_bound:

data_train.loc[data_train.Age >upper_bound, 'Age'] = upper_bound 

data_test.loc[data_test.Age >upper_bound, 'Age'] = upper_bound



max(data_train.Age)
# 1. check if all values from test set are in train set 

columns = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked']



for column in columns:

    print(column)

    print(data_train[column].unique())

    print(data_test[column].unique())

    
# check if all values from test set are in train set for family_size:

print(data_train['family_size'].unique())

print(data_test['family_size'].unique())
# visualize SibSp, Parch and family size vs survival rate:

sns.factorplot('SibSp','Survived', data=data_train)

sns.factorplot('Parch','Survived', data=data_train)

sns.factorplot('family_size','Survived', data=data_train)
# check family size for rare lables:

print(data_train["family_size"].value_counts()/len(data_train))
print('Pclass')

print(data_train["Pclass"].value_counts()/len(data_train))

print(data_test["Pclass"].value_counts()/len(data_train))

print("------------------------------")



print('Sex')

print(data_train["Sex"].value_counts()/len(data_train))

print(data_test["Sex"].value_counts()/len(data_train))

print("------------------------------")



print('Embarked')

print(data_train["Embarked"].value_counts()/len(data_train))

print(data_test["Embarked"].value_counts()/len(data_train))

print("------------------------------")
# combine train and test dataset

data = pd.concat([data_train.drop(['Survived'], axis=1), data_test], axis =0, sort = False)

data.head()
# encode variables into numeric labels

le = LabelEncoder()



columns = ['Sex', 'Embarked', 'ticket_type', 'cabin_type', 'title']



for col in columns:

    le.fit(data[col])

    data[col] = le.transform(data[col])

    

data.head()
# drop columns that have information about age or are strongly correlated with other features

data = data.drop(['Age_mean', 'Age_NA'], axis =1)
data.head()
sum(data.Age.isnull())
sns.pairplot(data, x_vars= ['Pclass', 'Sex','Fare','Embarked','ticket_type','cabin_type',\

                            'title', 'family_survival_rate'], y_vars='Age', size = 5, kind='reg')
colormap = plt.cm.RdBu

plt.figure(figsize=(14,8))

plt.title('Pearson Correlation of Features', y=1.05, size=15)

sns.heatmap(data.corr(),linewidths=0.1,vmax=1.0, 

            square=True,linecolor='white',cmap=colormap, annot=True)
data.head()
x_train_age = data.dropna().drop(['Age'], axis =1)

y_train_age = data.dropna()['Age']
x_test_age = data[pd.isnull(data.Age)].drop(['Age'], axis =1)
model_lin = make_pipeline(StandardScaler(),KernelRidge())

kfold = model_selection.KFold(n_splits=10, random_state=4, shuffle = True)

#model_lin.get_params().keys()

parameters = {'kernelridge__gamma' : [0.001, 0.01, 0.1, 1, 10, 100, 1000],

              'kernelridge__kernel': ['rbf', 'linear'],

               'kernelridge__alpha' :[0.001, 0.01, 0.1, 1, 10, 100, 1000],

              

             }

search_lin = GridSearchCV(model_lin, parameters, n_jobs = -1, cv = kfold, scoring = 'r2',verbose=1)

search_lin.fit(x_train_age, y_train_age)
print("Best parameters are:", search_lin.best_params_)

print("Best accuracy achieved:",search_lin.cv_results_['mean_test_score'].mean())
y_test_age = search_lin.predict(x_test_age)
data.loc[data['Age'].isnull(), 'Age'] = y_test_age
data_train.shape[0]
idx = int(data_train.shape[0])

data_train['Age'] = data.iloc[:idx].Age

data_test['Age'] = data.iloc[idx:].Age
data_train.head()
# plot distribution of available Age vs survival rate

a = sns.FacetGrid(data_train, hue = 'Survived', aspect=4 )

a.map(sns.kdeplot, 'Age', shade= True )

a.set(xlim=(0 , data_train['Age'].max()))

a.add_legend()
data_train.head()
# encode 'cabin_type' into numeric labels

le = LabelEncoder()

data_train_LE = data_train.copy()

data_test_LE = data_test.copy()



columns = ['Sex', 'Embarked', 'ticket_type', 'cabin_type', 'title']



for col in columns:

    le.fit(data_train_LE[col])

    data_train_LE[col] = le.fit_transform(data_train_LE[col])

    data_test_LE[col] = le.transform(data_test_LE[col])

    

data_train_LE.head()
plt.figure(figsize=(17,12))

plt.title('Pearson Correlation of Features', y=1.05, size=15)

sns.heatmap(data_train_LE.corr(),linewidths=0.1,vmax=1.0, 

            square=True,linecolor='white',cmap=colormap, annot=True)
data_train_LE.columns
for col in ['Pclass', 'Sex', 'SibSp', 'Parch','Embarked', 'Age_NA', 'ticket_type', 'cabin_type', 'title',

       'family_size', 'family_survival_rate', 'family_survival_rate_NA']:

    sns.barplot(x = col, y = 'Survived', data = data_train_LE)

    plt.show()
drop_col = ['Age_mean', 'SibSp', 'Parch']

data_train_LE = data_train_LE.drop(drop_col, axis=1)

data_test_LE = data_test_LE.drop(drop_col, axis=1)
plt.figure(figsize=(14,10))

plt.title('Pearson Correlation of Features', y=1.05, size=15)

sns.heatmap(data_train_LE.corr(),linewidths=0.1,vmax=1.0, 

            square=True,linecolor='white',cmap=colormap, annot=True)
plt.figure(figsize=(14,10))

plt.title('Pearson Correlation of Features', y=1.05, size=15)

sns.heatmap(data_test_LE.corr(),linewidths=0.1,vmax=1.0, 

            square=True,linecolor='white',cmap=colormap, annot=True)
X_train_onehot = data_train.drop(drop_col, axis=1)

X_test_onehot = data_test.drop(drop_col, axis=1)
X_train_onehot.head()
columns = ['cabin_type', 'title',  'Sex', 'Embarked', 'ticket_type', 'Pclass']



for col in columns:

    #X_train = pd.concat([X_train, pd.get_dummies(data_train[col])], axis =1)

    #X_test = pd.concat([X_test, pd.get_dummies(data_test[col])], axis =1)

    X_train_onehot = pd.concat([X_train_onehot, pd.get_dummies(X_train_onehot[col], drop_first = True)], axis =1)

    X_test_onehot = pd.concat([X_test_onehot, pd.get_dummies(X_test_onehot[col], drop_first = True)], axis =1)

    
X_train_onehot = X_train_onehot.drop(columns, axis=1)

X_test_onehot = X_test_onehot.drop(columns, axis=1)
X_train_onehot.head()
sns.clustermap(X_train_onehot.corr(),linewidths=0.1,vmax=1.0, 

            square=True,linecolor='white',cmap=colormap)
X_train_lab = data_train.drop(drop_col, axis=1)

X_test_lab = data_test.drop(drop_col, axis=1)
X_train_lab.head()
# encode 'cabin_type' into numeric labels

le = LabelEncoder()

columns = ['Sex', 'Embarked', 'ticket_type', 'cabin_type', 'title']



for col in columns:

    le.fit(data_train[col])

    X_train_lab[col] = le.transform(X_train_lab[col])

    X_test_lab[col] = le.transform(X_test_lab[col])

    

X_test_lab.head()
sns.clustermap(X_train_lab.corr(),linewidths=0.1,vmax=1.0, 

            square=True,linecolor='white',cmap=colormap)
X_train_mean = data_train.drop(drop_col, axis=1)

X_test_mean = data_test.drop(drop_col, axis=1)
X_train_mean.head()
columns = ['cabin_type', 'title',  'Sex', 'Embarked', 'ticket_type']



for col in columns:

    ordered_labels = X_train_mean.groupby([col])['Survived'].mean().to_dict()

    X_train_mean[col] = X_train_mean[col].map(ordered_labels)

    X_test_mean[col] = X_test_mean[col].map(ordered_labels)
X_train_mean.head()
sns.clustermap(X_train_mean.corr(),linewidths=0.1,vmax=1.0, 

            square=True,linecolor='white',cmap=colormap)
X_train_freq = data_train.drop(drop_col, axis=1)

X_test_freq = data_test.drop(drop_col, axis=1)
X_train_freq.head()
columns = ['cabin_type', 'title',  'Sex', 'Embarked', 'ticket_type']



for col in columns:

    ordered_labels = X_train_freq[col].value_counts().to_dict()

    X_train_freq[col] = X_train_freq[col].map(ordered_labels)

    X_test_freq[col] = X_test_freq[col].map(ordered_labels)
X_train_freq.head()
sns.clustermap(X_train_freq.corr(),linewidths=0.1,vmax=1.0, 

            square=True,linecolor='white',cmap=colormap)
random_state = 4
# Cross validate model with Kfold stratified cross val

kfold = StratifiedKFold(n_splits=5)
#separate dataset into X_train and Y_train:

def separate(X_train):

    X = X_train.drop(columns= ['Survived'])

    Y = X_train['Survived']

    return X, Y
X_onehot, Y_onehot  = separate(X_train_onehot)

X_lab, Y_lab  = separate(X_train_lab)

X_mean, Y_mean  = separate(X_train_mean)

X_freq, Y_freq  = separate(X_train_freq)
# Modeling step Test differents algorithms 

random_state = 4

classifiers = []



classifiers.append(('SVC', make_pipeline(StandardScaler(),SVC(random_state=random_state))))

classifiers.append(('DecisionTree', DecisionTreeClassifier(random_state=random_state)))

classifiers.append(('AdaBoost', AdaBoostClassifier(DecisionTreeClassifier(random_state=random_state),\

                                                  random_state=random_state,learning_rate=0.1)))

classifiers.append(('RandomForest', RandomForestClassifier(random_state=random_state)))

classifiers.append(('GradientBoost', GradientBoostingClassifier(random_state=random_state)))

classifiers.append(('MPL', make_pipeline(StandardScaler(), MLPClassifier(random_state=random_state))))

classifiers.append(('KNN',make_pipeline(MinMaxScaler(),KNeighborsClassifier(n_neighbors=7))))



# evaluate each model 

results = []

names = []

for name, classifier in classifiers:

    kfold = model_selection.KFold(n_splits= 3, random_state=random_state, shuffle = True)

    cv_results = model_selection.cross_val_score(classifier, X_onehot, y = Y_onehot, cv=kfold, scoring='accuracy')

    results.append(cv_results)

    names.append(name)

    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())

    print(msg)

def random_forest(X, Y, X_test):

    parameters = {'max_depth' : [2, 4, 5, 10], 

                  'n_estimators' : [200, 500, 1000, 2000], 

                  'min_samples_split' : [3, 4, 5], 



                 }

    kfold = model_selection.KFold(n_splits=3, random_state=random_state, shuffle = True)

    model_RFC = RandomForestClassifier(random_state = 4, n_jobs = -1)

    search_RFC = GridSearchCV(model_RFC, parameters, n_jobs = -1, cv = kfold, scoring = 'accuracy',verbose=1)

    search_RFC.fit(X, Y)

    predicted= search_RFC.predict(X_test)

    

    print("Best parameters are:", search_RFC.best_params_)

    print("Best accuracy achieved:",search_RFC.best_score_)

    

    return search_RFC.best_params_, model_RFC, search_RFC, predicted
param_RFC_onehot, model_RFC_onehot, search_RFC_onehot, predicted_cv_RFC_onehot = random_forest(X_onehot, Y_onehot, X_test_onehot)

param_RFC_lab, model_RFC_lab, search_RFC_lab, predicted_cv_RFC_lab = random_forest(X_lab, Y_lab, X_test_lab)

param_RFC_mean, model_RFC_mean, search_RFC_mean, predicted_cv_RFC_mean = random_forest(X_mean, Y_mean,  X_test_mean)

param_RFC_freq, model_RFC_freq, search_RFC_freq, predicted_cv_RFC_freq = random_forest(X_freq, Y_freq, X_test_freq)
def fit_pred_RF(X, Y, X_test):



    model_RFC = RandomForestClassifier(max_depth =2,  min_samples_split =3, n_estimators = 5000,

                                     random_state = 4, n_jobs = -1)

    model_RFC.fit(X, Y)

    

    predicted= model_RFC.predict(X_test)

    

    return predicted, model_RFC

# predict lables for all encoding strategies:

predicted_RFC_onehot, model_RFC_onehot = fit_pred_RF(X_onehot, Y_onehot, X_test_onehot)

predicted_RFC_lab, model_RFC_lab = fit_pred_RF(X_lab, Y_lab, X_test_lab)

predicted_RFC_mean, model_RFC_mean = fit_pred_RF(X_mean, Y_mean, X_test_mean)

predicted_RFC_freq, model_RFC_freq = fit_pred_RF(X_freq, Y_freq, X_test_freq)
pd.Series(model_RFC_onehot.feature_importances_,X_onehot.columns).sort_values(ascending=True).plot.barh(width=0.8)
def grad_boost(X, Y, X_test):



    parameters = {'max_depth' : [2, 4, 10, 15], 

                  'n_estimators' : [10, 50, 100], 

                  'min_samples_split' : [5, 10, 15],

                 }

    kfold = model_selection.KFold(n_splits=3, random_state=random_state, shuffle = True)

    model_GBC = GradientBoostingClassifier(random_state = 4)

    search_GBC = GridSearchCV(model_GBC, parameters, n_jobs = -1, cv = kfold, scoring = 'accuracy',verbose=1)

    search_GBC.fit(X, Y)

    predicted= search_GBC.predict(X_test)

    

    print("Best parameters are:", search_GBC.best_params_)

    print("Best accuracy achieved:",search_GBC.cv_results_['mean_test_score'].mean())

    

    return search_GBC.best_params_, model_GBC, search_GBC, predicted

    
param_GBC_onehot, model_GBC_onehot, search_GBC_onehot, predicted_cv_GBC_onehot = grad_boost(X_onehot, Y_onehot, X_test_onehot)

param_GBC_lab, model_GBC_lab, search_GBC_lab, predicted_cv_GBC_lab = grad_boost(X_lab, Y_lab, X_test_lab)

param_GBC_mean, model_GBC_mean, search_GBC_mean, predicted_cv_GBC_mean = grad_boost(X_mean, Y_mean, X_test_mean)

param_GBC_freq, model_GBC_freq, search_GBC_freq, predicted_cv_GBC_freq = grad_boost(X_freq, Y_freq, X_test_freq)
def fit_pred_GBC(X, Y, X_test):



    model_GBC = GradientBoostingClassifier(max_depth = 2, min_samples_split = 15, n_estimators = 10,\

                                 random_state = 4, max_features= 'auto')

    model_GBC.fit(X, Y)

    

    predicted= model_GBC.predict(X_test)

    

    return predicted, model_GBC
# predict lables for all encoding strategies:

predicted_GBC_onehot, model_GBC_onehot = fit_pred_GBC(X_onehot, Y_onehot, X_test_onehot)

predicted_GBC_lab, model_GBC_lab = fit_pred_GBC(X_lab, Y_lab, X_test_lab)

predicted_GBC_mean, model_GBC_mean = fit_pred_GBC(X_mean, Y_mean, X_test_mean)

predicted_GBC_freq, model_GBC_freq = fit_pred_GBC(X_freq, Y_freq, X_test_freq)
def mod_KNN(X, Y, X_test):

    

    model_KNN=make_pipeline(MinMaxScaler(),KNeighborsClassifier())

    #KNN.get_params().keys()

    kfold = model_selection.KFold(n_splits=3, random_state=random_state, shuffle = True)

    parameters=[{'kneighborsclassifier__n_neighbors': [2,3,4,5,6,7,8,9,10]}]

    search_KNN = GridSearchCV(estimator=model_KNN, param_grid=parameters, scoring='accuracy', cv=kfold)

    scores_KNN=cross_val_score(search_KNN, X, Y,scoring='accuracy', cv=kfold, verbose=1)

    search_KNN.fit(X, Y)

    predicted= search_KNN.predict(X_test)

    

    print("Best parameters are:", search_KNN.best_params_)

    print("Best accuracy achieved:",search_KNN.cv_results_['mean_test_score'].mean())

    

    return search_KNN.best_params_, model_KNN, search_KNN, predicted

param_KNN_onehot, model_KNN_onehot, search_KNN_onehot, predicted_cv_KNN_onehot = mod_KNN(X_onehot, Y_onehot, X_test_onehot)

param_KNN_lab, model_KNN_lab, search_KNN_lab, predicted_cv_KNN_lab = mod_KNN(X_lab, Y_lab, X_test_lab)

param_KNN_mean, model_KNN_mean, search_KNN_mean, predicted_cv_KNN_mean = mod_KNN(X_mean, Y_mean, X_test_mean)

param_KNN_freq, model_KNN_freq, search_KNN_freq, predicted_cv_KNN_freq = mod_KNN(X_freq, Y_freq, X_test_freq)
def fit_pred_KNN(X, Y, X_test):



    model_KNN = make_pipeline(MinMaxScaler(),KNeighborsClassifier(n_neighbors=11))

    

    model_KNN.fit(X, Y)

    

    predicted= model_KNN.predict(X_test)

    

    return predicted, model_KNN
# predict lables for all encoding strategies:

predicted_KNN_onehot, model_KNN_onehot = fit_pred_KNN(X_onehot, Y_onehot, X_test_onehot)

predicted_KNN_lab, model_KNN_lab = fit_pred_KNN(X_lab, Y_lab, X_test_lab)

predicted_KNN_mean, model_KNN_mean = fit_pred_KNN(X_mean, Y_mean, X_test_mean)

predicted_KNN_freq, model_KNN_freq = fit_pred_KNN(X_freq, Y_freq, X_test_freq)
def mod_SVC(X, Y, X_test):



    model_SVC=make_pipeline(StandardScaler(),SVC(random_state=1))

    parameters=[{'svc__C': [0.0001,0.001,0.1,1, 10, 100], 

           'svc__gamma':[0.0001,0.001,0.1,1,10,50,100],

           'svc__kernel':['rbf'],

           'svc__degree' : [1,2,3,4]

          }]

    kfold = model_selection.KFold(n_splits=3, random_state=random_state, shuffle = True)

    search_SVC = GridSearchCV(estimator=model_SVC, param_grid = parameters, scoring='accuracy', cv=kfold)

    scores_SVC=cross_val_score(search_SVC, X, Y,scoring='accuracy', cv=kfold, verbose =1)

    search_SVC.fit(X, Y)

    predicted= search_SVC.predict(X_test)

    

    print("Best parameters are:", search_SVC.best_params_)

    print("Best accuracy achieved:",search_SVC.cv_results_['mean_test_score'].mean())

    

    return search_SVC.best_params_, model_SVC, search_SVC, predicted
param_SVC_onehot, model_SVC_onehot, search_SVC_onehot, predicted_cv_SVC_onehot = mod_SVC(X_onehot, Y_onehot, X_test_onehot)

param_SVC_lab, model_SVC_lab, search_SVC_lab, predicted_cv_SVC_lab = mod_SVC(X_lab, Y_lab, X_test_lab)

param_SVC_mean, model_SVC_mean, search_SVC_mean, predicted_cv_SVC_mean = mod_SVC(X_mean, Y_mean, X_test_mean)

param_SVC_freq, model_SVC_freq, search_SVC_freq, predicted_cv_SVC_freq = mod_SVC(X_freq, Y_freq, X_test_freq)
def fit_pred_SVC(X, Y, X_test):



    model_SVC = make_pipeline(StandardScaler(),SVC(random_state=random_state, C= 1, gamma = 0.001, kernel = 'rbf', degree =1))

    

    model_SVC.fit(X, Y)

    

    predicted= model_SVC.predict(X_test)

    

    return predicted, model_SVC
# predict lables for all encoding strategies:

predicted_SVC_onehot, model_SVC_onehot = fit_pred_SVC(X_onehot, Y_onehot, X_test_onehot)

predicted_SVC_lab, model_SVC_lab = fit_pred_SVC(X_lab, Y_lab, X_test_lab)

predicted_SVC_mean, model_SVC_mean = fit_pred_SVC(X_mean, Y_mean, X_test_mean)

predicted_SVC_freq, model_SVC_freq = fit_pred_SVC(X_freq, Y_freq, X_test_freq)
predicted = np.where(((predicted_SVC_mean + predicted_KNN_onehot+predicted_RFC_onehot+predicted_RFC_freq+ predicted_RFC_mean )/5) > 0.5, 1, 0)

test =pd.read_csv("../input/test.csv")

submission = pd.DataFrame({'PassengerId': test['PassengerId'],'Survived':predicted})



submission.head()



filename = 'Titanic Predictions Public.csv'



submission.to_csv(filename,index=False)



print('Saved file: ' + filename)