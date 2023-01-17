# Load data wrangling libraries
import pandas as pd # library for data manipulation and analysis
import numpy as np  # library for scientific computing
import re           # regular expression operations

# Load visualization libraries
import matplotlib.pyplot as plt # library for plotting scientific and publication-ready visualization
# enable inline backend usage with IPython
%matplotlib inline
import plotly.offline as py     # library for offline composing, editing, and sharing interactive data visualization
py.init_notebook_mode(connected=True) # initiate the Plotly Notebook mode for offline plot
import plotly.tools as tls # module that communicates with plotly 
import plotly.graph_objs as go # module contains all of the class definitions for the object graph objects
from collections import Counter # import dict subclass for counting hashable objects from module implements specialized container datatypes
import seaborn as sns  # Visualization library based on matplotlib, provides interface for drawing attractive statistical graphics

# Load machine learning libraries
import xgboost as xgb  # Implementation of gradient boosted decision trees designed for speed and performance that is dominative competitive machine learning
import sklearn         # Collection of machine learning algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, 
                              GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier)
from sklearn.cross_validation import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report, precision_recall_curve, confusion_matrix

# Load python runtime services
import warnings # module for issuing warning
warnings.filterwarnings('ignore') # warnings filter controls for never print matching warnings

# Load train and test datasets from CSV files
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

# Store our passenger ID for easy access
PassengerId = test['PassengerId']

# Create 3 row by 4 col tuple and unpack into the figure (f) and axes objects (ax)
f,ax = plt.subplots(3,4,figsize=(20,15))
sns.countplot('Pclass',data=train,ax=ax[0,0])
sns.countplot('Sex',data=train,ax=ax[0,1])
sns.countplot('Embarked',data=train,ax=ax[0,2])
sns.boxplot(x='Pclass',y='Age',data=train,ax=ax[0,3])

sns.countplot('Pclass',hue='Survived',data=train,ax=ax[1,0],palette='husl')
sns.countplot('Sex',hue='Survived',data=train,ax=ax[1,1],palette='husl')
sns.countplot('Embarked',hue='Survived',data=train,ax=ax[1,2],palette='husl')
sns.distplot(train[train['Survived']==0]['Age'].dropna(),ax=ax[1,3],kde=False,color='r',bins=5)
sns.distplot(train[train['Survived']==1]['Age'].dropna(),ax=ax[1,3],kde=False,color='g',bins=5)

sns.swarmplot(x='Pclass',y='Fare',hue='Survived',data=train,palette='husl',ax=ax[2,0])
sns.distplot(train['Fare'].dropna(),ax=ax[2,1],kde=False,color='b')
sns.countplot('SibSp',hue='Survived',data=train,ax=ax[2,2],palette='husl')
sns.countplot('Parch',hue='Survived',data=train,ax=ax[2,3],palette='husl')

ax[0,0].set_title('Total Passengers by Pclass')
ax[0,1].set_title('Total Passengers by Gender')
ax[0,2].set_title('Total Passengers by Embarked')
ax[0,3].set_title('Age Box Plot By Class')

ax[1,0].set_title('Survival Rate by Pclass')
ax[1,1].set_title('Survival Rate by Gender')
ax[1,2].set_title('Survival Rate by Embarked')
ax[1,3].set_title('Survival Rate by Age')

ax[2,0].set_title('Survival Rate by Fare and Pclass')
ax[2,1].set_title('Fare Distribution')
ax[2,2].set_title('Survival Rate by SibSp')
ax[2,3].set_title('Survival Rate by Parch')
# Outlier detection
def detect_outliers(data,n,features):
    outlier_indices = []
    # iterate over features (columns)
    for x in features:
        Q1 = np.percentile(data[x],25)
        Q3 = np.percentile(data[x],75)
        IQR = Q3 - Q1
        Q1_outlier = Q1 - (1.5 * IQR)
        Q3_outlier = Q3 + (1.5 * IQR)
        #determine a list of indices of outliers (data less than Q1_outlier OR more than Q3_outlier) for feature col
        outlier_list_col = data[(data[x] < Q1_outlier) | (data[x] > Q3_outlier)].index
        # append the found outlier indices for col to the list of outlier indices 
        outlier_indices.extend(outlier_list_col)
    
    # select observations containing more than 2 outliers
    outlier_indices = Counter(outlier_indices)        
    multiple_outliers = list( k for k, v in outlier_indices.items() if v > n )
    return multiple_outliers  

# detect outliers from Age, SibSp, Parch and Fare
Outliers_to_drop = detect_outliers(train,2,["Age","SibSp","Parch","Fare"])
train.loc[Outliers_to_drop] # Show the outliers rows

# Drop outliers
# train = train.drop(Outliers_to_drop, axis = 'index').reset_index(drop=True)

# Descriptive analysis (univariate)
full_data = [train, test]
Survival = train['Survived']
Survival.describe()

# Titles feature
# Define function to extract titles from passenger names
def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
 # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""
for dataset in full_data:
    
# Create a new feature Title, containing the titles of passenger names
    dataset['Title'] = dataset['Name'].apply(get_title)

fig, (axis1) = plt.subplots(1,figsize=(18,6))
sns.barplot(x="Title", y="Survived", data=train, ax=axis1);

# Descriptive analysis (univariate)
full_data = [train, test]
Survival = train['Survived']
Survival.describe()

for dataset in full_data:
    dataset['Title'] = dataset['Title'].replace(['Mrs', 'Miss'], 'MM')
    dataset['Title'] = dataset['Title'].replace(['Dr', 'Major', 'Col'], 'DMC')
    dataset['Title'] = dataset['Title'].replace(['Don', 'Dona', 'Rev', 'Capt', 'Jonkheer'],'DRCJ')
    dataset['Title'] = dataset['Title'].replace(['Mme', 'Ms', 'Lady', 'Sir', 'Mlle', 'Countess'],'MMLSMC' )
    
    # Mapping titles
    title_mapping = {"Mr": 0, "MM": 1, "Master":2, "DMC": 3, "DRCJ": 4, "MMLSMC": 5}
    dataset['Title'] = dataset['Title'].map(title_mapping)

train[["Title", "Survived"]].groupby(['Title'], as_index=False).mean().sort_values(by='Survived', ascending=False)

fig, (axis1) = plt.subplots(1,figsize=(18,6))
sns.barplot(x="Title", y="Survived", data=train, ax=axis1);

# delete unnecessary feature from dataset
train.drop('Name', axis=1, inplace=True)
test.drop('Name', axis=1, inplace=True)

# Sex Feature
for dataset in full_data:# Mapping Gender
    dataset['Sex'] = dataset['Sex'].map( {'female': 0, 'male': 1} ).astype(int) 

fig, (axis1) = plt.subplots(1,figsize=(18,6))
sns.barplot(x="Sex", y="Survived", data=train, ax=axis1);

# Age Feature
# fill missing age with median age for each title
for dataset in full_data:
    dataset["Age"].fillna(train.groupby("Title")["Age"].transform("median"), inplace=True)
    
# plot distributions of age of passengers who survived or did not survive
a = sns.FacetGrid( train, hue = 'Survived', aspect=4 )
a.map(sns.kdeplot, 'Age', shade= True )
a.set(xlim=(0 , train['Age'].max()))
a.add_legend()        

#facet = sns.FacetGrid(train, hue="Survived",aspect=4)
#facet.map(sns.kdeplot,'Age',shade= True)
#facet.set(xlim=(0, train['Age'].max()))
#facet.add_legend()
#plt.xlim(43,60)

# Binning
for  dataset  in  full_data:
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0,
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 34), 'Age'] = 1,
    dataset.loc[(dataset['Age'] > 34) & (dataset['Age'] <= 42), 'Age'] = 2,
    dataset.loc[(dataset['Age'] > 42) & (dataset['Age'] <= 60), 'Age'] = 3,
    dataset.loc[ dataset['Age'] > 60, 'Age'] = 4
    dataset['Age'] = dataset['Age'].astype(int)

fig, (axis1) = plt.subplots(1,figsize=(18,6))
sns.barplot(x="Age", y="Survived", data=train, ax=axis1);

# Embarked feature
# filling missing values
for dataset in full_data:
# Replace all NULLS in the Embarked column with majority embark = S 
    dataset['Embarked'] = dataset['Embarked'].fillna('S')
# Mapping Embarked
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

fig, (axis1) = plt.subplots(1,figsize=(18,6))
sns.barplot(x="Embarked", y="Survived", data=train, ax=axis1);

# Fare feature
# Remove all NULLS in the Fare column and create a new feature Categorical Fare
for dataset in full_data:
    dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())

# Explore Fare distribution 
fig, (axis1) = plt.subplots(1,figsize=(18,6))
g = sns.distplot(dataset["Fare"], color="m", label="Skewness : %.2f"%(dataset["Fare"].skew()))
g = g.legend(loc="best")

# Apply log to Fare to reduce skewness distribution
for dataset in full_data:
    dataset["Fare"] = dataset["Fare"].map(lambda i: np.log(i) if i > 0 else 0)
a4_dims = (20, 6)
fig, ax = plt.subplots(figsize=a4_dims)
g = sns.distplot(train["Fare"][train["Survived"] == 0], color="r", label="Skewness : %.2f"%(train["Fare"].skew()), ax=ax)
g = sns.distplot(train["Fare"][train["Survived"] == 1], color="b", label="Skewness : %.2f"%(train["Fare"].skew()))
g = g.legend(["Not Survived","Survived"])

# Binning
for dataset in full_data:
    dataset.loc[ dataset['Fare'] <= 2.7, 'Fare']      = 0
    dataset.loc[ dataset['Fare'] > 2.7, 'Fare']       = 1
    dataset['Fare'] = dataset['Fare'].astype(int)

fig, (axis1) = plt.subplots(1,figsize=(18,6))
sns.barplot(x="Fare", y="Survived", data=train, ax=axis1);

# Deck feature
for dataset in full_data:
    dataset['Deck'] = dataset['Cabin'].str[:1]

Pclass1 = train[train['Pclass']==1]['Deck'].value_counts()
Pclass2 = train[train['Pclass']==2]['Deck'].value_counts()
Pclass3 = train[train['Pclass']==3]['Deck'].value_counts()
df = pd.DataFrame([Pclass1, Pclass2, Pclass3])
df.index = ['1st class','2nd class', '3rd class']
df.plot(kind='bar',stacked=True, figsize=(10,5))

cabin_mapping = {"A": 0, "B": 1, "C": 2, "D": 1, "E": 1, "F": 3, "G": 4, "T": 1}
for dataset in full_data:
    dataset['Deck'] = dataset['Deck'].map(cabin_mapping)
    
# fill missing Fare with median fare for each Pclass
for dataset in full_data:
    dataset["Deck"].fillna(train.groupby("Pclass")["Deck"].transform("median"), inplace=True)
    dataset['Deck'] = dataset['Deck'].astype(int)
    
fig, (axis1) = plt.subplots(1,figsize=(18,6))
sns.barplot(x="Deck", y="Survived", data=train, ax=axis1);

# Cabin feature
# Feature that tells whether a passenger had a cabin on the Titanic (O if no cabin number, 1 otherwise)
for dataset in full_data:
    dataset['Cabin'] = dataset["Cabin"].apply(lambda x: 0 if type(x) == float else 1)

train[["Cabin", "Survived"]].groupby(['Cabin'], as_index=False).sum().sort_values(by='Survived', ascending=False)
train[["Cabin", "Survived"]].groupby(['Cabin'], as_index=False).mean().sort_values(by='Survived', ascending=False)

fig, (axis1) = plt.subplots(1,figsize=(18,6))
sns.barplot(x="Cabin", y="Survived", data=train, ax=axis1);

# FamilySize Feature
for dataset in full_data:
# Create new feature FamilySize as a combination of SibSp and Parch
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch']+1
      
fig, (axis1) = plt.subplots(1,figsize=(18,6))
sns.barplot(x="FamilySize", y="Survived", hue="Sex", data=train, ax = axis1);

# Features drop
features_drop = ['PassengerId', 'SibSp', 'Parch', 'Ticket']
train = train.drop(features_drop, axis = 1)
test  = test.drop(features_drop, axis = 1)

train_data = train.drop('Survived', axis=1)
target = train['Survived']

# Pearson Correlation Heatmap
colormap = plt.cm.RdBu
plt.figure(figsize=(14,12))
plt.title('Pearson Correlation of Features', y=1.01, size=15)
sns.heatmap(train.astype(float).corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white',annot=True)

#Pairplots
g = sns.pairplot(train[[u'Survived', u'Pclass', u'Sex', u'Age', u'Fare',
       u'FamilySize', u'Title']], hue='Survived', palette = 'seismic',size=1.2,diag_kind = 'kde',diag_kws=dict(shade=True),plot_kws=dict(s=10) )
g.set(xticklabels=[])
# Cross Validation (K-fold)
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
k_fold = KFold(n_splits=10, shuffle=True, random_state=0)

# kNN
clf = KNeighborsClassifier(n_neighbors = 13)
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)
print('kNN =',(round(np.mean(score)*100, 2)))

# Decision Tree
clf = DecisionTreeClassifier()
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)
print('Decision Tree =',(round(np.mean(score)*100, 2)))

# Random Forest
clf = RandomForestClassifier(n_estimators=13)
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)
print('Random Forest =',(round(np.mean(score)*100, 2)))

# Naive Bayes
clf = GaussianNB()
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)
print('Naive Bayes =',(round(np.mean(score)*100, 2)))

# SVM
clf = SVC()
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)
print('SVC =',(round(np.mean(score)*100, 2)))
# Testing
clf = SVC()
clf.fit(train_data, target)
prediction = clf.predict(test)
# Preparing data for Submission
test_Survived = pd.Series(prediction, name="Survived")
Submission = pd.concat([PassengerId,test_Survived],axis=1)
Submission.head(15)

filename = 'Titanic Predictions 3K.csv'
Submission.to_csv(filename,index=False)
print('Saved file: ' + filename)