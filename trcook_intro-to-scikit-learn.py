#Add packages

#These are my standard packages I load for almost every project

%matplotlib inline 

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

#From Scikit Learn

from sklearn import preprocessing

from sklearn.model_selection  import train_test_split, cross_val_score, KFold

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report



from sklearn import tree 

from IPython.display import Image

!pip install pydotplus
# Then we just import as usual

import pydotplus
pwd
cd /kaggle/working


bank = pd.read_csv("../input/week1-data/bank_data.csv", sep=",")

#print type of object for target

print("Data type", bank.savings_acct.dtype)

#Dimensions of dataset

print("Shape of Data", bank.shape)

#Column names

print("Colums Names", bank.columns)

#See top few rows of dataset

bank.head(10)
# designate target variable name

targetName = 'savings_acct'

targetSeries = bank[targetName] # copy the target series to its own object (a series object)



#remove target from current location

bank.drop(columns=targetName,inplace=True)



# reinsert column back into the dataframe in the 0-th (i.e. first) position

bank.insert(0, targetName, targetSeries)

#reprint dataframe and see target is in position 0

bank.head(10)
#Note: axis=1 denotes that we are referring to a column, not a row

bank.drop('id',axis=1,inplace=True)

bank.head(10)
#Basic bar chart since the target is binominal

groups = bank.groupby(targetName) # create an object that  groups observations by the target variable

targetEDA=groups[targetName].count() # get counts by groups, focusing on the target variable

fig,ax=plt.subplots(1)

ax.bar(x=bank.loc[:,targetName].unique(),height=targetEDA) # call to plt

ax.set_title("Empirical Distribution of Savings Account"); # the semicolon at the end suppresses output

fig.savefig(fname="savings_accounts_bar.png") # saves the figure (optional)





# you could also get a similar plot by:

# targetEDA.plot(kind='bar', grid=False)

# see which variables are categorical, integer, etc. `object` types are typically categorical variables

bank.dtypes
# This code turns a text target into numeric to some scikit learn algorithms can process it



le_dep = preprocessing.LabelEncoder()

# to convert into numbers

bank['savings_acct'] = le_dep.fit_transform(bank['savings_acct'])
bank['savings_acct']
# perform data transformation. Creates dummies of any categorical feature

for col in bank.columns[1:]:

	attName = col

	dType = bank[col].dtype

	missing = pd.isnull(bank[col]).any()

	uniqueCount = len(bank[attName].value_counts(normalize=False))

	# discretize (create dummies)

	if dType == object:

		bank = pd.concat([bank, pd.get_dummies(bank[col], prefix=col)], axis=1)

		bank.drop(columns=[attName],inplace=True)

bank

# pandas also has a function `pd.get_dummies` that is useful for this task
bank.shape
bank.head(10)
# split dataset into testing and training

features_train, features_test, target_train, target_test = train_test_split(

    bank.iloc[:,1:].values, bank.iloc[:,0].values, test_size=0.40, random_state=0)



print(features_test.shape)

print(features_train.shape)

print(target_test.shape)

print(target_train.shape)
#Decision Tree train model. Call up my model and name it clf



clf_dt = tree.DecisionTreeClassifier()

#Call up the model to see the parameters you can tune (and their default setting)

print(clf_dt)

#Fit clf to the training data

clf_dt = clf_dt.fit(features_train, target_train)

#Predict clf DT model again test data

target_predicted_dt = clf_dt.predict(features_test)

print("DT Accuracy Score", accuracy_score(target_test, target_predicted_dt))

print(classification_report(target_test, target_predicted_dt))

print(confusion_matrix(target_test, target_predicted_dt))

#verify DT with Cross Validation

scores = cross_val_score(clf_dt, features_train, target_train, cv=10)

print("Cross Validation Score for each K",scores)

scores.mean()          
# this creates our graph data

dot_data = tree.export_graphviz(clf_dt, out_file=None, 

                         filled=True, rounded=True,  

                         special_characters=True)



graph = pydotplus.graph_from_dot_data(dot_data)

Image(graph.create_png()) 
from sklearn.neighbors import KNeighborsClassifier