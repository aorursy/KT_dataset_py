# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import pandas as pd

import numpy as np

import seaborn as sns

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

import math



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



df = pd.read_csv("../input/train.csv")

testDF = pd.read_csv("../input/test.csv")

#print (df.describe)

# Variable	Definition	Key



# survival 	Survival 	0 = No, 1 = Yes

# pclass 	Ticket class 	1 = 1st, 2 = 2nd, 3 = 3rd

# sex 	Sex 	

# Age 	Age in years 	

# sibsp 	# of siblings / spouses aboard the Titanic 	

# parch 	# of parents / children aboard the Titanic 	

# ticket 	Ticket number 	

# fare 	Passenger fare 	

# cabin 	Cabin number 	

# embarked 	Port of Embarkation 	C = Cherbourg, Q = Queenstown, S = Southampton



# Get an idea of the structure of the df

#print (df.shape)

# There are 891 rows and 12 columns



# Get an idea of the values across all the columns

#print (df.describe())

# The 'Age' column only has 714/891 column entries.



# Check to see which columns have null values

#print (df.isnull().sum())

# The 'Age', 'Cabin' and 'Embarked' column have some null entries. We need to perform some clean-up for these columns



# Before we begin cleaning up, let's visualize the data we have so far.

#sns.boxplot(data=df, palette="deep")

#plt.show()

# Boxplots of interest are for PassengerId, Age and Fare. Let's dive a bit deeper into the boxplots for these values

#sns.boxplot(x="PassengerId",data=df)

#plt.show()

# Age and Fare has outliers



#Check to see which columns have a positive correlation;

# sns.heatmap(df.corr(), 

#             xticklabels=df.corr().columns.values,

#             yticklabels=df.corr().columns.values)

# plt.show()

# From the heatmap, we observe a strong correlation for Parch and Fare

# Variables that aren't so correlated are SibSp and Age



#====================================================================

# CONSIDER CATEGORICAL VARIABLES

#====================================================================

# Create a temporary DataFrame to store only the age and place of embarkment

tmp_df = df[['Age', 'Embarked']].copy()

emb = tmp_df.groupby('Embarked')

#print (emb['Age'].agg(np.mean))



# Embarked

# C    30.814769

# Q    28.089286

# S    29.445397



# Function to fill the age NaN entries with the average age depending on the place of embarkment

def fillAgeNa (df):

    if (math.isnan(df['Age'])):

        #print ("NAN!")

        if (df['Embarked'] == 'C'):

            age = 30.0

        elif (df['Embarked'] == 'Q'):

            age = 28.0

        elif (df['Embarked'] == 'S'):

            age = 29.0

    else:

        age = df['Age']

    return age





def dummifySex (df):

    if (df['Sex'] == "male"):

        out = 0

    else:

        out = 1

    return out



#print (df['Age'].dtypes)

df['Age'] = df.apply(fillAgeNa, axis=1)

#print (df.head())

#print (df.isnull().sum())

# def getCabin (df):

#     if (df['Cabin'].isnull()):

#         out = np.nan

#     elif ('A' in df['Cabin']):

#         out = 1

#     elif ('B' in df['Cabin']):

#         out = 2

#     elif ('C' in df['Cabin']):

#         out = 3

#     elif ('D' in df['Cabin']):

#         out = 4

#     elif ('E' in df['Cabin']):

#         out = 5

#     elif ('F' in df['Cabin']):

#         out = 6

#     else:

#         out = np.nan

#     

#     return out



def haveSiblings (df):

    if (df['SibSp'] > 0):

        have = 1

    else:

        have = 0

    return have

    

def haveParents (df):

    if (df['Parch'] > 0):

        have = 1

    else:

        have = 0

    return have             

        

#df['SibSp'] = df.apply(haveSiblings, axis=1)

#df['Parch'] = df.apply(haveParents, axis=1)        

df['Sex'] = df.apply(dummifySex, axis=1)

#df['Embarked'] = df['Embarked'].replace({0:'S', 1:'C', 2:'Q'})

df_emb_dummies = pd.get_dummies(df['Embarked'])

#print (df_emb_dummies.head())

df = pd.concat([df, df_emb_dummies], axis=1)

#df['Cabin'] = df.apply(getCabin, axis=1)

df = df.drop(['Name', 'Ticket', 'Embarked', 'Cabin'], axis=1)

#df['Age'].fillna(0, inplace=True)



#print (df.head())

#print (df.isnull().sum())



# Age has 177 null entries, cabin has 687 null entires and embarked has 2 null entries



#Min age = 0.42 and max age is 80

#print (df['Age'].describe())





#print (df.isnull().sum())







X = df.drop('Survived', axis=1)

Y = df['Survived']



#print (df.describe)



X_train, X_test, Y_train,Y_test = train_test_split(X,Y,test_size=0.50, random_state=1)

logit = LogisticRegression()

result = logit.fit(X_train, Y_train)

#result = logit.fit(X, Y)

Y_pred = logit.predict(X_test)

#print (logit.score(X_test, Y_test))

# #print (result.coef_)

result = logit.fit(df.drop('Survived', axis=1), df['Survived'])

confusion_matrix = confusion_matrix(Y_test, Y_pred)

#print (confusion_matrix)





# 50% training split

# 0.780269058296

# [[231  32]

#  [ 66 117]]







#print (result)

#my_submission = pd.DataFrame({'PassengerId': df.PassengerId, 'Survived': Y_pred})

#========================================================================================





tmp_testDF = testDF[['Age', 'Embarked']].copy()

emb = tmp_testDF.groupby('Embarked')



testDF['Age'] = testDF.apply(fillAgeNa, axis=1)





testDF['Sex'] = testDF.apply(dummifySex, axis=1)

testDF_emb_dummies = pd.get_dummies(testDF['Embarked'])

testDF = pd.concat([testDF, testDF_emb_dummies], axis=1)

pIDs = testDF.PassengerId

testDF = testDF.drop(['Name', 'Ticket', 'Embarked', 'Cabin'], axis=1)

testDF['Fare'].fillna(0, inplace=True)

#print (testDF.isnull().sum())

pred = logit.predict(testDF)

submission = pd.DataFrame({"PassengerId": pIDs, "Survived": pred})

#print (submission.describe())

submission.to_csv('titanic_predictions.csv', index=False)