import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

#first read the csv data file
df=pd.read_csv('/kaggle/input/titanic/train.csv')
# Check if there is any value is null for Sex and Survived
print('Number of null value in Sex: ', df['Sex'].isnull().sum())
print('Number of null value in Survived: ', df['Survived'].isnull().sum())

# convert Sex into numeric intSex 
df['intSex'] = df['Sex'].map({'male': 1,'female': 0})
X = df[['intSex']]
y = df['Survived']

logReg = LogisticRegression(random_state=0).fit(X, y)
print(logReg.coef_)
print(logReg.intercept_)
print('Score: ', logReg.score(X,y))
#first read the csv data file
dfTest=pd.read_csv('/kaggle/input/titanic/test.csv')

# Check if there is any value is null for Sex and Survived
print('Number of null value in Sex: ', dfTest['Sex'].isnull().sum())

# convert Sex into numeric intSex 
dfTest['intSex'] = dfTest['Sex'].map({'male': 1,'female': 0})
# predict Y
X_test  = dfTest[['intSex']]
predicted = logReg.predict(X_test)

# create a Survived column
dfTest['Survived'] = predicted

# check it out and write it out
dfTest[['PassengerId', 'Survived']]
dfTest[['PassengerId', 'Survived']].to_csv('/kaggle/working/simplest.csv', index=False)
# filter out null values for all the variables
df2 = df.copy()
df2 = df2[df2['Age'].notna()]
df2 = df2[df2['Fare'].notna()]
df2 = df2[df2['Pclass'].notna()]
df2 = df2[df2['intSex'].notna()]
df2 = df2[df2['Parch'].notna()]
df2 = df2[df2['SibSp'].notna()]
df2 = df2[['Age','Fare', 'Pclass', 'intSex', 'Survived', 'Parch', 'SibSp']]

featureList = ['Age','Fare', 'Pclass', 'intSex', 'Parch', 'SibSp']
# to model with all possible combination of features
import itertools
# to hold the scores for all the models
scoreDict = {}
for j in range(1, len(featureList)+1):
    # find all the combinations of n fatures and build model
    for item in itertools.combinations(featureList, j):
        variableList = list(item)
        X = df2[variableList]
        y = df2['Survived']
        logReg = LogisticRegression(random_state=0).fit(X, y)
        #print(logReg.coef_)
        #print(logReg.intercept_)
        scoreDict[str(variableList)] = logReg.score(X,y)

# sort the score
sortedDict = {k: v for k, v in sorted(scoreDict.items(), key=lambda item: item[1])}

#print the top 10 the scores
for key in list(sortedDict)[-10:]:
    print ('Score for ', key, ' : ', sortedDict[key])



# fill na with individual mean
df3 = df.copy()
for i in featureList:
    # could use mean or meadian here
    df3[i].fillna(df3[i].median(),inplace=True)

# to model with all possible combination of features
import itertools
scoreDict2 = {}
for j in range(1, len(featureList)+1):
    for item in itertools.combinations(featureList, j):
        variableList = list(item)
        X = df3[variableList]
        y = df3['Survived']
        logReg = LogisticRegression(random_state=0).fit(X, y)
        scoreDict2[str(variableList)] = logReg.score(X,y)

# sort the score
sortedDict2 = {k: v for k, v in sorted(scoreDict2.items(), key=lambda item: item[1])}

#print the top 10 the scores
for key in list(sortedDict2)[-10:]:
    print ('Score for ', key, ' : ', sortedDict2[key])

#first read the csv data file
dfTest2=pd.read_csv('/kaggle/input/titanic/test.csv')

# Check if there is any value is null for Sex and Survived
print('Number of null value in Age: ', dfTest2['Age'].isnull().sum())
print('Number of null value in Pclass: ', dfTest2['Pclass'].isnull().sum())
print('Number of null value in Sex: ', dfTest2['Sex'].isnull().sum())
print('Number of null value in Parch: ', dfTest2['Parch'].isnull().sum())
print('Number of null value in SibSp: ', dfTest2['SibSp'].isnull().sum())

# convert Sex into numeric intSex 
dfTest2['intSex'] = dfTest2['Sex'].map({'male': 1,'female': 0})

print('Precentage of data with missing Age: ', 86/418)
# Given that there are lots of na data in Age, thus the model ['Pclass', 'intSex', 'Parch', 'SibSp']  :  0.8002244668911336 with df3 might be better
X = df3[['Pclass', 'intSex', 'Parch', 'SibSp']]
y = df3['Survived']

logReg2 = LogisticRegression(random_state=0).fit(X, y)
print(logReg2.coef_)
print(logReg2.intercept_)
print('Score: ', logReg2.score(X,y))

# predict Y
X_test  = dfTest2[['Pclass', 'intSex', 'Parch', 'SibSp']]
predicted = logReg2.predict(X_test)

# create a Survived column
dfTest2['Survived'] = predicted

# check it out and write it out
dfTest2[['PassengerId', 'Survived']]
dfTest2[['PassengerId', 'Survived']].to_csv('/kaggle/working/bestOutcome.csv', index=False)