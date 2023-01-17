import os

os.listdir('../input')
import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import pandas as pd



df = pd.read_csv("../input/train.csv")

df.info()
#--------------------------------------Survived/Died by class------------------------------------------

sns.set()

survived_class = df[df['Survived']==1]['Pclass'].value_counts()

dead_class = df[df['Survived']==0]['Pclass'].value_counts()

df_class = pd.DataFrame([survived_class, dead_class])

df_class.index = ['Survived', 'Dead']

df_class











df_class.plot(kind='bar', title='Survived/Died by class')

plt.show()
class1_survived = df_class.iloc[0,0]/df_class.iloc[:,0].sum()*100

class2_survived = df_class.iloc[0,1]/df_class.iloc[:,1].sum()*100

class3_survived = df_class.iloc[0,2]/df_class.iloc[:,2].sum()*100



print(round(class1_survived, 2), "% of class 1 survived")

print(round(class2_survived, 2), "% of class 2 survived")

print(round(class3_survived, 2), "% of class 3 survived")

#------------------------------------Survived/Died by sex---------------------------------------------

survived = df[df['Survived']==1]['Sex'].value_counts()

died = df[df['Survived']==0]['Sex'].value_counts()

df_sex = pd.DataFrame([survived, died], index=['Survived', 'Died'])

df_sex
df_sex.plot(kind='bar', title='Survived/Died by class')

plt.show()
female_survived = df_sex.iloc[0,0]/df_sex.iloc[:,0].sum()*100

male_survived = df_sex.iloc[0,1]/df_sex.iloc[:,1].sum()*100



print(round(female_survived, 2), "% of females survived")

print(round(male_survived, 2), "% of males survived")
#----------------------------------Survived/Died by Embarked----------------------------------------

survived_embarked = df[df['Survived']==1]['Embarked'].value_counts()

died_embarked = df[df['Survived']==0]['Embarked'].value_counts()

survived_embarked



df_embarked = pd.DataFrame([survived_embarked, died_embarked], index=['Survived', 'Died'])

df_embarked
df_embarked.plot(kind='bar', title='Survived/Died by embarked')

plt.show()
embark_s = df_embarked.iloc[0,0]/df_embarked.iloc[:,0].sum()*100

embark_c = df_embarked.iloc[0,1]/df_embarked.iloc[:,1].sum()*100

embark_q = df_embarked.iloc[0,2]/df_embarked.iloc[:,2].sum()*100



print(round(embark_s, 2), "% of Embark S survived")

print(round(embark_c, 2), "% of Embark C survived")

print(round(embark_q, 2), "% of Embark Q survived")
#----------------------------------Feature Selection-----------------------------------#

X = df.drop(['PassengerId', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin'], axis=1)

y = X.Survived

X = X.drop(['Survived'], axis=1)

X.head(20)
#----------------------------------Encoding Categorical Data---------------------------------#

from sklearn.preprocessing import LabelEncoder

labelEncoder_X = LabelEncoder()



#encode Sex

X.Sex = labelEncoder_X.fit_transform(X.Sex)



#encode Embarked

print('Number of null values in embarked : ', sum(X.Embarked.isnull()))



row_index = X.Embarked.isnull()

X.loc[row_index, 'Embarked'] = 'S'



Embarked = pd.get_dummies(X.Embarked, prefix='Embarked')

X = X.drop(['Embarked'], axis=1)

X = pd.concat([X, Embarked], axis=1)

X = X.drop(['Embarked_S'], axis=1)



X.head()
#----------------------------------Changing names to title---------------------------------#

print('Number of null values in age : ', sum(X.Age.isnull()))

got = df.Name.str.split(',').str[1]

X.iloc[:,1] = pd.DataFrame(got).Name.str.split('\s+').str[1]

X.head(10)
#----------------------------------Age by title---------------------------------#

ax = plt.subplot()

ax.set_ylabel('Average age')

X.groupby('Name').mean()['Age'].plot(kind='bar', figsize=(13,8))

#--------------------------Setting uniques values and transforming to list-------------------------#

title_mean_age = []

title_mean_age.append(list(set(X.Name)))

title_mean_age.append(X.groupby('Name').Age.mean())

title_mean_age
#--------------------------------------Setting values to NaNs ----------------------------------#

n_training = df.shape[0]

n_titles = len(title_mean_age[1])

for i in range(0, n_training):

    if (np.isnan(X.Age[i]) == True):

        for j in range(0, n_titles):

            if (X.Name[i] == title_mean_age[0][j]):

                X.Age[i] = title_mean_age[1][j]

                

print('Number of null values in age : ', sum(X.Age.isnull()))

X = X.drop('Name', axis=1)

print(X.head(10))
#--------------------------------Simplifying age=-----------------------------------#

for i in range(0, n_training):

    if X.loc[i, 'Age'] > 18:

        X.loc[i,'Age'] = 0

    else:

        X.loc[i, 'Age'] = 1

        

X.head(10)
#-----------------------------Linear Regression--------------------------#



#from sklearn.linear_model import LogisticRegression

#classifier = LogisticRegression(penalty = 'l2', random_state = 0, solver='liblinear')



#from sklearn.model_selection import cross_val_score

#accuracies = cross_val_score(estimator = classifier, X=X, y=y, cv=10)

#print("Logistic Regression:\n Accuracy : ", accuracies.mean(), "+/-", accuracies.std(), "\n")

#classifier.fit(X, y)



#k = classifier.predict(y)

#---------------------------------------Test Data-------------------------------------#

df2 = pd.read_csv("../input/test.csv")

df2.info()

X_test = df2.drop(['PassengerId', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin'], axis=1)

X_test.head(10)
#-------------------------------------Preparing test data-----------------------------------#

labelEncoder_X_test = LabelEncoder()



#encode Sex

X_test.Sex = labelEncoder_X_test.fit_transform(X_test.Sex)



#encode Embarked

print('Number of null values in embarked : ', sum(X_test.Embarked.isnull()))



Embarked = pd.get_dummies(X_test.Embarked, prefix='Embarked')

X_test = X_test.drop(['Embarked'], axis=1)

X_test = pd.concat([X_test, Embarked], axis=1)

X_test = X_test.drop(['Embarked_S'], axis=1)



#encode age

n_test_training = df2.shape[0]

title_mean_age2 = []

title_mean_age2.append(list(set(X_test.Name)))

title_mean_age2.append(X_test.groupby('Name').Age.mean())





n_test_training = df2.shape[0]

n_test_titles = len(title_mean_age2[1])



for i in range(0, n_test_training):

    if (np.isnan(X_test.Age[i]) == True):

        for j in range(0, n_test_titles):

            if (X_test.Name[i] == title_mean_age2[0][j]):

                X_test.Age[i] = title_mean_age2[1][j]



for i in range(0, n_test_training):

    if (X_test.loc[i, 'Age'] > 18):

        X_test.loc[i,'Age'] = 0

    else:

        X_test.loc[i, 'Age'] = 1

        

X_test = X_test.drop('Name', axis=1)



X_test.head()
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(penalty = 'l2', random_state = 0, solver='liblinear')



#from sklearn.model_selection import cross_val_score

#accuracies = cross_val_score(estimator = classifier, X=X, y=y, cv=10)

#print("Logistic Regression:\n Accuracy : ", accuracies.mean(), "+/-", accuracies.std(), "\n")

classifier.fit(X, y)

k = classifier.predict(X_test)

k

y_pred = pd.DataFrame(k, columns=['Survived'])

predictions = pd.concat([df2, y_pred], axis=1)

predictions.head(15)