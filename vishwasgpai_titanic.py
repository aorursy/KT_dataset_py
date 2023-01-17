# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

'''

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

'''



data = pd.read_csv("../input/train.csv")



X = data.drop(['PassengerId','Cabin','Ticket','Fare', 'Parch', 'SibSp'], axis=1)



y = X.Survived 



X=X.drop(['Survived'], axis=1) 





#X.head(20)





from sklearn.preprocessing import LabelEncoder

labelEncoder_X = LabelEncoder()

X.Sex=labelEncoder_X.fit_transform(X.Sex)





row_index = X.Embarked.isnull()

X.loc[row_index,'Embarked']='S' 





Embarked  = pd.get_dummies(  X.Embarked , prefix='Embarked'  )

#X.head(20)

#Embarked.head()

X = X.drop(['Embarked'], axis=1)



X= pd.concat([X, Embarked], axis=1) 



X = X.drop(['Embarked_S'], axis=1)



#X.head()



#got= data.Name.str.split(',').str[1]

#X.iloc[:,1]=pd.DataFrame(got).Name.str.split('\s+').str[1]





'''got = got.add_prefix("Name")



#got = got.Name.str.split('.').str[0]

got.head()

def sl(n):

    return n.split('.') [0]

got = [sl(str(i)) for i in got ]

#got = got.split('.').str[0]

#print(got)





Name  = pd.get_dummies(  got, prefix='Name'  )



X = X.drop(['Name'],axis = 1)



X = pd.concat([X,Name],axis = 1)



X = X.drop(['Name_ Col'],axis = 1)



#X.head()

'''



print ('Number of null values in Age:', sum(X.Age.isnull()))

 



# -------- Change Name -> Title ----------------------------

got= data.Name.str.split(',').str[1]

X.iloc[:,1]=pd.DataFrame(got).Name.str.split('\s+').str[1]

# ---------------------------------------------------------- 





#------------------ Average Age per title -------------------------------------------------------------

#ax = plt.subplot()

#ax.set_ylabel('Average age')

#X.groupby('Name').mean()['Age'].plot(kind='bar',figsize=(13,8), ax = ax)



title_mean_age=[]

title_mean_age.append(list(set(X.Name)))  #set for unique values of the title, and transform into list

title_mean_age.append(X.groupby('Name').Age.mean())

title_mean_age

#------------------------------------------------------------------------------------------------------





#------------------ Fill the missing Ages ---------------------------

n_traning= data.shape[0]   #number of rows

n_titles= len(title_mean_age[1])

for i in range(0, n_traning):

    if np.isnan(X.Age[i])==True:

        for j in range(0, n_titles):

            if X.Name[i] == title_mean_age[0][j]:

                X.Age[i] = title_mean_age[1][j]

#--------------------------------------------------------------------    



X=X.drop(['Name'], axis=1)





for i in range(0, n_traning):

    if X.Age[i] > 18:

        X.Age[i]= 0

    else:

        X.Age[i]= 1



X.head()







from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0)



from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator = classifier, X=X , y=y , cv = 10)

print("Random Forest:\n Accuracy:", accuracies.mean(), "+/-", accuracies.std())



# Any results you write to the current directory are saved as output.