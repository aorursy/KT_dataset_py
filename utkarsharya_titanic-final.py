import pandas as pd

import seaborn as sns

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline
train_data=pd.read_csv('../input/titanic/train.csv')

test_data=pd.read_csv('../input/titanic/test.csv')
def impute_age(cols):

    Age=cols[0]

    Pclass=cols[1]

    if pd.isnull(Age):

        if Pclass==1:

            return 37

        elif Pclass==2:

            return 29

        else:

            return 24

    else:

        return Age



train_data['Age']=train_data[['Age','Pclass']].apply(impute_age,axis=1)
train_data.drop('Cabin',axis=1,inplace=True)
sex=pd.get_dummies(train_data['Sex'],drop_first=True)

embark=pd.get_dummies(train_data['Embarked'],drop_first=True)

train_data=pd.concat([train_data,sex,embark],axis=1)

train_data.drop(['Name','Sex','Ticket','Embarked'],axis=1,inplace=True)
# x=train_data.drop('Survived',axis=1)

# y=train_data['Survived']



# from sklearn.model_selection import train_test_split

# X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=.3,random_state=101)



# from sklearn.linear_model import LogisticRegression

# logregmodel=LogisticRegression()

# logregmodel.fit(X_train,y_train)

# predictions=logregmodel.predict(X_test)
# from sklearn.metrics import classification_report

# classification_report(predictions,y_test)
# from sklearn.metrics import confusion_matrix

# confusion_matrix(predictions,y_test)
def impute_age1(cols):

    Age=cols[0]

    Pclass=cols[1]

    if pd.isnull(Age):

        if Pclass==1:

            return 42

        elif Pclass==2:

            return 26

        else:

            return 23

    else:

        return Age



test_data['Age']=test_data[['Age','Pclass']].apply(impute_age1,axis=1)
test_data.drop('Cabin',axis=1,inplace=True)

sex=pd.get_dummies(test_data['Sex'],drop_first=True)

embark=pd.get_dummies(test_data['Embarked'],drop_first=True)

test_data=pd.concat([test_data,sex,embark],axis=1)

test_data.drop(['Name','Sex','Ticket','Embarked'],axis=1,inplace=True)

test_data['Fare'].loc[152]=np.mean(test_data['Fare'])
x=train_data.drop('Survived',axis=1)

y=train_data['Survived']



from sklearn.linear_model import LogisticRegression

logregmodel=LogisticRegression()

logregmodel.fit(x,y)



predictions=logregmodel.predict(test_data)
def createList(r1,r2):

    return [item for item in range(r1,r2)]

PassengerId=createList(892,1310)

PassengerId=pd.DataFrame({'':PassengerId}).rename(columns={'':'PassengerId'})

PassengerId

Survived=pd.DataFrame({'':list(predictions)})
Survived=pd.DataFrame({'':list(predictions)}).rename(columns={'':'Survived'})

Survived
result=pd.concat([PassengerId,Survived],axis=1)
result.to_csv('result_submission.csv',index=False)