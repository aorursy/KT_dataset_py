# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
raw_data=pd.read_csv('/kaggle/input/titanic/train.csv')
raw_data
data_f=raw_data.copy()
pd.options.display.max_rows=None

pd.options.display.max_columns=None
data_f.loc[data_f.Pclass==1,"Pclass"]="First"

data_f.loc[data_f.Pclass==2,"Pclass"]="Second"

data_f.loc[data_f.Pclass==3,"Pclass"]="Third"
data_f['Age']=data_f['Age'].fillna(data_f['Age'].median())

data_f_drop_cabin=data_f.drop(['Name','Ticket','Cabin'],axis=1)
data_f_drop_cabin
data_f_dummies=pd.get_dummies(data_f_drop_cabin[['Sex','Embarked','Pclass']])
data_f_dummies
data_after_dummies=pd.concat([data_f_drop_cabin,data_f_dummies],axis=1)
data_after_dummies
data_after_dummies=data_after_dummies.drop(['Sex','Embarked','Pclass'],axis=1)
data_after_dummies
data_after_dummies['Age_Child']=data_after_dummies['Age']<10

data_after_dummies['Age_Teen']=((data_after_dummies['Age']>=10) & (data_after_dummies['Age']<20))

data_after_dummies['Age_Adult']=((data_after_dummies['Age']>=20) & (data_after_dummies['Age']<40))

data_after_dummies['Age_Old']=data_after_dummies['Age']>40
data_after_dummies['Age_Child']=data_after_dummies['Age_Child'].replace({True: 1, False: 0})

data_after_dummies['Age_Teen']=data_after_dummies['Age_Teen'].replace({True: 1, False: 0})

data_after_dummies['Age_Adult']=data_after_dummies['Age_Adult'].replace({True: 1, False: 0})

data_after_dummies['Age_Old']=data_after_dummies['Age_Old'].replace({True: 1, False: 0})
data_after_dummies['Sib_Alone']=data_after_dummies['SibSp']==0

data_after_dummies['Sib_Alone']=data_after_dummies['Sib_Alone'].replace({True:1,False:0})
data_after_dummies['Parch_Alone']=data_after_dummies['Parch']==0

data_after_dummies['Parch_Alone']=data_after_dummies['Parch_Alone'].replace({True:1,False:0})
outputs=data_after_dummies['Survived']

inputs=data_after_dummies.drop(['Survived','Fare','Age','PassengerId','SibSp','Parch'],axis=1)
outputs
inputs
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import validation_curve



model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=1)

model.fit(inputs, outputs)
model.score(inputs,outputs)
test_data=pd.read_csv('/kaggle/input/titanic/test.csv')
test_data
test_data=test_data.drop(['Name','Ticket','Cabin'],axis=1)
test_data
test_data.loc[test_data.Pclass==1,"Pclass"]="First"

test_data.loc[test_data.Pclass==2,"Pclass"]="Second"

test_data.loc[test_data.Pclass==3,"Pclass"]="Third"
test_data_dum=pd.get_dummies(test_data[['Sex','Embarked','Pclass']])
test_data_fin=pd.concat([test_data,test_data_dum],axis=1)
test_data_fin
test_data_fin=test_data_fin.drop(['Sex','Embarked','Fare','Pclass'],axis=1)
test_data_fin['Age']=test_data_fin['Age'].fillna(test_data_fin['Age'].median())
test_data_fin['Age_Child']=test_data_fin['Age']<10

test_data_fin['Age_Teen']=((test_data_fin['Age']>=10) & (test_data_fin['Age']<20))

test_data_fin['Age_Adult']=((test_data_fin['Age']>=20) & (test_data_fin['Age']<40))

test_data_fin['Age_Old']=test_data_fin['Age']>40
test_data_fin['Age_Child']=test_data_fin['Age_Child'].replace({True: 1, False: 0})

test_data_fin['Age_Teen']=test_data_fin['Age_Teen'].replace({True: 1, False: 0})

test_data_fin['Age_Adult']=test_data_fin['Age_Adult'].replace({True: 1, False: 0})

test_data_fin['Age_Old']=test_data_fin['Age_Old'].replace({True: 1, False: 0})
test_data_fin['Sib_Alone']=test_data_fin['SibSp']==0

test_data_fin['Sib_Alone']=test_data_fin['Sib_Alone'].replace({True:1,False:0})
test_data_fin['Parch_Alone']=test_data_fin['Parch']==0

test_data_fin['Parch_Alone']=test_data_fin['Parch_Alone'].replace({True:1,False:0})
test_data_fin=test_data_fin.drop(['PassengerId','Age','SibSp','Parch'],axis=1)
test_data_fin.isna().any()
predictions=model.predict(test_data_fin)



output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)
