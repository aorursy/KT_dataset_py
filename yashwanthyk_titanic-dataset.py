#titanicDataset
#predictingscore
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.tree import DecisionTreeClassifier
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#LoadData

train_df= pd.read_csv(r'../input/train.csv')
test_df=pd.read_csv(r'../input/test.csv')
#DropColumns

drop_columns=['Name','Ticket', 'Cabin']
train_df_X= train_df.drop(drop_columns, axis=1)
train_df_X= train_df_X.drop('Survived', axis=1)

print(train_df_X.info())


########Train data####

#Age
train_df_X['Age']= train_df_X['Age'].interpolate()

train_df_X['Embarked']=train_df_X['Embarked'].fillna('S')
train_df_X['Pclass']=train_df_X['Pclass'].astype('str')
X= pd.get_dummies(train_df_X)
print(X.info())
X_train= X.values

y_train= train_df['Survived'].values

######test data#####
test_df_X=test_df.drop(drop_columns, axis=1)
test_df_X['Fare']= test_df_X['Fare'].fillna(0.00)
test_df_X['Fare']=pd.to_numeric(test_df_X['Fare'])
test_df_X['Pclass']=test_df_X['Pclass'].astype('str')
test_df_X['Age']=test_df['Age'].interpolate()

test_df_X=pd.get_dummies(test_df_X)

X_test=test_df_X.values
######model#####

tree= DecisionTreeClassifier(max_depth=5)
tree.fit(X_train,y_train)
y_pred=tree.predict(X_test)

print('score: {}'.format(tree.score(X_test,y_pred)))
output = np.column_stack((X_test[:,0],y_pred))
df = pd.DataFrame(output.astype('int'),columns=['PassengerID','Survived'])
df=df.to_csv('Results.csv', index=False)

