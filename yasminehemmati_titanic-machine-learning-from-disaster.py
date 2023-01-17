# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv("../input/titanic/train.csv")

test = pd.read_csv("../input/titanic/test.csv")

train.head() # for the name variable we can change it do it only has Miss or Mrs etc 

print(train.shape)

has_null = train.isnull().sum()

has_null[has_null>0] # we see most are missing for cabin so we can just drop it 
#first 891 are train

data = pd.concat([train,test ],ignore_index=True) #put them together so we can make changes together

data = data.drop(['Cabin'], axis=1)

has_null = data.isnull().sum()

has_null[has_null>0] #we ignore the missing feature for survived since they are all from the test set
data['Name']
#We see that every status ends with a dot



#for name in data['Name']:

#    by_comma = name.split(',')

#    name = by_comma[1]

#    name = name.split('.')[0].strip()



data['Status'] = data['Name'].str.extract(' ([A-Za-z]+\.)')

data= data.drop(['Name','Ticket'], axis=1) #Ticket is the Ticket Number and this information does not seem relevent so we can drop it

data['Status']
print(data['Status'].unique())

data.head() 
#fill missing age with mean of each status

data["Age"] = data.groupby("Status").transform(lambda x: x.fillna(x.mean()))
embark_mode = data['Embarked'].mode()[0]

fare_mean = data['Fare'].mean()

print(embark_mode)

print(fare_mean)

data['Embarked']= data['Embarked'].fillna(embark_mode)

data['Fare'] = data['Fare'].fillna(fare_mean)
has_null = data.isnull().sum()

has_null[has_null>0] #So the only null is Survived which would all be in the test set which will later be seperated
#now to deal with categorical variables

data = pd.get_dummies(data, columns = ['Sex', 'Status', 'Embarked'], drop_first = True)

data
#now we seperatet the train and test data set recall first 891 rows are train data rest would be test data

df_train = data.iloc[:891,:]

df_test = data.iloc[891:,:]

has_null = df_train.isnull().sum()

has_null[has_null>0] # no missing survived features
print(df_test.shape)

has_null = df_test.isnull().sum()

has_null[has_null>0] #so all Survived missing as it should be so we can drop it
df_test= df_test.drop(['Survived'], axis=1)

from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score
Y = df_train['Survived']

X = df_train.drop(['Survived'], axis=1)

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, random_state = 0)
#from sklearn.ensemble import RandomForestClassifier



#rfc = RandomForestClassifier()

#rfc.fit(X_train,Y_train)
#rfc_pred = rfc.predict(X_test)

#r2_score(Y_test, rfc_pred)
from sklearn.neighbors import KNeighborsClassifier



knn = KNeighborsClassifier(n_neighbors= 7, weights='distance')



#Train the model using the training sets

knn.fit(X_train, Y_train)



#Predict the response for test dataset

y_pred = knn.predict(X_test)

from sklearn import metrics



metrics.accuracy_score(Y_test, y_pred)
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(random_state=42, n_estimators=700, max_leaf_nodes=32)

rfc.fit(X_train, Y_train)

y_pred = rfc.predict(X_test)

metrics.accuracy_score(Y_test, y_pred)
from xgboost import XGBClassifier

xgb =  XGBClassifier(max_depth = 3, n_estimators = 700,learning_rate=0.02)

xgb.fit(X_train, Y_train)

y_pred = xgb.predict(X_test)

metrics.accuracy_score(Y_test, y_pred)
from sklearn.ensemble import GradientBoostingClassifier

gbc =  GradientBoostingClassifier(learning_rate=0.11)

gbc.fit(X_train, Y_train)

y_pred = gbc.predict(X_test)

metrics.accuracy_score(Y_test, y_pred)
from sklearn.ensemble import AdaBoostClassifier

from sklearn.tree import DecisionTreeClassifier

ada =  AdaBoostClassifier(DecisionTreeClassifier(max_depth=3),n_estimators=200)

ada.fit(X_train, Y_train)

y_pred = ada.predict(X_test)

metrics.accuracy_score(Y_test, y_pred)
pred_xgb = xgb.predict(df_test)

pred_knn = knn.predict(df_test)

pred_ada = ada.predict(df_test)

pred_gbc = gbc.predict(df_test)

pred_rfc = rfc.predict(df_test)

#use all 5 or all but knn and ada model since we want odd number of classifier and those 2 have the lowest accuracy
Models = {'XGB': pred_xgb, 

        'ADA': pred_ada, 

        'RFC': pred_rfc} 



df = pd.DataFrame(Models, columns= ['XGB', 'ADA','RFC'])
df
final = df.mode(axis=1)
pred_xgb.shape
predicted_survival = final.astype(int)

predicted_survival
ans= np.reshape(predicted_survival, (418,))

ans = ans.values

ans =ans.reshape((418,))

ans
ans.shape
Y_test.shape


submission = pd.DataFrame(columns=['PassengerId', 'Survived'])

submission['PassengerId'] = df_test['PassengerId']

submission['Survived'] = ans

submission.to_csv('submissions.csv', header=True, index=False)