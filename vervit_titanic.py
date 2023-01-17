import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import pandas as pd

import seaborn as sns
import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import pandas as pd

import seaborn as sns
missing_values=['NAN','NaN','Nan',"na",np.nan]

df=pd.read_csv('/kaggle/input/titanic/train.csv',na_values=missing_values )

df.shape
k_test = pd.read_csv('/kaggle/input/titanic/test.csv',na_values=missing_values )

k_test.shape
k_train=pd.concat([df,k_test],axis=0)

k_train.shape
df_copy=k_train.copy(deep=True)
df_copy.fillna(df_copy['Age'].mean(),inplace=True)

df_copy.fillna(df_copy['Fare'].mean(),inplace=True)
df_copy.drop(['Name','Ticket','Cabin'],axis=1,inplace=True)
dummy_list=['Sex','Pclass','Parch','Embarked','SibSp']
#function to create dummy variables in dataset



def dummy_df(df, dummy_list):

  for df_copy in dummy_list:

    dummies = pd.get_dummies(df[df_copy], prefix=df_copy, dummy_na=False,drop_first=True)

    df=df.drop(df_copy, 1)

    df=pd.concat([df,dummies],axis =1)

  return df 
df_copy= dummy_df(df_copy,dummy_list)
#spliting the data in train and test set

df_train=df_copy.iloc[:891,:]

y_test=df_copy.iloc[891:,:]
y_test.drop(['Survived'],axis=1,inplace=True)
y_test.shape
X_train=df_train.drop(['Survived'],axis=1)

y_train=df_train['Survived']
from sklearn.ensemble import RandomForestClassifier

clf=RandomForestClassifier(bootstrap= True,

 criterion= 'gini',

 max_depth=80,

 max_features= 'auto',

 min_samples_leaf= 4,

 min_samples_split= 5,

 n_estimators= 670,n_jobs=-1)
clf.fit(X_train,y_train)
X_train.head()
y_preds=clf.predict(y_test)
y_preds
preds=pd.DataFrame(y_preds)

sub_df=pd.read_csv('/kaggle/input/titanic/gender_submission.csv')

datasets=pd.concat((sub_df['PassengerId'],preds),axis=1)

datasets.columns=['PassengerId','Survived']

datasets.to_csv('gender_submission.csv',index=False)