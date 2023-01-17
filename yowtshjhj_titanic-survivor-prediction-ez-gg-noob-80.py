import pandas as pd 

import numpy as np 

import matplotlib.pyplot as plt

import seaborn as sns



data=pd.read_csv('../input/titanic/train.csv')

test=pd.read_csv('../input/titanic/test.csv')



labels=data['Survived']

df=pd.concat([data,test],axis=0)

df=df.drop(labels='Survived',axis=1,inplace=False)

#df=df.drop(labels=['Age','Cabin','Name','PassengerId','Pclass'],axis=1,inplace=False)

df=df.drop(labels=['Name','PassengerId'],axis=1,inplace=False)



df=pd.get_dummies(df,prefix_sep='_',drop_first=True)

df.fillna(df.mean(),inplace=True)

train_df=df.iloc[:data.shape[0],:]

test_df=df.iloc[data.shape[0]:,:]



from sklearn.ensemble import RandomForestClassifier

rfc=RandomForestClassifier(n_estimators=100)

rfc.fit(train_df,labels)



predictions=rfc.predict(test_df)

predictions=pd.Series(predictions,name='Survived')

submission=pd.concat([pd.Series(range(int(892),int(1310)),name="PassengerId"),predictions],axis=1)

submission.to_csv('gender_submission.csv',index=False)


