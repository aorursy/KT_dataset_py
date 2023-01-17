import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
df=pd.read_csv('../input/predicting-who-pays-back-loans/loan_data.csv')
df.info()
df.describe()
df.head()
plt.figure(figsize=(10,6))

df[df['credit.policy']==1]['fico'].hist(alpha=0.5,color='blue',

                                              bins=30,label='Credit.Policy=1')

df[df['credit.policy']==0]['fico'].hist(alpha=0.5,color='red',

                                              bins=30,label='Credit.Policy=0')

plt.legend()

plt.xlabel('FICO')
plt.figure(figsize=(10,6))

df[df['not.fully.paid']==1]['fico'].hist(alpha=0.5,color='blue',

                                              bins=30,label='Credit.Policy=1')

df[df['not.fully.paid']==0]['fico'].hist(alpha=0.5,color='red',

                                              bins=30,label='Credit.Policy=0')

plt.legend()

plt.xlabel('FICO')
plt.figure(figsize=(10,6))

sns.countplot(x='purpose',hue='not.fully.paid',data=df,palette='Set1')
sns.set_style("darkgrid")

sns.jointplot(x='fico',y='int.rate',data=df,color='purple')
sns.lmplot(x='fico',y='int.rate',data=df,palette='Set1',hue='credit.policy',col='not.fully.paid')
df.info()
cat_feats=['purpose']
final_data=pd.get_dummies(df,columns=cat_feats,drop_first=True)
final_data.head()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(final_data.drop('not.fully.paid',axis=1),final_data['not.fully.paid'], test_size=0.3, random_state=101)
from sklearn.tree import DecisionTreeClassifier
dtree=DecisionTreeClassifier()
dtree.fit(X_train,y_train)
predict=dtree.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,predict))
print(confusion_matrix(y_test,predict))
from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier(n_estimators=600)
rfc.fit(X_train,y_train)
pred=rfc.predict(X_test)
print(classification_report(y_test,pred))
print(confusion_matrix(y_test,pred))
# Depends what metric you are trying to optimize for. 

# Notice the recall for each class for the models.

# Neither did very well, more feature engineering is needed.