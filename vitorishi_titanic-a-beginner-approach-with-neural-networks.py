import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

sns.set()
path='/kaggle/input/titanic/'

df=pd.read_csv(path+'train.csv')

df_test=pd.read_csv(path+'test.csv')
df.head()
sns.pairplot(df.drop('PassengerId',axis=1))
df.info()
df.describe().transpose()
sns.heatmap(df.isnull())
#Pencentage of missing values in the dataframe

(df.isnull().sum()*100)/df.shape[0]
#Cabin has to much missing information. I find that dropping it will be better

df.drop('Cabin',axis=1,inplace=True)



#Doing the same for test dataset

df_test.drop('Cabin',axis=1,inplace=True)
df['Last Name']=df['Name'].apply(lambda name:name.split(',')[0])
df[df['Last Name']=='Abelson']
family_survival=df.groupby('Last Name')[['Survived']].sum()
family_survival
family_survival['number of people']=df.groupby('Last Name').count()['PassengerId']
family_survival['ratio survived']=family_survival['Survived']/family_survival['number of people']
family_survival.head()
family_survival['number of people'].value_counts()
sns.boxplot(x='number of people',y='ratio survived', data=family_survival)
sns.scatterplot(x='Age',y='Fare',data=df)
g=sns.FacetGrid(df,col='Pclass')

g.map(plt.hist,'Age',bins=50)
df['Name'].apply(lambda name:name.split(',')[1].split('.')[0])
df['title name']=df['Name'].apply(lambda name:name.split(',')[1].split('.')[0])
other_titles=df['title name'].value_counts().index[4:].values
df[df['title name'].isin(other_titles)].groupby('title name').describe()['Age']
df['title name'] = df['title name'].replace('Mlle', 'Miss')

df['title name'] = df['title name'].replace('Ms', 'Miss')

df['title name'] = df['title name'].replace('Mme', 'Mrs')

df['title name']=df['title name'].replace(other_titles,'Rare')
#Doing the same for test

df_test['title name']=df_test['Name'].apply(lambda name:name.split(',')[1].split('.')[0])

other_titles=df_test['title name'].value_counts().index[4:].values

df_test[df_test['title name'].isin(other_titles)].groupby('title name').describe()['Age']
df_test['title name'] = df_test['title name'].replace('Mlle', 'Miss')

df_test['title name'] = df_test['title name'].replace('Ms', 'Miss')

df_test['title name'] = df_test['title name'].replace('Mme', 'Mrs')

df_test['title name']=df_test['title name'].replace(other_titles,'Rare')
df_test['title name'].value_counts()
df['title name'].value_counts()
plt.figure(figsize=(12,9))

sns.boxplot(x='title name', y='Age', data=df)
df['Age'] = df.groupby('title name')['Age'].apply(lambda x: x.fillna(x.mean()))

#Doing the same for test dataset

df_test['Age'] = df_test.groupby('title name')['Age'].apply(lambda x: x.fillna(x.mean()))
df['Age'] = df['Age'].astype(int)

df_test['Age'] = df_test['Age'].astype(int)
 # Mapping Age

df.loc[ df['Age'] <= 16, 'Age']= 0

df.loc[(df['Age'] > 16) & (df['Age'] <= 32), 'Age'] = 1

df.loc[(df['Age'] > 32) & (df['Age'] <= 48), 'Age'] = 2

df.loc[(df['Age'] > 48) & (df['Age'] <= 64), 'Age'] = 3

df.loc[ df['Age'] > 64, 'Age'] = 4 ;



df_test.loc[ df_test['Age'] <= 16, 'Age']= 0

df_test.loc[(df_test['Age'] > 16) & (df_test['Age'] <= 32), 'Age'] = 1

df_test.loc[(df_test['Age'] > 32) & (df_test['Age'] <= 48), 'Age'] = 2

df_test.loc[(df_test['Age'] > 48) & (df_test['Age'] <= 64), 'Age'] = 3

df_test.loc[ df_test['Age'] > 64, 'Age'] = 4 ;

# Mapping titles

title_mapping = {" Mr": 1, " Miss": 2, " Mrs": 3, " Master": 4, "Rare": 5}

df['title name'] = df['title name'].map(title_mapping)

df['title name'] = df['title name'].fillna(0)



df_test['title name'] = df_test['title name'].map(title_mapping)

df_test['title name'] = df_test['title name'].fillna(0)
# Mapping Embarked

df['Embarked'] = df['Embarked'].fillna('S')

df_test['Embarked'] = df_test['Embarked'].fillna('S')



df['Embarked'] = df['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

df_test['Embarked'] = df_test['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
df['Fare'] = df['Fare'].fillna(df['Fare'].median())

df_test['Fare'] = df_test['Fare'].fillna(df['Fare'].median())

# Mapping Fare

df.loc[ df['Fare'] <= 7.91, 'Fare']= 0

df.loc[(df['Fare'] > 7.91) & (df['Fare'] <= 14.454), 'Fare'] = 1

df.loc[(df['Fare'] > 14.454) & (df['Fare'] <= 31), 'Fare']   = 2

df.loc[ df['Fare'] > 31, 'Fare']= 3

df['Fare'] = df['Fare'].astype(int)



df_test.loc[ df['Fare'] <= 7.91, 'Fare']= 0

df_test.loc[(df['Fare'] > 7.91) & (df_test['Fare'] <= 14.454), 'Fare'] = 1

df_test.loc[(df['Fare'] > 14.454) & (df_test['Fare'] <= 31), 'Fare']   = 2

df_test.loc[ df_test['Fare'] > 31, 'Fare']= 3

df_test['Fare'] = df_test['Fare'].astype(int)
df.head()
sns.heatmap(df.isnull())
(df.isnull().sum()*100)/df.shape[0]
df_test.shape
(df_test.isnull().sum()*100)/df_test.shape[0]
(df_test.isnull().sum()*100)/df_test.shape[0]
df['size of family'] = df['SibSp'] + df['Parch'] + 1

df.drop('SibSp',inplace=True,axis=1)



df_test['size of family'] = df_test['SibSp'] + df_test['Parch'] + 1

df_test.drop('SibSp',inplace=True,axis=1)
df['is alone']=df.apply((lambda row: 1 if (row['size of family']==1) else 0),axis=1)

df_test['is alone']=df_test.apply((lambda row: 1 if (row['size of family']==1) else 0),axis=1)
plt.figure(figsize=(12,8))

sns.heatmap(df.corr(),cmap = sns.diverging_palette(220, 10, as_cmap=True),center=0,annot=True)
features=['Pclass','Sex','Age','Fare','Embarked','title name', 'size of family', 'is alone', 'Survived']

df_dummies= pd.get_dummies(df[features])

features=['Pclass','Sex','Age','Fare','Embarked','title name','size of family', 'is alone']

df_dummies_test = pd.get_dummies(df_test[features])
df_dummies.head()
df_dummies.drop(['Sex_male'],axis=1,inplace=True)
df_dummies_test.drop(['Sex_male'],axis=1,inplace=True)
df_dummies_test.head()
from sklearn.model_selection import train_test_split

X = df_dummies.drop('Survived',axis=1).values

y = df_dummies['Survived'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

scaler.fit(X_train)

X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)
import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Activation,Dropout
X_train.shape
X_test.shape
model= Sequential()



model.add(Dense(units=12,activation='relu'))

model.add(Dropout(0.4))



model.add(Dense(units=10,activation='relu'))

model.add(Dropout(0.4))



model.add(Dense(units=6,activation='relu'))

model.add(Dropout(0.4))



model.add(Dense(units=1,activation='sigmoid'))



model.compile(loss='binary_crossentropy',optimizer='adam')
from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=100)
model.fit(x=X_train, 

          y=y_train,

          epochs=600,

          validation_data=(X_test, y_test), verbose=1,

          callbacks=[early_stop]

          )
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=200,max_depth=12)

rfc.fit(X_train,y_train)
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()

logreg.fit(X_train, y_train)
model_loss = pd.DataFrame(model.history.history)

model_loss.plot()
# 
predictions = model.predict_classes(X_test)

from sklearn.metrics import classification_report,confusion_matrix

print(classification_report(y_test,predictions),'\n')

print(confusion_matrix(y_test,predictions))
predictions = rfc.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix

print(classification_report(y_test,predictions),'\n')

print(confusion_matrix(y_test,predictions))
predictions = logreg.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix

print(classification_report(y_test,predictions),'\n')

print(confusion_matrix(y_test,predictions))
df_dummies_test.head()
X_val = scaler.transform(df_dummies_test.values)

predictions = model.predict_classes(X_val)
output = pd.DataFrame(columns=['PassengerId','Survived'],data=zip(df_test['PassengerId'].values,np.hstack(predictions)))
output.reset_index(inplace=True,drop=True)
output.to_csv('my_submission.csv', index=False)