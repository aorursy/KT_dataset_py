import pandas as pd
from sklearn import svm
import matplotlib.pyplot as plt
import numpy as np
t_train = pd.read_csv('../input/train.csv')
print(t_train.head())
print(len(t_train.columns))
t_test = pd.read_csv('../input/test.csv')
t_test.head()
plt.hist(t_train['Survived'])
plt.show()
plt.scatter(t_train['Fare'],t_train['Survived'])
plt.show()
plt.hist([t_train[t_train['Survived']==1]['Fare'],t_train[t_train['Survived']==0]['Fare']],label=t_train['Survived'])
plt.xlabel('Fare')
plt.ylabel('Passenger Count')
plt.show()
plt.scatter(t_train['SibSp'],t_train['Survived'])
plt.show()
print(t_train.isnull().sum())
print(t_test.isnull().sum())
import seaborn as sns
sns.countplot('Survived',data = t_train)
plt.show()
sns.countplot('Sex',data=t_train)
plt.show()
sns.countplot('Embarked',data=t_train)
plt.show()
t_train.groupby(['Sex','Survived'])['Survived'].count()
t_train.groupby(['Embarked','Survived'])['Survived'].count()
sns.countplot('Sex',hue='Survived',data=t_train)
plt.show()
pd.crosstab(t_train.Pclass,t_train.Survived,margins=True)
t_train['Pclass'].value_counts()
sns.countplot('Pclass',hue='Survived',data=t_train)
plt.show()
pd.crosstab([t_train.Sex,t_train.Survived],t_train.Pclass,margins=True).style.background_gradient(cmap='summer_r')
sns.factorplot('Pclass','Survived',hue='Sex',data=t_train)
plt.show()
print(t_train['Age'].min())
sns.violinplot("Pclass","Age", hue="Survived", data=t_train,split=True)
plt.show()
sns.violinplot("Sex","Age", hue="Survived", data=t_train,split=True)
plt.show()
t_train['Initial']=0
for i in t_train:
    t_train['Initial']=t_train.Name.str.extract('([A-Za-z]+)\.')
print(t_train['Initial'])  

t_test['Initial']=0
for i in t_test:
    t_test['Initial']=t_test.Name.str.extract('([A-Za-z]+)\.')
print(t_test['Initial'])  
pd.crosstab(t_train.Initial,t_train.Sex).T.style.background_gradient(cmap='summer_r')

pd.crosstab(t_test.Initial,t_test.Sex).T.style.background_gradient(cmap='summer_r')
t_train['Initial'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don'],['Miss','Miss','Miss','Sir','Capt','Nobal','Nobal','Nobal','Capt','Sir','Capt','Sir','Nobal'],inplace=True)
t_test['Initial'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don','Dona'],['Miss','Miss','Miss','Sir','Capt','Nobal','Nobal','Nobal','Capt','Sir','Capt','Sir','Nobal','Nobal'],inplace=True)

t_test.groupby('Initial')['Age'].mean()
t_train.groupby('Initial')['Age'].mean()
t_train.loc[(t_train.Age.isnull())&(t_train.Initial=='Mr'),'Age']=33
t_train.loc[(t_train.Age.isnull())&(t_train.Initial=='Mrs'),'Age']=36
t_train.loc[(t_train.Age.isnull())&(t_train.Initial=='Master'),'Age']=5
t_train.loc[(t_train.Age.isnull())&(t_train.Initial=='Miss'),'Age']=22
t_train.loc[(t_train.Age.isnull())&(t_train.Initial=='Sir'),'Age']=43
t_train.loc[(t_train.Age.isnull())&(t_train.Initial=='Nobal'),'Age']=40
t_train.loc[(t_train.Age.isnull())&(t_train.Initial=='Capt'),'Age']=57
t_train.Age.isnull().any()
t_test.loc[(t_test.Age.isnull())&(t_test.Initial=='Mr'),'Age']=33
t_test.loc[(t_test.Age.isnull())&(t_test.Initial=='Mrs'),'Age']=36
t_test.loc[(t_test.Age.isnull())&(t_test.Initial=='Master'),'Age']=5
t_test.loc[(t_test.Age.isnull())&(t_test.Initial=='Miss'),'Age']=22
t_test.loc[(t_test.Age.isnull())&(t_test.Initial=='Sir'),'Age']=43
t_test.loc[(t_test.Age.isnull())&(t_test.Initial=='Nobal'),'Age']=40
t_test.loc[(t_test.Age.isnull())&(t_test.Initial=='Capt'),'Age']=57
t_test.Age.isnull().any()
sns.factorplot('Pclass','Survived',col='Initial',data=t_train)
plt.show()
sns.factorplot('Embarked','Survived',data=t_train)
plt.show()
pd.crosstab([t_train['Embarked'],t_train['Survived']],t_train['Pclass'])
t_train['Embarked'].fillna('S',inplace=True)
t_train.Embarked.isnull().any()
pd.crosstab([t_train.SibSp,t_train.Pclass],t_train.Survived)
sns.factorplot('SibSp','Survived',hue='Pclass',data=t_train)
plt.show()
sns.heatmap(t_train.corr(),annot=True)
plt.show()
t_train['Age_band']=0
t_train.loc[t_train['Age']<=16,'Age_band']=0
t_train.loc[(t_train['Age']>16)&(t_train['Age']<=32),'Age_band']=1
t_train.loc[(t_train['Age']>32)&(t_train['Age']<=48),'Age_band']=2
t_train.loc[(t_train['Age']>48)&(t_train['Age']<=64),'Age_band']=3
t_train.loc[t_train['Age']>64,'Age_band']=4
t_train.head(5)
t_test['Age_band']=0
t_test.loc[t_test['Age']<=16,'Age_band']=0
t_test.loc[(t_test['Age']>16)&(t_test['Age']<=32),'Age_band']=1
t_test.loc[(t_test['Age']>32)&(t_test['Age']<=48),'Age_band']=2
t_test.loc[(t_test['Age']>48)&(t_test['Age']<=64),'Age_band']=3
t_test.loc[t_test['Age']>64,'Age_band']=4
t_test.head(5)
t_train['Age_band'].value_counts()
#sns.factorplot('Age_band','Survived',hue='Pclass',data=t_train)
#plt.show()
t_train['Fare_Range']=pd.qcut(t_train['Fare'],4)
t_test['Fare_Range']=pd.qcut(t_test['Fare'],4)
t_train.groupby(['Fare_Range'])['Survived'].mean().to_frame()
t_train['Fare_cat']=0
t_train.loc[t_train['Fare']<=7.91,'Fare_cat']=0
t_train.loc[(t_train['Fare']>7.91)&(t_train['Fare']<=14.454),'Fare_cat']=1
t_train.loc[(t_train['Fare']>14.454)&(t_train['Fare']<=31),'Fare_cat']=2
t_train.loc[(t_train['Fare']>31)&(t_train['Fare']<=513),'Fare_cat']=3
sns.factorplot('Fare_cat','Survived',data=t_train,hue='Sex')
plt.show()
t_test['Fare_cat']=0
t_test.loc[t_test['Fare']<=7.91,'Fare_cat']=0
t_test.loc[(t_test['Fare']>7.91)&(t_test['Fare']<=14.454),'Fare_cat']=1
t_test.loc[(t_test['Fare']>14.454)&(t_test['Fare']<=31),'Fare_cat']=2
t_test.loc[(t_test['Fare']>31)&(t_test['Fare']<=513),'Fare_cat']=3
#sns.factorplot('Fare_cat','Survived',data=t_train,hue='Sex')
#plt.show()
t_train.loc[(t_train['Sex']=='male'),'Sex']=0
t_train.loc[(t_train['Sex']=='female'),'Sex']=1
t_train=pd.get_dummies(t_train, columns=['Embarked','Initial','Pclass','Fare_cat'])
#t_train['Embarked'].replace(['S','C','Q'],[0,1,2],inplace=True)
#t_train['Initial'].replace(['Mr','Mrs','Miss','Master','Other'],[0,1,2,3,4],inplace=True)
t_test.loc[(t_test['Sex']=='male'),'Sex']=0
t_test.loc[(t_test['Sex']=='female'),'Sex']=0
t_test=pd.get_dummies(t_test, columns=['Embarked','Initial','Pclass','Fare_cat'])
#t_test['Embarked'].replace(['S','C','Q'],[0,1,2],inplace=True)
#t_test['Initial'].replace(['Mr','Mrs','Miss','Master','Other'],[0,1,2,3,4],inplace=True)
t_train.head()
t_train.drop(['Name','Ticket','Fare','Cabin','PassengerId'],axis=1,inplace=True)
t_test1 = t_test.drop(['Name','Ticket','Fare','Cabin','PassengerId'],axis=1)
sns.heatmap(t_train.corr(),annot=True)
fig=plt.gcf()
fig.set_size_inches(14,14)
plt.show()
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
train,cv = train_test_split(t_train,test_size=0.3,random_state=3)
train_y=train['Survived']
train_x = train.drop(['Survived','Fare_Range'],axis=1)
train_x.head()
t_test.head()
train_x.head()
cv_y=cv['Survived']
cv_x = cv.drop(['Survived','Fare_Range'],axis=1)
cv_x.head()
print(cv.shape)
print(train.shape)
print(t_test1.shape)
t_test1 = t_test1.drop(['Fare_Range'],axis=1)
t_test1.head()
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectFromModel
parameters = {'bootstrap': False, 'min_samples_leaf': 3, 'n_estimators': 50, 
                  'min_samples_split': 10, 'max_features': 'sqrt', 'max_depth': 6}
    
model = RandomForestClassifier(bootstrap= False, min_samples_leaf= 3, n_estimators= 50, 
                  min_samples_split= 10, max_features= 'sqrt', max_depth= 6)
model.fit(train_x, train_y)
print(train_x.columns)
print(t_test1.columns)
pred = model.predict(t_test1)
submission = pd.DataFrame({"PassengerId":t_test['PassengerId'],"Survived":pred})
submission.to_csv("titan_random_forest_result.csv",index=False)