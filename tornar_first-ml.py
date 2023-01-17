import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



df=pd.read_csv('../input/train.csv')

df_test=pd.read_csv('../input/test.csv')
df.describe()
df.tail()
#check if there is any 0 value

for column in df.columns.values:

    print('0 values found in {0} {1}'.format(column,len(df.loc[df[column]==0])))

print('Null value found'

      ''

      '{}'.format(df.isnull().sum()))
df.hist(bins=10,figsize=(9,7),grid=False)
#try to visually detect any pattern



%matplotlib inline

import seaborn as sns

sns.set(font_scale=1)

g=sns.FacetGrid(df,col='Sex',row='Survived',margin_titles=True)

g.map(plt.hist,'Age',color='Orange')
g=sns.FacetGrid(df,hue='Survived',col="Pclass",margin_titles=True,palette={1:'red',0:'black'})

g=g.map(plt.scatter,"Fare","Age",edgecolor='w').add_legend()
#three people paid 500 to buy the same class 1 ticket which largely skewed the data

df.loc[df['Fare']>500,'Fare']=0 #redue to 0 and later i will replace with median
g = sns.FacetGrid(df, hue="Survived", col="Sex", margin_titles=True,

                palette="Set1",hue_kws=dict(marker=["^", "v"]))

g.map(plt.scatter, "Fare", "Age",edgecolor="w").add_legend()

plt.subplots_adjust(top=0.8)

g.fig.suptitle('Survival by Gender , Age and Fare');
g=sns.FacetGrid(df,hue='Survived',col="Embarked",margin_titles=True,palette={1:'red',0:'black'})

g=g.map(plt.scatter,"Fare","Age",edgecolor='w').add_legend()
corr=df.corr()#["Survived"]

plt.figure(figsize=(10, 10))



sns.heatmap(corr, vmax=.8, linewidths=0.01,

            square=True,annot=True,cmap='YlGnBu',linecolor="white")

plt.title('Correlation Between Features');
#checking null and 0 values

df.loc[df['Embarked'].isnull()]
fillembarked=df[df['Ticket'].str.contains('113')]

g = sns.FacetGrid(fillembarked, col="Embarked", row="Pclass", margin_titles=True)

g.map(plt.hist, "Fare",color="red");
#I see that tickets with similar series are all Pclass 1 and most are from S'

df['Embarked']=df['Embarked'].fillna('S')
df.groupby('Pclass', as_index=False)['Fare'].median()
#replace value with ticket class'

fill_median = lambda x: x.fillna(x.median())   # Write function that Fills NAs in subset

df.groupby('Pclass')['Fare'].apply(fill_median)  # Apply function to grouped column 



#replace age with median'

df['Age']=df['Age'].fillna(df['Age'].median())



#fill up 0 fare value as well

df.loc[(df['Pclass']==1) & (df['Fare']==0),'Fare']=57.49

df.loc[(df['Pclass']==2) & (df['Fare']==0),'Fare']=14.25

df.loc[(df['Pclass']==3) & (df['Fare']==0),'Fare']=8.05
#check if there is any 0 and Null value



for column in df.columns.values:

    print('0 values found in {0} {1}'.format(column,len(df.loc[df[column]==0])))

print('Null value found'

      ''

      '{}'.format(df.isnull().sum()))
#check if people got mutiple cabin also got spouse or children'

df[df['Cabin'].str.len()>4]

df['Alone']=df['SibSp']+df['Parch']

#'single person booked multiple rooms?'

#1st class room paid only 5?? G rooms is engine room. 

df.loc[(df['Alone']==0) & (df['Cabin'].str.len()>4)]
#Cabin G is lower class, which is below F. I am not differentiate them

#I will add a SibSp to ID 790 and 873 and change fare of 873 to median 57

df.loc[790,'SibSp']=1

df.loc[873,'SibSp']=1

df.loc[873,'Fare']=57.48
#get the cabin letter, which determines the location of the passenger

df['CabinL']=df['Cabin'].str[:1]


df.Pclass.value_counts().plot(kind='bar', alpha=0.55)

plt.title("Passengers by Deck Location");



g = sns.FacetGrid(df, col="CabinL", row="Pclass", margin_titles=True)

g.map(plt.hist, "Fare",color="red")
#Try to determine the CabinL from Pclass, referred to floor plan titanic online

df.loc[(df['Cabin'].isnull()) & (df['Fare']>60),'CabinL']='B'

df.loc[(df['Cabin'].isnull()) & (df['Fare']>14),'CabinL']='E'

df.loc[(df['Cabin'].isnull()),'CabinL']='F'
df.Pclass.value_counts().plot(kind='bar', alpha=0.55)

plt.title("Passengers by Deck Location");



g = sns.FacetGrid(df, col="CabinL", row="Pclass", margin_titles=True)

g.map(plt.hist, "Fare",color="red")
#from my research i don't really see cabin T, so i am allocating him to C based on his Pclass and Fare

df.loc[(df['CabinL']=='T')]

df.loc[(339,'CabinL')]='C'
#final check of any missing data



for column in df.columns.values:

    print('0 values found in {0} {1}'.format(column,len(df.loc[df[column]==0])))

print('Null value found'

      ''

      '{}'.format(df.isnull().sum()))
from sklearn.preprocessing import LabelEncoder

sex_le=LabelEncoder()

df['Sex']=sex_le.fit_transform(df['Sex'].values) #convert nominal value to integer

df['Embarked']=sex_le.fit_transform(df['Embarked'].values)#convert nominal value to integer
cabinL_map={'A':1,'B':2,'C':3,'D':4,'E':5,'F':6,'G':7}#convert cabin according to floor level. 

df['CabinL']=df['CabinL'].map(cabinL_map)
#create one-hot encoding

from sklearn.preprocessing import OneHotEncoder as ohe

ohe_1=ohe(categorical_features=[4],sparse=False)

ohe_1.fit_transform(df)

#need to create dummy sex and embarked categories, but got error. will debug further later
df_train_ready=df.drop(['Name','Ticket','Alone','Cabin','PassengerId','CabinL'],axis=1)



df_train_ready.head()
y=df_train_ready['Survived']

x=df_train_ready.drop(['Survived'],axis=1)
#evaluate the feature importance

from sklearn.ensemble import RandomForestClassifier

feat_labels=x.columns

forest=RandomForestClassifier(n_estimators=1000,random_state=42,n_jobs=-1)

forest.fit(x,y)

importances=forest.feature_importances_

indices=np.argsort(importances)[::-1]

plt.title('Feature Importances')

plt.bar(range(x.shape[1]),

       importances[indices],

       color='lightblue',

       align='center')

plt.xticks(range(x.shape[1]),

          feat_labels[indices],rotation=90)

plt.xlim([-1,x.shape[1]])

plt.tight_layout()

plt.show()
#spliting the training data to test and train

from sklearn import datasets

from sklearn import metrics

from sklearn.model_selection import train_test_split

split_test_size=0.30

x_train, x_test,y_train,y_test=train_test_split(x,y,test_size=split_test_size,random_state=42)
from sklearn.linear_model import LogisticRegressionCV

#run multiple runs loggistic regression by using cross validation technique'

lr_cv_model=LogisticRegressionCV(n_jobs=-1,Cs=2,cv=5,refit=True,class_weight='balanced',random_state=42)

lr_cv_model.fit(x_train,y_train)
#using basic Gaussian

from sklearn.naive_bayes import GaussianNB

nb_model=GaussianNB()

nb_model.fit(x_train,y_train)

GaussianNB()
#check the accuracy

nb_predict_train=lr_cv_model.predict(x_train)

nb_predict_test=lr_cv_model.predict(x_test)

print('Accuracy Report\n'

          )

print('Accuracy on test: {0:4f}'.format(metrics.accuracy_score(y_test,nb_predict_test)))

print('Accuracy on train: {0:4f}'.format(metrics.accuracy_score(y_train,nb_predict_train)))

print("\n"

          'Confusion Matrix\n'

          'TP   FP\n'

          'FN   TN\n'

          '{0}'.format(metrics.confusion_matrix(y_test,nb_predict_test,labels=[1,0])))

print('\n*CLASSIFICATION REPORT:\n')

print(metrics.classification_report(y_test,nb_predict_test,labels=[1,0]))
df_test.describe() #missing date in fare, Cabin and Age

from sklearn.preprocessing import LabelEncoder

sex_le=LabelEncoder()

df_test['Sex']=sex_le.fit_transform(df_test['Sex'].values) #convert nominal value to integer

df_test['Embarked']=sex_le.fit_transform(df_test['Embarked'].values)#convert nominal value to integer
#check if there is any 0 value

for column in df_test.columns.values:

    print('0 values found in {0} {1}'.format(column,len(df_test.loc[df_test[column]==0])))

print('Null value found'

      ''

      '{}'.format(df_test.isnull().sum()))
df_test['Fare']=df_test['Fare'].fillna(57) #filling up the 0 with Pclass 1 median

df_test['Age']=df_test['Age'].fillna(df_test['Age'].median())

df_test['Cabin'] #too many Nan, I have to drop this feature
df_test_ready=df_test.drop(['Name','Ticket','Cabin','PassengerId'],axis=1)

df_test_ready.info()
lr_cv_predict_test=lr_cv_model.predict(df_test_ready)
#generate the submission file

submission = pd.DataFrame({

        "PassengerId": df_test["PassengerId"],

        "Survived": lr_cv_predict_test

    })

submission.to_csv("titanic_submission.csv", index=False)