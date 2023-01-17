import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline

from pandas import read_csv

from pandas.plotting import scatter_matrix



from numpy import mean

from numpy import std



from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import StandardScaler

from sklearn.compose import ColumnTransformer



from sklearn import preprocessing

from sklearn.model_selection import cross_validate

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import RepeatedStratifiedKFold



from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import BaggingClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.linear_model import SGDClassifier





from sklearn import model_selection

from sklearn.metrics import accuracy_score,confusion_matrix, classification_report

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

df = pd.read_csv('../input/adult-census-income/adult.csv')

df.head()
# drop rows with missing

df = df.dropna()
# summarize the shape of the dataset

print(df.shape)
df.describe()
df.info()
df.isnull().values.any()
df['income']=LabelEncoder().fit_transform(df['income'])
fig=plt.figure(figsize=(10,6))

sns.countplot('workclass',data=df,hue="income" )

plt.tight_layout()

plt.show()
fig=plt.figure(figsize=(10,6))

sns.countplot('education',data=df)

plt.xticks(rotation=90)

plt.tight_layout()

plt.show()
fig=plt.figure(figsize=(10,6))

sns.countplot('marital.status',data=df )

plt.xticks(rotation=90)

plt.tight_layout()

plt.show()
fig=plt.figure(figsize=(10,6))

sns.countplot('occupation',data=df )

plt.xticks(rotation=90)

plt.tight_layout()

plt.show()
fig=plt.figure(figsize=(10,6))

sns.countplot('sex',data=df,hue="income")

plt.tight_layout()

plt.show()
fig=plt.figure(figsize=(10,6))

sns.countplot('race',data=df )

plt.tight_layout()

plt.show()
fig=plt.figure(figsize=(10,6))

sns.countplot('native.country',data=df.head(200) )

plt.tight_layout()

plt.show()
f,ax=plt.subplots(1,3,figsize=(25,5))

box1=sns.boxplot(data=df["fnlwgt"],ax=ax[0],color='m')

ax[0].set_xlabel('fnlwgt')

box1=sns.boxplot(data=df["hours.per.week"],ax=ax[1],color='m')

ax[1].set_xlabel('hours.per.week')

box1=sns.boxplot(data=df["age"],ax=ax[2],color='m')

ax[2].set_xlabel('age')
sns.boxplot(x="age",y="sex",hue="income",data=df)
#df1= df.corr()

corr = (df.corr())

plt.subplots(figsize=(9, 9))

sns.heatmap(corr, vmax=.8,annot=True,cmap="viridis", square=True);
df1=df.drop(['income'],axis=1)

df1.hist (bins=10,figsize=(20,20))

plt.show ()
sns.pairplot(data=df,kind='reg',size=5)
sns.pairplot(df,hue = 'income',vars = ['fnlwgt','hours.per.week','education.num'] )
ax = sns.violinplot(x="education.num", y="income", data=df, palette="muted")
df=df.dropna()
df['sex'] = LabelEncoder().fit_transform(df['sex'])
x = df.drop(['income','workclass','education','marital.status','occupation','relationship','race','native.country'],axis=1)

y= df['income']

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=101)
cat_ix = x.select_dtypes(include=['object', 'bool']).columns 

num_ix = x.select_dtypes(include=['int64', 'float64']).columns 
seed=101

models = []

models.append(('RF',RandomForestClassifier()))

models.append(('SGDC',SGDClassifier()))

models.append (('CART',DecisionTreeClassifier()))

models.append (('BAG',BaggingClassifier()))

models.append(('LR',LogisticRegression()))

models.append(('GBM',GradientBoostingClassifier()))

results = []

names = []

for name, model in models:

    cv=RepeatedStratifiedKFold(n_splits=10,n_repeats=3,random_state=seed)

    cv_results = cross_val_score(model, x_train, y_train,scoring='accuracy',cv=cv,n_jobs=-1)

    results.append(cv_results)

    names.append(name)

    msg = "%s: %f (%f)" %(name, cv_results.mean(), cv_results.std())

    print(msg)
from sklearn.metrics import classification_report, confusion_matrix

logistic = LogisticRegression()

logistic.fit(x_train,y_train)

y_pred=logistic.predict(x_test)

print(classification_report(y_test,y_pred))

accuracy1=logistic.score(x_test,y_test)

print (accuracy1*100,'%')

cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot= True)
des_class=DecisionTreeClassifier()

des_class.fit(x_train,y_train)

des_predict=des_class.predict(x_test)

print(classification_report(y_test,des_predict))

accuracy3=des_class.score(x_test,y_test)

print(accuracy3*100,'%')

cm = confusion_matrix(y_test, des_predict)

sns.heatmap(cm, annot= True)
Bag=BaggingClassifier()

Bag.fit(x_train,y_train)

Bag_predict=Bag.predict(x_test)

print(classification_report(y_test,Bag_predict))

accuracy3=Bag.score(x_test,y_test)

print(accuracy3*100,'%')

cm = confusion_matrix(y_test, Bag_predict)

sns.heatmap(cm, annot= True)
from sklearn.ensemble import RandomForestClassifier 

ran_class=RandomForestClassifier()

ran_class.fit(x_train,y_train)

ran_predict=ran_class.predict(x_test)

print(classification_report(y_test,ran_predict))

accuracy3=ran_class.score(x_test,y_test)

print(accuracy3*100,'%')

cm = confusion_matrix(y_test, ran_predict)

sns.heatmap(cm, annot= True)
Sgdc=SGDClassifier()

Sgdc.fit(x_train,y_train)

Sgdc_predict=Sgdc.predict(x_test)

print(classification_report(y_test,Sgdc_predict))

accuracy3=Sgdc.score(x_test,y_test)

print(accuracy3*100,'%')

cm = confusion_matrix(y_test, Sgdc_predict)

sns.heatmap(cm, annot= True)
gbc=GradientBoostingClassifier()

gbc.fit(x_train,y_train)

gbc_predict=gbc.predict(x_test)

print(classification_report(y_test,gbc_predict))

accuracy3=gbc.score(x_test,y_test)

print(accuracy3*100,'%')

cm = confusion_matrix(y_test, gbc_predict)

sns.heatmap(cm, annot= True)