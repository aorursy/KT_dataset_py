#Importing Libararies

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import sklearn

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.feature_selection import RFE

import statsmodels.api as sm

import warnings

warnings.filterwarnings('ignore')
# Reading the data

train_data=pd.read_csv('../input/titanic/train.csv')

train_data.head()
train_data.shape
train_data.describe()
# checking the percentage of missing values

round(100*train_data.isnull().sum()/train_data.shape[0],2)

# Dropping columns that are not required for analysis

train_data.drop(['Name','Ticket', 'Cabin'], axis=1, inplace=True)
# Imputing the NaN values in Age with the median value

train_data['Age'].fillna(train_data['Age'].median(), inplace=True)
# Only 2 values n Embarked column are missing and hence the corresponding rows can be dropped

train_data= train_data.dropna()

train_data.shape
# checking the percentage of missing values

round(100*train_data.isnull().sum()/train_data.shape[0],2)
train_data.columns
# Definning a function to plot barchart

import math

def func_bar(*args):                        

   

    m=math.ceil(len(args)/2)  # getting the length f arguments to determine the shape of subplots                   

    

    fig,axes = plt.subplots(m,2,squeeze=False, figsize = (16, 6*m))

    ax_li = axes.flatten()       # flattening the numpy array returned by subplots

    i=0

    for col in args:

        

        sns.countplot(x=col, data=train_data,ax=ax_li[i], order = train_data[col].value_counts().index)

        ax_li[i].set_title(col)

        plt.tight_layout()

        i=i+1
bar_list=['Survived', 'Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked']

func_bar(*bar_list)
survived=(sum(train_data['Survived'])/len(train_data['Survived'].index))*100

survived
# Checking the distribution of Age and fare

fig,axes = plt.subplots(1,2,squeeze=False, figsize = (16,6))

#plt.subplot(121)

sns.boxplot(y=train_data['Age'],ax=axes[0,0])

plt.yscale('log')

sns.boxplot(y=train_data['Fare'],ax=axes[0,1])

plt.yscale('log')

# function to plot stacked bar charts

import math

from textwrap import wrap

def func_stk(*args):                        

   

    m=math.ceil(len(args)/2)  # getting the length f arguments to determine the shape of subplots                   

    

    fig,axes = plt.subplots(m,2,squeeze=False, figsize = (16, 6*m))

    ax_li = axes.flatten()       # flattening the numpy array returned by subplots

    

    i=0

    for col in args:

        gf=train_data.groupby(col)['Survived'].value_counts(normalize=True)*100       #grouping by target and finding the percentage value of target and non target category

        gf=gf.unstack().plot(kind='bar',width=0.5, stacked=True,ax=ax_li[i])  # plotting a stcked bar chart

        ax_li[i].legend(['Dead','Survived'],bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.) # changing the legend 

        gf.set_ylabel('Percentage')

        i+=1

        
func_stk(*bar_list[1:])
# function to plot stackd histogram

import math

from textwrap import wrap

def func_hist(*args):                        

   

    m=len(args)  # getting the length f arguments to determine the shape of subplots                   

    

    fig,axes = plt.subplots(m,1,squeeze=False, figsize = (16, 6*m))

    ax_li = axes.flatten()       # flattening the numpy array returned by subplots

    

    i=0

    for col in args:

        train_data.pivot(columns='Survived')[col].plot(kind = 'hist', bins=10,rwidth=0.95,stacked=True,ax=ax_li[i])

        ax_li[i].legend(['Dead','Survived'],bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.) # changing the legend 

        ax_li[i].set_xlabel(col)

        i=i+1
func_hist(*['Age','Fare'])
#Converting sex column to numerical values 

# Assign 1 for female and 0 for male

train_data['Sex']=train_data[['Sex']].apply(lambda x :x.map({'female':1,'male':0}))
#Perform OneHot Encoding for embarked column

emb = pd.get_dummies(train_data['Embarked'], drop_first=True)

train_data.drop('Embarked',axis=1, inplace=True)

train_data=pd.concat([train_data,emb],axis=1)

train_data.head()
# Scaling the "Age" and "Fare" columns

scaler = MinMaxScaler()

train_data[["Age","Fare"]] = scaler.fit_transform(train_data[["Age","Fare"]])

train_data.head()
#Determing the correlation between variables

plt.figure(figsize = (12, 8))

sns.heatmap(train_data.corr(), annot = True, cmap="YlGnBu")

plt.show()
# Putting feature variable to X

X = train_data.drop(['PassengerId','Survived'], axis=1)

X.head()
# Putting response variable to y

y = train_data['Survived']

y.head()
# Logistic regression model

lgm1 = sm.GLM(y,(sm.add_constant(X)), family = sm.families.Binomial())

lgm1.fit().summary()
logreg = LogisticRegression()

rfe = RFE(logreg, 5)             # running RFE with 13 variables as output

rfe = rfe.fit(X, y)
list(zip(X.columns, rfe.support_, rfe.ranking_))
X = train_data.drop(['PassengerId','Survived','Q'], axis=1)

# Logistic regression model

lgm2 = sm.GLM(y,(sm.add_constant(X)), family = sm.families.Binomial())

lgm2.fit().summary()
X = train_data.drop(['PassengerId','Survived','Q','Parch'], axis=1)

# Logistic regression model

lgm3 = sm.GLM(y,(sm.add_constant(X)), family = sm.families.Binomial())

lgm3.fit().summary()
X = train_data.drop(['PassengerId','Survived','Q','Parch','Fare'], axis=1)



X_sm = sm.add_constant(X[X.columns])



lgm4 = sm.GLM(y,X_sm, family = sm.families.Binomial())

res = lgm4.fit()

res.summary()

# Check for the VIF values of the feature variables. 

from statsmodels.stats.outliers_influence import variance_inflation_factor
# Create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif = pd.DataFrame()

vif['Features'] = X.columns

vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
# Predicting the survival chance on train data

y_train_pred = res.predict(X_sm)

y_train_pred[:10]
thresh= 0.65  # fixed by trial and error

y_train_pred_val = y_train_pred.apply(lambda x: 1 if x > thresh else 0)



y_train_pred_final = pd.DataFrame({'Survived': y,'Predicted':y_train_pred_val})

y_train_pred_final['PassengerId'] = train_data['PassengerId'].astype(int)

col=["PassengerId","'Predicted'"]

y_train_pred_final.head()
y_train_pred_final.head()
#Confusion matrix

from sklearn import metrics

confusion = metrics.confusion_matrix(y_train_pred_final.Survived, y_train_pred_final.Predicted )

print(confusion)
#Checking Accuracy

print(round(metrics.accuracy_score(y_train_pred_final.Survived, y_train_pred_final.Predicted),4))
# Reading test data

test_data=pd.read_csv('../input/titanic/test.csv')

test_data.head()
#Converting sex column to numerical values 

# Assign 1 for female and 0 for male

test_data['Sex']=test_data[['Sex']].apply(lambda x :x.map({'female':1,'male':0}))

#Perform OneHot Encoding for embarked column

emb = pd.get_dummies(test_data['Embarked'], drop_first=True)

test_data.drop('Embarked',axis=1, inplace=True)

test_data=pd.concat([test_data,emb],axis=1)

test_data.head()
# Scaling the test data

test_data[["Age","Fare"]] = scaler.transform(test_data[["Age","Fare"]])

test_data.head()
X_test = test_data[['Pclass','Sex','Age','SibSp','S']]

X_test_sm = sm.add_constant(X_test)
y_pred = res.predict(X_test_sm)

y_pred[:10]
thresh= 0.65     

y_pred_val = y_pred.apply(lambda x: 1 if x > thresh else 0)



y_pred_final = pd.DataFrame({'Survived':y_pred_val})

y_pred_final['PassengerId'] = test_data['PassengerId'].astype(int)

col=["PassengerId","Survived"]

y_pred_final=y_pred_final.reindex()

y_pred_final.set_index('PassengerId')

y_pred_final.head()
cols = list(y_pred_final)

cols[1], cols[0] = cols[0], cols[1]

y_pred_final=y_pred_final.loc[:,cols]

y_pred_final.set_index('PassengerId',inplace=True)

y_pred_final.head()