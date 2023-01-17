# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# First we get the train data in
data_train=pd.read_csv('../input/train.csv')
#explore the data
data_train.head()
## check the missing value
data_train.isnull().sum()
# for the 'Embarked' column 
data_train.loc[data_train['Embarked'].isnull(),:]
## transform the Sex column into category variables
data_train['Sex']=data_train.loc[:,'Sex'].astype('category')
data_train['Sex'].head()
## view the Embared column
data_train['Embarked'].value_counts()
## drop the rows for Embarked column is NaN
data_train.dropna(how='any',subset=['Embarked'],inplace=True)
## check for missing value
data_train.isnull().sum()
# transfer Embarked column into category variables
data_train['Embarked']=data_train['Embarked'].astype('category')
data_train['Embarked'].head()
## Let's do some Explory data analysis
fig=plt.figure(figsize=[15,10])
## use plt.subplot2grid() function
plt.subplot2grid(shape=[2,3],loc=[0,0])
data_train['Survived'].value_counts().plot(kind='bar')
plt.ylabel('Survived number')
plt.title('Survived or not')

## Embarked place
plt.subplot2grid([2,3],[0,1])
data_train['Embarked'].value_counts().plot(kind='bar')
plt.ylabel('Survived number')
plt.title('Embarked place')

## ticket class
plt.subplot2grid([2,3],[0,2])
data_train['Pclass'].value_counts().plot(kind='bar')
plt.ylabel('Survived number')
plt.title('ticket class')

## age
plt.subplot2grid([2,3],[1,0],colspan=2)
data_train['Age'].loc[data_train['Pclass']==1].plot(kind='kde')
data_train['Age'].loc[data_train['Pclass']==2].plot(kind='kde')
data_train['Age'].loc[data_train['Pclass']==3].plot(kind='kde')
plt.title('Age distribution for every Ticket class')
plt.xlabel('Age')
plt.legend(['Upper','Middle','Lower'],loc='best')

## age distribution for survived or not
plt.subplot2grid([2,3],[1,2])
plt.scatter(data_train['Survived'],data_train['Age'],marker='*',s=20)
plt.grid(b=True,axis='y')
plt.show()
## look up the sex distribution for survied or not
survived_male=data_train['Survived'].loc[data_train['Sex']=='male'].value_counts()
survived_female=data_train['Survived'].loc[data_train['Sex']=='female'].value_counts()
df=pd.DataFrame({'Male':survived_male,'Female':survived_female})
df.plot(kind='bar',stacked=True)

survived_0=data_train['Sex'].loc[data_train['Survived']==0].value_counts()
survived_1=data_train['Sex'].loc[data_train['Survived']==1].value_counts()
df=pd.DataFrame({'Survived':survived_1,'NOT Survived':survived_0})
df.plot(kind='bar',stacked=True)
plt.show()
## it's obvious that sex is an importanct feature for Surviving
## Sex for detail,see the survival rate of female for very ticket class
fig,(ax1,ax2,ax3,ax4)=plt.subplots(1,4,figsize=[15,6],sharey=True)
fig.suptitle('Female/male Survival Situation for tiket class',fontsize=16)
data_train.loc[(data_train['Pclass']!=3) & (data_train['Sex']=='female'),'Survived'].value_counts().plot(kind='bar',ax=ax1,label='female upper class',color='#FA2479')

ax1.set_ylabel('Number')
ax1.set_xticklabels(['Survived','NOT Survived'],rotation=0)
ax1.legend(loc='best')

data_train.loc[(data_train['Pclass']==3) & (data_train['Sex']=='female'),'Survived'].value_counts().plot(kind='bar',ax=ax2,label='female lower class',color='pink')
ax2.set_xticklabels(['Survived','NOT Survived'],rotation=0)
ax2.legend(loc='best')

data_train.loc[(data_train['Pclass']!=3) & (data_train['Sex']=='male'),'Survived'].value_counts().plot(kind='bar',ax=ax3,label='male upper class',color='blue')
ax3.set_xticklabels(['Survived','NOT Survived'],rotation=0)
ax3.legend(loc='best')

data_train.loc[(data_train['Pclass']==3) & (data_train['Sex']=='male'),'Survived'].value_counts().plot(kind='bar',ax=ax4,label='male upper class',color='lightblue')
ax4.set_xticklabels(['Survived','NOT Survived'],rotation=0)
ax4.legend(loc='best')
plt.show()
## Another method of plotting
fig=plt.figure(figsize=[15,6])
fig.suptitle('Female/male Survival Situation for tiket class',fontsize=16)
# plt.title('Female/male Survival Situation')
ax1=fig.add_subplot(141)
data_train.loc[(data_train['Pclass']!=3) & (data_train['Sex']=='female'),'Survived'].value_counts().plot(kind='bar',ax=ax1,label='female upper class',color='#FA2479')
ax1.set_ylabel('Number')
ax1.set_xticklabels(['Survived','NOT Survived'],rotation=0)
ax1.legend(loc='best')

ax2=fig.add_subplot(142,sharey=ax1)
data_train.loc[(data_train['Pclass']==3) & (data_train['Sex']=='female'),'Survived'].value_counts().plot(kind='bar',ax=ax2,label='female lower class',color='pink')
ax2.set_xticklabels(['Survived','NOT Survived'],rotation=0)
ax2.legend(loc='best')

ax3=fig.add_subplot(143,sharey=ax1)
data_train.loc[(data_train['Pclass']!=3) & (data_train['Sex']=='male'),'Survived'].value_counts().plot(kind='bar',ax=ax3,label='male upper class',color='blue')
ax3.set_xticklabels(['Survived','NOT Survived'],rotation=0)
ax3.legend(loc='best')

ax4=fig.add_subplot(144,sharey=ax1)
data_train.loc[(data_train['Pclass']==3) & (data_train['Sex']=='male'),'Survived'].value_counts().plot(kind='bar',ax=ax4,label='male upper class',color='lightblue')
ax4.set_xticklabels(['Survived','NOT Survived'],rotation=0)
ax4.legend(loc='best')

plt.show()
# let's see the relationship between ticket class and survival

upper=data_train['Survived'].loc[data_train['Pclass']==1].value_counts()
middle=data_train['Survived'].loc[data_train['Pclass']==2].value_counts()
lower=data_train['Survived'].loc[data_train['Pclass']==3].value_counts()
df=pd.DataFrame({'Upper':upper,'Middle':middle,'Lower':lower})
df.plot(kind='bar',stacked=True)
plt.show()
## thus ticket class is another important feature that influence Survival
## Exploring Embarked place for Survival
C=data_train['Survived'].loc[data_train['Embarked']=='C'].value_counts()
S=data_train['Survived'].loc[data_train['Embarked']=='S'].value_counts()
Q=data_train['Survived'].loc[data_train['Embarked']=='Q'].value_counts()
total=C+S+Q
C_ratio=C/total*100
S_ratio=S/total*100
Q_ratio=Q/total*100
df=pd.DataFrame({'C':C_ratio,'S':S_ratio,'Q':Q_ratio})

df.plot(kind='bar',stacked=True)
plt.xticks([0,1],['Not Survived','Survived'],rotation=0)
plt.ylabel('ratio %')
plt.title('Embarked place')
plt.show()

## Another plot for Embarked place
survived_0=data_train['Embarked'].loc[data_train['Survived']==0].value_counts()
survived_1=data_train['Embarked'].loc[data_train['Survived']==1].value_counts()
df=pd.DataFrame({'Survived':survived_1,'not survived':survived_0})
df1=df.divide(df.sum(axis=1),axis=0)*100     ## py attention to df.divide(axis=0)
fig=plt.figure(figsize=[10,6])
fig.suptitle('Embarked place')
ax1=fig.add_subplot(121)
df1.plot(kind='bar',stacked=True,ax=ax1)
ax1.set_ylabel('ratio %')
ax1.set_xlabel('Embarked place')
ax1.set_xticklabels(df1.index,rotation=0)
ax1.legend(loc='best')

ax2=fig.add_subplot(122,sharey=ax1)
df2=df1.transpose()
df2=df2.divide(df2.sum(axis=1),axis=0)*100
df2.plot(kind='bar',stacked=True,ax=ax2)
ax2.legend(loc='best')
ax2.set_xticklabels(df2.index,rotation=0)
plt.show()

## it seems that people Embarked form place C are more likely to be survived，pepople from S are least liekly to be suvived
## Explore the Survival rate for parch and Sibsp
parch_0=data_train['Survived'].loc[(data_train['Parch']==0) & (data_train['SibSp']==0)].value_counts()
parch=data_train['Survived'].loc[~(data_train['Parch']==0) & (data_train['SibSp']==0)].value_counts()
df=pd.DataFrame({'have Parch or sibsp':parch,'no parch or sibsp':parch_0})
df=df.divide(df.sum(axis=1),axis=0)*100
df.plot(kind='bar',stacked=True)
plt.legend(loc='best')
plt.xticks([0,1],['not survived','survived'],rotation=0)
plt.ylabel('ratio %')
plt.title('Parch')
plt.show()
# it seems that have parch or sibsp help to survial
# lets see more detail for parch and sibsp
data_train.loc[:,['Survived','Parch']].groupby('Parch').mean().plot(kind='bar')
plt.title('Parch and Survived')
data_train.loc[:,['Survived','SibSp']].groupby('SibSp').mean().plot(kind='bar')   ## pay attention to the groupby() method
plt.title('Sibsp and Survived')
plt.show()
# it seems that too many sibsp will decrease the survival rate
## show survival situation with Fare
## plot the estmated kernel density for survived or not
data_train['Fare'].loc[data_train['Survived']==0].plot(kind='kde',label='not survival')
data_train['Fare'].loc[data_train['Survived']==1].plot(kind='kde',label='survived')
plt.legend(loc='best')
plt.xlabel('Fare')
plt.xlim([0,300])
plt.show()
## it seems ambiguous....I don't know..
# cabin
not_null=data_train['Cabin'].notnull().sum()
null=data_train['Cabin'].isnull().sum()
plt.bar([0,1],[null,not_null])
plt.xticks([0,1],['null','not_null'])
plt.title('Carbin')
plt.show()
## this column has too many missing value
## Next, we begin to build our model
data_train.head()
## what we need to do is that changing some string features into numerical features
## for example , change 'Sex'=='male' into 1.'Sex'=='female' into 0
## For the 'Cabin' columns , we decide to change it into binary,have cabin value or not
data_train.loc[data_train['Cabin'].notnull(),'Cabin']=1
data_train.loc[data_train['Cabin'].isnull(),'Cabin']=0
data_train['Cabin'].value_counts()
## for 'Sex' column and 'Embarked' column
data_train['Sex']=data_train['Sex'].cat.codes
data_train['Embarked']=data_train['Embarked'].cat.codes
data_train.head()
## obviously we don't need the 'Name' column and 'Ticket' column
df_train=data_train.drop(['Name','Ticket'],axis=1)
df_train.head()
## for the 'Age' column ,we need some trick called missing value imputation
from sklearn.ensemble import RandomForestRegressor
## define a function ,using random forest to fill data
def fill_age(dataframe):
    features=['Age','Pclass','Sex','SibSp','Fare','Parch','Embarked']
    # devide the dataframe into known_age and unknown_age
#######################################################################
# method one
#     known_age=dataframe.loc[dataframe['Age'].notnull(),features].values
#     unknown_age=dataframe.loc[dataframe['Age'].isnull(),features].values
#     # using known_age data as training set
#     x_train=known_age[:,1:]
#     y_train=known_age[:,0]
#     # using unknown_age data as predict set
#     x_test=unknown_age[:,1:]
#########################################################################
# method two
    known_age=dataframe.loc[dataframe['Age'].notnull(),features]
    unknown_age=dataframe.loc[dataframe['Age'].isnull(),features]
    # using known_age data as training set
    x_train=known_age.iloc[:,1:]
    y_train=known_age.iloc[:,0]
    # using unknown_age data as predict set
    x_test=unknown_age.iloc[:,1:]
    # build model
    model=RandomForestRegressor()
    model.fit(x_train,y_train)
    y_test=model.predict(x_test)
    # rebuild 'Age' column value as pd.Series
    return y_test.astype(int)
# fill the 'Age' column
df_train.loc[df_train['Age'].isnull(),'Age']=fill_age(df_train)
df_train.isnull().sum()
## Using logistic regression to build a model
# the first thing we need to do is nomalize 'Age' and 'Fare' column
from sklearn.preprocessing import StandardScaler
# subdata=df_train.loc[:,['Age','Fare']]
# subdata=(subdata-subdata.mean(axis=0))/subdata.var(axis=0)
# df_train.loc[:,['Age','Fare']]=subdata
temp=df_train.copy()
scaler=StandardScaler()
df_train.loc[:,['Age','Fare']]=scaler.fit_transform(df_train.loc[:,['Age','Fare']])
df_train.head()
# subdata=temp.loc[:,['Age','Fare']]
# subdata=(subdata-subdata.mean(axis=0)).divide(subdata.var(axis=0),axis=1)
# subdata.head()
subdata=temp.loc[:,['Age','Fare']]
print(scaler.mean_)
print(subdata.mean(axis=0))
print(scaler.var_)
print(subdata.var(axis=0))
print((subdata.iloc[0,0]-29.44226097 )/(186.14817858)**0.5)
df_train.head()
## load test set 
data_test=pd.read_csv('../input/test.csv')
data_test.head()
data_test.isnull().sum()
# the preprocessing of test set is just like the training data
# first drop the Name and ticket column
df_test=data_test.drop(['Name','Ticket'],axis=1)
df_test.head()
## change Sex column and Embarked column into category value
df_test.loc[:,['Sex','Embarked']]=df_test.loc[:,['Sex','Embarked']].astype('category')
df_test['Sex']=df_test['Sex'].cat.codes
df_test['Embarked']=df_test['Embarked'].cat.codes
df_test.head()
## change Carbin column into have carbin or not
df_test.loc[df_test['Cabin'].notnull(),'Cabin']=1
df_test.loc[df_test['Cabin'].isnull(),'Cabin']=0
df_test.head()
df_test.loc[df_test['Fare'].isnull(),'Fare']=0
temp1=df_test.copy()
## fill the age column
df_test.loc[df_test['Age'].isnull(),'Age']=fill_age(df_test)
df_test.isnull().sum()
df_test.head()
## normalize age and fare columns
df_test.loc[:,['Age','Embarked']]=scaler.transform(df_test.loc[:,['Age','Embarked']])
df_test.head()
## we build a model , for example logistic regression,to predict the result
from sklearn.linear_model import LogisticRegression
features=['Pclass','Sex','Age','SibSp','Parch','Cabin','Embarked']
x_train=df_train.loc[:,features]
y_train=df_train['Survived']
x_test=df_test.loc[:,features]
clf=LogisticRegression()
clf.fit(x_train,y_train)
pred=clf.predict(x_test)
result=pd.DataFrame({'PassengerId':df_test['PassengerId'],'Survived':pred})
result.head()
## ssometing we will do next is called cross-valildation
from sklearn.model_selection import cross_validate
## use training data to cross-validate
clf=LogisticRegression()

## using learning curves to see it overfitting or not
from sklearn.model_selection import learning_curve
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt 
import numpy as np  

def plot_learning_curve(estimator,title,X,y,ylim=None,cv=5,n_jobs=1,train_sizes=np.linspace(0.05,1,10),verbose=0,plot=True,figsize=[10,6]):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    train_sizes,train_scores,test_scores=learning_curve(estimator,X,y,cv=cv,n_jobs=n_jobs,train_sizes=train_sizes,verbose=verbose)
    train_scores_mean=np.mean(train_scores,axis=1)
    train_scores_std=np.std(train_scores,axis=1)
    test_scores_mean=np.mean(test_scores,axis=1)
    test_scores_std=np.std(test_scores,axis=1)

    if plot:
        plt.figure(figsize=figsize)
        plt.title(title)
        if ylim is not None:
            plt.ylim(*ylim)   
        plt.xlabel('training samples')
        plt.ylabel('score')
        plt.plot(train_sizes,train_scores_mean,label='training score',color='r',marker='o')
        plt.fill_between(train_sizes,train_scores_mean-train_scores_std,train_scores_mean+train_scores_std,color='r',alpha=0.1)
        plt.plot(train_sizes,test_scores_mean,color='b',label='cross_valildation score',marker='o')
        plt.fill_between(train_sizes,test_scores_mean-test_scores_std,test_scores_mean+test_scores_std,color='b',alpha=0.1)
        plt.legend(loc='best')
        plt.show()
    print('train_scores:',train_scores_mean)
    print('test_scores:',test_scores_mean)
plot_learning_curve(clf,title='Learning curves',X=x_train,y=y_train)
from sklearn import svm 
clf=svm.SVC(gamma='scale',C=5)
plot_learning_curve(clf,title='learing curve',X=df_train.iloc[:,2:],y=df_train['Survived'])
## using bagging


result=pd.DataFrame({'PassengerId':df_test['PassengerId'],'Survived':pred}).set_index('PassengerId')
result.to_csv('Titanic_prediction.csv')
