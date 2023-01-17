import numpy as np

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
titanic=pd.read_csv("../input/titanic_train.csv")
titanic.head(10)
del titanic["Name"] #bcz name variable isnt important thats why we remove this
del titanic["PassengerId"] #same pid also del
titanic.describe(include="all")
titanic.info()
titanic.columns
plt.figure(figsize=(9,9))

titanic.boxplot(["Age","Parch","Fare"])
titanic.head()
titanic.columns
titanic.nunique()  #show unique value in a every coloumn
sns.distplot(titanic["Survived"])
titanic.boxplot(column="Survived")
# age have nan value 
titanic.hist(column="Age") 
titanic.boxplot(column="Fare") #oputlier
titanic.hist(column="Fare")  #positive skewed   
pd.crosstab(titanic["Survived"],columns="count").plot(kind="bar") #survived count 0=549 more 1=342 less
titanic["Survived"].value_counts()
titanic["Pclass"].value_counts()   #group of people is more in 3 pclass 
titanic["Sex"].value_counts()    # male > female
titanic["SibSp"].value_counts()  # alone is more in sibsp
titanic["Parch"].value_counts() #alone is more in parch
titanic["Embarked"].value_counts() # people quantity s>c>q in Embarked
plt.scatter(titanic["Age"],titanic["Fare"])   #no relation
plt.scatter(titanic["Fare"],titanic["Age"]) #no relation 0-100 fare count high
c=titanic.groupby(["Sex","Embarked"],axis=0)

c.size()                                                # sex and embarked relation
c=titanic.groupby(["Sex","Survived"],axis=0)

c.size()                                                # survived ratio of gender [female survived more as compare to male]
c=titanic.groupby(["Pclass","Survived"],axis=0)

c.size()                                                # survived ratio is high in Pclass 1 and less in pclass 3
c=titanic.groupby(["Embarked","Survived"],axis=0)

c.size()      



#print("percentage c :", 93/75+93*100) 

#print("percentage q :", 30/47+30*100) 

#print("percentage s :", 217/427+217*100)                # survived and embarked relation
c=(93/168)*100

q=30/77*100

s=217/644*100

print("survived percentage by embarked \n")

print("percentage c :", c) 

print("percentage q :", q) 

print("percentage s :", s)               #survived percentage by embarked c>q>s @c on top
c=titanic.groupby(["Embarked","Sex"],axis=0)

c.size()                                                # sex and embarked relation
crr=titanic.corr()
sns.heatmap(crr,annot=True,cmap="RdBu_r")
titanic.head()
f=sns.FacetGrid(titanic,col="Survived")   # near 30 age group is more dead and live also more near by 30 age group

f.map(plt.hist, "Age")
f=sns.FacetGrid(titanic,col="Pclass")

f.map(plt.hist, "Age")
del titanic["Pclass"]
t=pd.read_csv("titanic_train.csv")
p=t["Pclass"]
t2=pd.concat([titanic,p],axis=1)
titanic.insert(loc=2,column="Pcl",value=p)
titanic.head()
#sns.pairplot(titanic)  not found good info from pairplot   
s=sns.FacetGrid(titanic,col="SibSp")    #live and dead ratio is high in sibsp alone(0)

s=s.map(plt.hist, "Survived")                         #but sibsp(1) live ratio is high as compare to dead

                                                     
s=sns.FacetGrid(titanic,col="Parch")    #live and dead ratio is high at the time of parch  is  alone(0)

s=s.map(plt.hist, "Survived")                         #but parch (1) live ratio is high as compare to dead

                                                     
s=sns.FacetGrid(titanic,col="Survived",row="Sex")    #gender survived ratio

s=s.map(plt.hist, "Age")
s=sns.FacetGrid(titanic,col="Embarked")    #gender survived ratio

s=s.map(plt.hist, "Fare")
titanic["Age"]=titanic["Age"].fillna(titanic["Age"].median())   #age fill with med.
titanic.info()
del titanic["Cabin"]
titanic.head()
titanic["Embarked"]=titanic["Embarked"].fillna(titanic["Embarked"].mode()[0])
titanic.isnull().sum()
titanic.describe()
iqr_age=titanic["Age"].quantile(0.75)-titanic["Age"].quantile(.25)
uppr_age=titanic["Age"].quantile(0.75)+1.5*iqr_age
lower_age=titanic["Age"].quantile(0.25)-1.5*iqr_age
uppr_age
lower_age
upproutlier_age=titanic[(titanic["Age"]>uppr_age)]
loweroutlier_age=titanic[(titanic["Age"]<lower_age)]
upproutlier_age.head()  # these are natural outlier  so we can not remove this type of outlier
loweroutlier_age.head() # these are also natural outlier
del titanic["Ticket"]
cate=titanic.dtypes==np.object
obj=titanic.columns[cate]
obj
dummy_dt=pd.DataFrame()
for i in obj:

    dummy=pd.get_dummies(titanic[i],drop_first=True)

    dummy_dt=pd.concat([dummy_dt,dummy],axis=1)
dummy_dt
titanic2=pd.concat([titanic,dummy_dt],axis=1)
titanic2
del titanic2["Sex"]
del titanic2["Embarked"]
titanic2
titanic2.info()
from IPython.core.interactiveshell import InteractiveShell 

InteractiveShell.ast_node_interactivity = "all"
import pandas_profiling
pandas_profiling.ProfileReport(titanic)