# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np# linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import matplotlib as plt
import os
#print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df=pd.read_csv("../input/train.csv")
df.head()
print (df.shape)
testdf=pd.read_csv("../input/test.csv")
print (testdf.shape)
testdf.head()
def missingvalues(df):
    miss=df.isnull().sum()
    misspercent=100*df.isnull().sum()/len(df)
    misvaltable=pd.concat([miss,misspercent],axis=1)
    misvaltable=misvaltable.rename(columns={0:"missing values",1:"missing percent"})
    return misvaltable
df.dtypes.value_counts()
    
categoryweather=df.groupby("holiday").nunique()
print(categoryweather)
df1=pd.get_dummies(df['weather'])
df1.head()
import matplotlib.pyplot as plt
df.head()
plt.figure(figsize=(20,20))
plt.subplot(4,2,1)
plt.hist(df["season"])
plt.xlabel("season")
plt.ylabel("count")
plt.subplot(4,2,2)
plt.hist(df["holiday"])
plt.xlabel("holiday")
plt.ylabel("count")
plt.subplot(4,2,3)
plt.hist(df["workingday"])
plt.xlabel("workingday")
plt.ylabel("count")
plt.subplot(4,2,4)
plt.hist(df["weather"])
plt.xlabel("weather")
plt.ylabel("count")
plt.subplot(4,2,5)
plt.hist(df["temp"])
plt.xlabel("temp")
plt.ylabel("count")
plt.subplot(4,2,6)
plt.hist(df["atemp"])
plt.xlabel("atemp")
plt.ylabel("count")
plt.subplot(4,2,7)
plt.hist(df["humidity"])
plt.xlabel("humidity")
plt.ylabel("count")
plt.subplot(4,2,8)
plt.hist(df["windspeed"])
plt.xlabel("windspeed")
plt.ylabel("count")
plt.show()
testdf.columns
l=[ 'season', 'holiday', 'workingday', 'weather', 'temp',
       'atemp', 'humidity', 'windspeed']
dftarget=df[["casual","registered","count"]].copy()
dftarget.head()

dfnew=df.copy()
dfnew.drop(["registered","casual","count"],axis=1,inplace=True)
dfnew.head()
import matplotlib.pyplot as plt
df2=pd.concat([dfnew,testdf])
print(df2.shape)
print(df.shape)
import matplotlib.pyplot as plt1
plt1.figure(figsize=(20,20))
plt1.subplot(4,2,1)
plt1.hist(df2["season"])
plt1.xlabel("season")
plt1.ylabel("count")
plt1.subplot(4,2,2)
plt1.hist(df2["holiday"])
plt1.xlabel("holiday")
plt1.ylabel("count")
plt1.subplot(4,2,3)
plt1.hist(df2["workingday"])
plt1.xlabel("workingday")
plt1.ylabel("count")
plt1.subplot(4,2,4)
plt1.hist(df2["weather"])
plt1.xlabel("weather")
plt1.ylabel("count")
plt1.subplot(4,2,5)
plt1.hist(df2["temp"])
plt1.xlabel("temp")
plt1.ylabel("count")
plt1.subplot(4,2,6)
plt1.hist(df2["atemp"])
plt1.xlabel("atemp")
plt1.ylabel("count")
plt1.subplot(4,2,7)
plt1.hist(df2["humidity"])
plt1.xlabel("humidity")
plt1.ylabel("count")
plt1.subplot(4,2,8)
plt1.hist(df2["windspeed"])
plt1.xlabel("windspeed")
plt1.ylabel("count")
plt1.show()
df=pd.read_csv("../input/train.csv")
df1=pd.get_dummies(df['weather'])
df1=df1.rename(columns={1:"clear",2:"misty",3:"snow",4:"heavy snow"})
df=df.drop(["weather"],axis=1)
df=pd.concat([df,df1],axis=1)
df.head()
df1=pd.get_dummies(df2["weather"])
df2.drop(["weather"],axis=1,inplace=True)
df2=pd.concat([df2,df1],axis=1)
df2.head()

#df2=df2.drop(["weather"],axis=1)
df2=df2.rename(columns={1:"clear",2:"misty",3:"snow",4:"heavy snow"})
df2.head()
df1=pd.get_dummies(df2["season"])
#df2=df2.drop(["season"],axis=1,inplace=True)
#df1.head()
df3=pd.concat([df2,df1],axis=1)
#df2.head()
df3=df3.rename(columns={1:"spring",2:"summer",3:"fall",4:"winter"})
df3.head()
df3.shape
df.shape
df0=pd.get_dummies(df["season"])
df4=pd.concat([df,df0],axis=1)
df4=df4.rename(columns={1:"spring",2:"summer",3:"fall",4:"winter"})
df4.head()
df4.groupby("spring").describe()
df4.groupby("fall")["registered"].describe()
df4["weekend"]=[abs(1-abs(x-y)) for x,y in zip(df4["workingday"],df4["holiday"]) ]
df4.head()
print(df4.groupby("weekend")["datetime"].nunique())
print(df4.groupby("holiday")["datetime"].nunique())
print(df4.groupby("workingday")["datetime"].nunique())
df4.groupby("weekend")['registered',"count","casual"].describe()
df3["weekend"]=[abs(1-abs(x-y)) for x,y in zip(df3["workingday"],df3["holiday"]) ]
df3.head()
df4.head(25)
df4["time"]=pd.to_datetime(df4["datetime"])
df4.head()
df4["hours"]=df4["time"].dt.hour
df4.head()
df4.head(25)
df3["time"]=pd.to_datetime(df3["datetime"])
df3.head()
df3["hours"]=df3["time"].dt.hour
df3.head()
df4.drop(["time"],axis=1,inplace=True)
df3.drop(["time"],axis=1,inplace=True)
df4.head()
import seaborn as sns
plt.figure(figsize=(20,20))
ax=plt.subplot(221)
sns.boxplot(data=df4,x="hours",y="registered",ax=ax)
ax=plt.subplot(222)
sns.boxplot(data=df4,x="hours",y="casual",ax=ax)
ax=plt.subplot(223)
sns.boxplot(data=df4,x="hours",y="count",ax=ax)
df4["logcasual"]=np.log(df4["casual"]+1)
df4["logcasual"]=np.log(df4["casual"]+1)
df4["logcasual"]=np.log(df4["casual"]+1)
df4.head()
df4["logregistered"]=np.log(df4["registered"]+1)
df4["logcount"]=np.log(df4["count"]+1)
df4.head()
#inspecting hourly trend
import seaborn as sns
plt.figure(figsize=(20,20))
ax=plt.subplot(221)
sns.boxplot(data=df4,x="hours",y="logregistered",ax=ax)
ax=plt.subplot(222)
sns.boxplot(data=df4,x="hours",y="logcasual",ax=ax)
ax=plt.subplot(223)
sns.boxplot(data=df4,x="hours",y="logcount",ax=ax)            
df4["time"]=pd.to_datetime(df4["datetime"])
df4.head()
df4["day"]=df4["time"].dt.day
df4.head(25)
df4["day"]=df4["time"].dt.dayofweek
df4.head()
df4["day"]=df4["time"].dt.dayofweek
df4.head(25)
#inspecting daily trend
import seaborn as sns
plt.figure(figsize=(20,20))
ax=plt.subplot(221)
sns.boxplot(data=df4,x="day",y="logregistered",ax=ax)
ax=plt.subplot(222)
sns.boxplot(data=df4,x="day",y="logcasual",ax=ax)
ax=plt.subplot(223)
sns.boxplot(data=df4,x="day",y="logcount",ax=ax) 
import seaborn as sns
plt.figure(figsize=(20,20))
ax=plt.subplot(221)
sns.boxplot(data=df4,x="day",y="registered",ax=ax)
ax=plt.subplot(222)
sns.boxplot(data=df4,x="day",y="casual",ax=ax)
ax=plt.subplot(223)
sns.boxplot(data=df4,x="day",y="count",ax=ax) 
df5=df.copy()
df5.drop(["holiday","workingday","season"],axis=1,inplace=True)
df5.corr()
df4["year"]=df4['time'].dt.year
df4.head()
plt.figure(figsize=(20,20))
ax=plt.subplot(2,2,1)
sns.boxplot(data=df4,x="year",y='registered',ax=ax)
#plt.figure(figsize=(20,20))
ax=plt.subplot(2,2,2)
sns.boxplot(data=df4,x="year",y='casual',ax=ax)
#plt.figure(figsize=(20,20))
ax=plt.subplot(2,2,3)
sns.boxplot(data=df4,x="year",y='count',ax=ax)

plt.figure(figsize=(10,10))
sns.boxplot(data=df4,x="weekend",y="casual")
df4.groupby("weekend")["datetime"].nunique()

df4["month"]=df4["time"].dt.month
df4.head()
#df4.drop(["year_bins"],axis=1,inplace=True)
df4["year_bin"]="y0"
df4["year_bin"].loc[(df4["year"]==2011) & (df4["month"]<=3)]="y1"
df4["year_bin"].loc[(df4["year"]==2011) & (df4["month"]>3) & (df4["month"]<=6)]="y2"
df4["year_bin"].loc[(df4["year"]==2011) & (df4["month"]>6) & (df4["month"]<=9)]="y3"
df4["year_bin"].loc[(df4["year"]==2011) & (df4["month"]>9) & (df4["month"]<=12)]="y4"
df4["year_bin"].loc[(df4["year"]==2012) & (df4["month"]<=3)]="y5"
df4["year_bin"].loc[(df4["year"]==2012) & (df4["month"]>3) & (df4["month"]<=6)]="y6"
df4["year_bin"].loc[(df4["year"]==2012) & (df4["month"]>6) & (df4["month"]<=9)]="y7"
df4["year_bin"].loc[(df4["year"]==2012) & (df4["month"]>9) & (df4["month"]<=12)]="y8"
df4.groupby('year_bin')["datetime"].nunique()
plt.figure(figsize=(20,20))
ax=plt.subplot(221)
sns.boxplot(data=df4,x="year_bin",y="casual",ax=ax)
ax=plt.subplot(222)
sns.boxplot(data=df4,x="year_bin",y="registered",ax=ax)
ax=plt.subplot(223)
sns.boxplot(data=df4,x="year_bin",y="count",ax=ax)
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor
l=["hours"]
X=df4[l]
Y=df4["casual"]
dtree=DecisionTreeRegressor(max_depth=3)
dtree.fit(X,Y)
from sklearn.externals.six import StringIO  
from IPython.display import Image 
from sklearn.tree import export_graphviz
import graphviz
data = export_graphviz(dtree,out_file=None,   
                         filled=True, rounded=True,  
                         special_characters=True)
graph = graphviz.Source(data)
graph
df4["casual"].describe()
l=["hours"]
X=df4[l]
Y=df4["registered"]
dtree1=DecisionTreeRegressor(max_depth=4)
dtree1.fit(X,Y)
data = export_graphviz(dtree1,out_file=None,   
                         filled=True, rounded=True,  
                         special_characters=True)
graph = graphviz.Source(data)
graph
df4["daycas"]="cas0"
df4["daycas"].loc[df4["hours"]<=6.5]="cas1"
df4["daycas"].loc[(df4["hours"]>6.5) & (df4["hours"]<=7.5)]="cas2"
df4["daycas"].loc[(df4["hours"]>7.5) & (df4["hours"]<=8.5)]="cas3"
df4["daycas"].loc[(df4["hours"]>8.5) & (df4["hours"]<=9.5)]="cas4"
df4["daycas"].loc[(df4["hours"]>9.5) & (df4["hours"]<=10.5)]="cas5"
df4["daycas"].loc[(df4["hours"]>10.5) & (df4["hours"]<=19.5)]="cas6"
df4["daycas"].loc[(df4["hours"]>19.5) & (df4["hours"]<=21.5)]="cas7"
df4["daycas"].loc[df4["hours"]>21.5]="cas8"
df4.groupby("daycas")["datetime"].nunique()
df4["dayreg"]="reg0"
df4["dayreg"].loc[df4["hours"]<=0.5]="reg1"
df4["dayreg"].loc[(df4["hours"]>0.5) & (df4["hours"]<=1.5)]="reg2"
df4["dayreg"].loc[(df4["hours"]>1.5) & (df4["hours"]<=4.5)]="reg3"
df4["dayreg"].loc[(df4["hours"]>4.5) & (df4["hours"]<=5.5)]="reg4"
df4["dayreg"].loc[(df4["hours"]>5.5) & (df4["hours"]<=6.5)]="reg5"
df4["dayreg"].loc[(df4["hours"]>6.5) & (df4["hours"]<=8.5)]="reg6"
df4["dayreg"].loc[(df4["hours"]>8.5) & (df4["hours"]<=16.5)]="reg7"
df4["dayreg"].loc[(df4["hours"]>16.5) & (df4["hours"]<=18.5)]="reg8"
df4["dayreg"].loc[(df4["hours"]>18.5) & (df4["hours"]<=20.5)]="reg9"
df4["dayreg"].loc[(df4["hours"]>20.5) & (df4["hours"]<=21.5)]="reg10"
df4["dayreg"].loc[(df4["hours"]>21.5) & (df4["hours"]<=22.5)]="reg11"
df4["dayreg"].loc[df4["hours"]>22.5]="reg12"
df4.groupby("dayreg")["datetime"].nunique()
plt.figure(figsize=(20,20))
plt.subplot(221)
plt.scatter(df.temp,df.casual)
plt.subplot(222)
plt.scatter(df.windspeed,df.casual)

df4.head()

df3.head()
df3["time"]=pd.to_datetime(df3["datetime"])
df3["year"]=df3['time'].dt.year
df3["month"]=df3["time"].dt.month
df3["day"]=df3["time"].dt.dayofweek
df3.head()
df3["year_bin"]="y0"
df3["year_bin"].loc[(df3["year"]==2011) & (df3["month"]<=3)]="y1"
df3["year_bin"].loc[(df3["year"]==2011) & (df3["month"]>3) & (df3["month"]<=6)]="y2"
df3["year_bin"].loc[(df3["year"]==2011) & (df3["month"]>6) & (df3["month"]<=9)]="y3"
df3["year_bin"].loc[(df3["year"]==2011) & (df3["month"]>9) & (df3["month"]<=12)]="y4"
df3["year_bin"].loc[(df3["year"]==2012) & (df3["month"]<=3)]="y5"
df3["year_bin"].loc[(df3["year"]==2012) & (df3["month"]>3) & (df3["month"]<=6)]="y6"
df3["year_bin"].loc[(df3["year"]==2012) & (df3["month"]>6) & (df3["month"]<=9)]="y7"
df3["year_bin"].loc[(df3["year"]==2012) & (df3["month"]>9) & (df3["month"]<=12)]="y8"
df3["daycas"]="cas0"
df3["daycas"].loc[df3["hours"]<=6.5]="cas1"
df3["daycas"].loc[(df3["hours"]>6.5) & (df3["hours"]<=7.5)]="cas2"
df3["daycas"].loc[(df3["hours"]>7.5) & (df3["hours"]<=8.5)]="cas3"
df3["daycas"].loc[(df3["hours"]>8.5) & (df3["hours"]<=9.5)]="cas4"
df3["daycas"].loc[(df3["hours"]>9.5) & (df3["hours"]<=10.5)]="cas5"
df3["daycas"].loc[(df3["hours"]>10.5) & (df3["hours"]<=19.5)]="cas6"
df3["daycas"].loc[(df3["hours"]>19.5) & (df3["hours"]<=21.5)]="cas7"
df3["daycas"].loc[df3["hours"]>21.5]="cas8"
df3["dayreg"]="reg0"
df3["dayreg"].loc[df3["hours"]<=0.5]="reg1"
df3["dayreg"].loc[(df3["hours"]>0.5) & (df3["hours"]<=1.5)]="reg2"
df3["dayreg"].loc[(df3["hours"]>1.5) & (df3["hours"]<=4.5)]="reg3"
df3["dayreg"].loc[(df3["hours"]>4.5) & (df3["hours"]<=5.5)]="reg4"
df3["dayreg"].loc[(df3["hours"]>5.5) & (df3["hours"]<=6.5)]="reg5"
df3["dayreg"].loc[(df3["hours"]>6.5) & (df3["hours"]<=8.5)]="reg6"
df3["dayreg"].loc[(df3["hours"]>8.5) & (df3["hours"]<=16.5)]="reg7"
df3["dayreg"].loc[(df3["hours"]>16.5) & (df3["hours"]<=18.5)]="reg8"
df3["dayreg"].loc[(df3["hours"]>18.5) & (df3["hours"]<=20.5)]="reg9"
df3["dayreg"].loc[(df3["hours"]>20.5) & (df3["hours"]<=21.5)]="reg10"
df3["dayreg"].loc[(df3["hours"]>21.5) & (df3["hours"]<=22.5)]="reg11"
df3["dayreg"].loc[df3["hours"]>22.5]="reg12"
df3.head()

print(df4.shape)
print(df3.shape)
df6=df4.copy()
df6.drop(["datetime","season","time","count","registered","casual","logcount"],axis=1,inplace=True)
df6.head()
df4["rtemp1"]=df4["temp"]+df4["atemp"]
df4["rtemp2"]=df4["temp"]-df4["atemp"]
df4["rtemp3"]=df4["temp"]*df4["atemp"]
print(df4["temp"].corr(df4["registered"]))
print(df4["rtemp1"].corr(df4["registered"]))
print(df4["rtemp2"].corr(df4["registered"]))
print(df4["rtemp3"].corr(df4["registered"]))
df3["rtemp3"]=df3["temp"]*df3["atemp"]

df6["rtemp3"]=df6["temp"]*df6["atemp"]
df6.head()
df3.drop(["datetime","season","time"],axis=1,inplace=True)
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
df6.head()

df7=df6.logcasual
df8=df6.logregistered
df7.head()
df6.drop(["logcasual",'logregistered'],axis=1,inplace=True)

df6.drop(["atemp"],axis=1,inplace=True)
df6.drop(["temp"],axis=1,inplace=True)
df6.drop(["rtemp3","month"],axis=1,inplace=True)
d=pd.read_csv("../input/train.csv")
df6["temp"]=d["temp"]
var=["holiday","workingday","weekend","hours"]
for v in var:
    df6[v]=df6[v].astype("category")

dfreg=df6.copy()
dfcas=df6.copy()
dfcas['year']=dfcas.year.replace({2011:0,2012:1})
dfreg['year']=dfcas.year.replace({2011:0,2012:1})
dfreg.drop(["daycas"],axis=1,inplace=True)
dureg1=pd.get_dummies(dfreg["dayreg"])
dfreg=pd.concat([dfreg,dureg1],axis=1)
dureg2=pd.get_dummies(dfreg["year_bin"])
dfreg=pd.concat([dfreg,dureg2],axis=1)
dureg3=pd.get_dummies(dfreg["day"])
dfreg=pd.concat([dfreg,dureg3],axis=1)
dfreg.drop(["dayreg","year_bin","day"],axis=1,inplace=True)
dfcas.drop(["dayreg"],axis=1,inplace=True)
ducas1=pd.get_dummies(dfcas["daycas"])
dfcas=pd.concat([dfcas,ducas1],axis=1)
ducas2=pd.get_dummies(dfcas["year_bin"])
dfcas=pd.concat([dfcas,ducas2],axis=1)
ducas3=pd.get_dummies(dfcas["day"])
dfcas=pd.concat([dfcas,ducas3],axis=1)
dfcas.drop(["daycas","year_bin","day"],axis=1,inplace=True)
from sklearn.model_selection import train_test_split 
x_train,x_test,y_train,y_test=train_test_split(dfreg,df7,random_state=42)
rf=RandomForestRegressor(n_estimators=500)
rf.fit(x_train,y_train)
predictions=rf.predict(x_test)
mean_squared_error(y_test, predictions)
rf1=RandomForestRegressor(n_estimators=500)
rf1.fit(dfcas,df7)
rf2=RandomForestRegressor(n_estimators=500)
rf2.fit(dfreg,df8)
df6.head()
df3.head()
df3.drop(["temp","atemp"],axis=1,inplace=True)
print(df3.shape)
print(df6.shape)
newtest=df3.tail(17379-10886)
newtest.head()
testdf.head()
newtest.head()
print(newtest.shape)
print(testdf.shape)
newtest.drop(["rtemp3","month"],axis=1,inplace=True)

f=pd.read_csv("../input/test.csv")
newtest['year']=newtest.year.replace({2011:0,2012:1})
newtest["temp"]=f["temp"]
var=["holiday","workingday","weekend","hours"]
for v in var:
    newtest[v]=newtest[v].astype("category")
newtestreg=newtest.copy()
newtestcas=newtest.copy()
newtestreg.drop(["daycas"],axis=1,inplace=True)
newreg1=pd.get_dummies(newtestreg["dayreg"])
newtestreg=pd.concat([newtestreg,newreg1],axis=1)
newreg2=pd.get_dummies(newtestreg["year_bin"])
newtestreg=pd.concat([newtestreg,newreg2],axis=1)
newreg3=pd.get_dummies(newtestreg["day"])
newtestreg=pd.concat([newtestreg,newreg3],axis=1)
newtestreg.drop(["dayreg","year_bin","day"],axis=1,inplace=True)
newtestcas.drop(["dayreg"],axis=1,inplace=True)
newcas1=pd.get_dummies(newtestcas["daycas"])
newtestcas=pd.concat([newtestcas,newcas1],axis=1)
newcas2=pd.get_dummies(newtestcas["year_bin"])
newtestcas=pd.concat([newtestcas,newcas2],axis=1)
newcas3=pd.get_dummies(newtestcas["day"])
newtestcas=pd.concat([newtestcas,newcas3],axis=1)
newtestcas.drop(["daycas","year_bin","day"],axis=1,inplace=True)
newtest.head()


predictcas=rf1.predict(newtestcas)
predictcas=np.exp(predictcas)-1
predictreg=rf2.predict(newtestreg)
predictreg=np.exp(predictreg)-1

print(type(predictcas))
predictcount=np.add(predictcas,predictreg)
final=pd.DataFrame({"datetime":testdf["datetime"],"count":predictcount})
final
final.to_csv("final2.csv",index=False)
dfcas.head()

