# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
%matplotlib inline
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
MedianHouseholdIncome=pd.read_csv("../input/MedianHouseholdIncome2015.csv",encoding="windows-1252")
PercentagePeopleBelowPovertyLevel=pd.read_csv("../input/PercentagePeopleBelowPovertyLevel.csv",encoding="windows-1252")
PercentOver25CompletedHighSchool=pd.read_csv("../input/PercentOver25CompletedHighSchool.csv",encoding="windows-1252")
PoliceKillingsUS=pd.read_csv("../input/PoliceKillingsUS.csv",encoding="windows-1252")
ShareRaceByCity=pd.read_csv("../input/ShareRaceByCity.csv",encoding="windows-1252")

PercentagePeopleBelowPovertyLevel.info()
PercentagePeopleBelowPovertyLevel.poverty_rate.value_counts()
PoliceKillingsUS.head()
PoliceKillingsUS.name.value_counts()
seperate=PoliceKillingsUS.name[PoliceKillingsUS.name!="TK TK"].str.split()
a,b=zip(*seperate)
name_list=a+b
name_count=Counter(name_list)
most_common_names=name_count.most_common(15)
x,y=zip(*most_common_names)
x,y=list(x), list(y)

plt.figure(figsize=(15,10))
ax=sns.barplot(x=x,y=y,palette=sns.cubehelix_palette(len(x)))
plt.xlabel("Name or Surname of killed people")
plt.ylabel("Frequency")
plt.title("Most Common 15 Name or Surname Of Killed People")
PercentagePeopleBelowPovertyLevel.poverty_rate.replace(["-"],0.0,inplace=True)
PercentagePeopleBelowPovertyLevel.poverty_rate=PercentagePeopleBelowPovertyLevel.poverty_rate.astype(float)
area_list=list(PercentagePeopleBelowPovertyLevel["Geographic Area"].unique())
area_poverty_ratio=[]
for i in area_list:
    x=PercentagePeopleBelowPovertyLevel[PercentagePeopleBelowPovertyLevel["Geographic Area"]==i]
    area_poverty_rate=sum(x.poverty_rate)/len(x)
    area_poverty_ratio.append(area_poverty_rate)
data=pd.DataFrame({"area_list":area_list, "area_poverty_ratio":area_poverty_ratio})
new_index=(data["area_poverty_ratio"].sort_values(ascending=False)).index.values
sorted_data=data.reindex(new_index)

plt.figure(figsize=(15,10))
ax=sns.barplot(x=sorted_data["area_list"],y=sorted_data["area_poverty_ratio"])
plt.xticks(rotation=90)
plt.xlabel("States")
plt.ylabel("Poverty Rate")
plt.title("Poverty Rate Given States")
plt.show()
PercentOver25CompletedHighSchool.head()

PercentOver25CompletedHighSchool["Geographic Area"].value_counts()

PercentOver25CompletedHighSchool.percent_completed_hs.replace(["-"], 0.0,inplace=True )
PercentOver25CompletedHighSchool.percent_completed_hs=PercentOver25CompletedHighSchool.percent_completed_hs.astype(float)
area_list=list(PercentOver25CompletedHighSchool["Geographic Area"].unique())
area_highschool=[]
for i in area_list:
    x=PercentOver25CompletedHighSchool[PercentOver25CompletedHighSchool["Geographic Area"]==i]
    area_highschool_rate=sum(x.percent_completed_hs)/len(x)
    area_highschool.append(area_highschool_rate)
data=pd.DataFrame({"area_list":area_list, "area_highschool_ratio":area_highschool})
new_index=(data["area_highschool_ratio"].sort_values(ascending=True)).index.values
sorted_data2=data.reindex(new_index)
plt.figure(figsize=(15,10))
ax=sns.barplot(x=sorted_data2["area_list"], y=sorted_data2["area_highschool_ratio"])
plt.xticks(rotation=45)
plt.xlabel("States")
plt.ylabel("High School Graduate Rate")
plt.title("Percent of Given State's Population Above 25 that Has Graduated High School")
plt.show()
ShareRaceByCity.head()

ShareRaceByCity.replace(["-"],0.0,inplace=True)
ShareRaceByCity.replace(["(X)"],0.0,inplace=True)
ShareRaceByCity.loc[:,["share_white","share_black","share_native_american","share_asian","share_hispanic"]]=ShareRaceByCity.loc[:,["share_white","share_black","share_native_american","share_asian","share_hispanic"]].astype(float)
area_list=list(ShareRaceByCity["Geographic area"].unique())
share_white=[]
share_black=[]
share_native_american=[]
share_asian=[]
share_hispanic=[]
for i in area_list:
    x=ShareRaceByCity[ShareRaceByCity["Geographic area"]==i]
    share_white.append(sum(x.share_white)/len(x))
    share_black.append(sum(x.share_black)/len(x))
    share_native_american.append(sum(x.share_native_american)/len(x))
    share_asian.append(sum(x.share_asian)/len(x))    
    share_hispanic.append(sum(x.share_hispanic)/len(x))

f,ax=plt.subplots(figsize=(9,15))
sns.barplot(x=share_white,y=area_list,color="red",alpha=0.5, label="White")
sns.barplot(x=share_black,y=area_list,color="blue",alpha=0.6, label="Black")
sns.barplot(x=share_native_american,y=area_list,color="cyan",alpha=0.7, label="Native American")
sns.barplot(x=share_asian,y=area_list,color="green",alpha=0.8, label="Asian")
sns.barplot(x=share_hispanic,y=area_list,color="black",alpha=0.9, label="Hispanic")

ax.legend(loc="lower right", frameon=True)
ax.set(xlabel="Percentage of Races", ylabel="States", title="Percentage of State's Population Acording to Races")
plt.show()
sorted_data["area_poverty_ratio"]=sorted_data["area_poverty_ratio"]/max(sorted_data["area_poverty_ratio"])
sorted_data2["area_highschool_ratio"]=sorted_data2["area_highschool_ratio"]/max(sorted_data2["area_highschool_ratio"])
data=pd.concat([sorted_data,sorted_data2["area_highschool_ratio"]],axis=1)
data.sort_values("area_poverty_ratio",inplace=True)

f,ax1=plt.subplots(figsize=(20,10))
sns.pointplot(x="area_list",y="area_poverty_ratio",data=data, color="red",alpha=0.5)
sns.pointplot(x="area_list", y="area_highschool_ratio",data=data, color="green",alpha=0.9)
plt.text(40,0.6,"high school graudate ratio", color="red",fontsize=17, style="italic")
plt.text(40,0.55, "poverty_rati", color="lime",fontsize=17,style="italic")
plt.xlabel("States",fontsize=15,color="blue")
plt.ylabel("Values",fontsize=15,color="blue")
plt.title("Highscool Graduate VS Provity Rate", fontsize=20,color="red")
plt.grid()
plt.show()
g=sns.jointplot(data.area_poverty_ratio,data.area_highschool_ratio, kind="kde", size=7)
plt.savefig("graph.png")
plt.show()
g=sns.jointplot("area_poverty_ratio","area_highschool_ratio",data=data,size=5,ratio=3,color="g")
plt.savefig("graph.png")
plt.show()
g=sns.jointplot("area_poverty_ratio","area_highschool_ratio",data=data,size=5,kind="reg",ratio=3,color="g")
plt.savefig("graph.png")
plt.show()
g=sns.jointplot("area_poverty_ratio","area_highschool_ratio",data=data,size=5,kind="hex",ratio=3,color="g")
plt.savefig("graph.png")
plt.show()
g=sns.jointplot("area_poverty_ratio","area_highschool_ratio",data=data,size=5,kind="resid",ratio=3,color="g")
plt.savefig("graph.png")
plt.show()
PoliceKillingsUS.race.dropna(inplace=True)
labels=PoliceKillingsUS.race.value_counts().index
colors=["grey","blue","red","yellow","green","blue"]
explode=[0,0,0,0,0,0]
sizes=PoliceKillingsUS.race.value_counts().values

plt.figure(figsize=(7,7))
plt.pie(sizes,explode=explode,labels=labels,colors=colors,autopct="%1.1f%%")
plt.title("Killed People Acording to Races",color="blue",fontsize=15)
plt.show()
sns.lmplot(x="area_poverty_ratio",y="area_highschool_ratio", data=data)
plt.show()
sns.kdeplot(data.area_poverty_ratio,data.area_highschool_ratio,color="red",shade=True,cut=5)
plt.show()
sns.kdeplot(data.area_poverty_ratio,data.area_highschool_ratio,color="red",shade=False,cut=5)
plt.show()
pal=sns.cubehelix_palette(2,rot=-.5,dark=.3)
sns.violinplot(data=data,palette=pal,inner="points")
plt.show()
f,ax=plt.subplots(figsize=(5,5))
sns.heatmap(data.corr(),annot=True, linewidths=.5,fmt="0.1f",ax=ax)
plt.show()
sns.boxplot(x="gender",y="age",hue="manner_of_death",data=PoliceKillingsUS,palette="PRGn")
plt.show()
sns.swarmplot(x="gender",y="age",hue="manner_of_death",data=PoliceKillingsUS,)
plt.show()
sns.pairplot(data)
plt.show()
sns.countplot(PoliceKillingsUS.gender)
sns.countplot(PoliceKillingsUS.manner_of_death)
plt.title("Manner Of Death",color="r",fontsize=15)
plt.show()
armed=PoliceKillingsUS.armed.value_counts()
#print(armed)
plt.figure(figsize=(10,7))
sns.barplot(x=armed[:7].index,y=armed[:7].values)
plt.ylabel("Number of weapon")
plt.xlabel("weapon types")
plt.title("kill weapon",color="r",fontsize=15)
plt.show()
above20=["above20" if i>30 else "below20" for i in PoliceKillingsUS.age]
df=pd.DataFrame({"age":above20})
sns.countplot(x=df.age)
plt.ylabel("Number of killed people")
plt.title("age of killed people", color="blue",fontsize=15)
plt.show()
