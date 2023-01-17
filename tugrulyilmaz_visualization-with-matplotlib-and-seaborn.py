# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

plt.style.use("seaborn-whitegrid")

import seaborn as sns

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import warnings

warnings.filterwarnings("ignore")



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data=pd.read_csv("/kaggle/input/120-years-of-olympic-history-athletes-and-results/athlete_events.csv")

df=data[data.Year>=1995]

s=len(df)

df.index=range(0,s,1)

                 
df.columns
df.describe()
df.head()
df.info()
# line plot

plt.plot(df.ID,df.Height,label="Height")

plt.xlabel("ID number")

plt.ylabel("Height of Athlets(cm)")

plt.legend()

plt.show()
#scatter plot

plt.scatter(df.Height,df.Weight)

plt.xlabel("Height")

plt.ylabel("Weight")

plt.legend()

plt.show()
#histogram

plt.hist(df.Year)

plt.ylabel("Frequency")

plt.xlabel("Years")

plt.legend()

plt.show()
#subplot

plt.subplot(3,1,1)

plt.plot(df.ID,df.Height,label="Height")

plt.legend()

plt.subplot(3,1,2)

plt.scatter(df.Height,df.Weight)

plt.legend()

plt.subplot(3,1,3)

plt.hist(df.Year)

plt.show()
# pie plot1

#sex rate

labels=df.Sex.value_counts().index

explode=[0,0]

sizes=df.Sex.value_counts().values

plt.pie(sizes,explode,labels,autopct="%1.1f%%")

plt.show()
#pie plot2

#seasons rate

labels=df.Season.value_counts().index

explode=[0.2,0.2]

sizes=df.Season.value_counts().values

plt.pie(sizes,explode,labels,autopct="%1.1f%%")

plt.show()
#bar plot1

team_list=list(df.NOC.unique())



tks=[]

for i in team_list:

    k=df[df.NOC==i]

    sx=len(k)

    tks.append(sx)



new_data=pd.DataFrame({"team_list":team_list,"TKS":tks})

sorted_data=new_data.sort_values("TKS",ascending=False)



plt.figure(figsize=(25,20))

sns.barplot(x=sorted_data.team_list,y=sorted_data.TKS)   

plt.xticks(rotation=90)    

plt.xlabel=("Teams")

plt.ylabel("Total Number of Athlets")

plt.title("1996 yılından 2016 yılına kadar olimpiyat sporcularının ülkelere göre dağılış grafiği")

plt.show()
#bar plot2

year_list=list(df.Year.unique())



yks=[]

for i in year_list:

    gecici=df[df.Year==i]

    sx=len(gecici)

    yks.append(sx)



new_data=pd.DataFrame({"year_list":year_list,"TKS":yks})

sorted_data=new_data.sort_values("TKS",ascending=False)

    

#visualization

plt.figure(figsize=(15,10))

sns.barplot(x=sorted_data.year_list,y=sorted_data.TKS)

plt.xticks(rotation=90)

plt.ylabel("TKS")

plt.title("1996'dan 2016'ya kadar yıllara göre katılımcı sayısı")

plt.show()
#bar plot3

sport_list=list(df.Sport.unique())



yks=[]

for i in sport_list:

    gecici=df[df.Sport==i]

    sx=len(gecici)

    yks.append(sx)



new_data=pd.DataFrame({"sport_list":sport_list,"TKS":yks})

sorted_data=new_data.sort_values("TKS",ascending=False)



plt.figure(figsize=(15,10))

sns.barplot(x=sorted_data.sport_list,y=sorted_data.TKS)

plt.xticks(rotation=90)

plt.ylabel("TKS")

plt.title("1996'dan 2016'ya spor dallarına göre katılımcı sayısı")

plt.show()
#point plot

plt.figure(figsize=(15,10))

sns.pointplot(x=sorted_data.sport_list,y=sorted_data.TKS)

plt.xticks(rotation=90)

plt.ylabel("TKS")

plt.show()
#heatmap

plt.figure(figsize=(25,20))

f,ax=plt.subplots(figsize=(5,5))

sns.heatmap(df.corr(),annot=True,linewidths=.5,fmt=".1f")

plt.show()
#boxplot

plt.figure(figsize=(15,10))

sns.boxplot(x=df.Sex,y=df.Age)

plt.show()