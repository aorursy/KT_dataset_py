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
#read the csv file
data=pd.read_csv("../input/fifa_ranking.csv")
#get some information about our csv file
data.info()
#get top of the 10 elements
data.head(10)
data.columns
#line plot 
data.previous_points.plot(kind="line",color="green",linestyle=":",alpha=0.5,label="previous_points")
data.rank_change.plot(kind="line",color="blue",linestyle="-",alpha=0.5,label="rank_change")
plt.legend("upper right")
plt.xlabel("previous_points")
plt.ylabel("rank_change")
plt.title("line plot")
plt.show()
#scatter plot
data.plot(kind="scatter" ,x="previous_points",y="rank_change",alpha=0.5,color="yellow" )
plt.legend("upper right")
plt.xlabel("previous_points")
plt.ylabel("rank_change")
plt.title("scatter plot")
plt.show()
#histogram
data.previous_points.plot(kind="hist",figsize=(10,10),bins=50,color="red")
plt.show()
#filtering data
data1=data["previous_points"]>50
data[data1]
#filtering data
data[np.logical_and(data["previous_points"]>50,data["rank_change"]>1)]
data.shape
print(data["country_full"].value_counts(dropna=False))
data.boxplot(column="rank")
plt.show()
data_new=data.head()
data_new
melted=pd.melt(frame=data_new,id_vars="rank",value_vars=["previous_point","rank_change"])
melted
melted.pivot(index="rank",columns="variable",values="value")
data2=data.head()
data3=data.tail()
con_cat=pd.concat([data2,data3],axis=0,ignore_index=True)
con_cat
data["rank"]=data["rank"].astype("category")
data["rank_change"]=data["rank_change"].astype("float")
data.dtypes
assert data["rank"].notnull().all()#returns nothing because we drop empty values
assert data.columns[0]=="rank"#returns nothing because we drop empty values
data.plot(subplots=True)
plt.show()
import warnings
warnings.filterwarnings("ignore")
data4=data.head()
data_list=["01.08.1995","15.04.1997","05.05.1986","14.04.1993","16.06.1998"]
datatime_obj=pd.to_datetime(data_list)
data4["date"]=datatime_obj
#lets change index
data4=data4.set_index("date")
data4
data4.resample("A").mean()
data4.resample("A").first().interpolate("linear")
data["TotalPoints"]=data.previous_points+data.rank_change
data.head()
dic={"treatment":["A","A","B","B"],"gender":["F","M","F","M"],"response":["10","11","12","13"],"age":["18","19","20","21"]}
df=pd.DataFrame(dic)
df
#pivoting
df.pivot(index="treatment",columns="gender",values="response")
df1=df.set_index(["treatment","gender"])
df1