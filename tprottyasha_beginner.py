# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#how to read a file and explore the first 5 elements of the dataset
#checking the physics score of the students
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
train=pd.read_csv("../input/AcademicScores.csv")
train.head()

#Visualizing the physics scores of the students
fig,ax=plt.subplots()
ax.bar(train["Names"],train["Physics"])
ax.set_title("Physics scores of the students")
ax.set_xticklabels(train["Names"],rotation=90)
plt.show()
#those who got higher marks in physics got total marks higher than others
sns.set_style("darkgrid")
ax1=sns.FacetGrid(train,col="Total",size=6)
ax1.map(sns.kdeplot,"Physics",shade="True")
ax1.add_legend()
sns.despine(left=True,bottom=True)
plt.xlabel("Total")
plt.show()
plt.savefig("Total vs Physics score.png")
#those who are good in mathematics are also good in physics.
sns.set_style("darkgrid")
ax1=sns.FacetGrid(train,col="Physics",size=6)
ax1.map(sns.kdeplot,"Mathematics",shade="True")
ax1.add_legend()
sns.despine(left=True,bottom=True)
plt.xlabel("Mathematics")
plt.show()
plt.savefig("Physics vs Mathematics.png")
sns.set_style("darkgrid")
ax1=sns.FacetGrid(train,col="Biology",size=6)
ax1.map(sns.kdeplot,"Physics",shade="True")
ax1.add_legend()
sns.despine(left=True,bottom=True)
plt.xlabel("Biology")
plt.show()

train.describe()
train["Physics"].sort_values(ascending=False)
##General analysis according to physics to other subjects
sns.set_style("darkgrid")
cols=["Mathematics","English","Chemistry","Biology","Economics","Computer std","Governmnet"]
for i in range(0,7):
    ax1=sns.FacetGrid(train,col=cols[i],size=6)
    ax1.map(sns.kdeplot,"Physics",shade="True")
    ax1.add_legend()
    sns.despine(left=True,bottom=True)
    plt.xlabel("Physics")

    plt.show()
train["math+physics"]=train["Mathematics"]+train["Physics"]

sns.set_style("darkgrid")
ax1=sns.FacetGrid(train,col="Biology",size=6)
ax1.map(sns.kdeplot,"math+physics",shade="True")
ax1.add_legend()
sns.despine(left=True,bottom=True)
plt.xlabel("math+physics")
plt.show()
