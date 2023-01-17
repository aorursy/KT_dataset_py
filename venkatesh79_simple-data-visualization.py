# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data=pd.read_csv("/kaggle/input/covid19-india-dataset/covid_india.csv")
data.head()
data.drop(["S. No."],axis=1,inplace=True)
data.rename(columns={"Name of State / UT":"state","Cured/Discharged/Migrated":"Discharged"},inplace=True)
data.head()
plt.figure(figsize=(18,10))

plt.bar(data["state"],data["Active Cases"],edgecolor="black")

plt.title("Total Active Cases By state",fontsize=15)

plt.xlabel("Region",fontsize=14)

plt.ylabel("Cases",fontsize=14)

plt.xticks(rotation=90,size=14)

plt.yticks(size=14)

plt.show()
plt.figure(figsize=(18,10))

plt.bar(data["state"],data["Discharged"],edgecolor="black")

plt.title("Total Discharged Patients",fontsize=15)

plt.xlabel("Region",fontsize=14)

plt.ylabel("Patients",fontsize=14)

plt.xticks(rotation=90,size=14)

plt.yticks(size=14)

plt.show()
plt.figure(figsize=(18,10))

plt.bar(data["state"],data["Deaths"],edgecolor="black")

plt.title("Total Deaths",fontsize=15)

plt.xlabel("Region",fontsize=14)

plt.ylabel("Patients",fontsize=14)

plt.xticks(rotation=90,size=14)

plt.yticks(size=14)

plt.show()
plt.figure(figsize=(18,10))

plt.bar(data["state"],data["Total Confirmed cases"],edgecolor="black")

plt.title("Total Deaths",fontsize=15)

plt.xlabel("Region",fontsize=14)

plt.ylabel("Patients",fontsize=14)

plt.xticks(rotation=90,size=14)

plt.yticks(size=14)

plt.show()
data1=data.groupby("state")["Active Cases","Deaths","Discharged","Total Confirmed cases"].sum()

data1.head()
m=data1.loc["Andhra Pradesh"]

plt.style.use("seaborn")

plt.title("Andhra Pradesh Covid Overview")

labels=["active Cases","Deaths","Discharged","Total Confirmed cases"]

colors=["#e3fc03","#fc3903","#03fc62","#036ffc"]

plt.pie(m,labels=labels,colors=colors,wedgeprops={'edgecolor':'black'},autopct='%1.1f%%')

plt.show()
Deaths=data["Deaths"].sum()

Discharged=data["Discharged"].sum()

Active=data["Active Cases"].sum()

Total_Cases=data["Total Confirmed cases"].sum()
slices=[Active,Deaths,Discharged,Total_Cases]

labels=["Active","Deaths","Discharged","Total_Cases"]

plt.title("India Covid19 OverView")

colors=["#e3fc03","#fc3903","#03fc62","#036ffc"]

plt.pie(slices,colors=colors,autopct='%1.1f%%',labels=labels,shadow=True)

plt.show()