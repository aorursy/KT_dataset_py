# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory









# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/PakistanSuicideAttacks Ver 6 (10-October-2017).csv", encoding="latin1")
data.head()
data.drop("S#",inplace=True,axis=1)
data.describe()
data.isnull().sum()
data.columns
# Lets see the missing values and their percentage

total=data.isnull().sum().sort_values(ascending=False)

percentage=(data.isnull().sum()/data.isnull().count()).sort_values(ascending=False)

missing_data=pd.concat([total,percentage],axis=1,keys=["Total","Percentage"])

missing_data
#Creating a new column Date which contain dates without -

data["Date"]=data["Date"].str.replace("-"," ")
data.head()

#Lets seperate day month and year from date

day_name=data.Date.str.split(' ').str[0]

month=data.Date.str.split(' ').str[1]

day=data.Date.str.split(' ').str[2]

year=data.Date.str.split(" ").str[3]

data["Day_name"]=day_name

data["Month"]=month

data["Day"]=day

data["Year"]=year

data.head()





data["Islamic Date"]=data["Islamic Date"].str.replace(" al","-al")

data.head()
#Lets seperate day month and year from Islamic date

day=data["Islamic Date"].str.split(' ').str[0]

month=data["Islamic Date"].str.split(' ').str[1]

year=data["Islamic Date"].str.split(' ').str[2]

data["Islamic_Day"]=day

data["Islamic_Month"]=month

data["Islamic_Year"]=year

data.drop("Islamic Date",axis=1,inplace=True)



data.drop("Date",axis=1,inplace=True)
data["Injured Max"].dtypes
data["Injured Max"]=data["Injured Max"].convert_objects(convert_numeric=True)

data["Injured Max"].dtypes
#Lets see the distribution of killed max,killed min ,Injured max and Injured min

plt.figure(figsize=(10,10))

plt.subplot(2,2,1)

sns.distplot(data["Killed Max"].dropna(),color="r")

plt.subplot(2,2,2)

sns.distplot(data["Killed Min"].dropna(),color="g")

plt.subplot(2,2,3)

sns.distplot(data["Injured Max"].dropna(),color="b")

plt.subplot(2,2,4)

sns.distplot(data["Injured Min"].dropna(),color="black");
mean_killed_max=data["Killed Max"].mean()

std_killed_max=data["Killed Max"].std()

null_killed_max=data["Killed Max"].isnull().sum()

rand1=np.random.randint(mean_killed_max-std_killed_max,mean_killed_max+std_killed_max,size=null_killed_max)
data["Killed Max"][np.isnan(data["Killed Max"])]=rand1

data["Killed Max"].isnull().sum();
mean_killed_min=data["Killed Min"].mean()

std_killed_min=data["Killed Min"].std()

null_killed_min=data["Killed Min"].isnull().sum()

rand2=np.random.randint(mean_killed_min-std_killed_min,mean_killed_min+std_killed_min,size=null_killed_min)
data["Killed Min"][np.isnan(data["Killed Min"])]=rand2

data["Killed Min"].isnull().sum();
mean_Injured_max=data["Injured Max"].mean()

std_Injured_max=data["Injured Max"].std()

null_Injured_max=data["Injured Max"].isnull().sum()

rand3=np.random.randint(mean_Injured_max-std_Injured_max,mean_Injured_max+std_Injured_max,size=null_Injured_max)
data["Injured Max"][np.isnan(data["Injured Max"])]=rand3

data["Injured Max"].isnull().sum()
mean_Injured_min=data["Injured Min"].mean()

std_Injured_min=data["Injured Min"].std()

null_Injured_min=data["Injured Min"].isnull().sum()

rand4=np.random.randint(mean_Injured_min-std_Injured_min,mean_Injured_min+std_Injured_min,size=null_Injured_min)
data["Injured Min"][np.isnan(data["Injured Min"])]=rand4

data["Injured Min"].isnull().sum();
plt.title("Number if Suicide Blasts")

plt.hist(data["No. of Suicide Blasts"].dropna());
#Filling the missing values No. of Suicide Blasts with 1

data["No. of Suicide Blasts"]=data["No. of Suicide Blasts"].fillna("1")
#Calculating average people killed and average people injured

data["Killed Avg"]=(data["Killed Max"]+data["Killed Min"])/2

data["Injured Avg"]=(data["Injured Max"]+data["Injured Min"]/2)
data["Location Sensitivity"]=data["Location Sensitivity"].str.replace("low","Low")
#lets draw a heatmap to get some insights

plt.figure(figsize=(10,5))

corr=data.corr()

sns.heatmap(corr,vmax=0.8,annot=True);
plt.figure(figsize=(10,10))

plt.subplot(2,1,1)

plt.title("Average People Killed per year")

sns.barplot("Year","Killed Avg",data=data)

plt.subplot(2,1,2)

plt.title("Average People Injured per year")

sns.barplot("Year","Injured Avg",data=data);

cleaned_influencing=data[pd.notnull(data["Influencing Event/Event"])]

cleaned_influencing[["Influencing Event/Event","Killed Avg"]].groupby("Influencing Event/Event",as_index=False).mean().sort_values(by="Killed Avg",ascending=False)    
#Correlation with Influencing events with average people injured

cleaned_influencing[["Influencing Event/Event","Injured Avg"]].groupby(["Influencing Event/Event"],as_index=False).mean().sort_values(by="Injured Avg",ascending=False)      
data["Explosive Weight (max)"]=data["Explosive Weight (max)"].str.replace("kg"," kg")

data["Explosive Weight (max)"]=data["Explosive Weight (max)"].str.replace("KG"," kg")

#correlation between blast/explosive weight and number of people killed and injured

data["Explosive Weight (max)"]=data["Explosive Weight (max)"].str.split(" ").str[0]

data["Explosive Weight (max)"]=data["Explosive Weight (max)"].str.split("-").str[0]

data["Explosive Weight (max)"]=data["Explosive Weight (max)"].convert_objects(convert_numeric=True)
plt.figure(figsize=(5,5))

plt.xlabel("Explosive Weight (max)")

plt.ylabel("Average people killed and injured")

plt.title("Correlation between Explosive weight and Average people Killed/Injured")

plt.scatter("Explosive Weight (max)","Killed Avg",data=data,label="Killed Average",color="r")

plt.scatter("Explosive Weight (max)","Injured Avg",data=data,label="Injured Average",color="b")

plt.legend(loc="upper right");
W=data["Holiday Type"]=="Weekend"

P=data["Holiday Type"]=="Pakistan Day"

L=data["Holiday Type"]=="Labour Day"

I=data["Holiday Type"]=="Iqbal Day"

EF=data["Holiday Type"]=="Eid-ul-Fitar"

EN=data["Holiday Type"]=="Eid Milad un-Nabi"

A=data["Holiday Type"]=="Ashura"
weekend=data[W]

pakistan_day=data[P]

labour_day=data[L]

iqbal_day=data[I]

eid_ul_fitar=data[W]

eid_milad_un_nabi=data[EN]

ashura=data[A]
#Average people killed on different holiday types

plt.figure(figsize=(20,20))

plt.subplot(4,2,1)

plt.title("Average people killed on Weekends")

plt.hist(weekend["Killed Avg"])

plt.subplot(4,2,2)

plt.title("Average people killed on Pakistan Day")

plt.hist(pakistan_day["Killed Avg"])

plt.subplot(4,2,3)

plt.title("Average people killed on Labour Day")

plt.hist(labour_day["Killed Avg"])

plt.subplot(4,2,4)

plt.title("Average people killed on Iqbal Day")

plt.hist(iqbal_day["Killed Avg"])

plt.subplot(4,2,5)

plt.title("Average people killed on Eid_ul_Fitar")

plt.hist(eid_ul_fitar["Killed Avg"])

plt.subplot(4,2,6)

plt.title("Average people killed on Eid Milad_un_Nabi")

plt.hist(eid_milad_un_nabi["Killed Avg"])

plt.subplot(4,2,7)

plt.title("Average people killed on Ashura")

plt.hist(ashura["Killed Avg"]);
#Average people injured on different holiday types

plt.figure(figsize=(20,20))

plt.subplot(4,2,1)

plt.title("Average people Injured on Weekends")

plt.hist(weekend["Injured Avg"])

plt.subplot(4,2,2)

plt.title("Average people Injured on Pakistan Day")

plt.hist(pakistan_day["Injured Avg"])

plt.subplot(4,2,3)

plt.title("Average people Injured on Labour Day")

plt.hist(labour_day["Injured Avg"])

plt.subplot(4,2,4)

plt.title("Average people Injured on Iqbal Day")

plt.hist(iqbal_day["Injured Avg"])

plt.subplot(4,2,5)

plt.title("Average people Injured on Eid_ul_Fitar")

plt.hist(eid_ul_fitar["Injured Avg"])

plt.subplot(4,2,6)

plt.title("Average people Injured on Eid Milad_un_Nabi")

plt.hist(eid_milad_un_nabi["Injured Avg"])

plt.subplot(4,2,7)

plt.title("Average people Injured on Ashura")

plt.hist(ashura["Injured Avg"])
data["Islamic_Month"]=data["Islamic_Month"].str.replace("SHawwal","Shawwal")

data["Islamic_Month"]=data["Islamic_Month"].str.replace("Jumada-al-awwal","Jamadi-ul-Awal")

data["Islamic_Month"]=data["Islamic_Month"].str.replace("Jumaada-al-awal","Jamadi-ul-Awal")

data["Islamic_Month"]=data["Islamic_Month"].str.replace("SHa`baan","Shaban")

data["Islamic_Month"]=data["Islamic_Month"].str.replace("Shawwal","Shawaal")

data["Islamic_Month"]=data["Islamic_Month"].str.replace("Thw-al-Qi`dah","Zeqad")

data["Islamic_Month"]=data["Islamic_Month"].str.replace("Thw-al-Hijjah","Zilhaj")
data["Islamic_Month"].unique()
#correlation between Islamic month and average people killed and average people injured

cleaned_Islamic_Date=data[pd.notnull(data["Islamic_Month"])]

cleaned_Date=data[pd.notnull(data["Month"])]
plt.figure(figsize=(10,5))

plt.subplot(1,2,1)

plt.title("Average people killed in each Islamic month")

sns.barplot("Islamic_Month","Killed Avg",data=cleaned_Islamic_Date)

plt.xticks(rotation=90)

plt.subplot(1,2,2)

plt.title("Average people injured in each Islamic month")

sns.barplot("Islamic_Month","Injured Avg",data=cleaned_Islamic_Date)

plt.xticks(rotation=90)
plt.figure(figsize=(10,5))

plt.xticks(rotation=90)

sns.countplot("Islamic_Month",hue="Location Sensitivity",data=cleaned_Islamic_Date);
plt.figure(figsize=(6,6))

sns.countplot(y="Location Category",data=data,order=data["Location Category"].value_counts().index)
data.Month=data.Month.replace("November","Nov")

data.Month=data.Month.replace("December","Dec")

data.Month=data.Month.replace("February","Feb")

data.Month=data.Month.replace("October","Oct")

data.Month=data.Month.replace("April","Apr")

data.Month=data.Month.replace("March","Mar")

data.Month=data.Month.replace("August","Aug")

data.Month=data.Month.replace("January","Jan")

data.Month=data.Month.replace("September","Sep")
sns.countplot(y="Month",data=data,order=data.Month.value_counts().index)
plt.figure(figsize=(15,15))

plt.subplot(2,1,1)

plt.title("Average People Killed per month")

sns.barplot("Month","Killed Avg",hue="Location Sensitivity",data=data)

plt.xticks(rotation=45)

plt.subplot(2,1,2)

plt.title("Average People Injured per month")

sns.barplot("Month","Injured Avg",hue="Location Sensitivity",data=data)

plt.xticks(rotation=45);

plt.figure(figsize=(10,10))

plt.title("Correlation between Explosive weight and location sensitivity")

sns.countplot(y="Explosive Weight (max)",hue="Location Sensitivity",data=data,order=data["Explosive Weight (max)"].value_counts().index)
sns.kdeplot(data.loc[data["Location Sensitivity"]=="High","Killed Avg"],shade=True,color="r",label="High")

sns.kdeplot(data.loc[data["Location Sensitivity"]=="Medium","Killed Avg"],shade=True,color="b",label="Medium")

sns.kdeplot(data.loc[data["Location Sensitivity"]=="low","Killed Avg"],shade=True,color="g",label="Low")
cleaned_holiday=data[pd.notnull(data["Holiday Type"])]
cleaned_holiday["Holiday Type"].isnull().sum()
cleaned_Islamic_Date.head()
plt.figure(figsize=(14,5))

plt.subplot(1,2,1)

plt.title("Average people killed each day")

sns.barplot("Day_name","Killed Avg",data=cleaned_Islamic_Date)

plt.subplot(1,2,2)

plt.title("Average people injured each day")

sns.barplot("Day_name","Injured Avg",data=cleaned_Islamic_Date);
cleaned_location=data[pd.notnull(data["Location Sensitivity"])]
sns.countplot(y="Day_name",hue="Location Sensitivity",data=cleaned_location)
data["City"].unique()
data["City"]=data.City.str.replace(" ","")

data["City"]=data.City.str.replace("karachi","Karachi")

data["City"]=data.City.str.replace("karachi","Karachi")
plt.figure(figsize=(20,5))

sns.barplot("City","Killed Avg",data=data)

plt.xticks(rotation=90);
plt.figure(figsize=(20,5))

sns.barplot("City","Killed Avg",data=data)

plt.xticks(rotation=90);
plt.figure(figsize=(20,5))

sns.countplot("City",hue="Location Sensitivity",data=data)

plt.xticks(rotation=90);
plt.figure(figsize=(15,5))

plt.subplot(1,2,1)

plt.xticks(rotation=45)

plt.title("Average People Killed in each Province")

sns.barplot("Province","Killed Avg",data=data)

plt.subplot(1,2,2)

plt.xticks(rotation=45)

plt.title("Average People Injured in each Province")

sns.barplot("Province","Injured Avg",data=data);
sns.countplot("Blast Day Type",data=data);
data["Open/Closed Space"]=data["Open/Closed Space"].str.replace("open","Open")

data["Open/Closed Space"]=data["Open/Closed Space"].str.replace(" ","")

data["Open/Closed Space"]=data["Open/Closed Space"].str.replace("closed","Closed")
sns.countplot("Open/Closed Space",data=data)
plt.figure(figsize=(15,5))

plt.xticks(rotation=90)

sns.barplot("Target Type","Killed Avg",data=data);

plt.figure(figsize=(15,5))

plt.xticks(rotation=90)

sns.barplot("Target Type","Injured Avg",data=data);
data["Longitude"]=data["Longitude"].replace(" ","")

data["Latitude"]=data["Latitude"].replace(" ","")
data["Longitude"]=data["Longitude"].astype(float)

data["Latitude"]=data["Latitude"].astype(float)
plt.figure(figsize=(5,5))

plt.scatter(x="Latitude",y="Longitude",data=data)

plt.ylabel('longitude')

plt.xlabel('latitude');
sns.countplot("Targeted Sect if any",data=data)

plt.xticks(rotation=45)
data["Hospital Names"]=data["Hospital Names"].str.extract('([a-zA-Z ]+)', expand=False).str.strip()
data[["Hospital Names","Killed Avg"]].groupby(["Hospital Names"],as_index=False).mean().sort_values(by="Killed Avg",ascending=False)   
data[["Hospital Names","Injured Avg"]].groupby(["Hospital Names"],as_index=False).mean().sort_values(by="Injured Avg",ascending=False)   