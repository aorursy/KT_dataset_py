import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns 

import warnings 

warnings.filterwarnings("ignore")

import os

print(os.listdir("../input"))
data = pd.read_csv("../input/StudentsPerformance.csv")
data.info()  
data.head()



#data.tail()
data.columns
data.isnull().sum()



data.dropna(inplace=True,axis=0) # Inplace for saving to the data after dropped.
data.shape
data.describe()
data.corr() 
data_head = data.head(3)

data_middle = data.iloc[500:503,:]

data_tail = data.tail(3)

concatenated_data = pd.concat([data_head,data_middle,data_tail],axis=0,ignore_index=False)

concatenated_data
f, ax = plt.subplots(figsize=(5,5))

sns.heatmap(data.corr(),annot=True,linewidths =.5,fmt=".1f",ax=ax)

plt.show()
data_random = data.sample(20,random_state=42)

data_random.index = np.arange(0,len(data_random))
data1 = data_random

data1["math score"].plot(figsize=(12,4.5),kind = "line",color = "red",label="Math Score",linewidth = 1,alpha=1,grid=True,linestyle='-.')

data1["reading score"].plot(kind = "line",color="green",label="Reading Score",linewidth=1,alpha=1,grid=True,linestyle="-")

data1["writing score"].plot(kind = "line",color = "black",label = "Writing Score",linewidth=1,alpha=1,grid=True,linestyle=":")

plt.legend(loc="upper right")

plt.xlabel('Students',FontSize = 10,color = "purple")

plt.ylabel("Scores",FontSize = 10, color = "green")

plt.title("Scores for 20 Students with Line Plot",FontSize = 12)

plt.savefig("Graphic.png")

plt.show() 

data2 = data.copy()

data2.gender = ["1" if each == "female" else "0" for each in data2.gender]

data2.gender = data2["gender"].astype(int)
data2.plot(kind = 'scatter',x="math score",y="gender",color="black",figsize=(17,8))

plt.xlabel("Mathematic Score",FontSize = 18)

plt.ylabel("Gender",FontSize = 18)

plt.title("Scatter Plot",FontSize = 20)

plt.show()
data2.gender = ["F" if each == 1 else "M" for each in data2.gender]
p = sns.countplot(x='gender', data = data2, palette='bright')

_ = plt.setp(p.get_xticklabels(), rotation=0) 

plt.xlabel("Gender")

plt.ylabel("Count")

plt.show()
data["Above_85"] = np.where(((data["math score"]>=85) & (data["reading score"]>=85) & (data["writing score"]>=85)),"A","U")
ploting = sns.countplot(x='parental level of education',data = data,hue = "Above_85", palette="Blues_d")

plt.setp(ploting.get_xticklabels(), rotation=45) 

plt.xlabel("Parental Level of\nEducation")

plt.ylabel("Count")

plt.show()
data["total_score"] = (data["math score"]+data["reading score"]+data["writing score"])/3
def Degree(mark):

    if mark>=90:

        return "AA"

    elif 90>mark>=85:

        return "BA"

    elif 85>mark>=80:

        return "BB"

    elif 80>mark>=75:

        return "CB"

    elif 75>mark>=65:

        return "CC"

    elif 65>mark>=58:

        return "DC"

    elif 58>mark>=50:

        return "DD"

    else:

        return "FF"

    

data["Mark_Degree"] = data.apply(lambda x: Degree(x["total_score"]),axis=1)

data.head()
p = sns.countplot(x='Mark_Degree',data = data,order=['AA','BA','BB','CB','CC','DC','DD','FF'],palette="Blues_d")

plt.setp(p.get_xticklabels(), rotation=0) 

plt.xlabel("Mark Degrees")

plt.ylabel("Number of Students")

plt.show()
data_melt = data

melting = pd.melt(frame = data_melt,id_vars= ["gender","lunch"],value_vars =["math score","reading score","writing score"])

melting.drop("lunch",axis=1,inplace=True)

melting.rename(index=str,columns={"variable":"ScoreTable","value":"Values","gender":"Gender"},inplace=True)

melting.Gender = ["F" if each == "female" else "M" for each in melting.Gender]

melting.head()
p = sns.countplot(x='ScoreTable', data = melting,hue="Gender" ,palette='bright')

_ = plt.setp(p.get_xticklabels(), rotation=90) 

plt.xlabel("Scores")

plt.ylabel("Count")

plt.show()
data2["math score"].plot(kind = 'hist',bins = 50,figsize = (9,5),alpha=0.9,color="gray")

data2["reading score"].plot(kind = 'hist',bins = 50,alpha=0.7,color="blue")

data2["writing score"].plot(kind = 'hist',bins = 50,alpha=0.5,color="green")

plt.xlabel("Math, Reading & Writing Scores")

plt.legend()

plt.show()
data["Total_Score"] = data["math score"]/3 + data["writing score"]/3 + data["reading score"]/3

data.head()
sns.stripplot(x="parental level of education",y='Total_Score',data=data)

plt.xticks(rotation=45)

plt.xlabel("Parental Level of Education")

plt.ylabel("Total Score")

plt.show()
sns.factorplot(x='gender', y='Total_Score', hue='test preparation course', data=data, kind='bar')

plt.xlabel("Gender")

plt.ylabel("Total Score")

plt.show()