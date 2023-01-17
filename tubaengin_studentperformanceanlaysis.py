import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
data = pd.read_csv("../input/StudentsPerformance.csv")
data.head() # Informations about the data
data.info() # Informations about the features of data
data.isnull().sum() # are there any "null" values in features ? 
data.describe() # Mathematical informations about the numerical features
data["gender"].value_counts()
data["race/ethnicity"].value_counts()
data["parental level of education"].value_counts()
data["lunch"].value_counts()
data["test preparation course"].value_counts()
sns.pairplot(data , hue="gender",palette="dark")
plt.figure(figsize=(15,5))



plt.subplot(1,3,1)

sns.boxplot(x = 'gender', y = 'math score', data = data,palette = "PRGn")



plt.subplot(1,3,2)

sns.boxplot(x = 'gender', y = 'reading score', data = data,palette = "PRGn")



plt.subplot(1,3,3)

sns.boxplot(x = 'gender', y = 'writing score', data = data,palette = "PRGn")



plt.tight_layout()

plt.show()
plt.figure(figsize=(9,8))

sns.heatmap(data.corr() , annot = True , cmap="inferno")

plt.show()
plt.figure(figsize=(13,4))



plt.subplot(1,3,1)

sns.barplot(x = "gender" , y=data["math score"] , data=data)

plt.title("Math Scores")



plt.subplot(1,3,2)

sns.barplot(x = "gender" , y=data["reading score"] , data=data)

plt.title("Reading Scores")



plt.subplot(1,3,3)

sns.barplot(x = "gender" , y=data["writing score"] , data=data)

plt.title("Writing Scores")



plt.tight_layout()

plt.show()
plt.figure(figsize=(13,4))



plt.subplot(1,3,1)

sns.barplot(x = "race/ethnicity" , y="reading score" , data=data)

plt.title("Reading Scores")



plt.subplot(1,3,2)

sns.barplot(x = "race/ethnicity" , y="writing score" , data=data)

plt.title("Writing Scores")



plt.subplot(1,3,3)

sns.barplot(x = "race/ethnicity" , y="math score" , data=data)

plt.title("Math Scores")



plt.tight_layout()

plt.show()
plt.figure(figsize=(13,4))



plt.subplot(1,3,1)

sns.barplot(x = "lunch" , y="reading score" , data=data)

plt.title("Reading Scores")



plt.subplot(1,3,2)

sns.barplot(x = "lunch" , y="writing score" , data=data)

plt.title("Writing Scores")



plt.subplot(1,3,3)

sns.barplot(x = "lunch" , y="math score" , data=data)

plt.title("Math Scores")



plt.tight_layout()

plt.show()
plt.figure(figsize=(13,4))



plt.subplot(1,3,1)

sns.barplot(x = "parental level of education" , y="reading score" , data=data)

plt.xticks(rotation = 90)

plt.title("Reading Scores")



plt.subplot(1,3,2)

sns.barplot(x = "parental level of education" , y="writing score" , data=data)

plt.xticks(rotation=90)

plt.title("Writing Scores")



plt.subplot(1,3,3)

sns.barplot(x = "parental level of education" , y="math score" , data=data)

plt.xticks(rotation=90)

plt.title("Math Scores")



plt.tight_layout()

plt.show()
plt.figure(figsize=(13,4))



plt.subplot(1,3,1)

sns.barplot(x = "test preparation course" , y="reading score" , data=data)

plt.title("Reading Scores")



plt.subplot(1,3,2)

sns.barplot(x = "test preparation course" , y="writing score" , data=data)

plt.title("Writing Scores")



plt.subplot(1,3,3)

sns.barplot(x = "test preparation course" , y="math score" , data=data)

plt.title("Math Scores")



plt.tight_layout()

plt.show()
plt.figure(figsize=(15,5))



plt.subplot(1,3,1)

sns.boxplot(x = 'parental level of education', y = 'math score', data = data,palette = "PRGn")

plt.xticks(rotation=90)



plt.subplot(1,3,2)

sns.boxplot(x = 'parental level of education', y = 'reading score', data = data,palette = "PRGn")

plt.xticks(rotation=90)



plt.subplot(1,3,3)

sns.boxplot(x = 'parental level of education', y = 'writing score', data = data,palette = "PRGn")

plt.xticks(rotation=90)



plt.tight_layout()

plt.show()
# We create a columns that gives informations us about whether the students passed their class or not

# If the grade is < 45 means Fail otherwise Pass.



data["math grade"] = np.where(data["math score"]<45,"Fail" , "Pass")

data["reading grade"] = np.where(data["reading score"]<45, "Fail" , "Pass")

data["writing grade"] = np.where(data["writing score"]<45, "Fail" , "Pass")
data.head()
plt.figure(figsize=(15,5))



plt.subplot(1,3,1)

sns.countplot(x = "parental level of education" ,hue="math grade" , data=data ,palette="dark")

plt.xticks(rotation=90)

plt.title("Math Grade")



plt.subplot(1,3,2)

sns.countplot(x = "parental level of education" ,hue="reading grade" , data=data ,palette="Greens_d")

plt.xticks(rotation=90)

plt.title("Reading Grade")



plt.subplot(1,3,3)

sns.countplot(x = "parental level of education" ,hue="writing grade" , data=data ,palette="GnBu_d")

plt.xticks(rotation=90)

plt.title("Writing Grade")





plt.show()