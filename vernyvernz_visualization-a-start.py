import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import warnings

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

warnings.filterwarnings("ignore")



#lets import our data into the environment

path = "../input/train.csv"

data = pd.read_csv(path)

data
#lets collect our useful features into a df

useful = data[["Survived","Pclass","Sex","Age","SibSp","Parch","Fare"]]

useful
#lets start with some basic visualisation

#lets dummy code the 'sex' column to be more useful



for i in range(len(useful)):

    if useful["Sex"][i] == "female":

        useful["Sex"][i] = 1

    else:

        useful["Sex"][i] = 0
#fix the column name we just broke

useful.columns = ["Survived","Pclass","female","Age","SibSp","Parch","Fare"]

useful
#now that all features of value are numeric , lets do some visualisation
females = useful.copy()

for i in range(len(females)):

    if females["female"][i] == 0:

        females["female"][i] = np.nan

females2 = females.dropna()



#same for males

males = useful.copy()

for i in range(len(males)):

    if males["female"][i] == 1:

        males["female"][i] = np.nan

males2 = males.dropna()
f = plt.figure(figsize=(10,10))



a = plt.scatter(males2["Age"],males2["Fare"] , c = males2["Pclass"] , marker = "*" , label = "males classes 1 - 3")

b = plt.scatter(females2["Age"],females2["Fare"] , c = females2["Pclass"] , marker = "^" , label = "females classes 1 - 3")

plt.xlabel("Age")

plt.ylabel("Fare")

plt.title("Age vs Fare for classes 1 - 3")

plt.legend()

plt.show()
#some visuals for survivors and non-survivors







#fix index

males2.index = np.arange(0,len(males2))



male_survivors  = males2.copy()

for i in range(len(male_survivors)):

    if male_survivors["Survived"][i] == 0:

        male_survivors["Survived"][i] = np.nan

male_survivors2 = male_survivors.dropna()



#fix index

females2.index = np.arange(len(females2))



female_survivors = females2.copy()

for i in range(len(female_survivors)):

    if female_survivors["Survived"][i] == 0:

        female_survivors["Survived"][i] = np.nan

female_survivors2 = female_survivors.dropna()
plt.pie([len(female_survivors2) , len(male_survivors2)] , labels = ["female survivors" , "male survivors"])

plt.legend()

plt.show()



#from the pie chart you can see that most of the survivors were female