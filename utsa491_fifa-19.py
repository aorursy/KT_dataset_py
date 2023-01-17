# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory





# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import seaborn as sns

import pandas as pd



df = pd.read_csv("../input/data.csv")

print(df.head())

# Printing out the columns of the DataSet

print(df.columns)



print(df.info)

df.tail()
df.dropna(axis  = "columns")
df.describe()
df.shape
#Extracting out the Statistics for the Players who Play for Chelsea 



chelsea = df[df["Club"] == "Chelsea"]

len(chelsea)

chelsea.sample(10)
#Extracting out the Statistics for the Players who Play for Chelsea  having the  England Nationality 



chelsea[chelsea["Nationality"] == "England"]

#Extracting out the Age and Nationality  for the players who play for Liverpool

df[df["Club"] == "Liverpool" ][["Age" , "Nationality"]].reset_index().head()
#Extracting out the  Statistics for the players who play for Liverpool and Chelsea  who are more than 30 years old



import operator



c = df[operator.or_(df["Club"] =="Chelsea" , df["Club"] == "Liverpool" )]

d= c [ c["Age"] > 30]



d.loc[:,["Name" ]]
#Extracting out the  Overall , Names , Wages  for the players who play for Liverpool , Chelsea , Arsenal , Napoli, Inter and Juventus

# who are more than 30 years old



import operator



teams =  ["Juventus" ,"Chelsea" ,"Arsenal" , "Liverpool" , "Napoli", "Inter"]

e = df[df["Club"].isin(teams)]

f = e[e["Age"] > 30]

g = f[[ "Name" ,"Overall" , "Wage" ]]. reset_index()



g = g.drop(columns = "index")



print(len(g.index))

h = g.sort_values( by  = "iWge", ascending = True)[["Name" , "Wage"]].sample(10)

h
#Plotting a Histogram to Compare the Number of Players having different overall playing for chelsea and liverpool 





plt.title("Comparing the Number of Players having different overall playing for chelsea and liverpool")

chelsea = df[df["Club"] == "Chelsea"]

chelsea = chelsea["Overall"]

liverpool = df[df["Club"] == "Liverpool"]

liverpool = liverpool["Overall"]

plt.hist( [chelsea, liverpool ],  color = ["Blue" ,"Red"])

plt.xlabel("The Overall of the Players playing for\n Cheslea and Liverpool")

plt.ylabel("The Number of Players falling\n in the overall range")

legend = ["Chelsea" ,"Liverpool"]

plt.legend(legend)

plt.show()
