import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns 

%matplotlib inline

# Any results you write to the current directory are saved as output.
df=pd.read_csv("/kaggle/input/nutrition-facts/menu.csv")

df
#checking the data types

df.dtypes
df.describe()
df.describe(include="all")
df.info
df[['Item','Sodium']].head(20)
plot=sns.swarmplot(x="Category", y="Sodium", data=df)

plt.setp(plot.get_xticklabels(),rotation=70)

plt.title("Sodium Content")

plt.show()
df[['Sodium']].describe()
pd.set_option('display.max_rows', 10)

df[['Sodium']].idxmax()
df.at[82,'Item']
df[['Sodium',"Item"]]
df[['Sodium']].max()
plot=sns.jointplot(x='Protein',y='Total Fat',data=df)

plt.show()
plot=sns.set_style("whitegrid")

ax=sns.boxplot(x=df["Sugars"])

plt.show()

# note the Sugars outliters, look for the items
df[['Calories','Item']]
df[['Calories']].idxmax()
df.at[82,'Calories']
plot=sns.swarmplot(x="Category", y="Calories", data=df)

plt.setp(plot.get_xticklabels(),rotation=70)

plt.title("Calories Content")

plt.show()