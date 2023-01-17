import numpy as np
import pandas as pd 
import os
data = pd.read_csv("../input/googleplaystore.csv")
data.head()
data.describe(include='all')
print('Before removing duplicates',len(data))
data = data.drop_duplicates(subset='App')
print('After removing duplicates',len(data))
print('Before removing Nan entries',len(data))
data = data.dropna()
print('After removing Nan entries',len(data))
print(data.Installs.unique())
print("Data type: ", data.Installs.dtypes)
data.Installs = data.Installs.str.replace('+','').str.replace(',','').astype(int)
print(data.Installs.head())
data['Category'].value_counts().plot.bar()
print(data['Category'].value_counts().describe())
category_sums = pd.DataFrame(columns=['Category','Total_Installs','Mean_Installs'])
for category in data['Category'].unique():
    sum_install = data.loc[data.Category == category].Installs.sum()
    mean_install = data.loc[data.Category == category].Installs.mean()
    category_sums = category_sums.append({"Category":category, "Total_Installs":sum_install, "Mean_Installs":mean_install}, ignore_index=True)
category_sums.plot.bar(x='Category', y='Total_Installs')
category_sums.sort_values(by='Mean_Installs', ascending=False)[:10].plot.bar(x="Category",y="Mean_Installs")
#Another method to find the average install/sum of installs for each category
data.groupby("Category").mean().sort_values(by="Installs",ascending=False)[:10].plot.bar()
import seaborn as sns
data.groupby('Rating')['Rating','Installs'].mean().sort_values(by='Installs',ascending=False).head(5).plot.bar()
data.Type.unique()
data.groupby('Type').Installs.mean().plot.bar()
print(data.groupby('Type').Installs.mean().round())
data.Price.unique()
paid = data.loc[data.Price != '0']
paid.groupby('Price')['Price','Installs'].mean().sort_values(by='Installs',ascending=False).head(5).plot.bar()
data['Content Rating'].unique()
data.groupby('Content Rating').Installs.mean().plot.bar()