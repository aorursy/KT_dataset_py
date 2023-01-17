import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



import os

print(os.listdir("../input"))



df = pd.read_csv("../input/BlackFriday.csv")

# Any results you write to the current directory are saved as output.
df.head()
row,column = df.shape

df.isna().sum()/row
df['Stay_In_Current_City_Years'] = df['Stay_In_Current_City_Years'].str.replace('4+','4',regex=False)
df['Stay_In_Current_City_Years'] = pd.to_numeric(df['Stay_In_Current_City_Years']) 
df['City_Category'] = df['City_Category'].replace({'A':'1','B':'2','C':'3','D':'4'})
plt.subplot(221)

df.Gender.value_counts().plot(kind='pie')

plt.axis('equal')



plt.subplot(222)

df.Age.value_counts().plot(kind='pie')

plt.axis('equal')



plt.subplot(223)

df.City_Category.value_counts().plot(kind='pie')

plt.axis('equal')



plt.subplot(224)

df.Stay_In_Current_City_Years.value_counts().plot(kind='pie')

plt.axis('equal')

user = df[['User_ID',"Occupation",'City_Category',"Stay_In_Current_City_Years","Purchase"]]
purchase_ = user['Purchase'].groupby(df['User_ID']).sum()
user = user.drop_duplicates(subset='User_ID')
#user.loc[:,-1] = df["Purchase"].groupby(df['User_ID']).sum()