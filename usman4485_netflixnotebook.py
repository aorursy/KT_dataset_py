#loading the packages
import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
os.listdir('../input/')
df1 = pd.read_csv('../input/netflix-prize-data/combined_data_1.txt',header=None ,
                  names = ['CusId','Rating'], usecols = [0,1])
df1
df1['Rating'].dtype
print(f"Total samples are { int(df1.shape[0]/1000000)}M")
print("reloading all the other files as well")
df2 = pd.read_csv('/content/combined_data_2.txt',header=None ,
                  names = ['CusId','Rating'], usecols = [0,1])
df3 = pd.read_csv('/content/combined_data_3.txt',header=None ,
                  names = ['CusId','Rating'], usecols = [0,1])
df4 = pd.read_csv('/content/combined_data_4.txt',header=None ,
                  names = ['CusId','Rating'], usecols = [0,1])

print(f"Total samples in the 2nd file are { int(df2.shape[0]/1000000)}M")
print(f"Total samples in the 3nd file are { int(df3.shape[0]/1000000)}M")
print(f"Total samples in the 4nd file are { int(df4.shape[0]/1000000)}M")
df = df1
df = df1.append(df2)
df = df.append(df3)
df = df.append(df4)
df
df.index = np.arange(0,len(df))

df.index = np.arange(0,len(df))
print('Full dataset shape: {}'.format(df.shape))
print('-Dataset examples-')
print(df.iloc[::5000000, :])
#plotting the data
#p = df.groupby('Rating')['Rating'].agg(['count'])

#getting movie count
movie_count = df.isnull().sum()
movie_count
df.groupby('Rating')['Rating'].sum()
count_nan =  df.groupby('Rating')['Rating'].agg(['count'])
count_nan
plt.title("No of null Ratings per rank")
sns.barplot(x = count_nan.index , y = count_nan['count'])
#removing missing values
df.dropna()
movies = pd.read_csv('movie_titles.csv',encoding='ISO-8859-1',names = ['YEAR','MOVIE'])
movies
#finding nan colums
movies.isnull()
movies.isnull().sum()
print("listing no of movies without years")
movies[pd.isnull(movies).any(axis =1 )]
movies.at[4388,'YEAR'] = 2001
movies.at[4794,'YEAR'] = 2001
movies.at[7241,'YEAR'] = 2001
movies.at[10782,'YEAR'] = 1974
movies.at[15918,'YEAR'] = 1999
movies.at[16678,'YEAR'] = 1994
movies.at[17667,'YEAR'] = 1999
movies.isnull().sum()
movies['YEAR'].dtype
#converting into Integer
movies['YEAR'].astype(np.int32)
