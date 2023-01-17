df = pd.read_csv('../input/bengaluru-house-price-data/Bengaluru_House_Data.csv')
df.head()
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df.shape
df.isnull().sum()
for feature in df.columns:
    if df[feature].isnull().sum() > 1:
        print("{} Feature has {}% Missing values ".format(feature,round(df[feature].isnull().mean()*100,1)))
df1 = df.copy()
df1.head()
df1['society'].fillna("Info Not available",inplace = True)
df1.head()
df1['size'].fillna('0',inplace = True)
df1['bath'].fillna(1.0,inplace = True)
df1['balcony'].fillna(0.0,inplace = True)
def is_float(x):
    try:
        float(x)
    except:
        return False
    return True


df1[~df1['total_sqft'].apply(is_float)]
def convert_sqft_to_num(x):
    tokens = x.split('-')
    if len(tokens) == 2:
        return (float(tokens[0])+float(tokens[1]))/2
    try:
        return float(x)
    except:
        return None
df1.total_sqft.isnull().sum()
df1.total_sqft = df1.total_sqft.apply(convert_sqft_to_num)
df1.total_sqft.isnull().sum() 
df1.total_sqft.dropna(axis='index',inplace=True)
df1.total_sqft.isnull().sum()
df1
df1 = df1.astype({'bath':np.int32, 'balcony':np.int32})
df1.info()
df1['bhk'] = df1['size'].apply(lambda x : int(x.split()[0]))
df1
df1['price_per_sqr'] = round(df1['price'] * 100000 / df1['total_sqft'],2) 

df1
df1.location.unique()
len(df1.location.unique())
location_stats = df1['location'].value_counts() 
location_stats
below_10_dp = location_stats[location_stats <= 10]
below_10_dp
df1['location'] = df1['location'].apply(lambda x : 'Others' if x in below_10_dp else x)
df1
df2 = df1.copy()
df3 = df2[~(df2.total_sqft/df2.bhk < 300)]
df3.shape
df3.price_per_sqr.describe()
def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m = np.mean(subdf.price_per_sqr)
        st = np.std(subdf.price_per_sqr)
        reduced_df = subdf[(subdf.price_per_sqr>(m-st)) & (subdf.price_per_sqr<=(m+st))]
        df_out = pd.concat([df_out,reduced_df],ignore_index=True)
    return df_out
df4 = remove_pps_outliers(df3)
df4.shape
def plot_scatter_chart(df,location):
    bhk2 = df[(df.location==location) & (df.bhk==2)]
    bhk3 = df[(df.location==location) & (df.bhk==3)]
    fig = plt.figure(figsize=(12,8))
    fig, plt.scatter(bhk2.total_sqft,bhk2.price,color='black',label='2 BHK', s=50)
    fig, plt.scatter(bhk3.total_sqft,bhk3.price,marker='+', color='red',label='3 BHK', s=50)
    plt.xlabel("Total Square Feet Area")
    plt.ylabel("Price (Lakh Indian Rupees)")
    plt.title(location)
    plt.legend()
    
plot_scatter_chart(df4,"Rajaji Nagar")
plot_scatter_chart(df4,"Hebbal")
def remove_bhk_outliers(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats = {}
        for bhk, bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_df.price_per_sqr),
                'std': np.std(bhk_df.price_per_sqr),
                'count': bhk_df.shape[0]
            }
        for bhk, bhk_df in location_df.groupby('bhk'):
            stats = bhk_stats.get(bhk-1)
            if stats and stats['count']>5:
                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqr<(stats['mean'])].index.values)
    return df.drop(exclude_indices,axis='index')
df5 = remove_bhk_outliers(df4)
df5.shape

plot_scatter_chart(df5,"Rajaji Nagar")
plot_scatter_chart(df5,"Hebbal")

import matplotlib

plt.hist(df5.price_per_sqr,rwidth=0.8)
plt.xlabel("Price Per Square Feet",size = 13)
plt.ylabel("Count", size = 13)
plt.title("Price per sqft distribution", size = 20)
df6 = df5.drop('location',axis='columns')
df6.head()
def LinearEquationPlot(df5,location):
    xy = df6[(df5.location==location)]
    fig = plt.figure(figsize=(20,10))
    sns.regplot(x='total_sqft', y='price', data=xy,ci = 68)
LinearEquationPlot(df5,'Hebbal')