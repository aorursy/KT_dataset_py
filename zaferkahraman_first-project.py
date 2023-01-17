# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
###############################
import matplotlib.pyplot as plt
import seaborn as sns   # visualization tools   # görselleştirme aracı
###############################
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

from subprocess import check_output  
print(check_output(["ls", "../input"]).decode("utf-8")) # bulunduğu dizindeki dosyaları listeler

# Any results you write to the current directory are saved as output.
data1 = pd.read_csv('../input/2015.csv')
data2 = pd.read_csv('../input/2016.csv')
data3 =  pd.read_csv('../input/2017.csv')
data1.info()
data2.info()
data3.info()
data1.corr()
data2.corr()
data3.corr()
# correlation map 1
f, ax = plt.subplots(figsize=(13, 13))
sns.heatmap(data1.corr(), annot=True, linewidths=0.5, fmt='.2f', ax=ax)
# annot = değerler  # linewidths = çizgi kalınlığı # fmt = değerin virgülden sonra kaç tane gösterileceği
plt.show()
# correlation map 2
f, ax = plt.subplots(figsize=(13, 13))
sns.heatmap(data2.corr(), annot=True, linewidths=0.5, fmt='.2f', ax=ax)
plt.show()
# correlation map 3
f, ax = plt.subplots(figsize=(13, 13))
sns.heatmap(data3.corr(), annot=True, linewidths=0.5, fmt='.2f', ax=ax)
plt.show()
# data1.head(-148) # 158 - gösterilmek istenen değer want to see values
# similar to 
#data1.head(11) 
data1[10::-1]

data2.head()
data3.head(7)
data1.columns = data1.columns.str.strip().str.lower().str.replace(' ', '_').str.replace("(","").str.replace(")","")
data1.columns
data2.columns = data2.columns.str.strip().str.lower().str.replace(" ","_").str.replace("(","").str.replace(")","")
data2.columns
data3.columns = data3.columns.str.strip().str.lower().str.replace(".","_").str.replace("__","_").str.replace("a_","a").str.replace("y_","y").str.replace("n_","n")
data3.columns

# Matplotlib

data1.economy_gdp_per_capita.plot(kind= 'line', color = 'b', label = 'Economy',linewidth=1,grid = True, linestyle=':')
data1.health_life_expectancy.plot(color= 'r', label= 'Health Life', linewidth=1, alpha=0.9, grid=True, linestyle= '-')
plt.legend(loc='upper right')
plt.xlabel('Economy')
plt.ylabel('Health Life')
plt.title('Line Plot')
plt.show()
data2.freedom.plot(kind = 'line', color = 'black', label='Freedom', linewidth=1, grid=True, linestyle='--' )
data2.trust_government_corruption.plot(color = 'cyan', label = 'Trust Goverment Corruption', linewidth = 1,grid=True, linestyle = '-.')
plt.legend(loc='upper right')
plt.title('Line Plot')
plt.xlabel('Freedom')
plt.ylabel('Trust Goverment Corruption')
plt.show()
data3.family.plot(kind='line', color='b',label='Family', linewidth=1, grid=True, linestyle=':')
data3.happiness_score.plot(color='r', label='Happiness_Score', linewidth=1,grid=True,linestyle='-.' )
plt.legend(loc='upper left')
plt.title('Line Plot')
plt.xlabel('Family')
plt.ylabel('Happiness Score')
plt.show()
# Scatter plot 
data1.plot(kind='scatter', x='economy_gdp_per_capita', y='health_life_expectancy', alpha=0.5, color='red')
plt.xlabel('Economy')
plt.ylabel('Health')
plt.title('Economy Health Scatter Plot')
plt.show()
data2.plot(kind='scatter', x='freedom', y='trust_government_corruption', color='blue', alpha=0.5)
plt.xlabel('Freedom')
plt.ylabel('Trust Goverment Corruption')
plt.title('Freedom - Trust Government')
plt.show()
data3.plot(kind='scatter', x='family', y='happiness_score', color='g', alpha=0.5)
plt.xlabel('Family')
plt.ylabel('Happiness Score')
plt.title('Family Happiness Score')
plt.show()
# Histogram
# bins = number of bar in figure 
# bins = figürdeki bar sayısı
# figsize figirün büyüklüğü
data1.economy_gdp_per_capita.plot(kind='hist', bins=50, figsize=(10,10))
# data1.economy_gdp_per_capita.plot(kind='hist', bins=60)
plt.show()
dictionary = {'crotaia' : 'zagreb', 'austria': 'vienna'}
print(dictionary.keys())
print(dictionary.values())
dictionary['crotaia'] = "dubrovnik"
print(dictionary)
dictionary['czech'] = "prague"
print(dictionary)
del dictionary['czech']
print(dictionary)
print('vienna' in dictionary.values())
dictionary.clear()
print(dictionary)
print(dictionary)
data1.columns
series = data1['happiness_rank']
print(type(series))
data_frame = data1['standard_error']
print(type(data_frame))
x = data1['happiness_score'] < 150
data1[x]
data1[np.logical_and(data1['happiness_score']< 5 , data1['family']< 0.5)]
data1[(data1['freedom'] < 0.5) & (data1['dystopia_residual']<1)]
print(data1['happiness_score'].value_counts(dropna=False))
data1.describe()
# data1.boxplot(column='happiness_score', by='country')
data_new = data1.head()
data_new
# melt
melted = pd.melt(frame = data_new, id_vars = 'country', value_vars = ['region', 'happiness_score'])
melted
melted.pivot(index='country', columns = 'variable', values = 'value')
data_ilk = data1.head(5)
data_ikinci = data2.tail(5)
conc_data_row = pd.concat([data_ilk, data_ikinci],axis = 0, sort=True)
conc_data_row
data_x = data1['happiness_rank'].head(5)
data_y = data2['happiness_score'].head(5)
conc_data_col= pd.concat([data_x, data_y],axis=1, sort=True)
conc_data_col
data1.dtypes
data2.dtypes
data3.dtypes
data1['country'] = data1['country'].astype('category')
data1['region'] = data1['region'].astype('category')
data1['happiness_rank'] = data1['happiness_rank'].astype('float')
data1.dtypes
data1.info()
# There are not any missing values
data1["happiness_rank"].value_counts(dropna = False)
country = ['Austria', 'Croatia']
population = ["15", "23"]
list_label = ["country", "population"]
list_col = [country, population]
zipped = list(zip(list_label, list_col))
data_dic = dict(zipped)
df = pd.DataFrame(data_dic)
df
# Add new columns 
# yeni sütun ekleme
df["capital"] = ["vienna","zagreb"]
df
# Broadcasting
# Yayın
df["income"] = 5,10
df
# Plotting all data
data_cizim = data1.loc[:, ["family", "happiness_score", "happiness_rank"]]
data_cizim.plot()
# subplots
data_cizim.plot(subplots=True)
plt.show()
# scatter plot
data_cizim.plot(kind="scatter", x="family", y="happiness_score")
plt.show()
# hist plot
data_cizim.plot(kind="hist", y="happiness_score", bins = 50, range = (2,8), normed = True)
# histogram subplot with non cumulative and cumulative
fig, axes = plt.subplots(nrows = 2, ncols = 1)
data_cizim.plot(kind="hist", y="happiness_rank", bins=40, range = (0,50), normed = True, ax=axes[0])
data_cizim.plot(kind="hist", y="happiness_score", bins=40, range = (0,50), normed = True, ax=axes[1], cumulative=True)
plt.savefig('graph.png')
plt
data2.describe()
time_list = ["1995-03-15", "1997-05-29"]
print(type(time_list[1]))
# As you can see date is string
# Gördügünüz gibi tarih string
# however we want it to be datetime object
datetime_object = pd.to_datetime(time_list)
print(type(datetime_object))
# close warning
import warnings
warnings.filterwarnings("ignore")
data2_new = data2.head()
date_list = ["1992-01-10","1992-02-10","1992-03-10","1993-03-15","1993-03-16"]
datetime_object = pd.to_datetime(date_list)
data2_new["date"] = datetime_object
# lets make date as index
data2_new = data2_new.set_index("date")
data2_new
# Now we can select according to our date index
print(data2_new.loc["1993-03-16"])
print(data2_new.loc["1992-03-10":"1993-03-16"])
data2_new.resample("A").mean()
# Lets resample with month
data2_new.resample("M").mean()
# We can interpolete from first value
# İlk değerden başlayarak aralara değer ekleyebiliriz
data2_new.resample("M").first().interpolate("linear")
# Or we can interpolate with mean()
# vaya ortalama ile değer ekleyebiliriz
data2_new.resample("M").mean().interpolate("linear")
# read data
# veriyi oku
data2_second = pd.read_csv('../input/2016.csv')
#data2_second = data2_second.set_index("month")
#data2_second.head()
data2_second
data2_second["Family"][15]
data2_second.Family[15]
data2_second.loc[15, ["Family"]]
data2_second[["Family","Freedom"]]
# Difference between selecting columns: series and dataframes
# sütun seçimi arasındaki fark
print(type(data2_second["Family"])) # series
print(type(data2_second[["Family"]])) # data frames
# Slicing and indexing series
data2_second.loc[1:10, "Family":"Freedom"]
# reverse slicing
data2_second.loc[10:1:-1, "Family":"Freedom"]
# From something to end
data2_second.loc[1:10, "Freedom":]
# creating boolen series
boolean = data2_second.Family > 1.12
data2_second[boolean]
# combining filters
first_filter = data2_second.Family > 1.12
second_filter = data2_second.Freedom > 0.57
data2_second[first_filter & second_filter]
# filtering column based other
data2_second.Family[data2_second.Freedom<0.1]
# Plain python functions
# sade pyhton fonksiyonları
def div(n):
    return n/2
data2_second.Family.apply(div)
# or we can use lambda function
# veya lambda fonksiyonunu kullanabiliriz
data2_second.Family.apply(lambda n : n/2)
# Defining column using other columns
# diğer sütunları kullanarak sütun tanımlama
data2_second["total"] = data2_second.Family + data2_second.Freedom
data2_second.head()
# our index name is this:
# bizim index imiz:
print(data2_second.index.name)
#lets change it
# değiştirelim
data2_second.index.name = "index_name"
data2_second.head()
data2_second.head()
second_data = data2_second.copy()
second_data.index = range(0,157,1)
second_data.head()
data2.head()
data2_third = data2.set_index(["region", "freedom"])
data2_third.head(100)
dic = {"treatment":["A","A","B","B"],"gender":["F","M","F","M"],"responce":[10, 45, 5, 9], "age":[1, 5, 72, 65]}
df = pd.DataFrame(dic)
df
# pivotting
df.pivot(index="treatment", columns="gender", values="responce")
df1 = df.set_index(["treatment", "gender"])
df1
df1.unstack(level=0)
df1.unstack(level=1)
df2 = df1.swaplevel(0,1)
df2
df
# df.pivot(index="treatmnt", columns = "gender", values="responce")
pd.melt(df, id_vars="treatment", value_vars=["age", "responce"])
# we will use df
df
df.groupby("treatment").mean()
df.groupby("treatment").age.max()
df.groupby("treatment")[["age", "responce"]].min()
df.info()