# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('/kaggle/input/suicide-rates-overview-1985-to-2016/master.csv')
data.info()
f,ax = plt.subplots(figsize=(17,17))
sns.heatmap(data.corr(),annot= True,linewidths = 0.75,fmt='.1f',ax=ax)
data.head(10)
data.columns
data.population.plot(kind='line',color='b',label='Nüfus',linewidth=1,alpha=0.75,grid=True,linestyle=":")
data.suicides_no.plot(color='r',label='İntihar Sayısı',linewidth=1,alpha=0.60,grid=True,linestyle="--")
plt.legend(loc="upper right")
plt.xlabel("Nüfus")
plt.ylabel("İntihar Sayısı")
plt.title("Nüfus ve İntihar Sayısı")
data.plot(kind = 'scatter',x='suicides_no',y='population',color='red',alpha=0.5)
plt.xlabel("Nüfus")
plt.ylabel("İntihar sayısı")
plt.title("Nüfus ve İntıhar Sayısı")
data.suicides_no.plot(kind='hist',bins=150,figsize=(15,15))
plt.show()
series = data["age"]
print(type(series))
df = data[["sex"]]
print(type(df))
desired = data[(data["sex"] == "female") & (data["age"] == "15-24 years") & (data["suicides_no"] > 800)]
desired2 = data[np.logical_and(data["country"] == "Turkey",data["age"] == "15-24 years")]
print(desired2)
liste1 = list(data)
i=0
while i+1 <= len(liste1) :
    print(f"{i+1}th column is {liste1[i]}")
    i+=1
def function(*args) :
    for i in args :
        print(i)
function(liste1)
result = map(lambda x : int((16000-x)/12),data[data["country"] == "Turkey"]["gdp_per_capita ($)"])
print(list(result))
z = zip(data["country-year"],data["generation"])
data["new"] = list(z)
print(data.new)

data["statementofCountries"] =["problematic" if i >2.5 else "normal" if i==2.5 else "better" for i in data["suicides/100k pop"]]
print(data[data["statementofCountries"]=="problematic"]["country-year"])