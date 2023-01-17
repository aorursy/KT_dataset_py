# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
%matplotlib inline
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#read data
data=pd.read_csv('../input/USD_TRY Gemi Verileri.csv')
data.head(10)
#Terms:
#Tarih -> Date
#Şimdi -> Now
#Açılış -> Opening
#Yüksek -> Highest
#Düşük -> Lowest
#Fark -> Difference
data.info()
#convert to
data['Tarih']=pd.to_datetime(data['Tarih'])
data['Şimdi']=pd.to_numeric(data['Şimdi'].str.replace(',', '.'), errors='coerce')
data['Yüksek']=pd.to_numeric(data['Yüksek'].str.replace(',', '.'), errors='coerce')
data['Açılış']=pd.to_numeric(data['Açılış'].str.replace(',', '.'), errors='coerce')
data['Düşük']=pd.to_numeric(data['Düşük'].str.replace(',', '.'), errors='coerce')

data.describe()
last_week_data=data.head(7)
last_week_data
# visualization
#barplot
sns.barplot(x=last_week_data['Tarih'], y=last_week_data['Şimdi'])
plt.xticks(rotation=90)
plt.xlabel('Date')
plt.ylabel('Highest value of dollar')
plt.title('Change')
# pointplot

sns.pointplot(x='Tarih',y='Açılış',data=last_week_data,color='blue',alpha=0.8)
sns.pointplot(x='Tarih',y='Yüksek',data=last_week_data,color='red',alpha=0.8)
plt.xticks(rotation=90)
plt.text(10,5,'opening value',color='blue',fontsize = 17,style = 'italic')
plt.text(10,5.5,'highest value',color='red',fontsize = 18,style = 'italic')
plt.xlabel('Date',fontsize = 15,color='blue')
plt.ylabel('Values of USD',fontsize = 15,color='blue')
plt.title('LAST WEEK VALUES OF USD',fontsize = 20,color='black')
plt.grid()
# Show the results of a linear regression within each dataset
sns.lmplot(x="Açılış", y="Şimdi", data=data)
plt.show()
# pair plot
sns.pairplot(data)
plt.show()
# highest currency value
list1=list(data['Yüksek'])
list2=list(data['Düşük'])
print(max(list1),min(list2))
df=pd.read_csv('../input/USD_TRY Gemi Verileri.csv')
df['Tarih'] = pd.to_datetime(df['Tarih'])
df['day'] = df.Tarih.dt.day
df['month'] = df.Tarih.dt.month
df['year'] = df.Tarih.dt.year

def get_nearest_time_data(df, day):
    newdf = pd.DataFrame()
    for month in range(1,13):
        daydf = df[(df.day==day) & (df.month==month)]
        while (daydf.shape[0]==0):
            day+=1
            daydf = df[(df.day==day) & (df.month==month)]  
        newdf = pd.concat([newdf,daydf], ignore_index=True)
    return newdf

data1=get_nearest_time_data(df, 5)
data1
#You can draw a scatterplot with the matplotlib plt.scatter function
#it is also the default kind of plot shown by the jointplot() function:
sns.jointplot(x="Açılış", y="Şimdi", data=data)
