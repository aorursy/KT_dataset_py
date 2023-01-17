# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt

from datetime import timedelta



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df=pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv")

df
week=2

df_italy=df[df['Country/Region']=='Italy']

df_italy.drop(["Province/State","Country/Region","Lat","Long"], axis=1, inplace=True)

df_italy=df_italy.T

df_italy.loc["3/12/20"]=15113 # This data was missing

df_italy=df_italy.loc["2/20/20":,] # Getting data after outbreak

df_italy=df_italy.iloc[:(week*7),] # Just checking first two weeks

df_italy
df_turkey=df[df['Country/Region']=='Turkey']

df_turkey.drop(["Province/State","Country/Region","Lat","Long"], axis=1, inplace=True)

df_turkey=df_turkey.T

df_turkey
df_turkey=df_turkey.loc["3/15/20":,] # Getting data after outbreak

df_turkey.loc["3/22/20"]=947

df_turkey.loc["3/23/20"]=1236

df_turkey
fig = plt.figure(figsize=(18, 10), dpi=80)

ax = fig.add_subplot(111)

y_pos = np.arange(len(df_italy.index))

ty_pos = np.arange(len(df_turkey.index))



w = 0.3

ax.bar(y_pos-w/2, df_italy[16], width=w, color='b', align='center')

ax.bar(ty_pos+w/2, df_turkey[207], width=w, color='r', align='center')



for i, v in enumerate(df_italy[16]):

    ax.text(i -0.35, v + 30, str(v), color='blue', fontweight='bold')



for i, v in enumerate(df_turkey[207]):

    ax.text(i , v + 30, str(v), color='red', fontweight='bold')



plt.ticklabel_format(useOffset=False,style='plain',axis='y')

plt.xticks(rotation=45)

plt.title("Türkiye-İtalya Vaka Sayısı Kıyası (20 Gün Gecikmeli)",fontsize=16)

ax.set_ylabel("Vaka Sayısı",fontsize=16)

ax.set_xlabel("Günler",fontsize=16)

ax.set_xticks(np.arange(0, len(df_italy.index), 1))

ax.set_xticklabels(np.arange(1, len(df_italy.index)+1, 1))



plt.gca().legend(["İtalya","Türkiye"],loc='upper left',fontsize=14)



plt.grid(linestyle='-', linewidth=0.6)



plt.show()
week=3

df_italy=df[df['Country/Region']=='Italy']

df_italy.drop(["Province/State","Country/Region","Lat","Long"], axis=1, inplace=True)

df_italy=df_italy.T

df_italy.loc["3/12/20"]=15113 # This data was missing

df_italy=df_italy.loc["2/20/20":,] # Getting data after outbreak

df_italy=df_italy.iloc[:(week*7),] # Just checking first two weeks

df_italy
fig = plt.figure(figsize=(18, 10), dpi=80)

ax = fig.add_subplot(111)

y_pos = np.arange(len(df_italy.index))

ty_pos = np.arange(len(df_turkey.index))



w = 0.3

ax.bar(y_pos-w/2, df_italy[16], width=w, color='b', align='center')

ax.bar(ty_pos+w/2, df_turkey[207], width=w, color='r', align='center')



for i, v in enumerate(df_italy[16]):

    ax.text(i -0.5, v + 60, str(v), color='blue', fontweight='bold')



for i, v in enumerate(df_turkey[207]):

    ax.text(i , v + 60, str(v), color='red', fontweight='bold')



plt.ticklabel_format(useOffset=False,style='plain',axis='y')

plt.xticks(rotation=45)

plt.title("Türkiye-İtalya Vaka Sayısı Kıyası (20 Gün Gecikmeli)",fontsize=16)

ax.set_ylabel("Vaka Sayısı",fontsize=16)

ax.set_xlabel("Günler",fontsize=16)

ax.set_xticks(np.arange(0, len(df_italy.index), 1))

ax.set_xticklabels(np.arange(1, len(df_italy.index)+1, 1))



plt.gca().legend(["İtalya","Türkiye"],loc='upper left',fontsize=14)



plt.grid(linestyle='-', linewidth=0.6)



plt.show()


df_italy=df[df['Country/Region']=='Italy']

df_italy.drop(["Province/State","Country/Region","Lat","Long"], axis=1, inplace=True)

df_italy=df_italy.T

df_italy.loc["3/12/20"]=15113 # This data was missing

df_italy=df_italy.loc["2/20/20":,] # Getting data after outbreak

df_italy
fig = plt.figure(figsize=(18, 10), dpi=80)

ax = fig.add_subplot(111)

y_pos = np.arange(len(df_italy.index))

ty_pos = np.arange(len(df_turkey.index))



w = 0.3

ax.bar(y_pos-w/2, df_italy[16], width=w, color='b', align='center')

ax.bar(ty_pos+w/2, df_turkey[207], width=w, color='r', align='center')



plt.ticklabel_format(useOffset=False,style='plain',axis='y')

plt.xticks(rotation=45)

plt.title("Türkiye-İtalya Vaka Sayısı Kıyası (20 Gün Gecikmeli)",fontsize=16)

ax.set_ylabel("Vaka Sayısı",fontsize=16)

ax.set_xlabel("Günler",fontsize=16)

ax.set_xticks(np.arange(0, len(df_italy.index), 1))

ax.set_xticklabels(np.arange(1, len(df_italy.index)+1, 1))



plt.gca().legend(["İtalya","Türkiye"],loc='upper left',fontsize=14)



plt.grid(linestyle='-', linewidth=0.6)



plt.show()
df_italy_all=pd.read_csv("/kaggle/input/coronavirus-in-italy/dati-andamento-nazionale/dpc-covid19-ita-andamento-nazionale.csv")

df_italy_test=df_italy_all[["data","tamponi"]] #date, cumulative test number

df_italy_test['data']=pd.to_datetime(df_italy_test['data'])

df_italy_test.set_index('data',inplace=True)

df_italy_test
df_italy_test["daily"]=df_italy_test["tamponi"]-df_italy_test["tamponi"].shift(1)

ffdaverage=df_italy_test.loc["2020-02-24",'tamponi']/5 #first five day average

lastdate=df_italy_test.loc["2020-02-24"].index

# We don't have first five days' test data, from the first available data we roughly estimate it

#df_italy_test.loc["2020-02-20",'daily']=round(ffdaverage*0.6)

#df_italy_test.loc["2020-02-21",'daily']=round(ffdaverage*0.8)

#df_italy_test.loc["2020-02-22",'daily']=round(ffdaverage)

#df_italy_test.loc["2020-02-23",'daily']=round(ffdaverage*1.2)

data={'daily':round(ffdaverage*1.2)}

df_italy_test=df_italy_test.append(pd.DataFrame(data, index=lastdate-timedelta(days=1)))

df_italy_test.loc["2020-02-24",'daily']=round(ffdaverage) #*1.4



print(lastdate-timedelta(days=1))

df_italy_test
fig = plt.figure(figsize=(18, 10), dpi=80)

ax = fig.add_subplot(111)

#ax.plot(df_italy_test["tamponi"])

ax.plot(df_italy_test["daily"])



plt.ticklabel_format(useOffset=False,style='plain',axis='y')

plt.xticks(rotation=90)

plt.title("İtalya Günlük Test Sayısı",fontsize=16)

ax.set_ylabel("Test Sayısı",fontsize=16)

ax.set_xlabel("Günler",fontsize=16)

ax.set_xticks(df_italy_test.index)

#ax.set_xticklabels(np.arange(1, len(df_italy.index)+1, 1))



plt.gca().legend(["İtalya","Türkiye"],loc='upper left',fontsize=14)



plt.grid(linestyle='-', linewidth=0.6)



plt.show()