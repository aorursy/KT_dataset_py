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
data = pd.read_csv('/kaggle/input/fifa19/data.csv')

data.info()
data.corr()
f,ax = plt.subplots(figsize=(16, 16))

sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()
data.head(10)
data.columns
data.Overall.plot(kind = 'line', color = 'g',label = 'Overall',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')

data.Age.plot(color = 'r',label = 'Age',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')

plt.legend(loc='upper right')     # legend = puts label into plot

plt.xlabel('x axis')              # label = name of label

plt.ylabel('y axis')

plt.title('Line Plot')            # title = title of plot

plt.show()
data.plot(kind='scatter', x='Age', y='Overall',alpha = 0.5,color = 'red')

plt.xlabel('Age')              # label = name of label

plt.ylabel('Overall')

plt.title('Age Overall Scatter Plot')            # title = title of plot

plt.show()
data.Overall.plot(kind = 'hist',bins = 70,figsize = (12,12))

plt.show()
x = data['Age']>35

x

data[x]
data.columns
threshold = sum(data.Age)/len(data.Age)

data["Age_level"] = ["high" if i > threshold else "low" for i in data.Age]

data.loc[:10,["Age_level","Age"]]
threshold = sum(data.Overall )/len(data.Overall )

data["Overall_level"] = ["high" if i > threshold else "low" for i in data.Overall]

data.loc[:10,["Overall_level","Overall"]]
data.info()
data.describe()
print(data['Nationality'].value_counts(dropna =False))
print(data['Position'].value_counts(dropna =False))
data['Position'].describe()


data.columns
data['Position_head'] = ["RF" if i=='RF' else 'LF' if  i=='LF' else "Other"  for i in data.Position]

sns.boxplot(x='Position_head',y='Overall',data = data)
data_new = data.head()    # I only take 5 rows into new data

data_new
melted = pd.melt(frame=data_new,id_vars = 'Name', value_vars= ['Club','Nationality'])

melted
melted.pivot(index = 'Name', columns = 'variable',values='value')
data1 = data.head()

data2= data.tail()

conc_data_row = pd.concat([data1,data2],axis =0,ignore_index =True) # axis = 0 : adds dataframes in row

conc_data_row
data3 = data['Name'].head()

data1 = data['Position'].head()

data2= data['Overall'].head()

conc_data_col = pd.concat([data3,data1,data2],axis =1) # axis = 0 : adds dataframes in row

conc_data_col
data.dtypes
data['Name'] = data['Name'].astype('category')

data['ID'] = data['ID'].astype('float')
data.info()

data.head(30)
#data["Club"].value_counts(dropna =False)
#data1=data

#data1.dropna(inplace = True)

#data1.info()
from collections import Counter

%matplotlib inline

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
data.info()
data.head()
data = pd.read_csv('/kaggle/input/fifa19/data.csv')
data.Overall.value_counts()
data.Club.value_counts()
club_list = list(data.Club.unique())

club_overall_ratio = []

for i in club_list:

    if type(i) == float:

        club_list.remove(i)

for i in club_list:

    x = data[data['Club'] == i]

    club_overall_rate = sum(x.Overall)/len(x)

    club_overall_ratio.append(club_overall_rate)

data1 = pd.DataFrame({'club_list': club_list,'club_overall_ratio':club_overall_ratio})

new_index = (data1['club_overall_ratio'].sort_values(ascending=False)).index.values

sorted_data = data1.reindex(new_index)
sorted_data
plt.figure(figsize=(15,10))

sns.barplot(x=sorted_data['club_list'].head(35), y=sorted_data['club_overall_ratio'].head(35))

plt.xticks(rotation= 90)

plt.xlabel('Club')

plt.ylabel('Overall Rate')

plt.title('club_overall_ratio_head')
plt.figure(figsize=(15,10))

sns.barplot(x=sorted_data['club_list'].tail(35), y=sorted_data['club_overall_ratio'].tail(35))

plt.xticks(rotation= 90)

plt.xlabel('Club')

plt.ylabel('Overall Rate')

plt.title('club_overall_ratio_tail')
Nationality = list(data.Nationality)

count_nat = Counter(Nationality)

most_common_nat = count_nat.most_common(15)  

x,y = zip(*most_common_nat)

x,y = list(x),list(y)



plt.figure(figsize=(15,10))

ax= sns.barplot(x=x, y=y,palette = sns.cubehelix_palette(len(x)))

plt.xticks(rotation= 90)

plt.xlabel('Nationalities')

plt.ylabel('Frequency')

plt.title('countries with footballer population')

print(most_common_nat)
data.columns
new_data = data[['Club','Release Clause']].copy()

new_data.dropna(inplace = True)

Club = list(new_data['Club'].unique())

Release = list(new_data['Release Clause'])

m = [ float(i[1:-1]) if 'M' == i[-1] else int(i[1:-1])/1000  for i in Release]

new_data['Release Clause'] = m

Release_Clause_list = []

for i in Club:

    x = new_data[new_data['Club'] == i]

    rate = sum(x['Release Clause'])/len(x['Release Clause'])

    Release_Clause_list.append(rate)



data_release = pd.DataFrame({'Club': Club,'Release_Clause_ratio': Release_Clause_list})

new_index = (data_release['Release_Clause_ratio'].sort_values(ascending = False)).index.values

sorted_data2 = data_release.reindex(new_index)



plt.figure(figsize = (15,10))

sns.barplot(x = sorted_data2['Club'].head(50), y = sorted_data2['Release_Clause_ratio'].head(50))

plt.xticks(rotation = 90)

plt.xlabel('Clubs')

plt.ylabel('Release Clause ratio')

plt.title("Average market value of players in clubs")
nat_ovr = data[['Nationality','Overall','Club']].copy()

nat_ovr.dropna(inplace=True)

club_list = list(nat_ovr['Club'].unique())[0:30]

most_common_nat = count_nat.most_common(5) # yukarida tanimli

n,f = zip(*most_common_nat) 

nat_list = list(n)



England = []

Germany = []

Spain = []

Argentina = []

France = []

nat_ovr_list = []



for i in nat_list:

    x = nat_ovr[nat_ovr['Nationality'] == i]

    for j in club_list:

        y = x[x['Club'] == j]

        if i == 'England' and not y.empty:

            England.append(sum(y['Overall'])/len(y['Overall']))

        elif y.empty:

            England.append(0)

        if i == 'Germany' and not y.empty:

            Germany.append(sum(y['Overall'])/len(y['Overall']))

        elif y.empty:

            Germany.append(0)

        if i == 'Spain' and not y.empty:

            Spain.append(sum(y['Overall'])/len(y['Overall']))

        elif y.empty:

            Spain.append(0)

        if i == 'Argentina' and not y.empty:

            Argentina.append(sum(y['Overall'])/len(y['Overall']))

        elif y.empty:

            Argentina.append(0)

        if i == 'France' and not y.empty:

            France.append(sum(y['Overall'])/len(y['Overall']))

        elif y.empty:

            France.append(0)


