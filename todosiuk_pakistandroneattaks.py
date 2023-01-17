# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
data = pd.read_csv('../input/PakistanDroneAttacksWithTemp Ver 10 (October 19, 2017).csv', encoding = 'cp1251')
# Any results you write to the current directory are saved as output.
data.drop(['S#','Comments','References', 'Special Mention (Site)'], 1, inplace = True)
data.columns = map(lambda x: x.replace(".", "").replace("_", ""), data.columns)
data.fillna(value = 0, inplace = True)
data = data[:-1]
data.describe()
max_killed = data['Total Died Mix'].sum()
min_killed = data['Total Died Min'].sum()
print ('Killed people for last 12 years (maximum count) ', max_killed)
print ('Killed people for last 12 years (minimum count)', min_killed)
attack_success = data.iloc[:, 6:8]
attack = attack_success[attack_success>0].count()
sum_attacks = attack.sum()
print('Attacks involved killing of actual terrorists from Al-Qaeeda and Taliban:', sum_attacks)

labels = 'Successful', 'Not successful'
sizes = [sum_attacks, 573]
explode = (0, 0.1)
fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')
plt.show()
data['Woman'] = data.iloc[:, 16:17]
woman_children_yes = (data[data.Woman == 'Y']).groupby(['Woman'])['Woman'].count()
woman_children_no = (data[data.Woman == 'N']).groupby(['Woman'])['Woman'].count()
woman_children_unknown = (data[data.Woman == 0]).groupby(['Woman'])['Woman'].count()
print('Attacks involved women and children (Y- yes, N - no, 0 - unknown)')
print(woman_children_yes)
print(woman_children_no)
print(woman_children_unknown)
from numpy import arange
data["Date"] = pd.to_datetime(data["Date"])
x = data["Date"]
y = np.array(data["No of Strike"])
%matplotlib inline
plt.figure(figsize=(25,15))

plt.plot(x,y)
plt.gcf().autofmt_xdate()
plt.show()
number_attacks_per_months = data[['Date','No of Strike']]
number_attacks_per_months['Date'] = number_attacks_per_months['Date'].dt.strftime('%B')
sum_number_attack = number_attacks_per_months.groupby(['Date'])['No of Strike'].sum()
sum_frame = sum_number_attack.to_frame()
sorted_sum = sum_frame.sort_values(by=['No of Strike'],  ascending=False)
print('Dependence between number of drone attacks and months (Date is the number of the month)')
print(sorted_sum)

df2 = pd.DataFrame(sum_frame['No of Strike'])
df2.plot(kind='bar')
Bush = data[(data['Date'] > '2000') & (data['Date'] <= '2009')]
Bush_attack = Bush['No of Strike'].sum()
Obama = data[(data['Date'] > '2009') & (data['Date'] <= '2017')]
Obama_attack = Obama['No of Strike'].sum()
print('Bush- the number of attacks: ', Bush_attack)
print('Obama- the number of attacks: ', Obama_attack)

labels = 'Bush', 'Obama'
sizes = [Bush_attack, Obama_attack]
explode = (0, 0.1)
fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()